# -*- coding: utf-8 -*-

import sys

sys.path.append('./')

from src import *
from src.trainer import *
from src.model import *
from src.model.template import *
from src.dataset import *
from src.metric import *
from src.utils.util_data import batch_to_cuda
from src.utils.util_eval import *
from src.eval import *
from src.utils.util_optimizer import create_scheduler
from src.trainer.summarization.unilang import SLTrainer
from src.metric.base import *
from run.util import *

import torch
import random


def gradient_accumulation(optimizer, train_step):
    for param_group in optimizer.param_groups:
        for p in param_group['params']:
            if p.grad is not None:
                p.grad.data.mul_(1.0 / train_step)
    optimizer.step()


class MAMLTrainer(Trainer):
    '''
    model-agnostic meta learning
    '''

    def __init__(self, config: Dict, ) -> None:
        super(MAMLTrainer, self).__init__(config)
        self.sl_trainer = SLTrainer(config)
        self.min_meta_loss = {lng: float('inf') for lng in self.config['dataset']['source']['dataset_lng']}
        self.best_performance = 0

    def support_train(self, support_bsz: int, model: IModel, task_train_iter: Iterator,
                      criterion: BaseLoss, meta_optimizer: Optimizer, ):
        for _ in range(support_bsz):
            meta_train_batch = next(task_train_iter)
            if model.config['common']['device'] is not None:
                meta_train_batch = batch_to_cuda(meta_train_batch)
            comment_loss = model.train_sl(meta_train_batch, criterion)
            meta_optimizer.zero_grad()
            comment_loss.backward()
        gradient_accumulation(meta_optimizer, support_bsz)

    def query_eval(self, query_bsz: int, model: IModel, task_train_iter: Iterator,
                   criterion: BaseLoss, ):
        meta_val_batch = next(task_train_iter)
        if model.config['common']['device'] is not None:
            meta_val_batch = batch_to_cuda(meta_val_batch)
        query_loss = model.train_sl(meta_val_batch, criterion)

        # query_loss = 0.0
        with torch.no_grad():
            loss_acc = 0.0
            for _ in range(query_bsz - 1):
                meta_val_batch = next(task_train_iter)
                if model.config['common']['device'] is not None:
                    meta_val_batch = batch_to_cuda(meta_val_batch)
                meta_val_loss = model.train_sl(meta_val_batch, criterion)
                # query_loss = query_loss + meta_val_loss
                loss_acc = loss_acc + meta_val_loss.item()
        query_loss.data.add_(float(loss_acc)).div_(query_bsz)
        return query_loss

    def _meta_train(self, model: IModel, dataset: XlangDataloader, criterion: BaseLoss, meta_optimizer: Optimizer,
                    gradient_steps: int = 1):
        """
        sample self.config['maml']['support_size'], self.config['maml']['query_size'] of dataset['source'][task_name]['train']
            as support dataset and query dataset for meta train
        """
        ori_weights = deepcopy(model.state_dict())
        sample_tasks = random.sample(dataset['source'].keys(), self.config['maml']['sample_task_num'])
        meta_loss_cache = {task_name: 0.0 for task_name in sample_tasks}

        # first task, store model graph of 1st task
        task_name = sample_tasks[0]
        # sample support dataset for task
        support_bsz = int(self.config['maml']['support_size'] * len(dataset['source'][task_name]['train']))
        support_bsz = max(support_bsz, 1)
        for _ in range(gradient_steps):
            task_train_iter = iter(dataset['source'][task_name]['train'])
            self.support_train(support_bsz, model, task_train_iter, criterion, meta_optimizer)

        # sample query dataset for task
        query_bsz = int(self.config['maml']['query_size'] * len(dataset['source'][task_name]['train']))
        query_bsz = max(query_bsz, 1)
        total_loss = self.query_eval(query_bsz, model, task_train_iter, criterion)
        meta_loss_cache[task_name] = round(total_loss.item(), 4)

        # add query losses of other tasks into 1st graph such that we can save some memory
        for task_idx in range(1, self.config['maml']['sample_task_num']):
            task_name = sample_tasks[task_idx]
            task_train_iter = iter(dataset['source'][task_name]['train'])
            # sample support dataset for task
            support_bsz = int(self.config['maml']['support_size'] * len(dataset['source'][task_name]['train']))
            support_bsz = max(support_bsz, 1)
            self.support_train(support_bsz, model, task_train_iter, criterion, meta_optimizer)
            # sample query dataset for task
            query_bsz = int(self.config['maml']['query_size'] * len(dataset['source'][task_name]['train']))
            query_bsz = max(query_bsz, 1)
            with torch.no_grad():
                qry_loss = self.query_eval(query_bsz, model, task_train_iter, criterion)
            # accumulate meta loss
            total_loss.data.add_(float(qry_loss.item()))
            meta_loss_cache[task_name] = round(qry_loss.item(), 4)
            # restore model state
            model.load_state_dict(ori_weights)

        total_loss.data.div_(self.config['maml']['sample_task_num'])
        return total_loss, meta_loss_cache

    def meta_train(self, model: IModel, dataset: XlangDataloader, criterion: BaseLoss,
                   optimizer: Optimizer, meta_optimizer: Optimizer,
                   SAVE_DIR=None, start_time=None, start_epoch=None, end_epoch=None,
                   ):
        super().train()
        start_time = time.time() if start_time is None else start_time
        # trg_lng = self.config['dataset']['target']['dataset_lng'][0]

        if start_epoch == None:
            start_epoch = 1
        if end_epoch == None:
            end_epoch = 1 + self.config['training']['train_epoch']

        for epoch in range(start_epoch, end_epoch):
            model.train()
            # meta train
            # inner gradient update steps
            # if epoch <= 500:
            #     gradient_steps = 1
            # else:
            #     gradient_steps = 2
            gradient_steps = 1
            total_loss, meta_loss_cache = self._meta_train(model, dataset, criterion, meta_optimizer,
                                                           gradient_steps=gradient_steps)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            LOGGER.info('Epoch: {:0>3d}/{:0>3d}, avg_loss: {:.2f}; meta loss: {}; time: {}'. \
                        format(epoch, self.config['training']['train_epoch'], total_loss.item(),
                               ",".join(["{}({:.2f})".format(task_name, meta_loss_cache[task_name])
                                         for task_name in sorted(meta_loss_cache.keys())]),
                               str(datetime.timedelta(seconds=int(time.time() - start_time))))
                        )

            # min_meta_loss_sum = sum([self.min_meta_loss[task_name] for task_name in meta_loss_cache.keys()])
            # cur_meta_loss_sum = sum(meta_loss_cache.values())
            # if (cur_meta_loss_sum <= min_meta_loss_sum) and (epoch % self.config['dataset']['save_iterval']) == 0:
            if (epoch % self.config['dataset']['save_iterval']) == 0:
                # record min meta loss
                for task_name, task_meta_loss in meta_loss_cache.items():
                    self.min_meta_loss[task_name] = task_meta_loss
                # compute valid loss on source domain
                src_valid_peformance = self.src_valid_peformance(model, dataset, criterion)
                avg_src_valid_peformance = sum(src_valid_peformance.values()) / len(src_valid_peformance)

                if self.best_performance <= avg_src_valid_peformance:
                    self.best_performance = avg_src_valid_peformance
                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, avg_src_performance: {:.2f}; src_performance: {}; time: {}'. \
                                format(epoch, self.config['training']['train_epoch'], avg_src_valid_peformance,
                                       ",".join(["{}({:.2f})".format(task_name, src_valid_peformance[task_name])
                                                 for task_name in sorted(src_valid_peformance.keys())]),
                                       str(datetime.timedelta(seconds=int(time.time() - start_time))))
                                )
                    # to save model checkpoint
                    model_name = '{}-bs{}-{}({})-m{}({})-EPOCH{}-{}-{}'.format(
                        '8'.join(self.config['training']['code_modalities']),
                        self.config['training']['batch_size'],
                        self.config['sl']['optim'], self.config['sl']['lr'],
                        self.config['maml']['meta_optim'], self.config['maml']['meta_lr'],
                        self.config['maml']['support_size'], self.config['maml']['query_size'], epoch)
                    model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name))
                    save_checkpoints(optimizer, meta_optimizer, model, epoch, model_path)
                    LOGGER.info('Dumping model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))

    def src_valid_peformance(self, model: IModel, dataset: XlangDataloader, criterion: BaseLoss) -> Dict:
        with torch.no_grad():
            model.eval()
            valid_peformance = {}
            for lang in dataset['source'].keys():
                bleu1 = Evaluator.summarization_eval(model, dataset['src'][lang]['valid'], dataset.token_dicts,
                                                     criterion, metrics=['bleu'])[0]
                valid_peformance[lang] = round(bleu1 * 100, 2)
            return valid_peformance

    def biased_meta_train(self, model: IModel, dataset: XlangDataloader, criterion: BaseLoss,
                          optimizer: Optimizer, meta_optimizer: Optimizer,
                          SAVE_DIR=None, start_time=None, start_epoch=None, end_epoch=None,
                          ):
        super().train()
        start_time = time.time() if start_time is None else start_time

        if start_epoch == None:
            start_epoch = 1
        if end_epoch == None:
            end_epoch = 1 + self.config['training']['train_epoch']

        for epoch in range(start_epoch, end_epoch):
            model.train()
            # meta train
            total_loss, meta_loss_cache = self._meta_train(model, dataset, criterion, meta_optimizer)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            LOGGER.info('Epoch: {:0>3d}/{:0>3d}, avg_loss: {:.2f}; meta loss: {}; time: {}'. \
                        format(epoch, self.config['training']['train_epoch'], total_loss.item(),
                               ",".join(["{}({:.2f})".format(task_name, meta_loss_cache[task_name])
                                         for task_name in sorted(meta_loss_cache.keys())]),
                               str(datetime.timedelta(seconds=int(time.time() - start_time))))
                        )

            if (epoch % self.config['dataset']['save_iterval']) == 0:
                # record min meta loss
                for task_name, task_meta_loss in meta_loss_cache.items():
                    self.min_meta_loss[task_name] = task_meta_loss

                # compute valid loss on source domain
                src_valid_peformance = self.src_valid_peformance(model, dataset, criterion)
                avg_src_valid_peformance = sum(src_valid_peformance.values()) / len(src_valid_peformance)

                if self.best_performance <= avg_src_valid_peformance:
                    self.best_performance = avg_src_valid_peformance

                    # compute valid loss on source domain
                    tgt_valid_peformance = self.tgt_valid_peformance(model, dataset, criterion)
                    avg_tgt_valid_peformance = sum(tgt_valid_peformance.values()) / len(tgt_valid_peformance)

                    LOGGER.info('Epoch: {:0>3d}/{:0>3d}, avg_tgt_performance: {:.2f}; tgt_performance: {}; time: {}'. \
                                format(epoch, self.config['training']['train_epoch'], avg_tgt_valid_peformance,
                                       ",".join(["{}({:.2f})".format(task_name, tgt_valid_peformance[task_name])
                                                 for task_name in sorted(tgt_valid_peformance.keys())]),
                                       str(datetime.timedelta(seconds=int(time.time() - start_time))))
                                )
                    # to save model checkpoint
                    model_name = '{}-bs{}-{}({})-m{}({})-EPOCH{}-{}-{}'.format(
                        '8'.join(self.config['training']['code_modalities']),
                        self.config['training']['batch_size'],
                        self.config['sl']['optim'], self.config['sl']['lr'],
                        self.config['maml']['meta_optim'], self.config['maml']['meta_lr'],
                        self.config['maml']['support_size'], self.config['maml']['query_size'], epoch)
                    model_path = os.path.join(SAVE_DIR, '{}.pt'.format(model_name))
                    save_checkpoints(optimizer, meta_optimizer, model, epoch, model_path)
                    LOGGER.info('Dumping model in {}'.format(model_path))

        LOGGER.info('{} train end'.format(self))

    def tgt_valid_peformance(self, model: IModel, dataset: XlangDataloader, criterion: BaseLoss) -> Dict:
        with torch.no_grad():
            model.eval()
            trg_lng = self.config['dataset']['target']['dataset_lng'][0]
            bleu1 = Evaluator.summarization_eval(model, dataset['target'][trg_lng]['valid'], dataset.token_dicts,
                                                 criterion, metrics=['bleu'])[0]
            valid_peformance = {trg_lng: round(bleu1 * 100, 2)}
            return valid_peformance
