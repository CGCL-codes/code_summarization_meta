# -*- coding: utf-8 -*-
import sys

sys.path.append('')

from run.util import *
from src.metric.base import *
from src.model.summarization.unilang import *
from src.trainer.summarization.unilang import *
from src.trainer.summarization.xlang import *
from src.metric.summarization.loss import PGCriterion_REINFORCE
from src.eval import *
from src.utils.constants import METRICS
from tabulate import tabulate


def main():
    args = get_args()
    args.yaml = os.path.join(os.path.dirname(__file__), 'config', args.yaml)
    LOGGER.info(args)

    # for debug
    # Argues = namedtuple('Argues', 'yaml task lang_mode method_name train_mode dataset_type multi_processing')

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_ppjj_r.log 2>&1 &
    # args = Argues('config.yml', 'summarization', 'xlang', 'maml', 'train_maml', 'all', True)  # train_sl

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_ppjj_ft_p0.log 2>&1 &
    # args = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'train_maml_ft', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/zero_shot_test.log 2>&1 &
    # args = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'train_select', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/case_study.log 2>&1 &
    # args = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'case_study', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_ppjj_sc_p0.log 2>&1 &
    # args = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'train_maml_sc', 'target',
    #               True)  # finetune

    # nohup python -u ./run/summarization/xlang/maml/main.py > ./run/summarization/xlang/maml/summarization_maml_test.log 2>&1 &
    # args = Argues('./ruby/tok8path-p1.0.yml', 'summarization', 'xlang', 'maml', 'test', 'target',
    #               True)  # test
    LOGGER.info(args)

    config, dataset, = load_config_dataset(args, XlangDataloader, sBaseDataset, sbase_collate_fn, )
    # config, dataset, = load_config_dataset(args, XlangMAMLDataloader, sBaseMAMLDataset, sbase_collate_fn, )
    LOGGER.info(config)
    model = build_model(config, MM2Seq(config))
    if (config['common']['init_weights'] is not None) and os.path.exists(config['common']['init_weights']):
        LOGGER.info('loading model parameters from {}'.format(config['common']['init_weights']))
        checkpoints = load_checkpoints(config['common']['init_weights'])
        if 'epoch' in checkpoints:
            model.load_state_dict(checkpoints['model'])
            checkpoints.pop('model')  # oom
            epoch = checkpoints['epoch'] + 1
        else:
            model.load_state_dict(checkpoints)
            del checkpoints
    else:
        epoch = 1

    trg_lng = config['dataset']['target']['dataset_lng'][0]  # ruby

    if args.train_mode == 'train_maml':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim'])(model.parameters(), config['sl']['lr'])
        meta_optimizer = getattr(torch.optim, config['maml']['meta_optim']) \
            (model.parameters(), config['maml']['meta_lr'])
        if (config['common']['init_weights'] is not None) and os.path.exists(config['common']['init_weights']):
            optimizer.load_state_dict(checkpoints['optimizer'])
            meta_optimizer.load_state_dict(checkpoints['meta_optimizer'])

        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                'maml_{}_{}_{}({})-{}({})_spt{}_qry{}'.format(
                                    '-'.join(sorted(config['dataset']['source']['dataset_lng'])),
                                    '-'.join(sorted(config['dataset']['target']['dataset_lng'])),
                                    config['sl']['optim'], config['sl']['lr'],
                                    config['maml']['meta_optim'], config['maml']['meta_lr'],
                                    config['maml']['support_size'], config['maml']['query_size'],
                                ),
                                args.train_mode)

        # save_dir = None
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.debug('save_dir: {}'.format(save_dir))
        maml_trainer = MAMLTrainer(config)
        maml_trainer.meta_train(model, dataset, lm_criterion, optimizer, meta_optimizer, SAVE_DIR=save_dir,
                                start_epoch=epoch)
        # maml_trainer.biased_meta_train(model, dataset, lm_criterion, optimizer, meta_optimizer, SAVE_DIR=save_dir,
        #                                start_epoch=epoch)

    elif args.train_mode == 'train_maml_ft':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sl']['optim'])(model.parameters(), config['sl']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'],
                                'maml_ft_{}_{}({})'.format(config['dataset']['portion'], config['sl']['optim'],
                                                           config['sl']['lr']))
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.debug('save_dir: {}'.format(save_dir))

        ft_trainer = FTTrainer(config)
        best_model = ft_trainer.finetune(model, dataset['target'][trg_lng], lm_criterion, optimizer,
                                         SAVE_DIR=save_dir)
        LOGGER.info(best_model)

    elif args.train_mode == 'test':
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        trg_lng = config['dataset']['target']['dataset_lng'][0]
        unilang_dataset = dataset['target'][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        # bleu1, bleu2, _, _, pycoco_bleu, meteor, pycoco_meteor, rouge1, rouge2, _, _, rougel, pycoco_rouge, \
        # cider = \
        #     Evaluator.summarization_eval(model, unilang_dataset['valid'], dataset.token_dicts, lm_criterion,
        #                                  metrics=['bleu', 'meteor', 'rouge', 'cider'])
        # headers = ['B1', 'B2', 'PycocoB4', 'Meteor', 'PycocoMeteor', 'R1', 'R2', 'RL', 'PycocoRL', 'Cider']
        # result_table = [[round(i, 4) for i in [bleu1, bleu2, pycoco_bleu, meteor, pycoco_meteor,
        #                                        rouge1, rouge2, rougel, pycoco_rouge, cider]]]
        # LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
        #                                                       tablefmt=config['common']['result_table_format'])))

        bleu1, bleu2, _, _, pycoco_bleu, meteor, pycoco_meteor, rouge1, rouge2, _, _, rougel, pycoco_rouge, \
        cider = \
            Evaluator.summarization_eval(model, unilang_dataset['test'], dataset.token_dicts, lm_criterion,
                                         metrics=['bleu', 'meteor', 'rouge', 'cider'])
        headers = ['B1', 'B2', 'PycocoB4', 'Meteor', 'PycocoMeteor', 'R1', 'R2', 'RL', 'PycocoRL', 'Cider']
        result_table = [[round(i, 4) for i in [bleu1, bleu2, pycoco_bleu, meteor, pycoco_meteor,
                                               rouge1, rouge2, rougel, pycoco_rouge, cider]]]
        LOGGER.info('Evaluation results:\n{}'.format(tabulate(result_table, headers=headers,
                                                              tablefmt=config['common']['result_table_format'])))


    elif args.train_mode == 'train_maml_sc':
        pg_criterion = PGCriterion_REINFORCE().cuda()  # TODO: to optimized like LMLoss
        lm_criterion = LMLoss(device=config['common']['device'] is not None, )
        optimizer = getattr(torch.optim, config['sc']['optim']) \
            (model.parameters(), config['sc']['lr'])
        save_dir = os.path.join(config['dataset']['save_dir'], model.__class__.__name__.lower(),
                                '-'.join(config['dataset']['source']['dataset_lng']) +
                                "_p{}_bi{}".format(config['dataset']['portion'],
                                                   config['training']['rnn_bidirectional']), args.train_mode)
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info('save_dir: {}'.format(save_dir))

        trg_lng = config['dataset']['target']['dataset_lng'][0]
        unilang_dataset = dataset['target'][trg_lng]

        sc_trainer = SCTrainer(config)
        sc_trainer.train(model, unilang_dataset, lm_criterion, pg_criterion, optimizer,
                         config['sc']['reward_func'], SAVE_DIR=save_dir, )


    elif args.train_mode == 'case_study':
        trg_lng = config['dataset']['target']['dataset_lng'][0]
        unilang_dataset = dataset['target'][trg_lng]
        LOGGER.info('evaluator on {} test dataset'.format(trg_lng))
        model_filename = config['common']['init_weights']
        Evaluator.case_study_eval(model, unilang_dataset['test'], dataset.token_dicts, model_filename=model_filename)


    else:
        raise NotImplementedError('No such train mode')


if __name__ == '__main__':
    main()
