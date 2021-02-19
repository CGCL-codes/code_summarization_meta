# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.abspath('.'))

import copy
import numpy as np

from eval.summarization.bleu.bleu import Bleu
from eval.summarization.cider.cider import Cider
from eval.summarization.meteor.meteor import Meteor
from eval.summarization.rouge.rouge import Rouge

from eval.pycocoevalcap.bleu import corpus_bleu as PycocoBleu
from eval.pycocoevalcap.rouge import Rouge as PycocoRouge
from eval.pycocoevalcap.meteor import Meteor as PycocoMeteor

from src.utils.util import *

from src.data.token_dicts import TokenDicts
import ujson

from src.log.log import get_logger

LOGGER = get_logger()


def dump_preds(src_comments: List, trg_comments: List, pred_comments: List,
               src_code_all: List, pred_filename: str, ) -> None:
    # write
    with open(pred_filename, 'w', encoding='utf-8') as f:
        for i, (src_cmt, pred_cmt, trg_cmt, src_code,) in \
            enumerate(zip(src_comments, pred_comments, trg_comments, src_code_all, )):
            f.write('=============================> {}\n'.format(i))
            # code
            f.write('[src code]\n{}\n'.format(src_code))
            # f.write('[trg code]\n{}\n'.format(' '.join(trg_code)))
            # comment
            f.write('[src cmnt]\n{}\n'.format(src_cmt))
            f.write('[pre cmnt]\n{}\n'.format(' '.join(pred_cmt)))
            f.write('[trg cmnt]\n{}\n'.format(' '.join(trg_cmt)))
            f.write('\n\n')
    LOGGER.info("Write source/predict/target comments into {}, size: {}".format(pred_filename, len(src_comments)))


def eval_metrics(src_comments: List, trg_comments: List, pred_comments: List,
                 src_code_all: List, oov_vocab: List,
                 token_dicts: TokenDicts, pred_filename=None, metrics=METRICS, ) -> Tuple:
    preds, trgs = {}, {}
    srcs_return, trgs_return, preds_return = {}, {}, {}
    new_pred_comments = [None] * len(pred_comments)

    for i, (src, trg, pred,) in enumerate(zip(src_comments, trg_comments, pred_comments, )):
        pred = clean_up_sentence(pred, remove_EOS=True)
        if oov_vocab is not None:
            pred = indices_to_words(pred, token_dicts['comment'], oov_vocab[i])
        else:
            pred = token_dicts['comment'].to_labels(pred, EOS_WORD)
        new_pred_comments[i] = pred

        preds_return[i] = copy.deepcopy(pred)
        trgs_return[i] = copy.deepcopy(trg)
        srcs_return[i] = src

        hypo = ' '.join(pred)
        preds[i] = [hypo if len(hypo) > 0 else '0']
        trgs[i] = [' '.join(trg)]

    # eval score
    if 'bleu' in metrics:
        _, bleu = Bleu(4).compute_score(trgs, preds)
        bleu1, bleu2, bleu3, bleu4, = bleu
        _, pycoco_bleu, _ = PycocoBleu(preds, trgs)
        pycoco_bleu = [pycoco_bleu]
    else:
        bleu1, bleu2, bleu3, bleu4, pycoco_bleu = \
            [0.] * len(src_comments), [0.] * len(src_comments), [0.] * len(src_comments), [0.] * len(src_comments), [0.]

    if 'meteor' in metrics:
        _, meteor = Meteor().compute_score(trgs, preds)

        meteor_calculator = PycocoMeteor()
        pycoco_meteor, _ = meteor_calculator.compute_score(trgs, preds)
        pycoco_meteor = [pycoco_meteor]
    else:
        meteor, pycoco_meteor = [0.0] * len(src_comments), [0.]

    if 'rouge' in metrics:
        rouge, _ = Rouge().compute_score(trgs, preds)  #
        rouge1, rouge2, rouge3, rouge4, rougel, _, _, _ = [[i] for i in rouge]

        rouge_calculator = PycocoRouge()
        pycoco_rouge, _ = rouge_calculator.compute_score(trgs, preds)
        pycoco_rouge = [pycoco_rouge]
    else:
        rouge1, rouge2, rouge3, rouge4, rougel, pycoco_rouge = \
            [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments), \
            [0.0] * len(src_comments), [0.]
    if 'cider' in metrics:
        _, cider = Cider().compute_score(trgs, preds)
    else:
        cider = [0.0] * len(src_comments)

    if pred_filename is not None:
        dump_preds(src_comments, trg_comments, new_pred_comments, src_code_all, pred_filename, )
    else:
        pass

    return bleu1, bleu2, bleu3, bleu4, pycoco_bleu, \
           meteor, pycoco_meteor, \
           rouge1, rouge2, rouge3, rouge4, rougel, pycoco_rouge, \
           cider, \
           srcs_return, trgs_return, preds_return,


def dump_all(src_comments, trg_comments, new_pred_comments, src_code_all,
             tok_len, comment_len, len_info,
             bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider,
             pred_filename, ):
    with open(pred_filename, 'w') as writer:
        for ind in range(len(src_comments)):
            tmp = {
                'src_comment': src_comments[ind],
                'trg_comment': trg_comments[ind],
                'pred_comment': new_pred_comments[ind],
                'src_code': src_code_all[ind],

                'tok_len': tok_len[ind],
                'comment_len': comment_len[ind],
                # 'ast_len': ast_len[ind],

                'bleu1': bleu1[ind],
                'bleu2': bleu2[ind],
                'bleu3': bleu3[ind],
                'bleu4': bleu4[ind],
                'meteor': meteor[ind],
                'rouge1': rouge1[ind],
                'rouge2': rouge2[ind],
                'rouge3': rouge3[ind],
                'rouge4': rouge4[ind],
                'rougel': rougel[ind],
                'cider': cider[ind],
            }
            for k, v in len_info.items():
                tmp[k + "_len"] = v[ind]
            code_snippet = ujson.dumps(tmp)
            writer.write(code_snippet + '\n')


def dump_all_ori_metric(src_comments, trg_comments, new_pred_comments, src_code_all,
                        tok_len, comment_len, len_info,
                        bleu1, bleu2, bleu3, bleu4, meteor, rougel, cider,
                        pred_filename, ):
    with open(pred_filename, 'w') as writer:
        for ind in range(len(src_comments)):
            tmp = {
                'src_comment': src_comments[ind],
                'trg_comment': trg_comments[ind],
                'pred_comment': new_pred_comments[ind],
                'src_code': src_code_all[ind],

                'tok_len': tok_len[ind],
                'comment_len': comment_len[ind],
                # 'ast_len': ast_len[ind],

                'bleu1': bleu1[ind],
                'bleu2': bleu2[ind],
                'bleu3': bleu3[ind],
                'bleu4': bleu4[ind],
                'meteor': meteor[ind],
                'rougel': rougel[ind],
                'cider': cider[ind],
            }
            for k, v in len_info.items():
                tmp[k + "_len"] = v[ind]
            code_snippet = ujson.dumps(tmp)
            writer.write(code_snippet + '\n')


def eval_per_metrics(src_comments: List, trg_comments: List, pred_comments: List,
                     src_code_all: List, oov_vocab: List, tok_len: List, comment_len: List, len_info: List,
                     token_dicts: TokenDicts, pred_filename=None, metrics=METRICS, ):
    # because calculate rouge one by one is too slow, so we only consider cider
    assert pred_filename is not None

    preds, trgs = {}, {}
    srcs_return, trgs_return, preds_return = {}, {}, {}
    # rouge1, rouge2, rouge3, rouge4, rougel = [], [], [], [], []
    new_pred_comments = [None] * len(pred_comments)

    for i, (src, trg, pred,) in enumerate(zip(src_comments, trg_comments, pred_comments, )):
        pred = clean_up_sentence(pred, remove_EOS=True)
        if oov_vocab is not None:
            pred = indices_to_words(pred, token_dicts['comment'], oov_vocab[i])
        else:
            pred = token_dicts['comment'].to_labels(pred, EOS_WORD)
        new_pred_comments[i] = pred

        preds_return[i] = copy.deepcopy(pred)
        trgs_return[i] = copy.deepcopy(trg)
        srcs_return[i] = src

        preds[i] = [' '.join(pred)]
        trgs[i] = [' '.join(trg)]

    # eval score
    if 'bleu' in metrics:
        _, bleu = Bleu(4).compute_score(trgs, preds)
        bleu1, bleu2, bleu3, bleu4, = bleu
        # print('bleu1-: ', bleu1)
    else:
        bleu1, bleu2, bleu3, bleu4 = \
            [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments), [0.0] * len(src_comments)
    if 'meteor' in metrics:
        _, meteor = Meteor().compute_score(trgs, preds)
    else:
        meteor = [0.0] * len(src_comments)
    if 'rouge' in metrics:
        # print('rouge-trgs: ', trgs)
        # print('rouge-preds: ', preds)
        rouge1, rouge2, rouge3, rouge4, rougel = [], [], [], [], []
        for ind in range(len(trgs)):
            rouge, _ = Rouge().compute_score({ind: trgs[ind]}, {ind: preds[ind]}, )  #
            # assert False
            # print('rouge: ', rouge)
            # print('_: ', _)
            _rouge1, _rouge2, _rouge3, _rouge4, _rougel, _, _, _ = [[i] for i in rouge]
            # print('rouge1-: ', rouge1)
            rouge1.extend(_rouge1)
            rouge2.extend(_rouge2)
            rouge3.extend(_rouge3)
            rouge4.extend(_rouge4)
            rougel.extend(_rougel)
    else:
        rouge1, rouge2, rouge3, rouge4, rougel = [0.0] * len(src_comments), [0.0] * len(src_comments), \
                                                 [0.0] * len(src_comments), [0.0] * len(src_comments), \
                                                 [0.0] * len(src_comments),
    if 'cider' in metrics:
        _, cider = Cider().compute_score(trgs, preds)
    else:
        cider = [0.0] * len(src_comments)

    dump_all(src_comments, trg_comments, new_pred_comments, src_code_all,
             tok_len, comment_len, len_info,
             bleu1, bleu2, bleu3, bleu4, meteor, rouge1, rouge2, rouge3, rouge4, rougel, cider,
             pred_filename, )
