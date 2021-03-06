# -*- coding: utf-8 -*-
import sys

sys.path.append('.')

from src import *
from src.data import *

import ujson

from joblib import Parallel, delayed
from multiprocessing import cpu_count
from src.dataset.base import *
# from src.utils.util import mpool
from src.utils.util_file import *
from src.utils.util import *
from src.utils.util_data import *
from src.dataset.unilang_dataloader import UnilangDataloader


def get_tree_max_pad_len(dst_dir: str, xlang: List[str], ) -> int:
    file_name = os.path.join(dst_dir, 'info.txt')
    while True:
        try:
            with open(file_name, 'r') as reader:
                sub_token_dict = ujson.loads(reader.read())
                max_sub_token_len = 0
                for lng in xlang:
                    max_sub_token_len = max(max_sub_token_len, sub_token_dict[lng])
                return max_sub_token_len
        except:
            if os.path.exists(file_name):
                LOGGER.info('other program is reading file: {}, waiting...'.format(file_name))
                continue
            else:
                raise Exception('No such file: {}'.format(file_name))


class XlangMAMLDataloader:
    __slots__ = ('src_datasets', 'trg_datasets', 'token_dicts',)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._construct(*args, **kwargs)

    def _construct(self,
                   # base parameters
                   base_dataset: Union[sBaseDataset],
                   collate_fn: Union[sbase_collate_fn],
                   params: Dict,

                   # loader parameters
                   src_dataset_info: Union[None, Dict], trg_dataset_info: Union[None, Dict],
                   batch_size: int,
                   thread_num: int,
                   # src_xlang=None,  # for dtrl, None is not to pad ast sub-token node
                   ) -> None:
        LOGGER.info("XlangDataloader TokenDicts begin ")
        self.token_dicts = TokenDicts(params.get('token_dicts'))
        LOGGER.info("XlangDataloader TokenDicts finished")
        params['token_dicts'] = self.token_dicts

        if 'src_xlang' in params:
            # load max pad sub-token len
            if params['src_xlang'] is None:
                max_sub_token_len = None
            else:
                max_sub_token_len = get_tree_max_pad_len(params['file_dir'], params['src_xlang'])
            params['pad_ast_sub_node'] = max_sub_token_len
            params.pop('src_xlang')

        # source dataset
        if src_dataset_info is not None:
            # TimeAnalyzer(self._load_dataset)(
            #     base_dataset, collate_fn, file_dir, code_modalities, src_dataset_info,
            #     batch_size, token_dicts, thread_num,
            #     portion, merge_xlng, leaf_path_k,
            # )
            # TimeAnalyzer.print_stats()

            src_params = deepcopy(params)
            if trg_dataset_info is not None:
                src_params['portion'] = 1.0

            LOGGER.info("load src_datasets")
            self.src_datasets = self._load_dataset(
                base_dataset, collate_fn, src_params,
                src_dataset_info, batch_size, thread_num,
            )
        else:
            self.src_datasets = None
        # target dataset
        if trg_dataset_info is not None:
            LOGGER.info("load trg_datasets")

            tgt_params = params
            self.trg_datasets = self._load_dataset(
                base_dataset, collate_fn, tgt_params,
                trg_dataset_info, params.get('tgt_batch_size'), thread_num,
            )
        else:
            self.trg_datasets = None

    def _load_dataset(self,
                      base_dataset: Union[sBaseDataset],
                      collate_fn: Union[sbase_collate_fn],
                      params: Dict,

                      # parameters
                      dataset_info: Dict,
                      batch_size: int, thread_num: int,
                      ) -> Dict:
        data_lngs, modes = dataset_info['dataset_lng'], dataset_info['mode']
        dataset_dict = {}

        # 53.6876
        LOGGER.debug('read xlang dataset loader')
        LOGGER.info('read xlang dataset loader')
        paralleler = Parallel(len(data_lngs))
        datasets = paralleler(delayed(UnilangDataloader)
                              (base_dataset, collate_fn, dict(params, **{'data_lng': lng}), batch_size, modes,
                               thread_num, )
                              for lng in data_lngs)
        for lng, ds in zip(data_lngs, datasets):
            assert lng == ds.lng
            dataset_dict[lng] = ds

        # # 116.549
        # for lng in data_lngs:
        #     dataset_dict[lng] = UnilangDataloader(
        #         base_dataset, collate_fn, dict(params, **{'data_lng': lng}),
        #         batch_size, modes, thread_num,
        #     )
        LOGGER.info("_load_dataset finished")
        return dataset_dict

    def __getitem__(self, key: str) -> Any:
        if key in ['src', 'source', ]:
            return self.src_datasets
        elif key in ['trg', 'target', ]:
            return self.trg_datasets
        else:
            raise NotImplementedError

    @property
    def size(self):
        return {
            'src': {lng: unilng_dataset.size for lng, unilng_dataset in self.src_datasets.items()},
            'trg': {lng: unilng_dataset.size for lng, unilng_dataset in self.trg_datasets.items()},
        }

    def __str__(self):
        return 'TLDataloader(\nsrc dataset: {}\ntrg dataset: {}\ntoken_dicts: {}\n)'.format(
            self.src_datasets, self.trg_datasets, self.token_dicts
        )

    def __repr__(self):
        return str(self)
