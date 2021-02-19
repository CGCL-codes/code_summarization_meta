# -*- coding: utf-8 -*-

import sys

sys.path.append('.')

from src.trainer.summarization.xlang.finetune_trainer import FTTrainer
from src.trainer.summarization.xlang.maml_trainer import MAMLTrainer

__all__ = [
    'FTTrainer',
    'MAMLTrainer',
]
