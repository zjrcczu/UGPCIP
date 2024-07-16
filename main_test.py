from utils.utils import load_yaml_config
from trainer.trainer import Trainer
import os.path as path

import torch
import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    torch.cuda.empty_cache()
    os.environ['PYTHONHASHSEED'] = str(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    cfg = load_yaml_config('config/cfg.yaml')
    dataset_cfg = load_yaml_config(path.join('config', cfg.dataset + '.yaml'))
    cfg.update(dataset_cfg)
    print(cfg)

    trainer = Trainer(cfg)
    dir = os.listdir("log/" + cfg.dataset + "/")
    print(dir[-1])
    loc = "log/" + cfg.dataset + "/" + dir[-1] + "/best_model.pth"
    test_loss, acc, auc, f1, precision, recall = trainer.test(pretrained_model=loc, single=True)
    print('test_acc: {:.4f}, test_auc: {:.4f}, test_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}'.format(acc, auc, f1, precision, recall))