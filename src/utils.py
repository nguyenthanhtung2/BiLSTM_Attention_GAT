import os
from typing import Tuple, Dict
import torch
from torch import nn, optim
import argparse
import yaml
from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    model_name: str,
    optimizer: optim.Optimizer,
    dataset_name: str,
    word_map: Dict[str, int],
    checkpoint_path: str,
    checkpoint_basename: str = 'checkpoint'
) -> None:

    state = {
        'epoch': epoch,
        'model': model,
        'model_name': model_name,
        'optimizer': optimizer,
        'dataset_name': dataset_name,
        'word_map': word_map
    }
    save_path = os.path.join(checkpoint_path, checkpoint_basename + '.pth.tar')
    torch.save(state, save_path)

def load_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[nn.Module, str, optim.Optimizer, str, Dict[str, int], int]:

    checkpoint = torch.load(checkpoint_path, map_location=str(device))

    model = checkpoint['model']
    model_name = checkpoint['model_name']
    optimizer = checkpoint['optimizer']
    dataset_name = checkpoint['dataset_name']
    word_map = checkpoint['word_map']
    start_epoch = checkpoint['epoch'] + 1

    return model, model_name, optimizer, dataset_name, word_map, start_epoch

def clip_gradient(optimizer: optim.Optimizer, grad_clip: float) -> None:

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class AverageMeter:

    def __init__(self, tag = None, writer = None):
        self.writer = writer
        self.tag = tag
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        # tensorboard
        if self.writer is not None:
            self.writer.add_scalar(self.tag, val)

def adjust_learning_rate(optimizer: optim.Optimizer, scale_factor: float) -> None:

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

class Config:
    """Convert a ``dict`` into a ``Class``"""
    def __init__(self, entries: dict = {}):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Config(v)
            else:
                self.__dict__[k] = v

def load_config(file_path: str) -> dict:

    f = open(file_path, 'r', encoding = 'utf-8')
    config = yaml.load(f.read(), Loader = yaml.FullLoader)
    return config

class TensorboardWriter:
    def __init__(self, log_dir, tensorboard=True):
        if tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

    def add_scalar(self, tag, scalar_value, global_step=None):
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag, img_tensor, global_step=None, dataformats='CHW'):
        if self.writer:
            self.writer.add_image(tag, img_tensor, global_step, dataformats)

    def add_graph(self, model, input_to_model):
        if self.writer:
            self.writer.add_graph(model, input_to_model)

    def close(self):
        if self.writer:
            self.writer.close()

def parse_opt() -> Config:
    parser = argparse.ArgumentParser()
    # config file
    parser.add_argument(
        '--config',
        type = str,
        default = 'configs/ag_news/han.yaml',
        help = 'path to the configuration file (yaml)'
    )
    args = parser.parse_args()
    config_dict = load_config(args.config)
    config = Config(config_dict)

    return config