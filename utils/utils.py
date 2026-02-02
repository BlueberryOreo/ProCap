import os
import random

import numpy as np
import time

import torch
import torch.nn as nn
from torchvision.io import write_video
from torchvision import utils


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True)
        return x / maxes


class Timer:
    _level = 0
    
    def __init__(self, name="Block", logger=print):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()
        Timer._level += 1
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = (time.time() - self.start_time) * 1000
        indent = "|  " * (Timer._level - 1)
        self.logger(f"{indent}[{self.name}] took {elapsed_time:.5f} ms")
        Timer._level -= 1


class DebugLogger:
    def __init__(self, log_path="logs/", file_name="debug.log", add_time=True, visible=True):
        self.log_path = log_path
        self.file_name = file_name
        self.add_time = add_time
        self.visible = visible
        os.makedirs(self.log_path, exist_ok=True)
        with open(os.path.join(self.log_path, self.file_name), "a") as f:
            f.write("----------------------------------------\n")
            f.write(f"Log file created at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    def __call__(self, msg):
        if not self.visible:
            return
        with open(os.path.join(self.log_path, self.file_name), "a") as f:
            # f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S') if self.add_time else ''} {msg}\n")
            if self.add_time:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - ")
            f.write(f"{msg}\n")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark=False


def save_image(ximg, path):
    n_sample = ximg.shape[0]
    utils.save_image(ximg,
                     path,
                     nrow=int(n_sample**0.5),
                     normalize=True,
                     range=(-1, 1))


def save_video(xseq, path):
    video = xseq.data.cpu().clamp(-1, 1)
    video = ((video + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
    write_video(path, video, fps=15)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


def clip_similarity(model, tokenizer, image, description):
    input_resolution = model.input_resolution.item()
    context_length = model.context_length.item()

    if image.shape[2] != input_resolution:
        image = F.interpolate(image, (input_resolution, input_resolution))
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    image_input = (image - image_mean[:, None, None]) / image_std[:, None,
                                                                  None]
    text_input = tokenizer.tokenize(
        description,
        context_length,
        truncate_text=True,
    ).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_input).float()

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (text_features.cpu().numpy() *
                  image_features.cpu().numpy()).sum(1)
    return similarity


def exists(val):
    return val is not None


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def sample_data(loader, sampler=None):
    epoch = -1
    while True:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch


"""
Copied from VQGAN main.py
"""
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

