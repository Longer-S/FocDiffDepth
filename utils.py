import numpy as np
from torchvision import transforms
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn

# 将tensor转换为img  nyuv2
def tensor2img(img):
    
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t * 10000.),
        transforms.Lambda(lambda t: t.squeeze().cpu().numpy().astype(np.uint16)),
    ])
    return reverse_transforms(img)

def weights_init(m):
    if isinstance(m, nn.Conv2d)or isinstance(m, nn.Conv3d)or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias!=None:
            m.bias.data.fill_(0.01)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def tensorboard_writer(timestr):
    log_path = os.path.join('logs', timestr)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    return writer


# txt 日志文件
def logger(timestr):
    log_dir = os.path.join('logs',timestr)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "log.txt")
    fw = open(log_path, "a+")
    return fw


def save_model(model, optimizer, scheduler, epoch, timestr):
    dir_path = os.path.join("weight", timestr)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    ckpt_name = "epoch_" + str(epoch) + ".pt"
    ckpt_path = os.path.join(dir_path, ckpt_name)
    
    # 选择是否保存调度器的状态
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, ckpt_path)


def save_model_tar(epoch,total_iters,best_loss,model,optimizer,scheduler,ckpt_path):

    # 选择是否保存调度器的状态
    checkpoint = {
        'epoch': epoch + 1,
        'iters': total_iters + 1,
        'best': best_loss,
        'state_dict': model.state_dict(),
        'optimize':optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, ckpt_path)
    
