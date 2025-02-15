# Function: generate fusion images using FusionDiff

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import torch
from torch.utils.data import DataLoader
# from dataset import MFI_Dataset
from Diffusion import GaussianDiffusion
from Condition_Noise_Predictor.UNet import NoisePred
import torch.nn as nn
from dataloader import FoD500Loader
from dataloader import NYULoader
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from torch.utils.data import Subset
# torch.manual_seed(2024)
# torch.cuda.manual_seed_all(2024)
# Gets the filename without the extension
def get_model_name(model_path):
    model_name_expend = os.path.basename(model_path)
    return model_name_expend.split(".")[0]


# Inference use pretrain_model
def valid(config_path, model_path, timestr):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # DDPM
    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    add_noise = config["diffusion_model"]["add_noise"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)

    # valid dataset
    valid_datasePath = config["dataset"]["valid"]["path"]
    valid_phase = config["dataset"]["valid"]["phase"]
    valid_batch_size = config["dataset"]["valid"]["batch_size"]
    valid_use_dataTransform = config["dataset"]["valid"]["use_dataTransform"]
    valid_dataset = config["dataset"]["valid"]["resize"]
    valid_imgSize = config["dataset"]["valid"]["imgSize"]
    valid_shuffle = config["dataset"]["valid"]["shuffle"]
    valid_drop_last = config["dataset"]["valid"]["drop_last"]

    # valid_dataset=DefocusNet(root_dir='./data/DefocusNet/', split='test', shuffle=False, img_num=5, tran=False,
    #                          visible_img=5, focus_dist=[0.1,.15,.3,0.7,1.5], recon_all=True, 
    #                 RGBFD=True, DPT=True, AIF=False,scale=2, norm=False, near=0.1, far=1., resize=224)
    
    # valid_dataset=NYUDataset(root_dir='./data/NYUv2/', split='test', shuffle=False, img_num=5, visible_img=5, focus_dist=[0.1,.15,.3,0.7,1.5], recon_all=True, 
    #                 RGBFD=True, DPT=True, AIF=False, scale=2, norm=True, near=0.1, far=1., trans=False, resize=256)
    train_dataset, valid_dataset = NYULoader()
    # valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=valid_shuffle,num_workers=1,
    #                               drop_last=valid_drop_last)


    # database = './fs_6/'
    # FoD500_train, FoD500_val = FoD500Loader(database, n_stack=5, scale=1)
    # # FoD500_train, FoD500_val =  [FoD500_train], [FoD500_val]

    # # train_dataset = torch.utils.data.ConcatDataset(DDFF12_train  + FoD500_train )

    # valid_dataset=FoD500_val

    # dataset_val = torch.utils.data.ConcatDataset(DDFF12_val) # we use the model perform better on  DDFF12_val

    # dataset_length = len(valid_dataset)
    # last_five_indices = range(625, dataset_length)
    # valid_dataset = Subset(valid_dataset, last_five_indices)
    valid_dataloader= DataLoader(dataset=valid_dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=False)

 

    assert len(valid_dataset) % valid_batch_size == 0, "please reset valid_batch_size"
    valid_step_sum = len(valid_dataloader)

    # Load noise_pred_model
    print(f"device = {device}")
    in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
    out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
    model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
    num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
    dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
    time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
    down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
    model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                      down_sample_mult)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [32, xxx] -> [8, ...], [8, ...], [8, ...] on 4 GPUs
        # 使用 DataParallel 把模型包装起来
        model = nn.DataParallel(model)
    checkpoint = torch.load(model_path, map_location=device)
    # if any('module.' in key for key in checkpoint.keys()):
    #     checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint['state_dict'])

    # model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
    model.to(device)
    model_name = get_model_name(model_path)
    concat_type = config["Condition_Noise_Predictor"]["concat_type"]

    # valid
    generat_imgs_num = config["dataset"]["valid"]["generat_imgs_num"]
    dataset_name = config["dataset"]["valid"]["dataset_name"]
    model.eval()
    with torch.no_grad():
        for valid_step, (valid_images,_,fd) in enumerate(valid_dataloader):
            valid_sourceImg1=valid_images.to(device)
            # valid_depth=valid_depth.to(device)
            fd=fd.to(device)
            diffusion.sample(model, valid_sourceImg1,fd, add_noise, concat_type, model_name, model_path,
                             generat_imgs_num, valid_step * valid_batch_size + 1, timestr, valid_step_sum, dataset_name)


if __name__ == '__main__':
    model_path = r"weight/20241231_115955/epoch_4000_3.tar"#weight/20241206_002636_3010可以/epoch_3010.tar
    timestr = time.strftime('%Y%m%d_%H%M%S')
    print(f"time: {timestr}")
    config_path = "config.json"
    valid(config_path, model_path, timestr)
    print("End of valid")
