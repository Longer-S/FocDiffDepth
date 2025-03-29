import json
import os
import time
import torch
from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from Diffusion import GaussianDiffusion
from net.net import NoisePred
import torch.nn as nn
from dataloader import NYULoader
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_model_name(model_path):
    model_name_expend = os.path.basename(model_path)
    return model_name_expend.split(".")[0]


def valid(config_path, model_path, timestr):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    add_noise = config["diffusion_model"]["add_noise"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)

    _, valid_dataset = NYULoader()

    valid_dataloader= DataLoader(dataset=valid_dataset, num_workers=1, batch_size=1, shuffle=False, drop_last=False)

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
        model = nn.DataParallel(model)
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model_name = get_model_name(model_path)
    # valid
    generat_imgs_num = 1
    dataset_name = 'NYUv2'
    model.eval()
    with torch.no_grad():
        for valid_step, (valid_images,_,fd) in enumerate(valid_dataloader):
            valid_stacksImg=valid_images.to(device)
            fd=fd.to(device)
            diffusion.sample(model, valid_stacksImg,fd, add_noise, model_name, model_path,
                             generat_imgs_num, valid_step, timestr, valid_step_sum, dataset_name)


if __name__ == '__main__':
    model_path = ""
    timestr = time.strftime('%Y%m%d_%H%M%S')
    print(f"time: {timestr}")
    config_path = "config.json"
    valid(config_path, model_path, timestr)
    print("End of valid")
