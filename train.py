import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import torch
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# from dataset import MFI_Dataset
from Diffusion import GaussianDiffusion
from Condition_Noise_Predictor.UNet import NoisePred
from utils import tensorboard_writer, logger, save_model,save_model_tar,weights_init
import torch.nn as nn
import torch.nn.functional as F
import os
from glob import glob
from dataloader import FoD500Loader
from dataloader import NYULoader
import torchvision.utils as vutils
# from torch.cuda.amp import GradScaler, autocast
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config_path = "./config.json"
timestr = time.strftime('%Y%m%d_%H%M%S')
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

# train dataset
train_datasePath = config["dataset"]["train"]["path"]
train_phase = config["dataset"]["train"]["phase"]
train_batch_size = config["dataset"]["train"]["batch_size"]
train_use_dataTransform = config["dataset"]["train"]["use_dataTransform"
                                                        ]
train_resize = config["dataset"]["train"]["resize"]
train_imgSize = config["dataset"]["train"]["imgSize"]
train_shuffle = config["dataset"]["train"]["shuffle"]
train_drop_last = config["dataset"]["train"]["drop_last"]

train_dataset, valid_dataset = NYULoader()



train_dataloader = DataLoader(dataset=train_dataset, num_workers=4, batch_size=train_batch_size, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(dataset=valid_dataset, num_workers=1, batch_size=10, shuffle=False, drop_last=True)
# Condition Noise Predictor
in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                    down_sample_mult)
model.apply(weights_init)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
# whether to use the pre-training model
use_preTrain_model = config["Condition_Noise_Predictor"]["use_preTrain_model"]


if use_preTrain_model:
    preTrain_Model_path = config["Condition_Noise_Predictor"]["preTrain_Model_path"]
    checkpoint = torch.load(preTrain_Model_path, map_location=device)

    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print(f"using pre-trained model：{preTrain_Model_path}")

        # 初始化优化器
    init_lr = config["optimizer"]["init_lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    
    # 加载优化器状态
    if 'optimize' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimize'])
        print("Loaded optimizer state.")

    # 初始化学习率调度器
    use_lr_scheduler = config["optimizer"]["use_lr_scheduler"]
    if use_lr_scheduler:
        StepLR_size = config["optimizer"]["StepLR_size"]
        StepLR_gamma = config["optimizer"]["StepLR_gamma"]
        learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size=StepLR_size, gamma=StepLR_gamma)
        
        # 加载调度器状态
        if 'scheduler_state_dict' in checkpoint:
            learningRate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded learning rate scheduler state.")
    else:
        learningRate_scheduler = None  # 不使用学习率调度器

else:
    # 如果没有使用预训练模型，初始化模型和优化器
    model = model.to(device)        
    use_lr_scheduler = config["optimizer"]["use_lr_scheduler"]
    init_lr = config["optimizer"]["init_lr"]
    StepLR_size = config["optimizer"]["StepLR_size"]
    StepLR_gamma = config["optimizer"]["StepLR_gamma"]
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr) #qz1e-3 (0.9,0.99)

    if use_lr_scheduler:
        learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size=StepLR_size, gamma=StepLR_gamma)
        # learningRate_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-6)
    else:
        learningRate_scheduler = None  # 不使用学习率调度器

# diffusion model
T = config["diffusion_model"]["T"]
beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
loss_scale = config["diffusion_model"]["loss_scale"]
diffusion = GaussianDiffusion(T, beta_schedule_type)

# log
writer_train = tensorboard_writer(timestr+ '/train')
writer_test = tensorboard_writer(timestr+ '/valid')
log = logger(timestr)
print(f"time: {timestr}")
log.write(f"time: {timestr} \n")
print(f"using {len(train_dataset)} images for train")
log.write(f"using {len(train_dataset)} images for train  \n\n")
log.write(f"config:  \n")
log.write(json.dumps(config, ensure_ascii=False, indent=4))
if use_lr_scheduler:
    log.write(
        f"\n learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size={StepLR_size}, gamma={StepLR_gamma})  \n\n")

# hyper-parameter
epochs = config["hyperParameter"]["epochs"]
start_epoch = config["hyperParameter"]["start_epoch"]
loss_step = config["hyperParameter"]["loss_step"]
save_model_epoch_step = config["hyperParameter"]["save_model_epoch_step"]
train_step_sum = len(train_dataloader)



def train():
    best_loss=1e4
    total_iters = 0
    num_train_step = 0
    # scaler = GradScaler()
    for epoch in range(start_epoch, epochs):
        
        # train
        model.train()
        loss_sum = 0
        writer_train.add_scalar('lr_epoch: ', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        for train_step, (train_images,train_depth,fd) in tqdm(enumerate(train_dataloader), desc="train step"):
            optimizer.zero_grad()

            train_stackImg=train_images.to(device)
            train_depth=train_depth.to(device)
            fd=fd.to(device)
            
            t = torch.randint(0, T, (train_batch_size,), device=device).long()
            # with autocast():
            scale_loss,const_loss,_psnr, _ssim, _sharp = diffusion.train_losses(model, train_stackImg,fd, train_depth, t, concat_type, loss_scale)

            writer_train.add_scalar('loss_step: ', scale_loss, num_train_step)

            if train_step % loss_step == 0:
                print(
                    f" [epoch] {epoch}/{epochs}    "
                    f"[epoch_step] {train_step}/{train_step_sum}     "
                    f"[train_step] {num_train_step}     "
                    f"[loss] {scale_loss.item() / loss_scale :.6f}     "
                    f"[scale_loss] {scale_loss.item() :.6f}     "
                    f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                    f"[t] {t.cpu().numpy()}")

                log.write(f" [epoch] {epoch}/{epochs}    "
                          f"[epoch_step] {train_step}/{train_step_sum}     "
                          f"[train_step] {num_train_step}     "
                          f"[loss] {scale_loss.item() / loss_scale :.6f}     "
                          f"[scale_loss] {scale_loss.item() :.6f}     "
                          f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                          f"[t] {t.cpu().numpy()}"
                          f"\n")

            scale_loss.backward()

            optimizer.step()

            total_iters += 1
            loss_sum += scale_loss
            num_train_step += 1

        aver_loss = loss_sum / train_step_sum


        dir_path = os.path.join("weight",timestr)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if epoch % 50 == 0:
            ckpt_name = "epoch_" + str(epoch) + ".tar"
            ckpt_path = os.path.join(dir_path, ckpt_name)
            save_model_tar(epoch,total_iters,best_loss,model,optimizer,learningRate_scheduler,ckpt_path)

        # update learning rate
        if use_lr_scheduler:
            learningRate_scheduler.step()
        
        # save top 5 ckpts only
        list_ckpt = glob(os.path.join( os.path.abspath(dir_path) + '/' , 'epoch_*'))
        list_ckpt.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        if len(list_ckpt) > 10:
            os.remove(list_ckpt[0])

        # Vaild
        if epoch % 5 == 0:
            total_val_loss = 0
            for train_step, (valid_images,valid_depth,fd) in tqdm(enumerate(valid_dataloader), desc="vild step"):

                with torch.no_grad():
                    start_time = time.time()
                    model.eval()
                    valid_stackImg=valid_images.to(device)
                    valid_depth=valid_depth.to(device)
                    fd=fd.to(device)
                    t = torch.randint(0, T, (10,), device=device).long()
                    val_loss,_psnr, _ssim, _mse, _sharp = diffusion.train_losses(model, valid_stackImg,fd, valid_depth, t, concat_type, loss_scale)
                    if epoch % 50 == 0 and train_step == 0:
                        pred=diffusion.ddim_sample(model,valid_stackImg[:5],fd[:5],valid_stackImg.shape[-1],batch_size=5,channels=1,ddim_timesteps=20,
                                            ddim_discr_method="uniform",ddim_eta=0.0,clip_denoised=True)
                if train_step %2 == 0:
                    torch.cuda.synchronize()
                    print('[val] epoch %d : %d/%d val_loss = %.6f , time = %.2f' % (epoch, train_step, len(valid_dataloader), val_loss, time.time() - start_time))
                total_val_loss += val_loss
                if epoch % 50 == 0 and train_step == 0:
                    writer_test.add_image('Predicted Image', vutils.make_grid(pred[:5].detach().cpu(), normalize=True, scale_each=True), epoch)
                    writer_test.add_image('Ground Truth Image', vutils.make_grid(valid_depth[:5].detach().cpu(), normalize=True, scale_each=True), epoch)   
            avg_val_loss = total_val_loss / len(valid_dataloader)
 
            writer_test.add_scalar('avg_loss', avg_val_loss, epoch)

            # save best
            if avg_val_loss < best_loss:
                best_name = "best_" + str(epoch) + ".tar"
                best_path = os.path.join(dir_path, best_name)
                best_loss = avg_val_loss
                save_model_tar(epoch,total_iters,best_loss,model,optimizer,learningRate_scheduler,best_path)

            list_ckpt = glob(os.path.join( os.path.abspath(dir_path) + '/' , 'best_*'))
            list_ckpt.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            if len(list_ckpt) > 10:
                os.remove(list_ckpt[0])
        torch.cuda.empty_cache()

        writer_train.add_scalar('aver_loss_epoch: ', aver_loss, epoch)
        log.write("\n")

    print("End of training")
    log.write("End of training \n")
    writer_train.close()


if __name__ == '__main__':
    config_path = "config.json"
    train()
