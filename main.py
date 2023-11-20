import argparse
import math
import os
import time
import warnings
import imageio
import numpy as np
import lpips
import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
import re
import pytorch_msssim

from dataset.dataset_vimeo90k import Vimeo_data
from network_model import Video_semantic

warnings.filterwarnings("ignore")

global args

def save_model(model, step):
    if os.path.exists(args.save_model_dir):
        torch.save(model.state_dict(), args.save_model_dir+"step{}.model".format(step))
    else:
        os.mkdir(args.save_model_dir)
        torch.save(model.state_dict(), args.save_model_dir + "step{}.model".format(step))

def load_model(model):
    if os.path.exists(args.load_model_dir):
        if len(os.listdir(args.load_model_dir)) == 0:
            return 0
        else:
            step = 0
            for item in os.listdir(args.load_model_dir):
                index_step = item.split('p')[1]
                index_step = index_step.split('.')[0]
                if step < int(index_step):
                    step = int(index_step)
            model_name = "step" + str(step) + ".model"
            model.load_state_dict(torch.load(args.load_model_dir+model_name), strict=False)
            return step
    else:
        os.mkdir(args.save_model_dir)
        return 0

def clip_gradient(model_optimizer, grad_clip):
    for group in model_optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

# 定义回调函数来打印梯度
def print_grad(grad):
    print(grad)

def extract_num(s):
    nums = re.findall(r'\d+', s[-7:])  # 在字符串的最后五位中查找数字子串
    return int(nums[0]) if nums else 0  # 将第一个数字子串转换为整数，如果没有则返回0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-channel_type", default='awgn', help="awgn/rayleigh/rician/fisher_snedecor")
    parser.add_argument("-save_model_dir", default='./model/', help="dir for model saving")
    parser.add_argument("-load_model_dir", default='./model/', help="dir for model loading")
    parser.add_argument("-loss_dir", default='loss/', help="dir for loss process saving")
    parser.add_argument("-eval_dir", default='eval/', help="dir for eval process saving")
    parser.add_argument("-batch_size", default=4, type=int, help="batch size for model training")
    parser.add_argument("-epochs", default=3, type=int, help="the epochs of model training")
    parser.add_argument("-steps", default=1000, type=int, help="the steps of model training epoch")
    parser.add_argument("-learning_rate", default=0.000025, type=float, help="the learning rate of model training")
    parser.add_argument("-train_snr_low", default=0, type=int, help="the adapt low snr during training, Unit:db")
    parser.add_argument("-train_snr_high", default=20, type=int, help="the adapt high snr during training, Unit:db")
    parser.add_argument("-eval_snr_low", default=0, type=int, help="the low snr during evaluating")
    parser.add_argument("-eval_snr_high", default=20, type=int, help="the high snr during evaluating")
    parser.add_argument("-mode", default='train', help="the system mode, train or eval")
    args = parser.parse_args()
    print("#######################################")
    print("Current execution parameters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")

    VideoModel = Video_semantic().cuda()
    # print(VideoModel)
    train_dataset = Vimeo_data()
    global_step = load_model(VideoModel)
    optimizer = torch.optim.Adam(VideoModel.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    writer = SummaryWriter(log_dir='./logs')
    step_epoch = global_step // (train_dataset.__len__() // args.batch_size)
    snr_index = step_epoch // args.epochs
    lpips_loss = lpips.LPIPS(net='vgg', verbose=False).cuda()
    lpips_loss.eval()
    # for name, param in VideoModel.named_parameters():
    #     param.register_hook(print_grad)
    if args.mode == "train":
        print("####################################### Online Video Semantic Communication Will Be Trained! #######################################")
        VideoModel.train()
        for snr in range(args.train_snr_low + snr_index, args.train_snr_high + 1):
            print("################## training snr: ", snr, "dB", " ##################")
            for epoch in range(step_epoch - snr_index * args.epochs, args.epochs):
                loss_list = []
                mse_loss_list = []
                runtime_list_per_frame = []
                train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size)
                if epoch == 0:
                    start_index = global_step
                else:
                    start_index = 0
                for batch_index, input_info in enumerate(train_loader, start=start_index):
                    global_step = global_step + 1
                    input_frame, ref_frame, input_noisy_frame, ref_noisy_frame = Variable(input_info[0].cuda()), Variable(input_info[1].cuda()), Variable(input_info[2].cuda()), Variable(input_info[3].cuda())
                    start_runtime = time.time()
                    reconstructed_frame = VideoModel(input_noisy_frame, ref_noisy_frame, ref_frame, snr)
                    runtime = (time.time() - start_runtime) / args.batch_size
                    runtime_list_per_frame.append(runtime * 1000)
                    mse_loss = torch.mean((reconstructed_frame - input_frame).pow(2))
                    mse_loss_list.append(mse_loss.tolist())
                    lpips_loss_value = torch.mean(lpips_loss(reconstructed_frame, input_frame))
                    loss = mse_loss + 0.0001*lpips_loss_value
                    loss_list.append(loss.tolist())
                    optimizer.zero_grad()
                    loss.backward()
                    # 记录梯度信息
                    # for name, param in VideoModel.named_parameters():
                    #     writer.add_scalar('gradient/' + name, param.grad.norm(), global_step)
                    # clip_gradient(optimizer, 0.5)
                    optimizer.step()

                    writer.add_scalar('loss/' + "loss_value", loss, global_step)
                    # writer.add_scalar('loss/' + "cross_entropy_loss", cross_entropy_loss, global_step)
                    if global_step % 100 == 0 and global_step != 0:
                        print("training epoch: ", epoch, " ---training global step: ", global_step, " ---ave_loss_value: ", sum(loss_list[len(loss_list)-100:])/100, " PSNR: ", 10*math.log10(1 / (sum(loss_list[len(loss_list)-100:])/100)), "dB")
                    if global_step % 10000 == 0 and global_step != 0:
                        save_model(VideoModel, global_step)
                print("########## training epoch: ", epoch, "ave_loss: ", sum(loss_list)/len(loss_list), "ave_runtime: ", sum(runtime_list_per_frame)/len(runtime_list_per_frame), "ms", " ##########")
                save_model(VideoModel, global_step)
    elif args.mode == "eval":
        print("####################################### Online Video Semantic Communication Will Be Evaluated! #######################################")
        VideoModel.eval()
        clean_frames_path = "/workspace/2023/video_dataset/UVG/images/"
        noisy_original_frames_path = "/workspace/2023/video_dataset/UVG/100_noisy_images/"
        short = ['Beauty', 'HoneyBee', 'ReadySteadyGo', 'YachtRide', 'Bosphorus', 'Jockey', 'ShakeNDry']
        json_data = {'Beauty':{"psnr":[], "lpips":[], "msssim":[]}, 'HoneyBee':{"psnr":[], "lpips":[], "msssim":[]}, 'ReadySteadyGo':{"psnr":[], "lpips":[], "msssim":[]},
                     'YachtRide':{"psnr":[], "lpips":[], "msssim":[]}, 'Bosphorus':{"psnr":[], "lpips":[], "msssim":[]}, 'Jockey':{"psnr":[], "lpips":[], "msssim":[]},
                     'ShakeNDry':{"psnr":[], "lpips":[], "msssim":[]}}
        for short_name in short:
            all_clean_frames = []
            all_noisy_frames = []
            for file_name in os.listdir(clean_frames_path + short_name):
                all_clean_frames.append(clean_frames_path + short_name + "/" + file_name)
            for file_name in os.listdir(noisy_original_frames_path + short_name):
                all_noisy_frames.append(noisy_original_frames_path + short_name + "/" + file_name)
            ref_clean_frames_path = sorted(all_clean_frames, key=extract_num)[0:-1]
            input_clean_frames_path = sorted(all_clean_frames, key=extract_num)[1:]
            ref_noisy_frames_path = sorted(all_noisy_frames, key=extract_num)[0:-1]
            input_noisy_frames_path = sorted(all_noisy_frames, key=extract_num)[1:]
            # 测试
            input_noisy_frame = imageio.imread(input_noisy_frames_path[0]).astype(np.float32) / 255.0
            ref_noisy_frame = imageio.imread(ref_noisy_frames_path[0]).astype(np.float32) / 255.0
            ref_frame = imageio.imread(ref_clean_frames_path[0]).astype(np.float32) / 255.0
            input_frame = imageio.imread(input_clean_frames_path[0]).astype(np.float32) / 255.0
            input_noisy_frame = torch.from_numpy(input_noisy_frame.transpose(2, 0, 1)).float()
            ref_noisy_frame = torch.from_numpy(ref_noisy_frame.transpose(2, 0, 1)).float()
            ref_frame = torch.from_numpy(ref_frame.transpose(2, 0, 1)).float()
            input_frame = torch.from_numpy(input_frame.transpose(2, 0, 1)).float()
            input_frame = input_frame.view(1, input_frame.shape[0], input_frame.shape[1], input_frame.shape[2]).cuda()
            ref_frame = ref_frame.view(1, ref_frame.shape[0], ref_frame.shape[1], ref_frame.shape[2]).cuda()
            input_noisy_frame = input_noisy_frame.view(1, input_noisy_frame.shape[0], input_noisy_frame.shape[1],
                                                       input_noisy_frame.shape[2]).cuda()
            ref_noisy_frame = ref_noisy_frame.view(1, ref_noisy_frame.shape[0], ref_noisy_frame.shape[1],
                                                   ref_noisy_frame.shape[2]).cuda()
            input_frame = input_frame[:, :, 0:1024, :]
            ref_frame = ref_frame[:, :, 0:1024, :]
            input_noisy_frame = input_noisy_frame[:, :, 0:1024, :]
            ref_noisy_frame = ref_noisy_frame[:, :, 0:1024, :]
            for snr in range(-2, 22, 2):
                with torch.no_grad():
                    reconstructed_frame = VideoModel(input_frame, ref_noisy_frame, ref_frame, snr)
                mse_loss_1 = torch.mean((reconstructed_frame - input_noisy_frame).pow(2))
                mse_loss_2 = torch.mean((reconstructed_frame - input_frame).pow(2))
                psnr_1 = 10 * math.log10(1 / (mse_loss_1.item()))
                psnr_2 = 10 * math.log10(1 / (mse_loss_2.item()))
                lpips_loss_value_1 = torch.mean(lpips_loss(reconstructed_frame, input_noisy_frame))
                lpips_loss_value_2 = torch.mean(lpips_loss(reconstructed_frame, input_frame))
                msssim_val_1 = pytorch_msssim.ms_ssim(reconstructed_frame, input_noisy_frame, data_range=1.0, size_average=True)
                msssim_val_2 = pytorch_msssim.ms_ssim(reconstructed_frame, input_frame, data_range=1.0, size_average=True)
                print(short_name, ":", psnr_2, lpips_loss_value_2.item(), msssim_val_2.item())
                json_data[short_name]["psnr"].append(psnr_2)
                json_data[short_name]["lpips"].append(lpips_loss_value_2.item())
                json_data[short_name]["msssim"].append(msssim_val_2.item())
                torch.cuda.empty_cache()

        total_data = {"psnr":[], "lpips":[], "msssim":[]}
        i = 0
        for snr in range(-2, 22, 2):
            total_data["psnr"].append((json_data["Beauty"]["psnr"][i] + json_data["HoneyBee"]["psnr"][i] + json_data["ReadySteadyGo"]["psnr"][i] + json_data["YachtRide"]["psnr"][i] +
                                       json_data["Bosphorus"]["psnr"][i] + json_data["Jockey"]["psnr"][i] + + json_data["ShakeNDry"]["psnr"][i]) / 7)
            total_data["lpips"].append((json_data["Beauty"]["lpips"][i] + json_data["HoneyBee"]["lpips"][i] + json_data["ReadySteadyGo"]["lpips"][i] + json_data["YachtRide"]["lpips"][i] +
                                       json_data["Bosphorus"]["lpips"][i] + json_data["Jockey"]["lpips"][i] + + json_data["ShakeNDry"]["lpips"][i]) / 7)
            total_data["msssim"].append((json_data["Beauty"]["msssim"][i] + json_data["HoneyBee"]["msssim"][i] + json_data["ReadySteadyGo"]["msssim"][i] + json_data["YachtRide"]["msssim"][i] +
                                       json_data["Bosphorus"]["msssim"][i] + json_data["Jockey"]["msssim"][i] + + json_data["ShakeNDry"]["msssim"][i]) / 7)
            i = i + 1
        print(total_data)
        print(total_data['psnr'][5], total_data['lpips'][5], total_data['msssim'][5],)
    else:
        raise Exception("This input system mode not exist!")

