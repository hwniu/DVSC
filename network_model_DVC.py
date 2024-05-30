import torch
import functools
from torch import nn
from DVC.mv_estimate import *
from DVC.basics import *
from DVC.GDN import *
from channel import *
from dataset.dataset_vimeo90k import *
from pytorch_msssim import MS_SSIM
import numpy as np
from scipy.stats import laplace, uniform

class Video_semantic(nn.Module):
    def __init__(self, channel_type='awgn'):
        super(Video_semantic, self).__init__()
        self.mv_est_net = ME_Spynet()
        self.mv_refinement = Mv_refinement()
        self.transmit_warp_net = Warp_net()
        self.semantic_encoder = Semantic_encoder()
        self.channel_encoder = Channel_encoder()
        self.channel = Channel(channel_type=channel_type)
        self.semantic_decoder = Semantic_decoder()
        self.res_information_repair = Semantic_restoration(in_channel=3)
        self.channel_decoder = Channel_decoder()
        self.receive_warp_net = Warp_net()

        # 自适应编码测试
        # mv_channel = 2
        # res_channel = 3
        # self.semantic_aware_quantization = Semantic_aware_quantization_mechanism(in_channel=mv_channel + res_channel)

    def forward(self, input_frame, ref_frame, ref_no_noisy_frame, snr):
        mv = self.mv_est_net(input_frame, ref_frame)
        mv = self.mv_refinement(mv)  # 已验证其可行性
        transmit_predict_frame, _ = self.transmit_motion_compensation(ref_frame, mv)
        res_infor = input_frame - transmit_predict_frame
        transmission_infor = torch.cat([mv, res_infor], dim=1)
        semantic_feature = self.semantic_encoder(transmission_infor)
        # # 自适应量化测试
        # quantization_transmission_infor, cross_entropy_loss = self.semantic_aware_quantization(transmission_infor, semantic_feature)

        channel_in_feature = self.channel_encoder(semantic_feature, snr)
        channel_out_feature = self.channel(channel_in_feature, snr)
        channel_de_out_feature = self.channel_decoder(channel_out_feature, snr)
        semantic_de_feature = self.semantic_decoder(channel_de_out_feature)
        out_res_infor = semantic_de_feature[:, 2:, :, :]
        out_mv = semantic_de_feature[:, :2, :, :]
        out_res_infor = self.res_information_repair(out_res_infor)
        receive_predict_frame, _ = self.receive_motion_compensation(ref_no_noisy_frame, out_mv)
        reconstructed_frame = receive_predict_frame + out_res_infor
        clipped_recon_image = reconstructed_frame.clamp(0., 1.)
        return clipped_recon_image

    def transmit_motion_compensation(self, ref_frame, motion_vector):
        warp_frame = flow_warp(ref_frame, motion_vector)
        input_feature = torch.cat((warp_frame, ref_frame), 1)
        prediction_frame = self.transmit_warp_net(input_feature) + warp_frame
        return prediction_frame, warp_frame

    def receive_motion_compensation(self, ref_frame, motion_vector):
        warp_frame = flow_warp(ref_frame, motion_vector)
        input_feature = torch.cat((warp_frame, ref_frame), 1)
        prediction_frame = self.receive_warp_net(input_feature) + warp_frame
        return prediction_frame, warp_frame

class Semantic_encoder(nn.Module):
    def __init__(self, in_channels=5):
        super(Semantic_encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channel_N, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / 6)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 3, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        # 让输入和输出服从相同的分布，这样就能够避免后面层的激活函数的输出值趋向于0
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        # torch.nn.init.constant_(tensor, val) 用值val填充向量
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        x = self.conv4(x)
        return x

class Channel_encoder(nn.Module):
    def __init__(self, C_channel=16, norm_layer=nn.BatchNorm2d):
        super(Channel_encoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)
        # Resnet 抽取
        self.res1 = ResnetBlock(out_channel_M, padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.res2 = ResnetBlock(out_channel_M, padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)
        self.projection = nn.Conv2d(out_channel_M, C_channel, kernel_size=3, padding=1, stride=1)

    def forward(self, input_feature, snr):
        z = self.snr_attention1(self.res1(input_feature), snr)
        z = self.snr_attention2(self.res2(z), snr)
        output_feature = self.projection(z)
        return output_feature

class Channel_decoder(nn.Module):
    def __init__(self, input_channel=16, norm_layer=nn.BatchNorm2d):
        super(Channel_decoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # Resnet 抽取
        self.projection_transpose = nn.ConvTranspose2d(input_channel, out_channel_M, 3, stride=1, padding=1, output_padding=0)
        self.res1 = ResnetBlock(out_channel_M, padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.res2 = ResnetBlock(out_channel_M, padding_type='zero', norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)
        self.snr_attention1 = Snr_attention_module(out_channel_M)
        self.snr_attention2 = Snr_attention_module(out_channel_M)

    def forward(self, input_feature, snr):
        z = self.projection_transpose(input_feature)
        z = self.snr_attention1(self.res1(z), snr)
        output_feature = self.snr_attention2(self.res2(z), snr)
        return output_feature

class Semantic_decoder(nn.Module):
    def __init__(self, input_channel=out_channel_M, out_channels=5):
        super(Semantic_decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_channel, out_channel_N, 3, stride=1, padding=1, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (
            math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, out_channels, 3, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data,
                                     (math.sqrt(2 * 1 * (out_channel_N + 3) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x

class Snr_attention_module(nn.Module):
    def __init__(self, define_ch_num):
        super(Snr_attention_module, self).__init__()
        self.channel_num = define_ch_num
        self.layer1 = nn.Linear(self.channel_num+1, self.channel_num//16)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(self.channel_num//16, self.channel_num)
        self.activation2 = nn.Sigmoid()
        self.spatial_attention = SpatialAttention()

    def forward(self, input_feature, snr):
        (num, ch_num, width, height) = input_feature.shape
        if ch_num != self.channel_num:
            raise Exception("This input feature is illegal!")
        input_mean = torch.mean(input_feature.view(input_feature.shape[0], input_feature.shape[1], -1), dim=-1)
        input_cat = torch.cat((input_mean, torch.Tensor([snr for i in range(num)]).reshape(num, 1).cuda()), dim=1)
        factor = self.layer1(input_cat)
        factor = self.activation1(factor)
        factor = self.layer2(factor)
        factor = self.activation2(factor)
        result = torch.mul(input_feature, factor.view(input_feature.shape[0], input_feature.shape[1], 1, 1))
        spatial_attention_factor = self.spatial_attention(result)
        result = torch.mul(result, spatial_attention_factor)
        return result

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
class Mv_refinement(nn.Module):
    def __init__(self, in_feature_channel=2, out_channel=2, inter_channel=64, stride=2, kernel_size=3):
        super(Mv_refinement, self).__init__()
        self.conv1 = nn.Conv2d(in_feature_channel, inter_channel, kernel_size, stride=stride, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.gdn1 = GDN(inter_channel)
        self.conv2 = nn.Conv2d(inter_channel, inter_channel, kernel_size, stride=stride, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.deconv1 = nn.ConvTranspose2d(inter_channel, inter_channel, kernel_size, stride=stride, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(inter_channel, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(inter_channel, out_channel, kernel_size, stride=stride, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)

    def forward(self, mv_vector):
        feature = self.gdn1(self.conv1(mv_vector))
        feature = self.conv2(feature)
        feature = self.igdn1(self.deconv1(feature))
        mv_vector_refine = self.deconv2(feature)
        return mv_vector_refine

class Semantic_restoration(nn.Module):
    def __init__(self, in_channel, kernel_size=3):
        out_channel = in_channel
        super(Semantic_restoration, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.CAB1 = RCAB(32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.CAB2 = RCAB(64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.CAB3 = RCAB(128, 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.CAB4 = RCAB(256, 256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv5.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5.bias.data, 0.01)
        self.CAB5 = RCAB(512, 512)
        self.CAB6 = RCAB(512, 512)
        self.CAB7 = RCAB(512, 512)
        self.CAB8 = RCAB(512, 512)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.CAB9 = RCAB(256, 256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.CAB10 = RCAB(128, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.CAB11 = RCAB(64, 64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size, stride=2, padding=1, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.CAB12 = RCAB(32, 32)
        self.conv6 = nn.Conv2d(32, out_channel, kernel_size, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv6.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv6.bias.data, 0.01)

    def forward(self, input_feature):
        feature = self.CAB1(self.conv1(input_feature))
        feature = self.CAB2(self.conv2(feature))
        feature = self.CAB3(self.conv3(feature))
        feature = self.CAB4(self.conv4(feature))
        feature = self.conv5(feature)
        feature = self.CAB5(feature)
        feature = self.CAB6(feature)
        feature = self.CAB7(feature)
        feature = self.CAB8(feature)
        feature = self.CAB9(self.deconv1(feature))
        feature = self.CAB10(self.deconv2(feature))
        feature = self.CAB11(self.deconv3(feature))
        c_feature = self.CAB12(self.deconv4(feature))

        out_frame = self.conv6(c_feature)
        return out_frame

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        conv_layer_1 = nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True)
        torch.nn.init.xavier_normal_(conv_layer_1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(conv_layer_1.bias.data, 0.01)
        conv_layer_2 = nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)
        torch.nn.init.xavier_normal_(conv_layer_2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(conv_layer_2.bias.data, 0.01)
        self.conv_du = nn.Sequential(
            conv_layer_1,
            nn.ReLU(inplace=True),
            conv_layer_2,
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, in_channel, inter_channel, reduction=1, kernel_size=3, bias=True, bn=False, act=nn.ReLU(True),
                 res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            conv_layer = nn.Conv2d(in_channel, inter_channel, kernel_size, padding=(kernel_size // 2), bias=bias)
            torch.nn.init.xavier_normal_(conv_layer.weight.data, math.sqrt(2))
            torch.nn.init.constant_(conv_layer.bias.data, 0.01)
            modules_body.append(conv_layer)
            if bn: modules_body.append(nn.BatchNorm2d(inter_channel))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(inter_channel, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class Prior_model(nn.Module):
    def __init__(self, in_channel, kernel_size=3):
        super(Prior_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size, stride=2, padding=1)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv4 = nn.Conv2d(64, 192, kernel_size, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, input_feature):
        feature = self.relu1(self.conv1(input_feature))
        feature = self.relu2(self.conv2(feature))
        feature = self.relu3(self.conv3(feature))
        out_feature = self.conv4(feature)

        return out_feature[:, :64, :, :].clamp(1e-5, 1e10), out_feature[:, 64:128, :, :].clamp(1e-5, 1e10), out_feature[:, 128:192, :, :].clamp(1e-5, 1e10),

class Semantic_aware_quantization_mechanism(nn.Module):
    def __init__(self, in_channel=3):
        super(Semantic_aware_quantization_mechanism, self).__init__()
        self.prior_model = Prior_model(in_channel=in_channel)

    def forward(self, uncompress_semantic, compressed_semantic):
        mu, sigma, step = self.prior_model(uncompress_semantic)
        gaussian_function = torch.distributions.laplace.Laplace(mu, sigma)
        quantized_semantic_feature = torch.round((compressed_semantic - mu) / step) * step + mu
        probs = gaussian_function.cdf(quantized_semantic_feature + step) - gaussian_function.cdf(quantized_semantic_feature - step)
        cross_entropy_loss = torch.mean(-1.0 * torch.log(probs + 1e-5) / math.log(2.0))
        return quantized_semantic_feature, cross_entropy_loss
