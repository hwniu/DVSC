from DVC.basics import *
import numpy as np
import imageio
from DVC.flowlib import flow_to_image, read_flow, evaluate_flow
from torch import nn
# sys.path.append('/home/user321/tf_flownet2-master/FlowNet2_src/')
modelspath = './DVC/flow_pretrain_np/'


# from flow_warp import flow_warp

def gather_nd(img, idx):
    """
    same as tf.gather_nd in pytorch
    """
    idx = idx.long()
    idx1, idx2, idx3 = idx.chunk(3, dim=3)
    output = img[idx1, idx2, idx3].squeeze(3)
    return output


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = x.size()
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = torch.arange(0, batch_size).int().cuda()
    batch_idx = batch_idx.view(batch_size, 1, 1)
    b = batch_idx.repeat(1, height, width)

    indices = torch.stack([b, y, x], 3)
    # print(indices.size())

    return gather_nd(img, indices)


Backward_tensorGrid = [{} for i in range(8)]


def torch_warp(tensorInput, tensorFlow):
    device_id = tensorInput.device.index
    if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(
            tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(
            tensorFlow.size(0), -1, -1, tensorFlow.size(3))
        Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat([tensorHorizontal, tensorVertical],
                                                                           1).cuda().to(device_id)

    tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(
                Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border')

def log10(x):
    numerator = torch.log(x)
    denominator = torch.log(10)
    return numerator / denominator

def flow_warp(im, flow):
    warp = torch_warp(im, flow)

    return warp

def loadweightformnp(layername):
    index = layername.find('modelL')
    if index == -1:
        print('laod models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = modelspath + name + '-weight.npy'
        modelbias = modelspath + name + '-bias.npy'
        weightnp = np.load(modelweight)
        biasnp = np.load(modelbias)
        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)

class MEBasic(nn.Module):
    def __init__(self, layername):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv1.weight.data, self.conv1.bias.data = loadweightformnp(layername + '_F-1')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv2.weight.data, self.conv2.bias.data = loadweightformnp(layername + '_F-2')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv3.weight.data, self.conv3.bias.data = loadweightformnp(layername + '_F-3')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv4.weight.data, self.conv4.bias.data = loadweightformnp(layername + '_F-4')
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        self.conv5.weight.data, self.conv5.bias.data = loadweightformnp(layername + '_F-5')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x

def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    # print(inputfeature.size())
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear')
    # print(outfeature.size())
    return outfeature

def bilinearupsacling2(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=True)
    return outfeature

class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        # self.resblock = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Conv2d(inputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2),
        #     nn.ReLU(),
        #     nn.Conv2d(outputchannel, outputchannel, kernel_size, stride, padding=kernel_size//2),
        # )
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
            torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
            torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


class Warp_net(nn.Module):
    def __init__(self, channelnum = 64):
        super(Warp_net, self).__init__()
        self.feature_ext = nn.Conv2d(6, channelnum, 3, padding=1)  # feature_ext
        self.f_relu = nn.ReLU()
        torch.nn.init.xavier_uniform_(self.feature_ext.weight.data)
        torch.nn.init.constant_(self.feature_ext.bias.data, 0.0)
        self.conv0 = ResBlock(channelnum, channelnum, 3)  # c0
        self.conv0_p = nn.AvgPool2d(2, 2)  # c0p
        self.conv1 = ResBlock(channelnum, channelnum, 3)  # c1
        self.conv1_p = nn.AvgPool2d(2, 2)  # c1p
        self.conv2 = ResBlock(channelnum, channelnum, 3)  # c2
        self.conv3 = ResBlock(channelnum, channelnum, 3)  # c3
        self.conv4 = ResBlock(channelnum, channelnum, 3)  # c4
        self.conv5 = ResBlock(channelnum, channelnum, 3)  # c5
        self.conv6 = nn.Conv2d(channelnum, 3, 3, padding=1)
        torch.nn.init.xavier_uniform_(self.conv6.weight.data)
        torch.nn.init.constant_(self.conv6.bias.data, 0.0)

    def forward(self, x):
        feature_ext = self.f_relu(self.feature_ext(x))
        c0 = self.conv0(feature_ext)
        c0_p = self.conv0_p(c0)
        c1 = self.conv1(c0_p)
        c1_p = self.conv1_p(c1)
        c2 = self.conv2(c1_p)
        c3 = self.conv3(c2)
        c3_u = c1 + bilinearupsacling2(
            c3)  # torch.nn.functional.interpolate(input=c3, scale_factor=2, mode='bilinear', align_corners=True)
        c4 = self.conv4(c3_u)
        c4_u = c0 + bilinearupsacling2(
            c4)  # torch.nn.functional.interpolate(input=c4, scale_factor=2, mode='bilinear', align_corners=True)
        c5 = self.conv5(c4_u)
        res = self.conv6(c5)
        return res

flowfiledsSamples = [{} for i in range(8)]


class ME_Spynet(nn.Module):
    '''
    Get flow
    '''

    def __init__(self, layername='motion_estimation'):
        super(ME_Spynet, self).__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList(
            [MEBasic(layername + 'modelL' + str(intLevel + 1)) for intLevel in range(4)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1list = [im1_pre]
        im2list = [im2_pre]
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))
            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))  # , count_include_pad=False))

        shape_fine = im2list[self.L - 1].size()
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device_id = im1.device.index
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device_id)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](torch.cat(
                [im1list[self.L - 1 - intLevel], flow_warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample),
                 flowfiledsUpsample], 1))  # residualflow

        return flowfileds

def build_model():
    net = ME_Spynet().cuda()
    # read images
    im1 = imageio.imread('input.png')
    im1 = im1 / 255.0
    im1 = np.expand_dims(im1, axis=0)
    im2 = imageio.imread('ref.png')
    im2 = im2 / 255.0
    im2 = np.expand_dims(im2, axis=0)
    im1 = np.transpose(im1, [0, 3, 1, 2])
    im2 = np.transpose(im2, [0, 3, 1, 2])
    im1 = torch.from_numpy(im1).float().cuda()
    im2 = torch.from_numpy(im2).float().cuda()
    net.eval()
    flow, warp_frame = net(im1, im2)
    flow = flow.detach().cpu().numpy()
    warp_frame = warp_frame.cpu().detach().numpy()
    rgb_my = flow_to_image(flow[0, :, :, :])
    imageio.imwrite('flow.png', rgb_my)
    # print(psnr)
    imageio.imwrite('warp2.png', warp_frame[0, :, :, :].transpose(1, 2, 0))
    # warp_frame_rgb = flow_warp(im2, flow)
    psnrwapr = CalcuPSNR(im1[0].cpu().numpy().transpose(1, 2, 0), warp_frame[0, :, :, :].transpose(1, 2, 0))
    print(psnrwapr)


if __name__ == '__main__':
    build_model()