import os
import torch.utils.data as data
import random
from DVC.basics import *

def random_crop_and_pad_image_and_labels(image, labels, size):
    combined = torch.cat([image, labels], 0)
    last_image_dim = image.size()[0]
    image_shape = image.size()
    combined_pad = F.pad(combined, (
    0, max(size[1], image_shape[2]) - image_shape[2], 0, max(size[0], image_shape[1]) - image_shape[1]))
    freesize0 = random.randint(0, max(size[0], image_shape[1]) - size[0])
    freesize1 = random.randint(0, max(size[1], image_shape[2]) - size[1])
    combined_crop = combined_pad[:, freesize0:freesize0 + size[0], freesize1:freesize1 + size[1]]
    return combined_crop[:last_image_dim, :, :], combined_crop[last_image_dim:, :, :]


def random_flip(images, labels):
    # augmentation setting....
    horizontal_flip = 1
    vertical_flip = 1
    transforms = 1

    if transforms and vertical_flip and random.randint(0, 1) == 1:
        # 对tensor进行翻转，torch.flip(input,dim):第一个参数是输入，第二个参数是输入的第几维度，按照维度对输入进行翻转
        images = torch.flip(images, [1])
        labels = torch.flip(labels, [1])
    if transforms and horizontal_flip and random.randint(0, 1) == 1:
        images = torch.flip(images, [2])
        labels = torch.flip(labels, [2])

    return images, labels

def gaussian(img, mean, std):
    c, h, w = img.shape
    noise = torch.randn([c, h, w])*std + mean
    return noise

class Vimeo_data(data.Dataset):
    def __init__(self, path="/workspace/2023/PyTorchVideoCompression/DVC/data/vimeo_septuplet/test.txt", im_height=256, im_width=256):
        self.image_input_list, self.image_ref_list = self.get_vimeo(filefolderlist=path)
        self.im_height = im_height
        self.im_width = im_width

        self.featurenoise = torch.zeros([out_channel_M, self.im_height // 16, self.im_width // 16])
        self.znoise = torch.zeros([out_channel_N, self.im_height // 64, self.im_width // 64])
        self.mvnois = torch.zeros([out_channel_mv, self.im_height // 16, self.im_width // 16])
        print("dataset find image: ", len(self.image_input_list))

    def get_vimeo(self, rootdir="/workspace/2023/PyTorchVideoCompression/DVC/data/vimeo_septuplet/sequences/",
                  filefolderlist="/workspace/2023/PyTorchVideoCompression/DVC/data/vimeo_septuplet/test.txt"):
        with open(filefolderlist) as f:
            data = f.readlines()

        fns_train_input = []
        fns_train_ref = []

        for n, line in enumerate(data, 1):
            y = os.path.join(rootdir, line.rstrip())
            fns_train_input += [y]
            refnumber = int(y[-5:-4]) - 2
            refname = y[0:-5] + str(refnumber) + '.png'
            fns_train_ref += [refname]

        return fns_train_input, fns_train_ref

    def __len__(self):
        return len(self.image_input_list)

    def __getitem__(self, index):
        input_image = imageio.imread(self.image_input_list[index])
        ref_image = imageio.imread(self.image_ref_list[index])

        input_image = input_image.astype(np.float32) / 255.0
        ref_image = ref_image.astype(np.float32) / 255.0

        input_image = input_image.transpose(2, 0, 1)
        ref_image = ref_image.transpose(2, 0, 1)

        input_image = torch.from_numpy(input_image).float()
        ref_image = torch.from_numpy(ref_image).float()

        input_image, ref_image = random_crop_and_pad_image_and_labels(input_image, ref_image,
                                                                      [self.im_height, self.im_width])
        # 上述random_crop_and_pad_image_and_labels生成的input_image, ref_image 均为(3,256,256)
        input_image, ref_image = random_flip(input_image, ref_image)

        quant_noise_feature, quant_noise_z, quant_noise_mv = torch.nn.init.uniform_(torch.zeros_like(self.featurenoise),
                                                                                    -0.5, 0.5), torch.nn.init.uniform_(
            torch.zeros_like(self.znoise), -0.5, 0.5), torch.nn.init.uniform_(torch.zeros_like(self.mvnois), -0.5, 0.5)
        noise_tensor = gaussian(input_image, 0, 0.05)
        noise_img_tensor = input_image + noise_tensor
        for i in range(input_image.shape[0]):  # min-max normalization
            noise_tensor[i] = (noise_tensor[i] - noise_tensor[i].min()) / (
                        noise_tensor[i].max() - noise_tensor[i].min())
            noise_img_tensor[i] = (noise_img_tensor[i] - noise_img_tensor[i].min()) / (
                        noise_img_tensor[i].max() - noise_img_tensor[i].min())
        ref_noise_tensor = gaussian(ref_image, 0, 0.05)
        noise_ref_img_tensor = ref_image + ref_noise_tensor
        for i in range(input_image.shape[0]):  # min-max normalization
            ref_noise_tensor[i] = (ref_noise_tensor[i] - ref_noise_tensor[i].min()) / (
                        ref_noise_tensor[i].max() - ref_noise_tensor[i].min())
            noise_ref_img_tensor[i] = (noise_ref_img_tensor[i] - noise_ref_img_tensor[i].min()) / (
                        noise_ref_img_tensor[i].max() - noise_ref_img_tensor[i].min())
        return input_image, ref_image, noise_img_tensor, noise_ref_img_tensor
