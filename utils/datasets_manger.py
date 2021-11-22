import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps


class Mydataset(Dataset):
    def __init__(self, imgs_dir, patch_size=128, patch_num_per_images=1):
        self.imgs_dir = imgs_dir
        self.patch_size = patch_size
        self.patch_num_per_images = patch_num_per_images
        self.imgfiles = [[], []]
        path_gt = os.path.join(imgs_dir, 'gt')
        path_input = os.path.join(imgs_dir, 'input')
        for file in os.listdir(path_input):  # combine中包含了合成图片的路径索引
            self.imgfiles[0].append(os.path.join(path_input, file))   # 添加网络输入图片路径
            self.imgfiles[1].append(os.path.join(path_gt, file.replace('RGBN', 'RGB')))   # 添加rgb图像路径

    def __len__(self):
        return len(self.imgfiles[0])

    @classmethod
    def preprocess(cls, pil_img, patch_size, patch_coords, flip_op):  # 对操作的图片进行预处理，截取部分区域，进行镜像反转
        if flip_op == 1:
            pil_img = ImageOps.mirror(pil_img)  # 水平方向镜像
        elif flip_op == 2:
            pil_img = ImageOps.flip(pil_img)  # 垂直方向镜像
        img_nd = np.array(pil_img)  # 创建numpy数组
        assert len(img_nd.shape) == 3, 'Training/validation images should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size, patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i): 
        in_img = Image.open(self.imgfiles[0][i])
        gt_img = Image.open(self.imgfiles[1][i])
        w, h = in_img.size

        flip_op = np.random.randint(3)
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)
        in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), flip_op)
        gt_img_patches = self.preprocess(gt_img, self.patch_size, (patch_x, patch_y), flip_op)
        return {'input': torch.from_numpy(in_img_patches), 'gt': torch.from_numpy(gt_img_patches)}


