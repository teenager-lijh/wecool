import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np


class TrainDataset(Dataset):
    def __init__(self, data_dir, resize=(480, 320)):
        super().__init__()
        self.data_dir = data_dir
        self.resize = resize
        files = os.listdir(os.path.join(self.data_dir, 'imgs'))
        self.files = [item.split('.')[0] for item in files]
        self.mask_values = [0, 1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self._load_img(self.files[i], is_mask=False)
        mask = self._load_img(self.files[i], is_mask=True)

        # 转化成 np.array 格式
        img = np.asarray(img, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.long)

        # 转换成 tensor; # C, H, W; 把通道放在前面; 并且把像素值归一化到 0 到 1 之间
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.
        mask = torch.from_numpy(mask)

        return img, mask

    # def process(self, img, is_mask=False):
    #     mask_values = self.mask_values
    #     height, width = self.resize[0], self.resize[1]
    #
    #     mask = torch.zeros((height, width), dtype=torch.long)




    def _load_img(self, file_name, is_mask=False):
        data_dir = self.data_dir

        # 构造图像的路径
        if is_mask:
            file_name = file_name + '_mask.gif'
            data_dir = os.path.join(data_dir, 'masks')
        else:
            file_name = file_name + '.jpg'
            data_dir = os.path.join(data_dir, 'imgs')

        obj = Image.open(os.path.join(data_dir, file_name))  # 从磁盘读取一张图像
        obj = obj.resize(size=self.resize)  # 调整图像大小

        return obj


if __name__ == '__main__':
    data_dir = '/home/blueberry/cache/data/UNet'  # your data path

    # 创建数据集对象
    dataset = TrainDataset(data_dir=data_dir, resize=(480, 320))
    # 创建数据加载器  ==> 关于多进程 dataloader 报错 ==> 要在 py 文件中使用 __main__
    # num_workers 指代加载数据的进程数, 并行度越高则加载速度越快 ==> 有时候加载数据的速度是训练速度的瓶颈
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)

    for images, masks in dataloader:
        print(f'images.shape: {images.shape}, dtype:{images.dtype}')
        print(f'masks.shape: {masks.shape}, dtype:{masks.dtype}')
        print(torch.max(masks))
        print(torch.min(masks))
        mask = masks[0]


        break

