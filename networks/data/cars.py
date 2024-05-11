import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class TrainDataset(Dataset):
    def __init__(self, data_dir, resize=(480, 320)):
        super().__init__()
        self.data_dir = data_dir
        self.resize = resize
        files = os.listdir(os.path.join(self.data_dir, 'imgs'))
        self.files = [item.split('.')[0] for item in files]
        self.convert = ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = self._load_img(self.files[i], is_mask=False)
        mask = self._load_img(self.files[i], is_mask=True)

        img = self.convert(img)
        mask = self.convert(mask)

        # 去掉 mask 通道的维度
        mask = torch.squeeze(mask, dim=0)

        return img, mask

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
    data_dir = '/Volumes/SSD/SSD/blueberry/datasets/UNet'  # your data path

    # 创建数据集对象
    dataset = TrainDataset(data_dir=data_dir, resize=(480, 320))
    # 创建数据加载器  ==> 关于多进程 dataloader 报错 ==> 要在 py 文件中使用 __main__
    # num_workers 指代加载数据的进程数, 并行度越高则加载速度越快 ==> 有时候加载数据的速度是训练速度的瓶颈
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)

    for images, masks in dataloader:
        print(f'images.shape: {images.shape}')
        print(f'masks.shape: {masks.shape}')
        break

