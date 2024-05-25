import torch
import os
from data.cars import TrainDataset
import matplotlib.pyplot as plt
from utils.model import load_checkpoint


def predict(net, inputs):
    """
    预测
    :param net: 模型的实力对象
    :param inputs: (b, c, h, w)
    :return: mask_pred
    """
    net.eval()  # 调整到推理模式

    pred = net(inputs)  # b, num_classes, h, w
    pred = torch.argmax(pred, dim=1)  # b, h, w

    return pred


def main():
    from network.net import UNet as Model
    # from bluenet2.net import BlueNet as Model

    model_name = 'unet'
    version = 'sgd'
    weight = '001.pth'
    checkpoint_home = '/home/blueberry/cache/checkpoints/wecool'  # 模型权重家目录
    data_root = '/home/blueberry/cache/data/UNet'  # 数据集的根目录

    checkpoint_home = os.path.join(checkpoint_home, f"{model_name}/{version}")
    dataset = TrainDataset(data_dir=data_root, resize=(480, 320))

    # 选一个数据
    case_num = 20
    case_name = dataset.files[case_num]
    image, mask = dataset[case_num]
    print(f'case_name: {case_name}')
    print(f'image.shape: {image.shape}')
    print(f'mask.shape: {mask.shape}')

    # 为了能输入到模型中进行预测, 调整一下 image 的维度
    inputs = image.unsqueeze(0)
    print(f'inputs.shape: {inputs.shape}')

    # 创建模型
    net = Model(in_channels=3, out_channels=2)

    # 加载模型权重
    load_checkpoint(net, weight, checkpoint_home)

    # 预测
    pred = net(inputs)
    pred_mask = torch.argmax(pred, dim=1)  # 在通道这个维度上做 argmax 操作
    print(f'pred.shape: {pred.shape}')
    print(f'pred_mask.shape: {pred_mask.shape}')

    # 显示出来
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

    plt.imshow(mask)
    plt.show()

    plt.imshow(pred_mask[0])
    plt.show()


if __name__ == '__main__':
    main()
