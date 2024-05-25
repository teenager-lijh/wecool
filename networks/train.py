import os
import torch
from torch import nn
from torch import optim
from utils.model import save_checkpoint
from utils.model import load_checkpoint
from torch.utils.data import DataLoader
from utils.loss_func import CrossEntropyDiceLoss
from data.cars import TrainDataset
from utils.logger import get_logger


def net_trainer(
    net,
    resume_from,
    optimizer,
    loss_func,
    num_epochs,
    batch_size,
    dataset,
    logger,
    saved_interval,
    checkpoint_home,
    device,
    num_workers
):
    """
    :param net: 模型
    :param resume_from: 使用 resume_from 作为预训练权重
    :param optimizer: 优化器的实例对象
    :param loss_func: 损失函数的实例对象
    :param num_epochs: 训练多少轮
    :param batch_size: dataloader 的批量大小
    :param dataset: 数据集的实例对象
    :param logger: 日志的实例对象
    :param saved_interval: 每隔 saved_interval 次保存一次权重到 checkpoint_home
    :param checkpoint_home: 模型权重的保存路径
    :param device: 在哪个设备上进行训练
    :param num_workers: dataloader 使用几个进程进行数据集加载
    :return:
    """

    if resume_from is None:
        resume_from = 0  # 确实应该从 (resume_from, max_epochs); resume_from 这轮已经训练完成了

    # 设置网络为训练模式
    net.train()

    # 将网络移动到指定的设备上
    net.to(device)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    # 统计迭代次数
    max_iterations = (len(dataset) // batch_size) * num_epochs
    iter_nums = 1

    logger.info(f'Max Iter:{max_iterations}')

    # 开始训练
    for epoch in range(resume_from, num_epochs):

        # 遍历数据加载器
        for inputs, labels in dataloader:
            # 将数据移动到指定的设备上
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 修改学习率 训练过程中逐渐减小学习率
            lr = None
            score = (1.0 - iter_nums / (max_iterations + max_iterations * 0.1)) ** 0.9
            for param_group in optimizer.param_groups:
                lr = param_group['lr'] * score

            # 前向传播
            outputs = net(inputs)

            # 计算损失
            loss = loss_func(outputs, labels).sum()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            logger.info(f'Epoch:{epoch+1}, Iter:{iter_nums}, Loss:{float(loss):.4f}, Lr:{float(lr)}, score:{score}')
            iter_nums = iter_nums + 1

        if (epoch+1) % saved_interval == 0 or (epoch+1) == num_epochs:
            logger.info(f'try to save checkpoint {epoch+1:03}.pth')
            save_checkpoint(net, f'{epoch+1:03}.pth', checkpoint_home)
            logger.info(f'saved checkpoint {epoch+1:03}.pth')

    logger.info(f'training done!')


def train_net():
    # from network.net import UNet as Model
    from bluenet2.net import BlueNet as Model

    model_name = 'bluenet2'
    train_version = 'sgd'  # 这个模型训练的版本号
    resume_from = None  # 是否要在 resume_from 这个权重的基础上继续训练; None 则代表不使用任何预训练权重; 填权重的前缀,如 66 整型值
    device = torch.device('cuda:2')  # 使用第 0 号显卡进行训练
    logger_name = 'train_logger'
    num_workers = 4

    # =============
    lr = 0.05  # 学习率
    max_epochs = 100  # 在数据集上训练多少轮
    num_classes = 2  # 这是几个类别的图像分割问题 (背景也算一个类别)
    batch_size = 8  # 数据集的批量大小
    data_root = '/home/blueberry/cache/data/UNet'  # 数据集的根目录
    checkpoint_home = f'/home/blueberry/cache/checkpoints/wecool'  # 保存模型权重和日志的根目录

    saved_interval = 1  # 每训练多少轮, 保存一次权重到 checkpoint_home 目录

    # =============
    # 确保目录存在 ==> 不存在的话自动创建目录
    checkpoint_home = os.path.join(checkpoint_home, f'{model_name}/{train_version}')  # 模型权重和日志的保存目录
    os.makedirs(checkpoint_home, exist_ok=True)


    # 创建日志; 如果是从头训练则覆盖日志文件中的内容, 如果是在预训练权重的基础上训练则追加日志内容
    if resume_from is None:
        logger = get_logger(name=logger_name, file_path=os.path.join(checkpoint_home, f'{train_version}.log'), mode='w')
    else:
        logger = get_logger(name=logger_name, file_path=os.path.join(checkpoint_home, f'{train_version}.log'), mode='a')

    # 创建模型
    net = Model()

    # 创建数据集
    dataset = TrainDataset(data_root, resize=(480, 320))

    # 定义损失函数
    loss_func = CrossEntropyDiceLoss()

    # 定义优化器
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # 输出参数信息
    logger.info(locals())

    if resume_from is not None:
        logger.info(f'try to load checkpoint {resume_from}.pth')
        load_checkpoint(net, f'{resume_from:03}.pth', checkpoint_home)
        logger.info(f'load checkpoint {resume_from}.pth successfully!')

    net_trainer(
        net=net,
        resume_from=resume_from,
        optimizer=optimizer,
        loss_func=loss_func,
        num_epochs=max_epochs,
        batch_size=batch_size,
        dataset=dataset,
        logger=logger,
        saved_interval=saved_interval,
        checkpoint_home=checkpoint_home,
        device=device,
        num_workers=num_workers
    )


if __name__ == '__main__':
    train_net()
