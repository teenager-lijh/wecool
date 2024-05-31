import numpy as np
import torch
from torch import nn

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(label: list, pred: list, epsilon=1e-4, func=None):
    label = np.array(label, dtype=np.float32)
    pred = np.array(pred, dtype=np.float32)
    if func:
        pred = func(pred)
    pred[pred < epsilon] = epsilon
    pred = np.log(pred)
    loss = - np.sum(label * pred)
    return loss

def main():
    loss = nn.CrossEntropyLoss()
    pred = torch.tensor([5, 2, 2, 2, 2], dtype=torch.float).view(1, 5, 1)
    label = torch.tensor([0], dtype=torch.long).view(1, 1)
    print(f'pred: {pred.shape}, label: {label.shape}, loss: {loss(pred, label)}')

    pred = [5, 2, 2, 2, 2]
    label = [1, 0, 0, 0, 0]
    print(f'The Softmax(pred): {softmax(np.array(pred))}')
    print(f'label: {label}, pred: {pred}, loss: {cross_entropy_loss(label, pred, func=softmax)}')
    print('=================== line ===================')


    # 相对于上一组来说；这两个概率分布表达的更加接近；因为混乱程度更低；确定性更强
    pred = [5, 1, 1, 1, 1]
    label = [1, 0, 0, 0, 0]
    print(f'The Softmax(pred): {softmax(np.array(pred))}')
    print(f'label: {label}, pred: {pred}, loss: {cross_entropy_loss(label, pred, func=softmax)}')

    loss = nn.CrossEntropyLoss()
    pred = torch.tensor([5, 1, 1, 1, 1], dtype=torch.float).view(1, 5, 1)
    label = torch.tensor([0], dtype=torch.long).view(1, 1)
    print(f'pred: {pred.shape}, label: {label.shape}, loss: {loss(pred, label)}')
    print('=================== line ===================')


if __name__ == '__main__':
    main()
