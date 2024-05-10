import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor, n_classes):
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        n_classes = inputs.size(1)

        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        target = self._one_hot_encoder(target, n_classes)

        if weight is None:
            weight = [1] * n_classes

        assert inputs.size() == target.size(), \
            'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        class_wise_dice = []
        loss = 0.0
        for i in range(0, n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / n_classes


class CrossEntropyDiceLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dice_loss = DiceLoss()

    def forward(self, pred, label):
        label = label.long()
        loss = super().forward(pred, label)
        loss += self.dice_loss(pred, label, softmax=True)

        return loss


if __name__ == '__main__':
    # inputs
    pred = torch.rand(size=(8, 2, 224, 224), dtype=torch.float)
    label = torch.ones(size=(8, 224, 224), dtype=torch.long)

    # CrossEntropyLoss
    cross_entropy_loss = nn.CrossEntropyLoss()
    cl = cross_entropy_loss(pred, label)
    print(f"cross_entropy_loss: {cl}")

    # DiceLoss
    dice_loss = DiceLoss()
    dl = dice_loss(pred, label)
    print(f"dice_loss: {dl}")

    # CrossEntropyDiceLoss
    both_loss = CrossEntropyDiceLoss()
    bl = both_loss(pred, label)
    print(f"both_loss: {bl}")
