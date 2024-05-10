import os
import torch


def save_checkpoint(model, fname, checkpoint_home):
    # 保存模型权重
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(checkpoint_home, fname))


def load_checkpoint(model, weight_file, checkpoint_home):

    state_dict = torch.load(os.path.join(checkpoint_home, weight_file), map_location='cpu')
    model.load_state_dict(state_dict)
