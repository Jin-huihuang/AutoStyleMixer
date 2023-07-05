import torch
from torchvision.utils import save_image
import numpy as np

def denorm(tensor, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    std = torch.Tensor(std).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor(mean).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def save_image_from_tensor_batch(batch, column, path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], device='cpu'):
    batch = denorm(batch, device, mean, std)
    save_image(batch, path, nrow=column)


def mean_teacher(model, teacher, momentum=0.9995):
    model_dict = model.state_dict()
    teacher_dict = teacher.state_dict()
    for k, v in teacher_dict.items():
        teacher_dict[k] = v * momentum + (1 - momentum) * model_dict[k]

    teacher.load_state_dict(teacher_dict)


def update_teacher(model, teacher, momentum=0.9995):
    model_dict = model.state_dict()
    teacher_dict = teacher.state_dict()
    for (k_q, v_q), (k_k, v_k) in zip(model_dict.items(), teacher_dict.items()):
        assert k_k == k_q, "state_dict names are different!"
        # if k_q.endswith('statistics'): continue
        if 'num_batches_tracked' in k_k:
            v_k.copy_(v_q)
        else:
            v_k.copy_(v_k * momentum + (1. - momentum) * v_q)


def warm_update_teacher(model, teacher, momentum=0.9995, global_step=2000, warm_up=0):
    if global_step > warm_up - 1:
        momentum = min(1 - 1 / (global_step + 2 - warm_up), momentum)
    else:
        momentum = 1 - 0.5 * (global_step + 1) / warm_up
    model_dict = model.state_dict()
    teacher_dict = teacher.state_dict()
    for (k_q, v_q), (k_k, v_k) in zip(model_dict.items(), teacher_dict.items()):
        assert k_k == k_q, "state_dict names are different!"
        # if k_q.endswith('statistics'): continue
        if 'num_batches_tracked' in k_k:
            v_k.copy_(v_q)
        else:
            v_k.copy_(v_k * momentum + (1. - momentum) * v_q)


def preprocess_teacher(model, teacher):
    for param_m, param_t in zip(model.parameters(), teacher.parameters()):
        param_t.data.copy_(param_m.data)  # initialize
        param_t.requires_grad = False  # not update by gradient

def calculate_correct(scores, labels):
    assert scores.size(0) == labels.size(0)
    _, pred = scores.max(dim=1)
    correct = torch.sum(pred.eq(labels)).item()
    return correct


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def step_rampup(current, rampup_length):
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return 0.0


def get_current_consistency_weight(epoch, weight, rampup_length, rampup_type='step'):
    if rampup_type == 'step':
        rampup_func = step_rampup
    elif rampup_type == 'linear':
        rampup_func = linear_rampup
    elif rampup_type == 'sigmoid':
        rampup_func = sigmoid_rampup
    else:
        raise ValueError("Rampup schedule not implemented")

    return weight * rampup_func(epoch, rampup_length)