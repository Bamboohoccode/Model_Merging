import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from collections import OrderedDict
from typing import Sequence




def LinearMerging(models : Sequence[nn.Module],
                 weights : Sequence[float]) -> OrderedDict:
    if(len(models) == 0):
        raise ValueError("Không có model trong list")
    sum_w = 0
    for weight in weights:
        sum_w += weight
    if(sum_w != 1):
        raise ValueError("Tổng input của weights phải bằng 1")
    if(weights is None):
        weights = [1.0/len(models) for _ in range(len(model))]

    if(len(weights) != len(models)):
        raise ValueError("Input không hợp lệ(Số lượng weights và models không giống nhau)")

    state_dicts = [model.state_dict() for model in models]
    model_keys = state_dicts[0].keys()

    for state_dict in state_dicts[1:]:
        if(state_dict.keys() != model_keys):
            raise ValueError("Kiến trúc của các model khác nhau")
    merged_SD = OrderedDict()

    for name in model_keys:        
        tensors = [model[name] for model in models]

        merged_tensor = torch.zero_like(tensors[0])

        for tensor,weight in zip(tensors,weights):
            merged_tensor.add_(tensor.float(),alpha = weight)
        merged_SD[name] = merged_tensor
    print(f"Đã gộp xong {len(models)} với nhau !")
    return merged_SD
