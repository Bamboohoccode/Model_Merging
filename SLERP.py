import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from collections import OrderedDict
from typing import Sequence


def Factor(tensor1 : torch.Tensor,
           tensor2 : torch.Tensor,
           t : float) -> torch.Tensor:
    flatten_1 = tensor1.flatten()
    flatten_2 = tensor2.flatten()
    norm_1 = torch.linalg.norm(flatten_1)
    norm_2 = torch.linalg.norm(flatten_2)
    flatten_1 = flatten_1 / norm_1
    flatten_2 = flatten_2 / norm_2
    cosin_S = torch.dot(flatten_1,flatten_2)
    cosin_S = cosin_S.clamp(-1.0,1.0)
    theta = torch.acos(cosin_S)
    sin_theta = torch.sin(theta)
    output_tensor = (torch.sin(theta * (1 - t)) * tensor1) / sin_theta + torch.sin(theta * t) * tensor2 / sin_theta
    return output_tensor

def SLERP_Merging(model_1 : nn.Module,
                  model_2 : nn.Module,
                  t: float) -> nn.Module:
    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()
    if(state_dict_1.keys() != state_dict_2.keys()):
        raise ValueError("2 Model Không có chung kiến trúc")
    keys = state_dict_1.keys()
    merged_SD = OrderedDict()
    for name in keys:
        merged_tensor = Factor(model_1[name],model_2[name],)

