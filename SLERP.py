import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from collections import OrderedDict
from typing import Sequence


def Factor(tensor1 : torch.tensor,
           tensor2 : torch.tensor,
           t : float) -> Sequence[float]:
    flatten_1 = tensor1.flatten()
    flatten_2 = tensor2.flatten()
    cosin_S = flatten_1 @ flatten_2.T
    cosin_S = cosin_S.clamp(-1.0,1.0)
    theta = torch.acos(cosin_S)

    return 

def SLERP_Merging():
    pass