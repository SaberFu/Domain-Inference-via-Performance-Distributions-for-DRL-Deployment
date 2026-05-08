import torch
import torch.nn as nn
import numpy as np

class KinematicBaseModel(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        #self.len_horizon = args.frame_predict
        #self.len_history = args.frame_condition
        pass
    
    def sample(self, past):
        __doc__ = r"""Inference, given past frame/frames, predict future
        Input:
            past: tensor of shape (Batch size, frame size)
        Output:
            tensor of shape (Batch size, frame size)
        """
        raise NotImplementedError("sample function not implemented!") 

    def forward(self, past, gt):
        raise NotImplementedError("loss bp not implemented!") 

    