import os, sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from processors.hed_net import Network
from PIL import Image
from .util import resize_batch

class HEDInference(nn.Module):
    def __init__(self, lg, device, model_path):
        super().__init__()
        print("initialize HED")
        self.lg = lg
        self.model_path = model_path
        self.device = device
        self.hed_model = self.load_model(self.model_path)
        self.lg.info('HED model loaded into {}.'.format(self.device))
        
    def load_model(self, model_path):                
        return Network(model_path).eval().to(self.device)

    def forward(self, x, return_pil=True):
        input_b, input_c, input_h, input_w = x.shape
        prediction = self.hed_model(x)
        prediction = (prediction.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        prediction = resize_batch(prediction, (input_h, input_w))
        
        if return_pil:
            detected_map = Image.fromarray(detected_map)

        return prediction

    def unload_hed_model(self,):
        global netNetwork
        if netNetwork is not None:
            netNetwork.cpu()

    def nms(self, x, t, s):
        x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

        f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
        f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
        f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

        y = np.zeros_like(x)

        for f in [f1, f2, f3, f4]:
            np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

        z = np.zeros_like(y, dtype=np.uint8)
        z[y > t] = 255
        return z

    
