import cv2
import numpy as np
import torch

from einops import rearrange
from modules.ldm.modules.midas.api import MiDaSInference
from PIL import Image
from .util import resize_batch, write_depth

class MidasDetector:
    def __init__(self, lg, device, model_type, model_path):
        self.lg = lg
        self.device = device
        print("initialize MiDaS")
        self.model = MiDaSInference(model_type=model_type, model_path=model_path).to(self.device)
        self.lg.info('MiDaS model loaded into {}.'.format(self.device))

    def forward(self, input_image, a=np.pi * 2.0, bg_th=0.1, use_normal=False, return_pil=True):
        input_b, input_c, input_h, input_w = input_image.shape
        input_image = input_image / 127.5 - 1.0
        depth = self.model(input_image)

        depth_pt = depth.clone()
        depth_pt -= torch.min(depth_pt)
        depth_pt /= torch.max(depth_pt)
        depth_pt = depth_pt.cpu().numpy()
        depth_image = (depth_pt * 255.0).clip(0, 255).astype(np.uint8)
        depth_image = resize_batch(np.expand_dims(depth_image, axis=1), (input_h, input_w))

        depth_np = depth.cpu().numpy()
        x = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        y = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)
        z = np.ones_like(x) * a
        x[depth_pt < bg_th] = 0
        y[depth_pt < bg_th] = 0

        if use_normal:
            normal = np.stack([x, y, z], axis=2)
            normal /= np.sum(normal ** 2.0, axis=2, keepdims=True) ** 0.5
            normal_image = (normal * 127.5 + 127.5).clip(0, 255).astype(np.uint8)

            if return_pil:
                depth_image = Image.fromarray(depth_image)
                normal_image = Image.fromarray(normal_image)

            return write_depth(depth_image, grayscale=True), normal_image
        else:
            if return_pil:
                depth_image = Image.fromarray(depth_image)

            return write_depth(depth_image, grayscale=True)





