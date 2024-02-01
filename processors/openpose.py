import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from PIL import Image
from einops import rearrange
from .openpose_models import Body, Hand
from . import util

class OpenposeDetector:
    def __init__(self, lg, device, body_modelpath, hand_modelpath, if_hand=False):
        self.lg=lg
        self.device = device
        print("initialize OpenPose")

        self.body_modelpath = body_modelpath
        self.hand_modelpath = hand_modelpath
        self.if_hand=if_hand
        self.body_estimation = Body(self.lg, self.device, self.body_modelpath)

        if self.if_hand:
            self.hand_estimation = Hand(self.lg, self.device, self.hand_modelpath)

    def forward(self, input_image, return_pil=True):        
        # input_image = rearrange(input_image, 'h w c -> 1 c h w')
        candidate, subset = self.body_estimation(input_image)
        canvas = np.zeros_like(input_image)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        if self.if_hand:
            hands_list = util.handDetect(candidate, subset, input_image)
            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                peaks = self.hand_estimation(input_image[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                all_hand_peaks.append(peaks)
            canvas = util.draw_handpose(canvas, all_hand_peaks)

        if return_pil:
            canvas = Image.fromarray(canvas)

        return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())