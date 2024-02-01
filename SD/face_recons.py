import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

class FaceReconstruction():
    def __init__(self, device, control_pipe):
        super().__init__()
        self.device = device
        self.control_pipe = control_pipe
        self.set_sd_pipeline(self.control_pipe)

    def set_sd_pipeline(self, control_pipe):
        self.pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path="runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16).to("cuda:0")
        self.pipe_inpaint.unet = control_pipe.unet
        self.pipe_inpaint.vae = control_pipe.vae
        self.pipe_inpaint.scheduler = control_pipe.scheduler

    def reconstruct_face(self, init_images, mask_imgs, prompt, n_prompt, seed, num_inference_steps=50, denoising_strength=0.2, cfg_scale=12, if_morph=False, dd_dilation_factor_b=0, dd_offset_x_b=0, dd_offset_y_b=0):
        generator = [torch.Generator(device="cuda:0").manual_seed(seed) for i in range(4)]
        print('aaa', type(init_images), type(mask_imgs))
        print('bb', init_images.size, mask_imgs.size)
        
        
        output_images = self.pipe_inpaint(
            prompt=prompt,
            image = init_images,
            mask_image = mask_imgs,
            negative_prompt=n_prompt,
            # generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
        )
        
        if if_morph: # need to be updated
            masks_b_pre = [self.dilate_masks(masks_b_pre, dd_dilation_factor_b, 1) for output_img in output_images]
            masks_b_pre = self.offset_masks(masks_b_pre,dd_offset_x_b, dd_offset_y_b)
        
        return output_images

    def dilate_masks(self, masks, dilation_factor, iter=1):
        if dilation_factor == 0:
            return masks

        dilated_masks = []
        kernel = np.ones((dilation_factor,dilation_factor), np.uint8)

        for i in range(len(masks)):
            cv2_mask = np.array(masks[i])
            dilated_mask = cv2.dilate(cv2_mask, kernel, iter)
            dilated_masks.append(Image.fromarray(dilated_mask))

        return dilated_masks

    def offset_masks(self, masks, offset_x, offset_y):
        if (offset_x == 0 and offset_y == 0):
            return masks

        offset_masks = []

        for i in range(len(masks)):
            cv2_mask = np.array(masks[i])
            offset_mask = cv2_mask.copy()
            offset_mask = np.roll(offset_mask, -offset_y, axis=0)
            offset_mask = np.roll(offset_mask, offset_x, axis=1)
            
            offset_masks.append(Image.fromarray(offset_mask))

        return offset_masks

    def update_result_masks(self, results, masks):
        for i in range(len(masks)):
            boolmask = np.array(masks[i], dtype=bool)
            results[2][i] = boolmask
        return results