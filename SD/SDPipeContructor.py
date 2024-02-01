import torch
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, AutoencoderKL, UNet2DConditionModel
from diffusers import ControlNetModel, UniPCMultistepScheduler
from diffusers import DiffusionPipeline, StableDiffusionUpscalePipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UniPCMultistepScheduler, KDPM2DiscreteScheduler
from processors.processor import preprocessConditions, image_grid, merge_imgs, load_input_imgs, read_json

class SDPipeContructor():
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.sd_params = self.args['sd_params']
        self.seed = self.sd_params['seed']
        
        self.modelname = self.sd_params['modelname']
        self.sd_model_path = self.sd_params['sd_model_path']
        self.lora_name = self.sd_params['lora_name']
        self.torch_dtype = self.sd_params['torch_dtype']
        self.init_pipe(self.args, self.sd_params['style'])

        if self.sd_params['life4cut']:
            self.batch_prompt_generator = self.set_batch_prompt_generator(
                batch_size=self.sd_params['batch_size'],
                prompt=self.sd_params['prompt'],
                n_prompt=self.sd_params['n_prompt'],
                seed=self.seed
            )

    def init_pipe(self, args, style):
        if style == 'anime_01':
            pass
            from prompts.anime01_prompts import animstyle_prompts
            self.base_prompts = animstyle_prompts['{}'.format(self.sd_params['prompt_version'])]
            self.anime_01_pipe = Anime01_Pipe(args)
            self.pipe = self.anime_01_pipe.set_pipe(self.device)
        elif style == 'real_01':
            from prompts.real01_prompts import real01_prompts
            self.base_prompts = real01_prompts['{}'.format(self.sd_params['prompt_version'])]
            self.real_01_pipe = Real01_Pipe(args)
            self.pipe = self.real_01_pipe.set_pipe(self.device)

    def sd_output(self, img_frames, conditions_desc_dict, gpt_response):
        output_img_lst = []
        for img_frame_idx in tqdm(iterable=range(len(img_frames)), total=len(img_frames), desc='SD Output', leave=False):
            print(type(conditions_desc_dict['hed'][img_frame_idx]), type(conditions_desc_dict['depth'][img_frame_idx]))
            
            print(conditions_desc_dict['hed'][img_frame_idx].shape)
            print(conditions_desc_dict['depth'][img_frame_idx].shape)
            condition_images = [conditions_desc_dict['hed'][img_frame_idx], conditions_desc_dict['depth'][img_frame_idx]] #conditions_desc_dict['pose'][img_frame_idx], 
            output = self.pipe(
                controlnet_conditioning_image=condition_images,
                image=img_frames[img_frame_idx],
                strength=self.sd_params['denoise_strength'],
                prompt=gpt_response['positive'],
                negative_prompt=gpt_response['negative'],
                num_inference_steps=self.sd_params['num_inference_steps'],
                guidance_scale=self.sd_params['cfg_scale'],
                controlnet_conditioning_scale=self.sd_params['controlnet_conditioning_scale'], #, 0.7 hed, pose, depth conditioning scale ratio
            )

            # prompt_embeds (`torch.FloatTensor`, *optional*):
            # Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            # provided, text embeddings will be generated from `prompt` input argument.

            if self.sd_params['if_recons_face']:
                output = self.face_pipe(
                    init_images=output.images[0], 
                    mask_imgs=conditions_desc_dict['facemask'][0],
                    prompt='dog face, cat face', 
                    n_prompt='girl face', 
                    seed=self.seed, 
                    num_inference_steps=50, 
                    denoising_strength=0.2, 
                    cfg_scale=12, 
                    if_morph=False, 
                    dd_dilation_factor_b=0, 
                    dd_offset_x_b=0, 
                    dd_offset_y_b=0)

            output_img_lst.append(output.images[0])
        
        return output_img_lst

    def set_batch_prompt_generator(self, batch_size=4, prompt=None, n_prompt=None, seed=None):
        generator = [torch.Generator(device=self.device).manual_seed(seed) for i in range(batch_size)]
        prompts = batch_size * [prompt]
        n_prompts = batch_size * [n_prompt]                                                                                                                                                                                       

        return {"prompt": prompts, "n_prompt":n_prompts, "generator": generator}

class Real01_Pipe():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sd_params = self.args['sd_params']
        self.model_path = './models/{}/'.format(self.sd_params['modelname'])
        self.lora_path = './models/Lora/{}/'.format(self.sd_params['lora_name']) # ulzzang-6500-v1.1는 수정 필요
        self.schedulername = 'UniPCMultistepScheduler' # UniPCMultistepScheduler, PNDMScheduler
        self.scheduler_config = read_json("./models/schedulers/{}.json".format(self.schedulername))
        self.scheduler = self.set_scheduler(self.schedulername, self.scheduler_config)
        if self.sd_params['torch_dtype']==16:
            self.torch_dtype = torch.float16
        elif self.sd_params['torch_dtype']==32:
            self.torch_dtype = torch.float32
        else:
            raise NotImplementedError('Please input correct fp: 16 or 32')

    def set_pipe(self, device):
        controlnet, vae, unet = self.set_models()
        pipe = DiffusionPipeline.from_pretrained(
            self.sd_params['sd_model_path'],
            vae=vae,
            # text_encoder=self.text_encoder,
            # tokenizer=self.tokenizer,
            controlnet=controlnet,
            unet=unet,
            scheduler=self.scheduler,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
            custom_pipeline=self.sd_params['custom_pipeline']
        ).to(device)

        if self.sd_params['if_lora'] == True:
            pipe.unet.load_attn_procs(self.lora_path)

        if self.sd_params['if_recons_face']:
            self.face_pipe = self.set_face_pipe(device, pipe)
            # face_reconstructor.pipe_inpaint.enable_model_cpu_offload()
            self.face_pipe.pipe_inpaint.enable_xformers_memory_efficient_attention()


        # pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        return pipe

    def set_models(self, ):
        controlnet = [ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=self.torch_dtype), 
                      ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=self.torch_dtype),
                      ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=self.torch_dtype)]
        vae = AutoencoderKL.from_pretrained(self.sd_params['vae_model_path'], torch_dtype=self.torch_dtype) # stabilityai/sd-vae-ft-ema
        unet = UNet2DConditionModel.from_pretrained(self.sd_params['unet_model_path'], torch_dtype=self.torch_dtype)

        return controlnet, vae, unet
    
    def set_scheduler(self, schedulername, scheduler_config):
        if schedulername == 'UniPCMultistepScheduler':        
            scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
        elif schedulername == 'KDPM2DiscreteScheduler':
            scheduler = KDPM2DiscreteScheduler.from_config(scheduler_config)

        return scheduler

    def set_face_pipe(self, ):
        pass


class Anime01_Pipe():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sd_params = self.args['sd_params']
        self.model_path = './models/{}/'.format(self.modelname)
        self.lora_path = './models/Lora/{}/'.format(self.lora_name) # ulzzang-6500-v1.1는 수정 필요
        self.schedulername = 'UniPCMultistepScheduler' # UniPCMultistepScheduler, PNDMScheduler
        self.scheduler_config = read_json("./models/schedulers/{}.json".format(self.schedulername))
        self.scheduler = self.set_scheduler(self.schedulername, self.scheduler_config)

    def set_pipe(self, device):
        controlnet, vae, unet = self.set_models()
        pipe = DiffusionPipeline.from_pretrained(
            self.sd_params['sd_model_path'],
            vae=vae,
            # text_encoder=self.text_encoder,
            # tokenizer=self.tokenizer,
            controlnet=controlnet,
            unet=unet,
            scheduler=self.scheduler,
            safety_checker=None,
            torch_dtype=self.sd_params['torch_dtype'],
            custom_pipeline=self.sd_params['custom_pipeline']
        ).to(device)

        if self.sd_params['if_lora'] == True:
            pipe.unet.load_attn_procs(self.lora_path)

        if self.sd_params['if_recons_face']:
            self.face_reconstructor = self.set_face_pipe(device, pipe)
            # face_reconstructor.pipe_inpaint.enable_model_cpu_offload()
            self.face_reconstructor.pipe_inpaint.enable_xformers_memory_efficient_attention()


        # pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        return pipe

    def set_models(self, ):
        controlnet = [ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed", torch_dtype=self.sd_params['torch_dtype']), 
                      ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=self.sd_params['torch_dtype']),
                      ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=self.sd_params['torch_dtype'])]
        vae = AutoencoderKL.from_pretrained(self.sd_params['vae_model_path'], torch_dtype=self.sd_params['torch_dtype']) # stabilityai/sd-vae-ft-ema
        unet = UNet2DConditionModel.from_pretrained(self.sd_params['unet_model_path'], torch_dtype=self.sd_params['torch_dtype'])

        return controlnet, vae, unet
    
    def set_scheduler(self, schedulername, scheduler_config):
        if schedulername == 'UniPCMultistepScheduler':        
            scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
        elif schedulername == 'KDPM2DiscreteScheduler':
            scheduler = KDPM2DiscreteScheduler.from_config(scheduler_config)

        return scheduler

    def set_face_pipe(self, ):
        pass