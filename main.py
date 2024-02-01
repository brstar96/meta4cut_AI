import os, sys, cv2, argparse
import torch, torch.hub
import warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from pytz import timezone
from datetime import datetime

from PIL import Image, ImageDraw
from image_captioning import ImageCaptioning, ConversationBot # ImageCaptioning 쓰게 해보기
from prompt_engineering import gen_prompt
from pytorch_lightning import seed_everything
from processors.processor import preprocessConditions, image_grid, merge_imgs, load_input_imgs, read_json
from SDPipeContructor import SDPipeContructor
from configs import parse


def main_debug(args, device, lg, mode='debug'):
    sd_params = args['sd_params']
    datasets_params = args['datasets']
    preprocess_params = datasets_params['preprocess']
    now = datetime.now(timezone('Asia/Seoul'))
    formattedDateToday = now.strftime("%Y%m%d_%H%M%S")
    bg_img = Image.open('./_assets/bg_imgs/download_image.png')
    
    if mode == 'debug':
        if sd_params['format']=='photos' or sd_params['format']=='life4cut':
            _input = load_input_imgs(lg=lg, input_path=datasets_params['input_src_root'], 
                                        input_type=sd_params['format'])
        elif sd_params['format']=='videos':
            available_cores = os.cpu_count()
            _input = load_input_imgs(lg=lg, input_path=datasets_params['input_src_root'], 
                                        input_type=sd_params['format'], 
                                        num_processes=available_cores, 
                                        frame_interval=preprocess_params['frame_interval'])
        else:
            raise ValueError('input_type should be `photos` or `life4cut` or `videos`.')
        
        lg.info('Input data loaded!')
 
    # set prompt
    if sd_params['style'] == 'anime':
        from prompts.anime01_prompts import animstyle_prompts
        base_prompts = animstyle_prompts['{}'.format(sd_params['prompt_version'])]
    elif sd_params['style'] == 'real_01':
        from prompts.real01_prompts import real01_prompts
        base_prompts = real01_prompts['{}'.format(sd_params['prompt_version'])]
    else:
        raise ValueError('Invalid style input.')
       
    # set preprocessor & SD pipeline
    visualgpt_bot = ConversationBot(device=device) if sd_params['visual_gpt'] else None
    
    preprocessor = preprocessConditions(
        args=args,
        lg=lg,
        device=device,
        visualgpt_bot=visualgpt_bot, 
        openai_api_key=args['OPENAI_API_KEY'],
        annotation_typ_lst=sd_params['annotation_typ_lst'], 
        face_recon=False)
    SD_Pipe = SDPipeContructor(args=args, device=device)

    # prepare conditioned images
    if sd_params['format'] == 'photos' or sd_params['format'] == 'life4cut':
        for group_idx, value in _input.items():
            filenames = list(value.keys())
            img_frames = list(value.values())
            merged_input = image_grid(img_frames, 2, 2)
            conditions_desc_dict = preprocessor.get_conditions_with_desc(imgs=img_frames, format=sd_params['format'])
            gpt_response = gen_prompt(image_caption=conditions_desc_dict['caption'], 
                              style=sd_params['style'], 
                              basic_prompt=base_prompts, 
                              model_name=sd_params['modelname'], 
                              lora_name=sd_params['lora_name'], 
                              visual_gpt=sd_params['visual_gpt'],
                              openai_api_key=args['OPENAI_API_KEY'])          
    else:
        for group_idx, img_frames in _input.items():
            conditions_desc_dict = preprocessor.get_conditions_with_desc(imgs=img_frames, format=sd_params['format'])
            gpt_response = gen_prompt(image_caption=None, 
                              style=sd_params['style'], 
                              basic_prompt=base_prompts, 
                              model_name=sd_params['modelname'], 
                              lora_name=sd_params['lora_name'], 
                              visual_gpt=sd_params['visual_gpt'],
                              openai_api_key=args['OPENAI_API_KEY'])

            output_img_lst = SD_Pipe.sd_output(img_frames, conditions_desc_dict, gpt_response)
            sample_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = '{}/{}/{}'.format(datasets_params['output_root'], sd_params['modelname'], formattedDateToday)
            filename = "sample_input_with_output_iter{}_{}_{}.jpg".format(sd_params['num_inference_steps'], sample_timestamp, os.path.basename(filenames[0]).replace('.jpg', '').replace('.png', ''))
            save_result(sd_params, output_img_lst, output_dir, filename, sample_timestamp, conditions_desc_dict, merged_input=None)
 

def save_result(sd_params, output_img_lst, output_dir, filename, sample_timestamp, conditions_desc_dict, merged_input=None): # save output    
    if format == 'life4cut':
        merged_output = image_grid(output_img_lst, 2, 2)
        os.makedirs(output_dir, exist_ok=True)

        if args['to_show_lst'] is not None:
            toshow_lst = []
            toshow_lst.append(merged_input)

            for to_show in args['to_show_lst']:
                merged_img = image_grid(conditions_desc_dict['{}'.format(to_show)], 2, 2)
                toshow_lst.append(merged_img)

            toshow_lst.append(merged_output)
            merge_imgs(toshow_lst).save("{}/{}".format(output_dir, filename))
        else:
            merge_imgs([merged_input, merged_output]).save("{}/{}".format(output_dir, filename))
    elif format == 'video':
        os.makedirs(output_dir, exist_ok=True)

        if format == 'videos':
            filename = "sample_input_with_output_iter{}_{}_{}.mp4".format(sd_params['num_inference_steps'], sample_timestamp, os.path.basename(filename[0]).replace('.jpg', '').replace('.png', ''))
            output_filepath = os.path.join(output_dir, filename)

            # 비디오 저장을 위한 설정
            img_array = np.array(output_img_lst[0])
            height, width, _ = img_array.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 코덱 설정
            fps = 30 # 프레임 수 설정
            video_writer = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

            # output_img_lst의 이미지를 비디오로 저장
            for img in output_img_lst:
                bgr_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_image)

            video_writer.release()
        else:
            raise ValueError('input_type should be videos when format is video')


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    warnings.filterwarnings(action='ignore')

    parser = argparse.ArgumentParser(description='SD RealDosMix')
    parser.add_argument('--yaml_path', default='./configs/RealDosMix_2023.04.1.0.1.yaml')  
    
    args = parser.parse_args()
    args, lg, current_time = parse(args)
    lg.info('Initialize Pipeline... \ncID:{}\nrunname:{}'.format(os.uname()[1], args['runname']))
    
    # Set device index
    if len(args['gpu']) > 1:
        gpus = ", ".join([x for x in args['gpu']])
    else:
        gpus = str(args['gpu'])

    # Get the available GPU IDs
    available_gpu_ids = [int(x) for x in gpus.split(", ")]
    world_size = len(available_gpu_ids)
    

    if world_size == 1:
        single_gpu_id = available_gpu_ids[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(single_gpu_id)
        device = torch.device(f"cuda:{single_gpu_id}")
        lg.info('Initialize Single GPU Training...')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpu_ids))
        devices = [torch.device(f"cuda:{gpu_id}") for gpu_id in available_gpu_ids]
    
    lg.info("Available device type: {}".format(device if world_size == 1 else devices))
    lg.info("Current CUDA device: cuda:{}".format(torch.cuda.current_device()))
    lg.info("Worker process count: {}".format(args['datasets']['num_workers']))

    # Set seed
    seed_everything(args['datasets']['seed'])
    main_debug(args, device, lg, args['mode'])