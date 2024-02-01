import os
import cv2
import random
import torch
import glob
import warnings
import dlib
import json
import numpy as np
warnings.filterwarnings('ignore')
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from torch.utils.data import Dataset, DataLoader
from annotator.util import resize_image, HWC3
from processors.midas import MidasDetector
from processors.hed import HEDInference
from processors.openpose import OpenposeDetector
from multiprocessing import Pool

class PreprocessDataset(Dataset): # for video gen pipeline
    def __init__(self, args, device, imgs, annotation_typ_lst):
        self.args = args
        self.device=device
        self.imgs = imgs
        self.annotation_typ_lst = annotation_typ_lst

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        input_image = self.imgs[idx]
        processed_images = dict()

        for annotation in self.annotation_typ_lst:
            if annotation == "hed":
                processed_images["hed"] = self.preprocess_hed(input_image)
            elif annotation == "depth":
                processed_images["depth"] = self.preprocess_midas(input_image)
            elif annotation == "pose":
                processed_images["pose"] = self.preprocess_openpose(input_image)
        
        for key in processed_images:
            if key == 'pose':
                processed_images[key] = processed_images[key].transpose(2, 0, 1).astype(np.float32)
            else:
                processed_images[key] = torch.from_numpy(processed_images[key].transpose(2, 0, 1).astype(np.float32)).to(self.device)

        return processed_images

    def preprocess_hed(self, input_image, detect_resolution=384):
        input_image = resize_image(HWC3(np.array(input_image)), detect_resolution)

        return input_image

    def preprocess_midas(self, input_image, detect_resolution=384):
        input_image = resize_image(HWC3(np.array(input_image)), detect_resolution)

        return input_image

    def preprocess_openpose(self, input_image, detect_resolution=384):
        input_image = resize_image(HWC3(np.array(input_image)), detect_resolution)
        print('aaaa', type(input_image))

        return input_image

    def HWC3(self, x):
        assert x.dtype == np.uint8
        if x.ndim == 2:
            x = x[:, :, None]
        assert x.ndim == 3
        H, W, C = x.shape
        assert C == 1 or C == 3 or C == 4
        if C == 3:
            return x
        if C == 1:
            return np.concatenate([x, x, x], axis=2)
        if C == 4:
            color = x[:, :, 0:3].astype(np.float32)
            alpha = x[:, :, 3:4].astype(np.float32) / 255.0
            y = color * alpha + 255.0 * (1.0 - alpha)
            y = y.clip(0, 255).astype(np.uint8)
            return y

    def resize_image(self, input_image, resolution):
        H, W, C = input_image.shape
        H = float(H)
        W = float(W)
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        
        return img

class preprocessConditions():
    def __init__(self, args, lg, device, visualgpt_bot, openai_api_key, annotation_typ_lst, face_recon=False):
        super().__init__()
        self.args=args
        self.lg=lg
        self.device=device
        self.preprocess_params = self.args['datasets']['preprocess']
        self.visualgpt_bot = visualgpt_bot
        self.openai_api_key = openai_api_key
        self.annotation_typ_lst = annotation_typ_lst
        self.face_recon=face_recon

        self.hed_model_path=self.preprocess_params['hed_model_path']
        self.midas_model_path=self.preprocess_params['midas_model_path']
        self.dlib_weight_path=self.preprocess_params['dlib_weight_path']
        self.body_openpose_weight_path = self.preprocess_params['body_openpose_weight_path']
        self.hand_openpose_weight_path = self.preprocess_params['hand_openpose_weight_path']

        self.models = {}
        if 'hed' in self.annotation_typ_lst and self.hed_model_path is not None:
            self.models['hed'] = HEDInference(lg=self.lg, device=self.device, model_path=self.hed_model_path)
        if 'depth' in self.annotation_typ_lst and self.midas_model_path is not None:
            self.models['depth'] = MidasDetector(lg=self.lg, device=self.device, model_type="dpt_hybrid", model_path=self.midas_model_path)
        if 'pose' in self.annotation_typ_lst and self.body_openpose_weight_path is not None:
            self.models['pose'] = OpenposeDetector(lg=self.lg, device=self.device, body_modelpath=self.body_openpose_weight_path, hand_modelpath=self.hand_openpose_weight_path, if_hand=False)

    def get_conditions_with_desc(self, imgs, format='video'):
        self.lg.info("Getting conditions with description")
        start_x, end_x = None, None
        output_dict = dict()
        captions = []

        if format == 'photos' or format == 'life4cut':
            for annotation in self.annotation_typ_lst:
                if annotation == 'canny':
                    start_x, end_x = None, None
                    annotation_imgs = [self.apply_canny(np.array(img), start_x, end_x) for img in imgs]
                elif annotation == 'hed':
                    annotation_imgs = [self.models['hed'].forward(np.array(img)).convert("RGB") for img in imgs]
                elif annotation == 'depth':
                    annotation_imgs = [self.models['midas'].forward(np.array(img), detect_resolution=512).convert("RGB") for img in imgs]
                elif annotation == 'pose':
                    annotation_imgs = [self.models['pose'].forward(img, detect_resolution=512).resize(imgs[0].size) for img in imgs]
                    start_x, end_x = self.get_start_fin_xcoord(np.array(annotation_imgs[0]))
                elif annotation == 'facemask':
                    detector = dlib.get_frontal_face_detector()
                    predictor = dlib.shape_predictor(self.dlib_weight_path)
                    annotation_imgs = [self.get_68landmarks_mask(np.array(img), detector, predictor) for img in imgs]
                else:
                    raise NotImplementedError(f"{annotation} is not implemented yet. Please choose from ('pose', 'hed', 'depth', 'facemask', 'canny")

                output_dict[annotation] = annotation_imgs

            if self.visualgpt_bot is not None:
                for image in imgs:
                    caption = self.visualgpt_bot.run_image(openai_api_key=self.openai_api_key, img = image)
                    captions.append(caption)
                output_dict.update({"caption": captions})

            return output_dict
        else: # format == 'video'
            output_dict = dict()
            self.dataset = PreprocessDataset(self.args, self.device, imgs, self.annotation_typ_lst)
            self.dataloader = DataLoader(self.dataset, batch_size=self.args['datasets']['batch_size'], shuffle=False)

            for model_key in self.models.keys():
                output_dict[model_key] = []

            for batch_dict in tqdm(self.dataloader, desc="Processing video frames"):
                for model_key, model in self.models.items():
                    with torch.no_grad():
                        outputs = model.forward(batch_dict[model_key], return_pil=False)
                        print(model_key, outputs[0])
                        output_dict[model_key].append(outputs) # append np arrays

            # output_dict의 값들을 각 key에 대해 flatten하기
            for key in output_dict:
                flattened_output_list = []
                for batch in output_dict[key]:
                    flattened_output_list.extend([Image.fromarray(np.transpose(img, (1, 2, 0))) for img in batch])
                output_dict[key] = flattened_output_list

            return output_dict

    def resize_image(self, image, new_resolution):
        return cv2.resize(image, new_resolution)
    

    def draw_landmarks(self, image, landmarks, color="white", radius=2.5):
        draw = ImageDraw.Draw(image)
        for dot in landmarks:
            x, y = dot
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color) 

    def draw_landmarks_mask(self, image_shape, landmarks, color="white"):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        pts = np.array(landmarks, dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        return mask

    def get_68landmarks_mask(self, img, detector, predictor):
        try:
            rect = detector(img)[0]
            sp = predictor(img, rect)
            landmarks = np.array([[p.x, p.y] for p in sp.parts()])

            outline = landmarks[[*range(17), *range(26,16,-1)]]
            vertices = ConvexHull(landmarks).vertices
            Y, X = polygon(landmarks[vertices, 1], landmarks[vertices, 0])
            mask_image = np.zeros(img.shape, dtype=np.uint8)
            mask_image[Y, X] = 255
        except:
            mask_image = np.zeros(img.shape, dtype=np.uint8)

        return Image.fromarray(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))

    def apply_canny(self, img, start_x=None, end_x=None, low_threshold=50, high_threshold=150, ):
        canny_img = cv2.Canny(img, low_threshold, high_threshold)
        
        if start_x is not None and end_x is not None:
            canny_img[:, start_x:end_x] = 0

        return Image.fromarray(canny_img)

    def get_start_fin_xcoord(self, img):
        # Calculate the histogram
        hist, edges = np.histogram(img, bins=256, range=(0, 255))

        # Find the x-coordinates where the values start to appear and where they don't
        start_x = None
        end_x = None
        for i, val in enumerate(hist):
            if val != 0 and start_x is None:
                start_x = i
            if val == 0 and start_x is not None and end_x is None:
                end_x = i

        return start_x, end_x

def extract_frame(args):
    video_path, frame_idx = args
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if ret:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return pil_image
    else:
        return None

def extract_frames(video_path, random_sampling=False, num_processes=None, frame_interval=3):
    frames = []
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if random_sampling: # Randomly sample 4 frames
        random_frame_indices = random.sample(range(num_frames), 4)
        
        for idx in random_frame_indices:
            frame = extract_frame((video_path, idx))
            if frame is not None:
                frames.append(frame)
    else: # Sample frames with specified interval
        with Pool(processes=num_processes) as p:
            args = [(video_path, frame_idx) for frame_idx in range(0, num_frames, frame_interval)]
            frame_list = list(tqdm(p.imap(extract_frame, args), total=len(args), desc="Extracting frames with num_processes={}, frame_interval={}".format(num_processes, frame_interval), leave=False, ncols=200, unit="frames"))
        
        # Remove None values from the list
        frames = [frame for frame in frame_list if frame is not None]

    return frames

def merge_imgs(imgs):
    # Find the maximum width and height of all the images
    max_width = max(img.size[0] for img in imgs)
    max_height = max(img.size[1] for img in imgs)

    # Resize all images to the maximum size
    resized_imgs = []
    for img in imgs:
        resized_img = img.resize((max_width, max_height))
        resized_imgs.append(resized_img)

    cell_width = max_width
    cell_height = max_height

    # Calculate the number of columns and rows in the grid
    num_imgs = len(resized_imgs)
    num_cols = num_imgs
    num_rows = 1

    # Create a new image with the appropriate size
    grid = Image.new("RGB", size=(num_cols*cell_width, num_rows*cell_height))

    # Paste each image into the grid
    for i, img in enumerate(resized_imgs):
        x = i * cell_width
        y = 0
        grid.paste(img, box=(x, y))

    return grid

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_input_imgs(lg, input_path, input_type='4cut_from_videos', num_processes=4, frame_interval=3):
    lg.info('Desired input type: {}'.format(input_type))
    
    if input_type == 'photos':        
        pil_images = []
        image_filenames = []
        image_filenames = glob.glob(input_path + "/*.png")
        image_filenames.sort()
        
        for filename in tqdm(image_filenames, desc="Reading images..."):
            pil_image = Image.open(filename).convert("RGB")
            pil_images.append(pil_image)

        sublists = [list(zip(sublist, sublist_filenames)) for sublist, sublist_filenames in zip([pil_images[i:i+4] for i in range(0, len(pil_images), 4)], [image_filenames[i:i+4] for i in range(0, len(image_filenames), 4)])]
        
        return {i: {filename: image for image, filename in sublist} for i, sublist in enumerate(sublists)}

    elif input_type == 'life4cut':
        video_filenames = glob.glob(input_path + "/*.mp4")
        video_filenames.sort()
        sublists_dict = {}

        for i, video_filename in enumerate(tqdm(video_filenames, desc="Sampling 4 frames from each video...")):
            filename = os.path.basename(video_filename)
            frames = extract_frames(video_filename, random_sampling=True)
            frame_filenames = [f"{video_filename[:-4]}_frame{j}.jpg" for j in range(len(frames))]
            sublist = list(zip(frames, frame_filenames))
            sublists_dict[i] = sublist

        return {i: {filename: image for image, filename in sublist} for i, sublist in sublists_dict.items()}

    elif input_type == 'videos':
        video_filenames = glob.glob(input_path + "/*.mp4")
        video_filenames.sort()
        sublists_dict = {}

        for i, video_filename in enumerate(video_filenames):
            filename = os.path.basename(video_filename)
            with tqdm(total=len(video_filenames), desc=f"Sampling frames from {filename}...") as pbar:
                frames = extract_frames(video_filename, random_sampling=False, num_processes=num_processes, frame_interval=frame_interval)
                sublists_dict[i] = frames
                
        return sublists_dict

    else:
        raise NotImplementedError('Please input correct input_type: photos or videos')
    
def read_json(json_path):
    with open(json_path, "r") as st_json:
        st_python = json.load(st_json)
        return st_python