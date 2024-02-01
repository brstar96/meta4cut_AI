import os.path
import tqdm
import torch
import safetensors.torch
from torch import Tensor
from modules import sd_models
from typing import List

def conv_fp16(t: Tensor):
    return t.half()

def conv_bf16(t: Tensor):
    return t.bfloat16()

def conv_full(t):
    return t

_g_precision_func = {
    "full": conv_full,
    "fp32": conv_full,
    "fp16": conv_fp16,
    "bf16": conv_bf16,
}

def check_weight_type(k: str) -> str:
    if k.startswith("model.diffusion_model"):
        return "unet"
    elif k.startswith("first_stage_model"):
        return "vae"
    elif k.startswith("cond_stage_model"):
        return "clip"
    return "other"

def load_model(path):
    if path.endswith(".safetensors"):
        m = safetensors.torch.load_file(path, device="cpu")
    else:
        m = torch.load(path, map_location="cpu")
    state_dict = m["state_dict"] if "state_dict" in m else m
    return state_dict

def do_convert(model, checkpoint_formats: List[str],
               precision: str, conv_type: str, custom_name: str,
               unet_conv, text_encoder_conv, vae_conv, others_conv):
    if model == "":
        return "Error: you must choose a model"
    if len(checkpoint_formats) == 0:
        return "Error: at least choose one model save format"

    extra_opt = {
        "unet": unet_conv,
        "clip": text_encoder_conv,
        "vae": vae_conv,
        "other": others_conv
    }
    model_info = sd_models.checkpoints_list[model]
    print(f"Loading {model_info.filename}...")
    state_dict = load_model(model_info.filename)

    ok = {}  # {"state_dict": {}}

    conv_func = _g_precision_func[precision]

    def _hf(wk: str, t: Tensor):
        if not isinstance(t, Tensor):
            return
        w_t = check_weight_type(wk)
        conv_t = extra_opt[w_t]
        if conv_t == "convert":
            ok[wk] = conv_func(t)
        elif conv_t == "copy":
            ok[wk] = t
        elif conv_t == "delete":
            return

    print("Converting model...")

    if conv_type == "ema-only":
        for k in tqdm.tqdm(state_dict):
            ema_k = "___"
            try:
                ema_k = "model_ema." + k[6:].replace(".", "")
            except:
                pass
            if ema_k in state_dict:
                _hf(k, state_dict[ema_k])
                # print("ema: " + ema_k + " > " + k)
            elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
                _hf(k, state_dict[k])
            #     print(k)
            # else:
            #     print("skipped: " + k)
    elif conv_type == "no-ema":
        for k, v in tqdm.tqdm(state_dict.items()):
            if "model_ema" not in k:
                _hf(k, v)
    else:
        for k, v in tqdm.tqdm(state_dict.items()):
            _hf(k, v)

    output = ""
    ckpt_dir = sd_models.model_path
    save_name = f"{model_info.model_name}-{precision}"
    if conv_type != "disabled":
        save_name += f"-{conv_type}"

    if custom_name != "":
        save_name = custom_name

    for fmt in checkpoint_formats:
        ext = ".safetensors" if fmt == "safetensors" else ".ckpt"
        _save_name = save_name + ext

        save_path = os.path.join(ckpt_dir, _save_name)
        print(f"Saving to {save_path}...")

        if fmt == "safetensors":
            safetensors.torch.save_file(ok, save_path)
        else:
            torch.save({"state_dict": ok}, save_path)
        output += f"Checkpoint saved to {save_path}\n"

    return output[:-1]

pt_model = do_convert(model='/projects/meta4cut_BE/models/Stable-diffusion/Chilloutmix/Chilloutmix-non-ema-fp32.safetensors',
                      checkpoint_formats=".safetensors", 
                      precision='fp32', 
                      conv_type='no-ema', 
                      custom_name='Chilloutmix-non-ema-fp32',
                      unet_conv="convert",
                      text_encoder_conv="convert", 
                      vae_conv="convert", 
                      others_conv="convert")