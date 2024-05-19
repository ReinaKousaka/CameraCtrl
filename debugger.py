# from cameractrl.data.dataset_epic import EpicKitchen
# from cameractrl.data.dataset import RealEstate10KPose

import torch
from cameractrl.models.sparse_controlnet import SparseControlNetModel
from cameractrl.models.unet import UNet3DConditionModelPoseCond
from omegaconf import OmegaConf


config = OmegaConf.load('configs/train_cameractrl/adv3_256_384_cameractrl_relora.yaml')
unet_additional_kwargs = config['unet_additional_kwargs']
controlnet_additional_kwargs = config['controlnet_additional_kwargs']
# controlnet_additional_kwargs['num_attention_heads'] = 1


unet = UNet3DConditionModelPoseCond.from_pretrained_2d(
    "./stable-diffusion-v1-5",
    subfolder="unet",
    unet_additional_kwargs=unet_additional_kwargs
)

# unet.set_all_attn_processor(add_spatial_lora=lora_ckpt is not None,
#                             add_motion_lora=motion_lora_rank > 0,
#                             lora_kwargs={"lora_rank": lora_rank, "lora_scale": lora_scale},
#                             motion_lora_kwargs={"lora_rank": motion_lora_rank, "lora_scale": motion_lora_scale},
#                             **attention_processor_kwargs)

unet.config.num_attention_heads = 8
unet.config.projection_class_embeddings_input_dim = None
# load v3_sd15_sparsectrl_rgb.ckpt
controlnet = SparseControlNetModel.from_unet(
    unet,
    controlnet_additional_kwargs=controlnet_additional_kwargs,
)
print(f"loading controlnet checkpoint from ./v3_sd15_sparsectrl_rgb.ckpt ...")
controlnet_state_dict = torch.load("v3_sd15_sparsectrl_rgb.ckpt", map_location="cpu")
controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
controlnet_state_dict.pop("animatediff_config", "")
controlnet.load_state_dict(controlnet_state_dict)
# controlnet.cuda()

t = 10
controlnet_noisy_latents = torch.zeros(2, 4, t, 32, 48)
controlnet_prompt_embeds = torch.zeros(2, 77, 768)
controlnet_cond = torch.zeros(1, 4, t, 32, 48)
controlnet_conditioning_mask = torch.zeros(1, 1, t, 32, 48)
controlnet_conditioning_scale = 1

a, b = controlnet(
    controlnet_noisy_latents, 961,
    encoder_hidden_states=controlnet_prompt_embeds,
    controlnet_cond=controlnet_cond,
    conditioning_mask=controlnet_conditioning_mask,
    conditioning_scale=controlnet_conditioning_scale,
    guess_mode=False, return_dict=False,
)

# print(a)
# print(b)