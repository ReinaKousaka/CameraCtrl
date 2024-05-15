import omegaconf.listconfig
import os
import math
import random
import time
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.models.attention_processor import AttnProcessor

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection,CLIPImageProcessor, CLIPVisionModel
from einops import rearrange

from cameractrl.data.dataset import RealEstate10KPose
from cameractrl.utils.util import setup_logger, format_time, save_videos_grid
from cameractrl.pipelines.pipeline_animation import CameraCtrlPipeline
from cameractrl.models.unet import UNet3DConditionModelPoseCond
from cameractrl.models.pose_adaptor import CameraPoseEncoder, PoseAdaptor
from cameractrl.models.attention_processor import AttnProcessor as CustomizedAttnProcessor
from cameractrl.geometry.projection import get_world_rays
from cameractrl.geometry.epipolar_lines import project_rays
from cameractrl.visualization.drawing.lines import draw_attn
from utils import _resize_with_antialiasing

def init_dist(launcher="slurm", backend='nccl', port=29500, **kwargs):
    """Initializes distributed environment."""
    if launcher == 'pytorch':
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        local_rank = rank % num_gpus
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, **kwargs)

    elif launcher == 'slurm':
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        port = os.environ.get('PORT', port)
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend=backend)

    else:
        raise NotImplementedError(f'Not implemented launcher type: `{launcher}`!')

    return local_rank


def main(name: str,
         launcher: str,
         port: int,

         output_dir: str,
         pretrained_model_path: str,

         train_data: Dict,
         validation_data: Dict,
         cfg_random_null_text: bool = True,
         cfg_random_null_text_ratio: float = 0.1,

         unet_additional_kwargs: Dict = {},
         unet_subfolder: str = "unet",

         lora_rank: int = 4,
         lora_scale: float = 1.0,
         lora_ckpt: str = None,
         motion_module_ckpt: str = "",
         motion_lora_rank: int = 0,
         motion_lora_scale: float = 1.0,

         pose_encoder_kwargs: Dict = None,
         attention_processor_kwargs: Dict = None,
         noise_scheduler_kwargs: Dict = None,

         do_sanity_check: bool = True,

         max_train_epoch: int = -1,
         max_train_steps: int = 100,
         validation_steps: int = 100,
         validation_steps_tuple: Tuple = (-1,),

         learning_rate: float = 3e-5,
         lr_warmup_steps: int = 0,
         lr_scheduler: str = "constant",

         num_workers: int = 32,
         train_batch_size: int = 1,
         adam_beta1: float = 0.9,
         adam_beta2: float = 0.999,
         adam_weight_decay: float = 1e-2,
         adam_epsilon: float = 1e-08,
         max_grad_norm: float = 1.0,
         gradient_accumulation_steps: int = 1,
         checkpointing_epochs: int = 5,
         checkpointing_steps: int = -1,

         mixed_precision_training: bool = True,

         global_seed: int = 42,
         logger_interval: int = 10,
         resume_from: str =
         None,
         ):
    check_min_version("0.10.0.dev0")

    # Initialize distributed training
    local_rank = init_dist(launcher=launcher, port=port)
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    is_main_process = global_rank == 0

    seed = global_seed + global_rank
    torch.manual_seed(seed)

    # Logging folder
    folder_name = name + datetime.datetime.now().strftime("-%Y-%m-%dT%H-%M-%S")
    output_dir = os.path.join(output_dir, folder_name)

    *_, config = inspect.getargvalues(inspect.currentframe())

    logger = setup_logger(output_dir, global_rank)

    # Handle the output folder creation
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/sanity_check", exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    # imageencoder
    pretrained_model_name_or_path = "openai/clip-vit-base-patch32"
    feature_extractor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_name_or_path)


    unet = UNet3DConditionModelPoseCond.from_pretrained_2d(pretrained_model_path, subfolder=unet_subfolder,
                                                           unet_additional_kwargs=unet_additional_kwargs)
    pose_encoder = CameraPoseEncoder(**pose_encoder_kwargs)

    # init attention processor
    logger.info(f"Setting the attention processors")
    unet.set_all_attn_processor(add_spatial_lora=lora_ckpt is not None,
                                add_motion_lora=motion_lora_rank > 0,
                                lora_kwargs={"lora_rank": lora_rank, "lora_scale": lora_scale},
                                motion_lora_kwargs={"lora_rank": motion_lora_rank, "lora_scale": motion_lora_scale},
                                **attention_processor_kwargs)

    if lora_ckpt is not None:
        logger.info(f"Loading the image lora checkpoint from {lora_ckpt}")
        lora_checkpoints = torch.load(lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        logger.info(f'Loading done')
    else:
        logger.info(f'We do not add image lora')

    if motion_module_ckpt != "":
        logger.info(f"Loading the motion module checkpoint from {motion_module_ckpt}")
        mm_checkpoints = torch.load(motion_module_ckpt, map_location=unet.device)
        if 'motion_module_state_dict' in mm_checkpoints:
            mm_checkpoints = {k.replace('module.', ''): v for k, v in mm_checkpoints['motion_module_state_dict'].items()}
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        logger.info("Loading done")
    else:
        logger.info(f"We do not load pretrained motion module checkpoint")
    
    # the following snippet is to load pre-trained CameraCtrl.ckpt
    # print(f"Loading pose adaptor")
    # pose_adaptor_checkpoint = torch.load('./CameraCtrl.ckpt', map_location='cpu')
    # pose_encoder_state_dict = pose_adaptor_checkpoint['pose_encoder_state_dict']
    # pose_encoder_m, pose_encoder_u = pose_encoder.load_state_dict(pose_encoder_state_dict)
    # assert len(pose_encoder_u) == 0 and len(pose_encoder_m) == 0
    # attention_processor_state_dict = pose_adaptor_checkpoint['attention_processor_state_dict']
    # _, attn_proc_u = unet.load_state_dict(attention_processor_state_dict, strict=False)
    # assert len(attn_proc_u) == 0


    def encode_image(pixel_values):
        pixel_values = pixel_values * 2.0 - 1.0
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0

        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(local_rank)
        image_embeddings = image_encoder(pixel_values).last_hidden_state  #.image_embeds
        return image_embeddings



    print(f"Loading done")

    # Freeze vae, and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # imageencoder
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # print(unet)

    spatial_attn_proc_modules = torch.nn.ModuleList([v for v in unet.attn_processors.values()
                                                     if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))])
    temporal_attn_proc_modules = torch.nn.ModuleList([v for v in unet.mm_attn_processors.values()
                                                      if not isinstance(v, (CustomizedAttnProcessor, AttnProcessor))])
    spatial_attn_proc_modules.requires_grad_(True)
    temporal_attn_proc_modules.requires_grad_(True)
    pose_encoder.requires_grad_(True)
    # set requires_grad of image lora to False
    for n, p in spatial_attn_proc_modules.named_parameters():
        if 'lora' in n:
            p.requires_grad = False
            logger.info(f'Setting the `requires_grad` of parameter {n} to false')
    pose_adaptor = PoseAdaptor(unet, pose_encoder)

    for name, para in unet.named_parameters():
        if 'epipolar' in name:
            para.requires_grad = True
        if 'attn3' in name:
            para.requires_grad = True
        if 'attn2' in name:
            para.requires_grad = True

    encoder_trainable_params = list(filter(lambda p: p.requires_grad, pose_encoder.parameters()))
    encoder_trainable_param_names = [p[0] for p in
                                     filter(lambda p: p[1].requires_grad, pose_encoder.named_parameters())]
    attention_trainable_params = [v for k, v in unet.named_parameters() if v.requires_grad and 'merge' in k and 'lora' not in k]
    attention_trainable_param_names = [k for k, v in unet.named_parameters() if v.requires_grad and 'merge' in k and 'lora' not in k]
    epipolar_trainable_params = [v for k, v in unet.named_parameters() if v.requires_grad and 'epipolar' in k]
    epipolar_trainable_params_names = [k for k, v in unet.named_parameters() if v.requires_grad and 'epipolar' in k]

    attn3_trainable_params = [v for k, v in unet.named_parameters() if v.requires_grad and ('attn3' in k or 'attn2' in k)]
    attn3_trainable_params_names = [k for k, v in unet.named_parameters() if v.requires_grad and ('attn3' in k or 'attn2' in k)]

    trainable_params = encoder_trainable_params + attention_trainable_params + epipolar_trainable_params + attn3_trainable_params
    trainable_param_names = encoder_trainable_param_names + attention_trainable_param_names + epipolar_trainable_params_names + attn3_trainable_params_names

    if is_main_process:
        logger.info(f"trainable parameter number: {len(trainable_params)}")
        logger.info(f"encoder trainable number: {len(encoder_trainable_params)}")
        logger.info(f"attention processor trainable number: {len(attention_trainable_params)}")
        logger.info(f"trainable parameter names: {trainable_param_names}")
        logger.info(f"encoder trainable scale: {sum(p.numel() for p in encoder_trainable_params) / 1e6:.3f} M")
        logger.info(f"attention processor trainable scale: {sum(p.numel() for p in attention_trainable_params) / 1e6:.3f} M")
        logger.info(f"trainable parameter scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    image_encoder.to(local_rank)

    # Get the training dataset
    logger.info(f'Building training datasets')
    train_dataset = RealEstate10KPose(**train_data)
    distributed_sampler = DistributedSampler(
        train_dataset,
        num_replicas=num_processes,
        rank=global_rank,
        shuffle=True,
        seed=global_seed,
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=distributed_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Get the validation dataset
    logger.info(f'Building validation datasets')
    validation_dataset = RealEstate10KPose(**validation_data)
    VALIDATION_BATCH_SIZE = 1
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Get the training iteration
    if max_train_steps == -1:
        assert max_train_epoch != -1
        max_train_steps = max_train_epoch * len(train_dataloader)

    if checkpointing_steps == -1:
        assert checkpointing_epochs != -1
        checkpointing_steps = checkpointing_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Validation pipeline
    validation_pipeline = CameraCtrlPipeline(
        vae=vae,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        feature_extractor = feature_extractor,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        pose_encoder=pose_encoder)
    validation_pipeline.enable_vae_slicing()

    # DDP wrapper
    pose_adaptor.to(local_rank)
    pose_adaptor = DDP(pose_adaptor, device_ids=[local_rank], output_device=local_rank)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * num_processes * gradient_accumulation_steps

    if is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    if resume_from is not None:
        logger.info(f"Resuming the training from the checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=pose_adaptor.device)
        global_step = ckpt['global_step']
        trained_iterations = (global_step % len(train_dataloader))
        first_epoch = int(global_step // len(train_dataloader))
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        pose_encoder_state_dict = ckpt['pose_encoder_state_dict']
        attention_processor_state_dict = ckpt['attention_processor_state_dict']
        epipolar_layer_state_dict = ckpt["epipolar_layer_state_dict"]
        attn3_layer_state_dict = ckpt["attn3_layer_state_dict"]
        _, _ = pose_adaptor.module.pose_encoder.load_state_dict(pose_encoder_state_dict, strict=False)
        # import pdb
        # pdb.set_trace()
        # assert len(pose_enc_m) == 0 and len(pose_enc_u) == 0
        _, _ = pose_adaptor.module.unet.load_state_dict(attention_processor_state_dict, strict=False)

        _, _ = pose_adaptor.module.unet.load_state_dict(epipolar_layer_state_dict,
                                                                            strict=False)

        _, _ = pose_adaptor.module.unet.load_state_dict(attn3_layer_state_dict,
                                                                            strict=False)
        del ckpt
        # assert len(attention_processor_u) == 0
        logger.info(f"Loading the pose encoder and attention processor weights done.")
        logger.info(f"Loading done, resuming training from the {global_step + 1}th iteration")
        lr_scheduler.last_epoch = first_epoch
    else:
        trained_iterations = 0

    # Support mixed-precision training
    scaler = torch.cuda.amp.GradScaler() if mixed_precision_training else None

    for epoch in range(first_epoch, num_train_epochs):
        print(f'Current epoch: {epoch}, Last: {num_train_epochs - 1}')
        train_dataloader.sampler.set_epoch(epoch)
        pose_adaptor.train()

        for batch_idx, batch in enumerate(train_dataloader):
            print(f'global step: {global_step}')
            # skip trained iterations
            # if batch_idx < trained_iterations: continue
            iter_start_time = time.time()
            data_end_time = time.time()
            if cfg_random_null_text:
                batch['text'] = [name if random.random() > cfg_random_null_text_ratio else "" for name in batch['text']]

            # Data batch sanity check
            if epoch == first_epoch and batch_idx == 0 and do_sanity_check:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                pixel_values = pixel_values
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    save_videos_grid(pixel_value,
                                     f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_rank}-{idx}'}.gif",
                                     rescale=True)

            ### >>>> Training >>>> ###

            # Convert videos to latent space
            pixel_values = batch["pixel_values"].to(local_rank)
            video_length = pixel_values.shape[1]

            with torch.no_grad(): # image_encoder
                image_hidden_states = encode_image(pixel_values[:, 0]) # image_encoder




            with torch.no_grad():
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215
            condition_image_latents = latents[:, :, 0] #1stframe

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents) # [b, c, f, h, w]
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # [b, c, f h, w]

            noisy_latents[:,:, 0] = condition_image_latents #1stframe

            # Get the text embedding for conditioning
            with torch.no_grad():
                prompt_ids = tokenizer(
                    batch['text'], max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                    return_tensors="pt"
                ).input_ids.to(latents.device)
                encoder_hidden_states = text_encoder(prompt_ids)[0]  # b l c

            encoder_hidden_states = torch.cat([encoder_hidden_states,image_hidden_states], dim = 1)
            # Predict the noise residual and compute loss
            # Mixed-precision training
            plucker_embedding = batch["plucker_embedding"].to(device=local_rank)  # [b, f, 6, h, w]
            plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")  # [b, 6, f h, w]

            # start_time = time.perf_counter()
            def _calculate_attn_mask(intrinsics, extrinsics, batch_size):
                """
                intrinsics: B, 6
                extrinsics: B, T, 4, 4
                """
                attn_masks = []
                for s in range(3):
                    w, h, t = latents.shape[4] // 2 ** (s + 1), latents.shape[3] // 2 ** (s + 1), latents.shape[2]
                    xs = torch.linspace(0, 1, steps=w)
                    ys = torch.linspace(0, 1, steps=h)
                    grid = torch.stack(
                        torch.meshgrid(xs, ys, indexing='xy'), dim=-1).float().to(
                            local_rank, non_blocking=True)

                    grid = rearrange(grid, "h w c  -> (h w) c")
                    grid = grid.repeat(t, 1)
                    attn_mask = []
                    for b in range(batch_size):
                        k = torch.eye(3).float().to(
                            local_rank, non_blocking=True)
                        k[0, 0] = intrinsics[b][0]
                        k[1, 1] = intrinsics[b][1]
                        k[0, 2] = 0.5
                        k[1, 2] = 0.5
                        source_intrinsics = k
                        source_intrinsics = source_intrinsics[None].repeat_interleave(t * w * h, 0)

                        source_extrinsics_all = []
                        target_extrinsics_all = []
                        for t1 in range(t):
                            source_extrinsics = torch.inverse(extrinsics[b][t1].to(
                                local_rank, non_blocking=True))
                            source_extrinsics_all.append(source_extrinsics[None].repeat_interleave(w * h, 0))
                            tmp_seq = []
                            for t2 in range(t):
                                target_extrinsics = torch.inverse(extrinsics[b][t2].to(
                                    local_rank, non_blocking=True))
                                tmp_seq.append(target_extrinsics[None])
                            target_extrinsics_all.append(torch.cat(tmp_seq).repeat(w*h,1,1))

                        source_extrinsics_all = torch.cat(source_extrinsics_all)
                        target_extrinsics_all = torch.cat(target_extrinsics_all)
                        origin, direction = get_world_rays(grid, source_extrinsics_all, source_intrinsics)
                        origin = origin.repeat_interleave(t, 0)
                        direction = direction.repeat_interleave(t, 0)
                        source_intrinsics = source_intrinsics.repeat_interleave(t, 0)
                        projection = project_rays(
                                    origin, direction, target_extrinsics_all, source_intrinsics
                                )

                        attn_image = torch.zeros((3, h, w)).to(
                            local_rank, non_blocking=True)

                        attn_image = draw_attn(
                            attn_image,
                            projection["xy_min"],
                            projection["xy_max"],
                            (1, 1, 1),
                            8 // 2 ** s,
                            x_range=(0, 1),
                            y_range=(0, 1),)
                        attn_image = attn_image
                        attn_image = rearrange(attn_image, '(t1 a t2) b-> (t1 a) (t2 b)', t1 =t, t2=t)
                        attn_mask.append(attn_image)
                    attn_mask = torch.stack(attn_mask)
                    attn_masks.append(attn_mask.float())
                attn_masks.append(attn_mask.float())
                return attn_masks
            
            attn_masks = _calculate_attn_mask(batch['intrinsics'], batch['extrinsics'], bsz)
            with torch.cuda.amp.autocast(enabled=mixed_precision_training):
                # print(attn_masks[-1].shape)
                model_pred = pose_adaptor(noisy_latents,
                                          timesteps,
                                          encoder_hidden_states=encoder_hidden_states,
                                          pose_embedding=plucker_embedding,
                                          attention_mask_epipolar=attn_masks)  # [b c f h w]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float()[:,:,1:], target.float()[:,:,1:], reduction="mean") #1stframe
                loss /= gradient_accumulation_steps

            # Backpropagate
            if mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, pose_adaptor.parameters()),
                                               max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()),
                                               max_grad_norm)



            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, unet.parameters()),
                                               max_grad_norm)
                lr_scheduler.step()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1
            iter_end_time = time.time()

            # Save checkpoint
            if is_main_process and (global_step % checkpointing_steps == 0):
                save_path = os.path.join(output_dir, f"checkpoints")
                state_dict = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "pose_encoder_state_dict": pose_adaptor.module.pose_encoder.state_dict(),
                    "attention_processor_state_dict": {k: v for k, v in unet.state_dict().items()
                                                       if k in attention_trainable_param_names},
                    "epipolar_layer_state_dict": {k: v for k, v in unet.state_dict().items()
                                                  if k in epipolar_trainable_params_names},

                    "attn3_layer_state_dict": {k: v for k, v in unet.state_dict().items()
                                                  if k in attn3_trainable_params_names},


                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(state_dict, os.path.join(save_path, f"checkpoint-step-{global_step}.ckpt"))
                logger.info(f"Saved state to {save_path} (global_step: {global_step})")

            # Periodically validation
            if is_main_process and (
                    (global_step + 950) % validation_steps == 0 or (global_step + 950) in validation_steps_tuple):

                generator = torch.Generator(device=latents.device)
                generator.manual_seed(global_seed)

                if isinstance(train_data, omegaconf.listconfig.ListConfig):
                    height = train_data[0].sample_size[0] if not isinstance(train_data[0].sample_size, int) else \
                    train_data[0].sample_size
                    width = train_data[0].sample_size[1] if not isinstance(train_data[0].sample_size, int) else \
                    train_data[0].sample_size
                else:
                    height = train_data.sample_size[0] if not isinstance(train_data.sample_size,
                                                                         int) else train_data.sample_size
                    width = train_data.sample_size[1] if not isinstance(train_data.sample_size,
                                                                        int) else train_data.sample_size

                validation_data_iter = iter(validation_dataloader)
                print(f'------ valid ------')
                for idx, validation_batch in enumerate(validation_data_iter):
                    plucker_embedding = validation_batch['plucker_embedding'].to(device=unet.device)
                    plucker_embedding = rearrange(plucker_embedding, "b f c h w -> b c f h w")
                    intrinsics = validation_batch['intrinsics'].to(device=unet.device)
                    extrinsics = validation_batch['extrinsics'].to(device=unet.device)
                    attention_mask = _calculate_attn_mask(intrinsics, extrinsics, VALIDATION_BATCH_SIZE)
                    # print(plucker_embedding.shape)

                    test_pixel_values = validation_batch["pixel_values"][:, 0].to(local_rank) #1stframe

                    with torch.no_grad(): #1stframe
                        # pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                        test_condition_image_latents = vae.encode(test_pixel_values).latent_dist.sample()#1stframe
                        # latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                        test_condition_image_latents = test_condition_image_latents * 0.18215 #1stframe
                    # test_condition_image_latents = latents[:, 0].copy
                    test_image_embedding = encode_image(test_pixel_values)
                    # print(test_image_embedding.shape)



                    sample = validation_pipeline(
                        prompt=validation_batch['text'],
                        pose_embedding=plucker_embedding,
                        attention_mask_epipolar=attention_mask,
                        video_length=video_length,
                        height=height,
                        width=width,
                        num_inference_steps=25,
                        guidance_scale=8.,
                        generator=generator,
                        test_condition_image_latents = test_condition_image_latents, #1stframe
                        test_image_embedding = test_image_embedding,
                    ).videos[0]  # [3 f h w]
                    sample_gt = torch.cat([sample, (validation_batch['pixel_values'][0].permute(1, 0, 2, 3) + 1.0) / 2.0], dim=2)  # [3, f, 2h, w]
                    if 'clip_name' in validation_batch:
                        save_path = f"{output_dir}/samples/sample-{global_step}/{validation_batch['clip_name'][0]}.gif"
                    else:
                        save_path = f"{output_dir}/samples/sample-{global_step}/{idx}.gif"
                    save_videos_grid(sample_gt[None, ...], save_path)
                    logger.info(f"Saved samples to {save_path}")
                    if idx == 6: break      # get 7 samples

            if (global_step % logger_interval) == 0 or global_step == 0:
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
                msg = f"Iter: {global_step}/{max_train_steps}, Loss: {loss.detach().item(): .4f}, " \
                      f"lr: {lr_scheduler.get_last_lr()}, Data time: {format_time(data_end_time - iter_start_time)}, " \
                      f"Iter time: {format_time(iter_end_time - data_end_time)}, " \
                      f"ETA: {format_time((iter_end_time - iter_start_time) * (max_train_steps - global_step))}, " \
                      f"GPU memory: {gpu_memory: .2f} G"
                logger.info(msg)

            if global_step >= max_train_steps:
                break

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--launcher", type=str, choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--port", type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    name = Path(args.config).stem
    config = OmegaConf.load(args.config)

    main(name=name, launcher=args.launcher, port=args.port, **config)