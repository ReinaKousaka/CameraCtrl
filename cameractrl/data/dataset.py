import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from decord import VideoReader
from torch.utils.data.dataset import Dataset
from packaging import version as pver
import scipy.io
from PIL import Image


class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    B, V = K.shape[:2]
    assert B == 1

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)  # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


class RealEstate10K(Dataset):
    def __init__(
            self,
            root_path,
            annotation_json,
            sample_stride=4,
            sample_n_frames=16,
            sample_size=[256, 384],
            is_image=False,
    ):
        self.root_path = root_path
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image

        self.dataset = json.load(open(os.path.join(root_path, annotation_json), 'r'))
        self.length = len(self.dataset)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        pixel_transforms = [transforms.Resize(sample_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]

        self.pixel_transforms = transforms.Compose(pixel_transforms)

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]

        video_path = os.path.join(self.root_path, video_dict['clip_path'])
        video_reader = VideoReader(video_path)
        return video_reader, video_dict['caption']

    def get_batch(self, idx):
        video_reader, video_caption = self.load_video_reader(idx)
        total_frames = len(video_reader)

        if self.is_image:
            frame_indice = [random.randint(0, total_frames - 1)]
        else:
            if isinstance(self.sample_stride, int):
                current_sample_stride = self.sample_stride
            else:
                assert len(self.sample_stride) == 2
                assert (self.sample_stride[0] >= 1) and (self.sample_stride[1] >= self.sample_stride[0])
                current_sample_stride = random.randint(self.sample_stride[0], self.sample_stride[1])

            cropped_length = self.sample_n_frames * current_sample_stride
            start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
            end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

            assert end_frame_ind - start_frame_ind >= self.sample_n_frames
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int)

        pixel_values = torch.from_numpy(video_reader.get_batch(frame_indice).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.

        if self.is_image:
            pixel_values = pixel_values[0]

        return pixel_values, video_caption

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        video = self.pixel_transforms(video)
        sample = dict(pixel_values=video, caption=video_caption)

        return sample


class RealEstate10KPose(Dataset):
    def __init__(
            self,
            root_path='./RealEstate',
            frame_dir='./train',
            image_json='train.json',  # video_id -> [image_file_names]
            caption_json='train_captions.json',  # video_id -> [video_caption]
            pose_mat='train.mat',
            sample_stride=4,
            minimum_sample_stride=1,
            num_channels=3,
            sample_n_frames=8,  # t
            relative_pose=False,
            zero_t_first_frame=False,
            sample_size=[256, 384],
            rescale_fxy=True,
            shuffle_frames=False,
            use_flip=False,
    ):
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.num_channels = num_channels
        self.t = sample_n_frames

        self.frame_dir = frame_dir
        self.dataset = json.load(open(os.path.join(root_path, image_json), 'r'))
        self.caption_dict = json.load(open(os.path.join(root_path, caption_json), 'r'))
        self.mat = scipy.io.loadmat(os.path.join(root_path, pose_mat))
        self.length = len(self.dataset)

        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.sample_size = sample_size
        self.w, self.h = sample_size

        self.pixel_transforms = [transforms.Resize(sample_size),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_cameras(self, idx):
        video_dict = self.dataset[idx]
        pose_file = os.path.join(self.root_path, video_dict['pose_file'])
        with open(pose_file, 'r') as f:
            poses = f.readlines()
        poses = [pose.strip().split(' ') for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_params = [Camera(cam_param) for cam_param in cam_params]
        return cam_params

    def get_batch(self, idx):
        video_id = list(self.dataset.keys())[idx]
        plist = self.mat[video_id]
        flist = self.dataset[video_id]
        flist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        video_caption = self.caption_dict[video_id + '.mp4'][0]
        total_frames = len(flist)

        current_sample_stride = self.sample_stride + random.randint(-2, 4)
        if total_frames < self.t * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.t)
            current_sample_stride = random.randint(self.minimum_sample_stride, maximum_sample_stride)

        cropped_length = self.t * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.t
        frame_indices = np.linspace(start_frame_ind, end_frame_ind - 1, self.t, dtype=int)
        if self.use_flip and random.randint(0, 1) == 1:
            frame_indices = np.flip(frame_indices)

        # stack images into a tensor
        pixel_values = []
        transformer = transforms.PILToTensor()

        for frame_idx in frame_indices:
            with Image.open(os.path.join(self.root_path, self.frame_dir, video_id, flist[frame_idx])) as img:
                pixel_values.append(transformer(img).float())

        pixel_values = torch.stack(pixel_values, dim=0)  # t, c, h, w
        pixel_values = pixel_values / 255.

        # load poses
        cam_params = [plist[indice] for indice in frame_indices]
        cam_params = [Camera(cam_param) for cam_param in cam_params]

        if self.rescale_fxy:
            ori_h, ori_w = pixel_values.shape[-2:]
            ori_wh_ratio = ori_w / ori_h
            if ori_wh_ratio > self.sample_wh_ratio:  # rescale fx
                resized_ori_w = self.sample_size[0] * ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fx = resized_ori_w * cam_param.fx / self.sample_size[1]
            else:  # rescale fy
                resized_ori_h = self.sample_size[1] / ori_wh_ratio
                for cam_param in cam_params:
                    cam_param.fy = resized_ori_h * cam_param.fy / self.sample_size[0]
        intrinsics = np.asarray([[cam_param.fx * self.sample_size[1],
                                  cam_param.fy * self.sample_size[0],
                                  cam_param.cx * self.sample_size[1],
                                  cam_param.cy * self.sample_size[0]]
                                 for cam_param in cam_params], dtype=np.float32)
        intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]
        if self.relative_pose:
            c2w_poses = self.get_relative_pose(cam_params)
        else:
            c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
        c2w = torch.as_tensor(c2w_poses)[None]  # [1, n_frame, 4, 4]

        flip_flag = torch.zeros(self.t, dtype=torch.bool, device=c2w.device)
        plucker_embedding = ray_condition(intrinsics, c2w, self.sample_size[0], self.sample_size[1], device='cpu',
                                          flip_flag=flip_flag)[0].permute(0, 3, 1, 2).contiguous()
        # [V, H, W, 6] --> [V, 6, H, W]
        extrinsics = torch.empty((self.t, 4, 4))
        camera_embeddings = torch.empty((self.t, 12))
        intrinsics = plist[0][1:7]
        for i in range(self.t):
            index = frame_indices[i]
            pose = np.identity(4)
            tmp = plist[index][7:].reshape((3, 4))
            pose[:3, :4] = tmp
            if i == 0:
                base_pose = np.linalg.inv(np.array(pose))

            extrinsics[i] = torch.tensor(pose)
            camera_embeddings[i] = torch.tensor(np.matmul(base_pose, pose)[:3, :].flatten())

        return pixel_values, video_caption, plucker_embedding, \
            plist[0][1: 7], extrinsics, camera_embeddings, flip_flag

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption, plucker_embedding, intrinsics, extrinsics,camera_embeddings, flip_flag = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length - 1)

        for transform in self.pixel_transforms:
            video = transform(video)
        # pixel_values: t, c, h, w
        # plucker_embedding: t, 6, h, w
        # intrinsics: 6,
        # extrinsics: t, 4, 4
        return dict(
            pixel_values=video,
            text=video_caption,
            plucker_embedding=plucker_embedding,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            camera_embeddings = camera_embeddings,
        )