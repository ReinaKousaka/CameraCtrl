import os
import random
import csv
import json
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def _get_extrinsic_matrix(qw, qx, qy, qz, tx, ty, tz) -> torch.Tensor:
    """
    Args: quaternion and translation
    """
    extrinsic = np.eye(4)
    r = qvec2rotmat([qx, qy, qz, qw])
    extrinsic[:3, :3] = r
    extrinsic[:3, 3] = [tx, ty, tz]

    return torch.from_numpy(extrinsic).float()


def _get_rays(H, W, intrinsics, c2w):
    """
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    """
    u, v = torch.meshgrid(
        torch.arange(W, device=c2w.device),
        torch.arange(H, device=c2w.device),
        indexing="ij",
    )
    B = c2w.shape[0]
    u, v = u.reshape(-1), v.reshape(-1)
    u_noise = v_noise = 0.5
    u, v = u + u_noise, v + v_noise  # add half pixel
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=0)  # (3, H*W)
    pixels = pixels.unsqueeze(0).repeat(B, 1, 1)  # (B, 3, H*W)
    if intrinsics.sum() == 0:
        inv_intrinsics = torch.eye(3, device=c2w.device).tile(B, 1, 1)
    else:
        inv_intrinsics = torch.linalg.inv(intrinsics)
    rays_d = inv_intrinsics @ pixels  # (B, 3, H*W)
    rays_d = c2w[:, :3, :3] @ rays_d
    rays_d = rays_d.transpose(-1, -2)  # (B, H*W, 3)
    rays_d = F.normalize(rays_d, dim=-1)

    rays_o = c2w[:, :3, 3].reshape((-1, 3))  # (B, 3)
    rays_o = rays_o.unsqueeze(1).repeat(1, H * W, 1)  # (B, H*W, 3)

    return rays_o, rays_d


# cite from: https://github.com/echen01/ray-conditioning/blob/8e1d5ae76d4747c771d770d1f042af77af4b9b5d/training/plucker.py#L9
def _get_plucker_embedding(H, W, intrinsics, c2w):
    """Computes the plucker coordinates from batched cam2world & intrinsics matrices, as well as pixel coordinates
    c2w: (B, 4, 4)
    intrinsics: (B, 3, 3)
    """    
    cam_pos, ray_dirs = _get_rays(H, W, intrinsics, c2w)
    cross = torch.cross(cam_pos, ray_dirs, dim=-1)
    plucker = torch.cat((ray_dirs, cross), dim=-1)

    plucker = plucker.view(-1, H, W, 6).permute(0, 3, 1, 2)
    return plucker  # B, 6, H, W


class EpicKitchen(Dataset):
    def __init__(self,
        root = '/root/workspace/CameraCtrl/Epic',
        image_subfolder = 'epic',
        posefile_subfolder = 'pose',
        meta_file = "EPIC_100_train.csv",
        caption_subfolder = 'caption_per5',
        h = 256,
        w = 448,
        sample_size = [256, 448],
        num_frames=8,
        sample_stride=5,
        is_image=False,         # set to true to return C, H, W instead of T, C, H, W
        sample_by_narration=True,       # true: use narration as key
        is_valid = False,
    ):
        """Define EpicKiten Dataset
        Args:
            root (str): path of images
            image_subfolder (str): relative path to the folder of video frames
            meta_file (str): relative path to the meta file, e.g. EPIC_100_train.csv
            num_frames (int): frames number of input images sequences
        """
        with open(os.path.join(root, meta_file), "r") as f:
            self.meta_file = csv.reader(f)
            self.meta_file = list(self.meta_file)[1:] # drop the head line
        self.root = root
        self.image_subfolder = image_subfolder
        self.posefile_subfolder = posefile_subfolder
        self.caption_subfolder = caption_subfolder
        self.h, self.w = h, w
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.datalist = []
        self.narration_to_videos = defaultdict(list)
        self.is_image = is_image
        self.is_valid = is_valid
        self.sample_by_narration = sample_by_narration
        
        # TODO: confirm the tranformer
        self.transformer = transforms.Compose([
            transforms.Resize([self.h, self.w]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self._check_from_data()

        # The following snippet is to save frame_to_ex.json
        # self.frame_to_ex = {}
        # with open(os.path.join(root, 'epic_cam-pre.json')) as f:
        #     data = json.load(f)
        #     for chunk in data:
        #         narration = chunk[0]   
        #         for piece in chunk[1]:
        #             path = piece[0]
        #             path = os.path.join(root, image_subfolder, path[26:])
        #             # extrinsic = torch.tensor(piece[1]).float()
        #             extrinsic = piece[1]
        #             # self.datalist2.append((narration, path, extrinsic))
        #             splits = path.split('/')
        #             if splits[-2] not in self.frame_to_ex:
        #                 self.frame_to_ex[splits[-2]] = {}
        #             self.frame_to_ex[splits[-2]][splits[-1]] = extrinsic
        # with open(os.path.join(root, 'frame_to_ex.json'), 'w') as f:
        #     # self.frame_to_ex = json.load(f)
        #     json.dump(self.frame_to_ex, f)
        # exit(0)

        with open(os.path.join(root, 'frame_to_ex.json')) as f:
            self.frame_to_ex = json.load(f)

    def _check_from_data(self):
        res = 0
        for line in self.meta_file:
            video_id = line[2]
            # skip if the video is missing
            frame_dir = os.path.join(self.root, self.image_subfolder, video_id)
            if not os.path.exists(frame_dir): continue
            start_frame = int(line[6])
            end_frame = int(line[7])
            narration = line[8]
            self.datalist.append((video_id, start_frame, end_frame, narration))
            res += end_frame - start_frame + 1
            self.narration_to_videos[narration].append((video_id, start_frame, end_frame,))

    def __len__(self):
        if self.sample_by_narration:
            return len(self.narration_to_videos.keys())
        else:
            return len(self.datalist)

    def __getitem__(self, index):
        while True:
            try:
                if not self.sample_by_narration:
                    video_id, start_frame, end_frame, narration = self.datalist[index]
                else:
                    narration = list(self.narration_to_videos.keys())[index]
                    if self.is_valid:
                        # for validation, use fixed first video
                        video_id, start_frame, end_frame = self.narration_to_videos[narration][0]
                    else:
                        video_id, start_frame, end_frame = random.sample(self.narration_to_videos[narration], 1)[0]
                if start_frame + (self.num_frames - 1) * self.sample_stride < end_frame:
                    indices = [x for x in range(start_frame, end_frame + 1, self.sample_stride)][:self.num_frames]
                else:
                    indices = random.sample(range(start_frame, end_frame + 1), self.num_frames)
                assert len(indices) == self.num_frames
                
                pixels = []
                extrinsics = []
                intrinsics = torch.zeros(6)
                captions = []
                for i, index in enumerate(indices):
                    filename = f'frame_{str(index).zfill(10)}'

                    # get captions
                    if i == 0 or i == self.num_frames - 1:
                        with open(os.path.join(self.root, self.caption_subfolder, video_id + '.json')) as f:
                            data = json.load(f)
                            # round frames to existing keys
                            if i == 0:
                                mods = [1, 0, 4, 3, 2]
                            else:
                                mods = [-4, 0, -1, -2, -3]
                            index_ = index + mods[index % 5]
                            captions.append(data[f'frame_{str(index_).zfill(10)}.jpg'])

                    # get pixels
                    with Image.open(os.path.join(self.root, self.image_subfolder, video_id, filename + '.jpg')) as img:
                        pixels.append(self.transformer(img))
                    
                    # get camera poses
                    extrinsics.append(torch.tensor(self.frame_to_ex[video_id][filename + '.jpg']).float())
                    # extrinsics.append(_get_extrinsic_matrix(*data['images'][filename + '.jpg']))

                with open(os.path.join(self.root, self.posefile_subfolder, video_id + '.json')) as f:
                    data = json.load(f)
                    for j in range(4):
                        intrinsics[j] = data['camera']['params'][j]
                break
            except Exception as err:
                # skip if the corresponding camera pose is missing, get a random idx instead
                index = random.randint(0, self.__len__() - 1)
                # print(f'err: {err}')

        pixels = torch.stack(pixels, dim = 0)
        extrinsics = torch.stack(extrinsics, dim = 0)

        intrinsic_mat = torch.tensor([
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1]
        ], dtype=torch.float32)
        plucker_embedding = _get_plucker_embedding(
            self.h, self.w,
            torch.unsqueeze(intrinsic_mat, 0).repeat(self.num_frames, 1, 1),      # 3, 3 --> T, 3, 3
            extrinsics
        )

        if self.is_image:
            pixels = pixels[0]

        return {
            'pixel_values': pixels,     # T, C, H, W
            'text': captions[0] + ',' + captions[1] + ',' + narration,     # str
            'intrinsics': intrinsics,       # 6,
            'extrinsics': extrinsics,       # T, 4, 4
            'plucker_embedding': plucker_embedding,     # T, 6, H, W
        }