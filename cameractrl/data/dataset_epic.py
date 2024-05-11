import os
import random
import csv
import json
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from packaging import version as pver

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F


def _get_extrinsic_matrix(qw, qx, qy, qz, tx, ty, tz) -> torch.Tensor:
    """
    Args: quaternion and translation
    """
    extrinsic = np.eye(4)
    # note the order: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html#scipy.spatial.transform.Rotation.from_quat
    r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # R = Rotation.from_quat([qx, qy, qz, qw]).as_euler('xyz')
    # R[1] *= -1
    # R[2] *= -1
    # r = Rotation.from_euler('xyz', R).as_matrix()
    extrinsic[:3, :3] = r
    # extrinsic[:3, 3] = [tx, -ty, -tz]
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
        root = '/root/CameraCtrl/Epic',
        image_subfolder = 'epic',
        posefile_subfolder = 'pose',
        meta_file = "EPIC_100_train.csv", 
        h = 256,
        w = 448,
        num_frames=8,
        sample_stride=4,
        is_image=False,         # set to true to return C, H, W instead of T, C, H, W
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
        self.h, self.w = h, w
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.datalist = []
        self.is_image = is_image
        
        # TODO: confirm the tranformer
        self.transformer = transforms.Compose([
            transforms.Resize([self.h, self.w]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self._check_from_data()

    def _check_from_data(self):
        for line in self.meta_file:
            video_id = line[2]
            # skip if the video is missing
            frame_dir = os.path.join(self.root, self.image_subfolder, video_id)
            if not os.path.exists(frame_dir): continue
            start_frame = int(line[6])
            end_frame = int(line[7])
            narration = line[8]
            self.datalist.append((video_id, start_frame, end_frame, narration))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        while True:
            try:
                video_id, start_frame, end_frame, narration = self.datalist[index]
                if start_frame + (self.num_frames - 1) * self.sample_stride < end_frame:
                    indices = [x for x in range(start_frame, end_frame + 1, self.sample_stride)][:self.num_frames]
                else:
                    indices = random.sample(range(start_frame, end_frame + 1), self.num_frames)
                assert len(indices) == self.num_frames
                
                pixels = []
                extrinsics = []
                intrinsics = torch.zeros(6)
                for i in indices:
                    filename = f'frame_{str(i).zfill(10)}'

                    # get pixels
                    with Image.open(os.path.join(self.root, self.image_subfolder, video_id, filename + '.jpg')) as img:
                        pixels.append(self.transformer(img))
                    
                    # get camera poses
                    with open(os.path.join(self.root, self.posefile_subfolder, video_id + '.json')) as f:
                        data = json.load(f)
                        extrinsics.append(_get_extrinsic_matrix(*data['images'][filename + '.jpg']))
                        if i == indices[0]:
                            for j in range(4):
                                intrinsics[j] = data['camera']['params'][j]
                break
            except Exception as err:
                # skip if the corresponding camera pose is missing, get a random idx instead
                index = random.randint(0, self.__len__() - 1)
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
            'caption': narration,          # str
            'intrinsics': intrinsics,       # 6,
            'extrinsics': extrinsics,       # T, 4, 4
            'plucker_embedding': plucker_embedding,     # T, 6, H, W
        }
