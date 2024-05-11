import os
import random
import torch
import csv
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class EpicKitchen(Dataset):
    def __init__(self,
        root = '/root/CameraCtrl/Epic',
        image_subfolder = 'epic',
        meta_file = "EPIC_100_train.csv", 
        h = 256,
        w = 448,
        num_frames=8,
        sample_stride=6,        # TODO: confirm appropriate sample stride
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
            self.datalist.append((frame_dir, start_frame, end_frame, narration))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        frame_dir, start_frame, end_frame, narration = self.datalist[index]
        if start_frame + (self.num_frames - 1) * self.sample_stride < end_frame:
            indices = [x for x in range(start_frame, end_frame + 1, self.sample_stride)][:self.num_frames]
        else:
            indices = random.sample(range(start_frame, end_frame + 1), self.num_frames)
        assert len(indices) == self.num_frames
        pixels = []
        for i in indices:
            with Image.open(os.path.join(frame_dir, f'frame_{str(i).zfill(10)}.jpg')) as img:
                pixels.append(self.transformer(img))
        pixels = torch.stack(pixels, dim = 0)
        if self.is_image:
            pixels = pixels[0]

        return {
            'pixel_values': pixels,     # T, C, H, W
            'caption': narration + " in the kitchen with hands, egocentric view, first person",          # str
        }
