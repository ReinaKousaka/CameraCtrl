import os
import json
from json.decoder import JSONDecodeError
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from lavis.models import load_model_and_preprocess


def chunks(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i + chunk_size]


def run(device):
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption",
        model_type="large_coco",
        is_eval=True,
        device=device
    )

    root_path = './Epic'
    frame_dir = 'epic'
    video_ids = sorted(os.listdir(os.path.join(root_path, frame_dir)))
    chunk_size = len(video_ids) // 8
    video_ids = video_ids[chunk_size * device: chunk_size * (device + 1)]
    print(f'running on device {device}, vidoes: {video_ids}')

    for video_id in tqdm(video_ids):
        json_path = os.path.join(root_path, 'caption_shift2', video_id + '.json')
        # the json has been created by someone else, skip
        if os.path.exists(json_path):
            continue
        img_names = sorted(os.listdir(os.path.join(root_path, frame_dir, video_id)))[2::5]
        for chunk in chunks(img_names, 15):
            img_tensors = []
            for img_name in chunk:
                raw_image = Image.open(os.path.join(root_path, frame_dir, video_id, img_name)).convert("RGB")
                image = vis_processors["eval"](raw_image).to(device)
                img_tensors.append(image)

            img_tensors = torch.stack(img_tensors).to(device)
            captions = model.generate({"image": img_tensors})

            try:
                with open(json_path) as f:
                    to_write = json.load(f)
                if to_write is None: to_write = {}
            except (IOError, JSONDecodeError):
                to_write = {}

            for idx, caption in enumerate(captions):
                to_write[chunk[idx]] = caption
            with open(json_path, 'w+') as f:
                json.dump(to_write, f)
        print(f'device {device} finishes writing {video_id}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    run(args.device)
