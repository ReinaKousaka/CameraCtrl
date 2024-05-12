import torch
from scipy.spatial.transform import Rotation
import numpy as np
from cameractrl.data.dataset_epic import EpicKitchen


from einops import rearrange
import random
from cameractrl.geometry.epipolar_lines import project_rays
from cameractrl.geometry.projection import get_world_rays
from cameractrl.visualization.annotation import add_label
from cameractrl.visualization.drawing.lines import draw_lines, draw_attn
from cameractrl.visualization.drawing.points import draw_points
from cameractrl.visualization.layout import add_border, hcat

from torchvision.utils import save_image


def helper(data, name):
    device = 0
    tmp = data
    h, w  = 256, 448
    x = 320
    y = 180
    xy = torch.tensor((x / w, y / h), dtype=torch.float32, device=device)

    xs = torch.linspace(0, 1, steps=7)
    ys = torch.linspace(0, 1, steps=4)
    grid = torch.stack(torch.meshgrid(xs, ys, indexing='xy'),dim=-1).float().to(device)

    grid = rearrange(grid, "h w c  -> (h w) c")

    frame2 = 3
    source_image = tmp['pixel_values'][0].to(("cuda:0"))
    target_image = tmp['pixel_values'][frame2].to(("cuda:0"))

    k = torch.tensor(np.identity(3)).float()
    k[0, 0] = tmp['intrinsics'][0] / 456.0
    k[1, 1] = tmp['intrinsics'][1] / 256.0
    k[0, 2] = 0.5
    k[1, 2] = 0.5

    target_intrinsics = source_intrinsics = k.to(("cuda:0"))

    source_extrinsics = tmp['extrinsics'][0].to(("cuda:0"))
    source_extrinsics = torch.inverse(source_extrinsics)

    origin, direction = get_world_rays(
        grid,
        source_extrinsics,
        source_intrinsics
    )
    target_extrinsics = tmp['extrinsics'][frame2].to(("cuda:0"))
    target_extrinsics = torch.inverse(target_extrinsics)


    projection = project_rays(
        origin,
        direction,
        target_extrinsics,
        source_intrinsics,
    )

    for i in range(28):
        color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

        source_image = draw_points(
            source_image, grid[i], color, 4, x_range=(0, 1), y_range=(0, 1)
        )

        target_image = draw_lines(
            target_image,
            projection["xy_min"][i],
            projection["xy_max"][i],
            color,
            4,
            x_range=(0, 1),
            y_range=(0, 1),
        )

    source_image = add_label(source_image, "Source")
    target_image = add_label(target_image, "Target")
    together = add_border(hcat(source_image, target_image))
    
    print(f'together shape: {together.shape}')
    save_image(together, f'./output_draw/{name}.png')



ds = EpicKitchen()
helper(ds[8900], 0)
helper(ds[700], 1)
helper(ds[2000], 2)