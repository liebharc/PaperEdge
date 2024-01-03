# -*- encoding: utf-8 -*-
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from networks.paperedge import GlobalWarper, LocalWarper, WarperUtil

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

script_dir = Path(__file__).parent.resolve()
model_dir = script_dir / 'models'

def load_img(img_path):
    im = cv2.imread(img_path).astype(np.float32) / 255.0
    im = im[:, :, (2, 1, 0)]
    im = cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
    return im

def limit_to_dimension(value, dimension):
    value = int(value)
    if value < 0:
        return 0
    if value >= dimension:
        return dimension - 1
    return value

def relative_range_to_abs(image_dimension, grid_dimension):
    """Converts tensors from [-1, 1] to [min_val, max_val]."""
    min_max = (np.min(grid_dimension), np.max(grid_dimension))
    min_val = ((min_max[0] + 1) / 2) * image_dimension
    max_val = ((min_max[1] + 1) / 2) * image_dimension
    return (limit_to_dimension(min_val, image_dimension), limit_to_dimension(max_val, image_dimension))

def crop_ranges(image, grid):
    x_range = relative_range_to_abs(image.shape[1], grid[0])
    y_range = relative_range_to_abs(image.shape[0], grid[1])
    if x_range[1] - x_range[0] < 100 or y_range[1] - y_range[0] < 100:
        print("Invalid crop ranges: ", x_range, y_range)
        return image
    print("Cropping image to: ", x_range, y_range)
    return image[y_range[0]:y_range[1], x_range[0]:x_range[1], :]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Enet_ckpt', type=str,
                        default=model_dir / 'G_w_checkpoint_13820.pt')
    parser.add_argument('--Tnet_ckpt', type=str,
                        default=model_dir / 'L_w_checkpoint_27640.pt')
    parser.add_argument('--img_path', type=str, default='images/3.jpg')
    parser.add_argument('--out_dir', type=str, default='output')
    parser.add_argument('--only_crop', default=False, action='store_true')
    args = parser.parse_args()

    img_path = args.img_path
    dst_dir = args.out_dir
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    netG = GlobalWarper().to('cuda')
    netG.load_state_dict(torch.load(args.Enet_ckpt)['G'])
    netG.eval()

    netL = LocalWarper().to('cuda')
    netL.load_state_dict(torch.load(args.Tnet_ckpt)['L'])
    netL.eval()

    warpUtil = WarperUtil(64).to('cuda')

    gs_d, ls_d = None, None
    with torch.no_grad():
        x = load_img(img_path)
        x = x.unsqueeze(0)
        x = x.to('cuda')
        d = netG(x)  # d_E the edged-based deformation field
        d = warpUtil.global_post_warp(d, 64)
        gs_d = copy.deepcopy(d)

        d = F.interpolate(d, size=256, mode='bilinear', align_corners=True)
        y0 = F.grid_sample(x, d.permute(0, 2, 3, 1), align_corners=True)
        ls_d = netL(y0)
        ls_d = F.interpolate(ls_d, size=256, mode='bilinear', align_corners=True)
        ls_d = ls_d.clamp(-1.0, 1.0)

    org_img = cv2.imread(img_path).astype(np.float32) / 255.0
    im = torch.from_numpy(np.transpose(org_img, (2, 0, 1)))
    im = im.to('cuda').unsqueeze(0)

    if args.only_crop:
        gs_d = F.interpolate(gs_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
        gs_y = F.grid_sample(im, gs_d.permute(0, 2, 3, 1), align_corners=True).detach()
        grid = gs_d.cpu().numpy()
        tmp_y = crop_ranges(org_img, grid[0])
        cv2.imwrite(f'{dst_dir}/result_crop.png', tmp_y * 255.)
        exit()

    gs_d = F.interpolate(gs_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
    gs_y = F.grid_sample(im, gs_d.permute(0, 2, 3, 1), align_corners=True).detach()
    tmp_y = gs_y.squeeze().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(f'{dst_dir}/result_gs.png', tmp_y * 255.)

    ls_d = F.interpolate(ls_d, (im.size(2), im.size(3)), mode='bilinear', align_corners=True)
    ls_y = F.grid_sample(gs_y, ls_d.permute(0, 2, 3, 1), align_corners=True).detach()
    ls_y = ls_y.squeeze().permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(f'{dst_dir}/result_ls.png', ls_y * 255.)
