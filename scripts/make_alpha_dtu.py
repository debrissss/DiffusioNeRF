#!/usr/bin/env python3
"""Build an RGBA (alpha-channel) DTU scan dataset for few-shot training.

Background: torch-ngp's native pipeline treats a 4-channel image as
"RGB + object-mask alpha". During training the collate path then trains with a
pixel-wise random background color (nerf/utils.py, train_step C==4 branch),
which forces alpha -> 0 on background rays regardless of the true backdrop
color. At evaluation the GT and renders are composited onto a fixed white
background. This removes any dependence on --bg_color for scenes whose far
background is bright or inconsistent.

Inputs:  <src>/images/*.png (RGB), <src>/masks/*.png (binary, white=object),
         <src>/transforms.json
Output:  <dst>/images_rgba/*.png (RGBA), <dst>/masks/*.png (copy),
         <dst>/transforms.json (file_path rewritten), and matching split
         copies under splits/dtu_{3v,9v}/<dst_name>.json when the source scan
         has split files.

Usage:
    python scripts/make_alpha_dtu.py data/DTU_standard/scan63 \
        [--dst data/DTU_standard/scan63_alpha]

    python scripts/make_alpha_dtu.py data/DTU/scan24 \
        --image-dir image --mask-dir mask --output-dir image_alpha
"""

import argparse
import glob
import json
import os
import shutil

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src', type=str, help='scan dir with images/, masks/, transforms.json')
    parser.add_argument('--dst', type=str, default=None,
                        help='output dir (default: <src>_alpha next to src)')
    parser.add_argument('--image-dir', type=str, default='images',
                        help='image directory relative to src')
    parser.add_argument('--mask-dir', type=str, default='masks',
                        help='mask directory relative to src')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='write only RGBA images to this directory relative to src')
    parser.add_argument('--size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help='resize RGB images and masks before composing RGBA')
    args = parser.parse_args()

    src = os.path.abspath(args.src)
    image_dir = os.path.join(src, args.image_dir)
    mask_dir = os.path.join(src, args.mask_dir)

    if args.output_dir is not None:
        output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                      else os.path.join(src, args.output_dir))
        os.makedirs(output_dir, exist_ok=True)

        n_img = 0
        for p in sorted(glob.glob(os.path.join(image_dir, '*.png'))):
            fid = os.path.basename(p)
            mask_p = os.path.join(mask_dir, fid)
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
            if args.size is not None:
                size = tuple(args.size)
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = mask
            cv2.imwrite(os.path.join(output_dir, fid), rgba)
            n_img += 1

        print(f'OK: {n_img} RGBA images -> {output_dir}')
        return

    dst = os.path.abspath(args.dst) if args.dst else src.rstrip('/') + '_alpha'
    name = os.path.basename(dst)
    os.makedirs(os.path.join(dst, 'images_rgba'), exist_ok=True)
    os.makedirs(os.path.join(dst, 'masks'), exist_ok=True)

    n_img = 0
    for p in sorted(glob.glob(os.path.join(image_dir, '*.png'))):
        fid = os.path.basename(p)
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        mask_p = os.path.join(mask_dir, fid)
        if not os.path.isfile(mask_p):
            raise FileNotFoundError(f'missing mask for {fid}: {mask_p}')
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        cv2.imwrite(os.path.join(dst, 'images_rgba', fid), rgba)
        n_img += 1

    n_mask = 0
    for p in glob.glob(os.path.join(mask_dir, '*.png')):
        shutil.copy(p, os.path.join(dst, 'masks'))
        n_mask += 1

    with open(os.path.join(src, 'transforms.json'), 'r', encoding='utf-8') as f:
        tr = json.load(f)
    for fr in tr['frames']:
        fr['file_path'] = fr['file_path'].replace('images/', 'images_rgba/')
    with open(os.path.join(dst, 'transforms.json'), 'w', encoding='utf-8') as f:
        json.dump(tr, f)

    for v in ('3v', '9v'):
        split_p = os.path.join('splits', f'dtu_{v}', os.path.basename(src) + '.json')
        if os.path.isfile(split_p):
            with open(split_p, 'r', encoding='utf-8') as f:
                spec = json.load(f)
            spec['scene'] = name
            with open(os.path.join('splits', f'dtu_{v}', name + '.json'), 'w', encoding='utf-8') as f:
                json.dump(spec, f, indent=1)
            print(f'wrote splits/dtu_{v}/{name}.json')

    sample = cv2.imread(os.path.join(dst, 'images_rgba',
                         sorted(os.listdir(os.path.join(dst, 'images_rgba')))[0]),
                        cv2.IMREAD_UNCHANGED)
    if sample is None or sample.shape[-1] != 4:
        raise RuntimeError('RGBA verification failed')
    print(f'OK: {n_img} RGBA images, {n_mask} masks -> {dst}')


if __name__ == '__main__':
    main()
