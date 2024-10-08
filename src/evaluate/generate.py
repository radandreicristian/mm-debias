# Source: https://github.com/felifri/MAGBIG/blob/main/generate_evaluate/generate_images.py

import torch
from diffusers import AltDiffusionPipeline
import os
import json
import numpy as np
from utils import face_existing
import dlib
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Bias in Diffusion Generation')
parser.add_argument('--model', default='AD', type=str,
                    help='which model to evaluate')
parser.add_argument('--model_version', default='m9', type=str,
                    help='which version of this model to evaluate')
parser.add_argument('--mode', default='generated', type=str, choices=['generated'],
                    help='which images to generate')
parser.add_argument('--dataset', default='occupations', type=str, choices=['occupations', 'adjectives'],
                    help='which dataset to evaluate')
parser.add_argument('--language', default='', type=str,
                    choices=['', 'english', 'arabic', 'chinese', 'chinese_traditional', 'spanish', 'italian',
                             'german', 'korean', 'russian', 'french', 'japanese', 'german_star'],
                    help='what category to evaluate')
parser.add_argument('--gender_neutral', default='', type=str,
                    help='whether to evaluate gender-neutral prompts')
parser.add_argument('--num_images', default=100, type=int,
                    help='how many images to generate')

args = parser.parse_args()

# TODO: add Multifusion, SD-Distill, Safe SD, DallE, Midjourney, etc.?
device = 'cuda'

if args.model == 'SD':
    if 'XL' in args.model_version:
        if args.model_version == 'XL':
            model_name = "/workspace/StableDiff/models/stable-diffusion-xl-base-0.9_"
        elif args.model_version == 'XLR':
            model_name = "/workspace/StableDiff/models/stable-diffusion-xl-refiner-0.9"
    else:
        if args.model_version == '1-4':
            model_name = "/workspace/StableDiff/models/stable-diffusion-v1-4"
        elif args.model_version == '1-5':
            model_name = "/workspace/StableDiff/models/stable-diffusion-v1-5"
        elif args.model_version == '2-0':
            model_name = "/workspace/StableDiff/models/stable-diffusion-2-base"
        elif args.model_version == '2-1':
            model_name = "/workspace/StableDiff/models/stable-diffusion-2-1-base"
        elif args.model_version == 'ED':
            model_name = "/workspace/StableDiff/models/epic-diffusion-v1.1"
            # model_name = "johnslegers/epic-diffusion-v1.1"
        elif args.model_version == 'CSRD':
            model_name = "/workspace/StableDiff/models/cutesexyrobutts-diffusion"
            # model_name = "andite/cutesexyrobutts-diffusion"
        elif args.model_version == 'DL':
            model_name = "/workspace/StableDiff/models/dreamlike-2-0"
        elif args.model_version == 'OJ':
            model_name = "/workspace/StableDiff/models/openjourney"
        elif args.model_version == 'RV':
            model_name = "/workspace/StableDiff/models/realistic-vision-v1-4"

    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        safety_checker=None
    ).to(device)
elif args.model == 'AD':
    if args.model_version == 'm9':
        model_name = "BAAI/AltDiffusion-m9"
    elif args.model_version == 'm18':
        model_name = "BAAI/AltDiffusion-m18"
    elif args.model_version == 'finetune_english':
        model_name = "/mnt/QNAP/viceco/scripts/mm-debias/src/output_fine_tune/alt_diffusion_m9_fine_tune_english_5e6_e5_poly"
    elif args.model_version == 'finetune_english_all_prompts':
        model_name = "/mnt/QNAP/viceco/scripts/mm-debias/src/output_fine_tune/alt_diffusion_m9_fine_tune_english_all_prompts_5e6_e5_poly"
    elif args.model_version == 'finetune_all_languages':
        model_name = "/mnt/QNAP/viceco/scripts/mm-debias/src/output_fine_tune/alt_diffusion_m9_fine_tune_all_languages_5e6_e5_poly"
    elif args.model_version == 'finetune_german_all_prompts':
        model_name = "/mnt/QNAP/viceco/scripts/mm-debias/src/output_fine_tune/alt_diffusion_m9_fine_tune_german_all_prompts_5e6_e5_poly"

    pipe = AltDiffusionPipeline.from_pretrained(
        model_name,
        safety_checker=None
    ).to(device)


cnn_face_detector = dlib.cnn_face_detection_model_v1('models/mmod_human_face_detector.dat')
gen = torch.Generator(device=device)

if args.dataset == 'occupations':
    if not args.gender_neutral:
        data = pd.read_csv(f'data/eval/prompts/occ_{args.language}.csv')
    else:
        data = pd.read_csv(f'data/eval/prompts/gender_neutral/occ_{args.language}.csv')
elif args.dataset == 'adjectives':
    data = pd.read_csv(f'data/eval/prompts/adj_{args.language}.csv')

if not args.language:
    pth_dir = f"data/eval/generated_images_base/{args.model}_{args.model_version}"
else:
    if not args.gender_neutral:
        pth_dir = f"data/eval/generated_images_base/multiling/{args.model}_{args.model_version}/{args.language}"
    else:
        pth_dir = f"data/eval/generated_images_base/multiling/gender_neutral/{args.model}_{args.model_version}/{args.language}"

if args.mode == 'generated':
    for _, cl in data.iterrows():
        pth = f"{pth_dir}/{args.mode}/{cl['name']}"
        os.makedirs(pth, exist_ok=True)
        i, j = 0, 0
        while j < args.num_images:
            gen.manual_seed(i)
            params = {'guidance_scale': 7,
                      'prompt': cl['prompt'],
                      'num_images_per_prompt': 1
                      }
            out = pipe(**params, generator=gen)
            image = out.images[0]
            # check if face exists in img with fairface detector
            if face_existing(np.array(image), cnn_face_detector) == 1 or i > 0:
                image.save(f"{pth}/image{j}.png")
                params['seed'] = i
                with open(f"{pth}/image{j}.json", 'w') as fp:
                    json.dump(params, fp)
                j += 1
            else:
                print(f'no Face - {i}')
            i += 1