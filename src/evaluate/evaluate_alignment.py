# Source: https://github.com/felifri/MAGBIG/blob/main/generate_evaluate/CLIPscore.py

import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from src.evaluate.utils import ensure_dir, get_img_list
import argparse
import clip
import torch

parser = argparse.ArgumentParser(description='Bias in Diffusion Evaluation')
parser.add_argument('--model', default='AD', type=str,
                    help='which model to evaluate')
parser.add_argument('--model_version', default='m9', type=str,
                    help='which version of this model to evaluate')
parser.add_argument('--mode', default='generated', type=str,
                    choices=['laion', 'sega', 'generated', 'baseline', 'baseline_ext', 'baseline_neg'],
                    help='which images to evaluate')
parser.add_argument('--classifier', default='fairface', type=str, choices=['fairface', 'clip'],
                    help='which classifier to use for evaluation')
parser.add_argument('--dataset', default='occupations', type=str, choices=['occupations', 'adjectives'],
                    help='which dataset to evaluate')
parser.add_argument('--non-binary', default=False, type=bool,
                    help='only binary evaluation?')
parser.add_argument('--language', default='', type=str,
                    choices=['', 'english', 'arabic', 'chinese_simplified', 'chinese_traditional', 'spanish', 'italian',
                             'german', 'german_star', 'korean', 'russian', 'french', 'japanese'],
                    help='what category to evaluate')
parser.add_argument('--gender_neutral', default='', type=str,
                    help='whether to evaluate gender-neutral prompts')
parser.add_argument('--postfix', default='', type=str)
parser.add_argument('--num_images', default=10, type=int,
                    help='how many images to generate')
args = parser.parse_args()

data = pd.read_csv('data/eval/prompts/indirect:gender_neutral/occ_english.csv')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model.eval()

if not args.language:
    pth = f"results/{args.model}_{args.model_version}"
else:
    if not args.gender_neutral:
        pth = f"data/results/multiling/{args.model}_{args.model_version}/{args.language}"
    else:
        pth = f"data/results/multiling/gender_neutral/{args.model}_{args.model_version}/{args.language}"

ensure_dir(pth)

txt_file = open(f'{pth}/{args.dataset}_{args.mode}_clipscore.txt', 'w+')

for i, d in data.iterrows():
    sim = 0
    # args.mode = generated, d['name']=accountant or idk
    # postfix = "_english"
    img_list = get_img_list(args.mode, d['name'], args, images_path_postfix=args.postfix)
    prompt_gn = d['prompt']
    prompt = f"a photo of the face of a {d['name']}"
    prompt_neutral = "a photo of the face of a person"
    prompt = [prompt_gn, prompt, prompt_neutral]

    text = clip.tokenize(prompt).to(device)
    text_features = model.encode_text(text)
    sims = []
    for im in tqdm(img_list):
        image = preprocess(Image.open(im)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        ti_sim = torch.nn.functional.cosine_similarity(image_features, text_features)
        sims.append(ti_sim.detach().cpu().numpy())

    sim = np.mean(sims, axis=0)
    txt_file.write(f"{d['name']}: {sim}\n")
txt_file.close()