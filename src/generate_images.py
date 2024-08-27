#This script is used to generate the training data for each language
from diffusers import AltDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import jsonlines
import csv

pipe = AltDiffusionPipeline.from_pretrained("BAAI/AltDiffusion-m9")
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

with open('./prompts_zh-cn.csv', 'r') as file:
    reader = csv.DictReader(file)
    prompts = [row['Gendered Prompt'] for row in reader]

with jsonlines.open(f"./dataset_fine_tune_ch/train/metadata.jsonl", "w") as writer:
    for idx, prompt in enumerate(prompts):
        if idx > 8083:
            image_path = f"./dataset_fine_tune_ch/train/{idx}.jpg"
            image = pipe(prompt).images[0]
            image.save(image_path)
            writer.write( {
                "file_name": f"{idx}.jpg",
                "text": prompt
            } )