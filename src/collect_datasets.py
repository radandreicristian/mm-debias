#This script is used to create the final datasets used for training (combine prompts in different languages with the generated images)
import os
import shutil
import jsonlines
import csv

images_folders = [
    'dataset_fine_tune_english/train',
    'dataset_fine_tune_english/train',
    'dataset_fine_tune_english/train',
    'dataset_fine_tune_english/train'
]
csv_files = [
    'prompts_en.csv',
    'prompts_ja.csv',
    'prompts_zh-cn.csv',
    'prompts_de.csv'
]
output_folder = 'dataset_fine_tune_english_all_prompts/train'
metadata_output = os.path.join(output_folder, 'metadata.jsonl')

os.makedirs(output_folder, exist_ok=True)

metadata_rows = []

with jsonlines.open(metadata_output, "w") as writer:
    new_image_index = 0
    for index_csv, csv_file in enumerate(csv_files):
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                old_image_path = os.path.join(images_folders[index_csv], f"{index}.jpg")
                new_image_name = f"{new_image_index}.jpg"
                new_image_path = os.path.join(output_folder, new_image_name)

                shutil.copyfile(old_image_path, new_image_path)

                writer.write( {
                    "file_name": f"{new_image_index}.jpg",
                    "text": row["Non-Gendered Prompt"]
                } )

                new_image_index += 1
