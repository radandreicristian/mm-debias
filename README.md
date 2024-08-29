# Debiasing Multi-Language Large Vision-Language Models

## Context

Large Vision-Language Models (LVLMs) are increasingly popular and influential in e-commerce, retail, content creation, and fashion. Despite their growing utility, these models are known to inherit social biases from their underlying vision and language components. The research aims to address these biases, particularly in the context of non-English-speaking users, and to make LVLMs more accessible and equitable.

# Setup
Create a virtual environment

```bash
python3 -m venv venv
source ./venv/bin/activate
(venv) pip install -r requirements.txt
```

## Train Data Generation

The first step in fine-tuning the Alt-Diffusion Model is creating the datasets required for training. This involves generating prompts in various languages, creating images based on those prompts, and finally, combining the results into cohesive datasets.

### Generate English Prompts
We begin by creating the English prompts using the following pattern:

```bash
<STYLE> photo of a <AGE> <GENDER> <PROFESSION> <DRESSING> <PLACE>
```

To generate these prompts, run the following command:

```bash
python -m src.generate_prompts
```

### Generate Translated Prompts
Translate the English prompts into German, Chinese, and Japanese:
```bash
python -m src.translate.translate_prompts
```

### Generate Images
Generate images using all prompts:
```bash
python -m src.generate_images
```

### Create the datasets
Combine the generated prompts and images to create a dataset for each experiment. You can choose the source of the prompts and the source of the images:
```bash
python -m src.collect_datasets
```


## Fine-tune Alt-Diffusion model
To fine-tune the Alt-Diffusion model, run the `train_text_to_image`  CLI.

CLI Parameters:

- —pretrained_model_name_or_path BAAI/AltDiffusion-m9
- —train_data_dir dataset_path
- —mixed_precision fp16
- —allow_tf32
- —gradient_checkpointing
- —resolution 256
- —center_crop
- —random_flip
- —train_batch_size 16
- —gradient_accumulation_steps 2
- —num_train_epochs 5
- —learning_rate 5e-06
- —lr_scheduler polynomial
- —lr_warmup_steps 0
- —output_dir "output_fine_tune/alt_diffusion_m9_fine_tune"

```bash
python -m src.tti-fine-tune.train_text_to_image
```

## Evaluation Results

### Bias
First, we need to run `classify.py` CLI to find faces and attributes (e.g. gender, race, etc.)

CLI Parameters:

- —model (default `AD`)
- —model_version (default `m9`)
- —classifier (default `fairface`)
- —dataset (we only use the default value, `occupations`)
- —language
- —num_images (default `25`)
- —postfix. This is a parameter that we added to manipulate paths if we move stuff around. For example, with the `base` postfix, we evaluate `data/eval/generated_images_base/...`

```bash
python -m src.evaluate.classify --language english
```

This will extract faces from images generated with `generate_images.py` and output a file like `occupations_fairface_generated.txt` with lines like

```bash
musician
man: X, woman: Y
```

To compute the bias metrics from the extracted faces, run the `evaluate_bias` CLI.

CLI Parameters:

- —model (default `AD`)
- —model_version (default `m9`)
- —classifier (default `fairface`)
- —dataset (we only use the default value, `occupations`)
- —language

```bash
python -m src.evaluate.evaluate_bias --model_version finetune_english --language english
```

### CLIP/Alignment
To evaluate for image/text alignment (CLIP score), run the `evaluate_alignment`  CLI.

CLI Parameters:

- —model (default `AD`)
- — model_version (default `m9`)
- —mode (we only use the default value, `generated`)
- —classifier (default `fairface`)
- —dataset (we only use the default value, `occupations`)
- —language
- —num_images (default `25`)
- —postfix

```bash
python -m src.evaluate.evaluate_alignment --language english --postfix base
```

Evaluates the images in `data/eval/generated_images_base/multilang/AD_m9/english/generated/[..]`  for all professions and outputs the results in `resutls/multilang/AD_m9/english/occupations_generated_clipscore.txt`