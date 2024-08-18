#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shutil
import sys
import tempfile

from diffusers import DiffusionPipeline, UNet2DConditionModel  # noqa: E402


sys.path.append("..")

from utils.test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class TextToImage(ExamplesTestsAccelerate):
    def test_text_to_image(self):
        output_directory="output_fine_tune/alt_diffusion_m9_fine_tune"
        test_args = f"""
            tti_fine_tune/train_text_to_image.py
            --pretrained_model_name_or_path BAAI/AltDiffusion-m9
            --train_data_dir dataset_fine_tune
            --mixed_precision fp16
            --allow_tf32
            --gradient_checkpointing
            --resolution 256
            --center_crop
            --random_flip
            --train_batch_size 1
            --gradient_accumulation_steps 1
            --num_train_epochs 2
            --learning_rate 5.0e-09
            --lr_scheduler constant
            --lr_warmup_steps 0
            --output_dir {output_directory}
            """.split()

        output = run_command(self._launch_args + test_args, return_stdout=True)
        print(output)
        # save_pretrained smoke test
        self.assertTrue(os.path.isfile(os.path.join(output_directory, "unet", "diffusion_pytorch_model.bin")))
        self.assertTrue(os.path.isfile(os.path.join(output_directory, "scheduler", "scheduler_config.json")))
