import copy
import logging
import time
import warnings
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import HfArgumentParser

from mu.mu_vision.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from mu.mu_vision.conversation import conv_templates
from mu.mu_vision.mm_utils import process_images, tokenizer_image_token
from mu.mu_vision.model.builder import load_pretrained_model

logger = logging.getLogger(__name__)


@dataclass
class GenerateTextImage2Text:
    def setup(self):
        warnings.filterwarnings("ignore")

        # Mu v2 model
        pretrained = "artifacts/models/mu_vision/"

        model_name = "llava_mu_v2"
        mu_version = "mu_v2"

        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained, None, model_name, mu_version=mu_version
        )
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.set_tokenizer(self.tokenizer)

        self.model.eval()

    def _run_inference(self, image_file: str, question: str):
        conv_template = "mu_v1"
        start_time = time.perf_counter()
        image = Image.open(image_file)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=self.dtype, device=self.device) for _image in image_tensor]

        image_question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.messages = []
        conv.append_message(conv.roles[0], image_question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        image_sizes = [image.size]

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

        end_time = time.perf_counter()

        logger.info(f"Image size: {image.size}")
        logger.info(f"Question: {question}")
        logger.info(f"Response: {text_outputs}")

        inference_time = end_time - start_time
        logger.info(f"Inference time (sec): {inference_time:.2f}")

    def __call__(self):
        self.setup()

        prompts = [
            dict(
                image_file="artifacts/images/kids_playing.jpg",
                question="What sport are they playing?",
            ),
            dict(
                image_file="artifacts/images/dog.jpg",
                question="What color is the dog?",
            ),
            dict(
                image_file="artifacts/images/desert_pic.jpg",
                question="Where is this place?",
            ),
        ]

        for prompt in prompts:
            self._run_inference(prompt["image_file"], prompt["question"])


def main():
    logging.basicConfig(level=logging.INFO)

    parser = HfArgumentParser(GenerateTextImage2Text)
    generate_text_image2text = parser.parse_args_into_dataclasses()[0]
    generate_text_image2text()


if __name__ == "__main__":
    main()
