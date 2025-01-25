from pathlib import Path

from transformers import AutoTokenizer


def _infer_which_tokenizer(tokenizer):
    canary = "hello world"
    tokenized = tokenizer(canary)
    if tokenized.input_ids == [22172, 3186]:
        return "phi35"
    elif tokenized.input_ids == [128000, 15339, 1917]:
        return "llama31"
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer}")


def _update_llama31_tokenizer_(tokenizer):
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    return tokenizer


def _update_phi35_tokenizer_(tokenizer, model_max_length: int):
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = model_max_length
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token_id = 1
    tokenizer.bos_token = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)

    return tokenizer


def get_tokenizer(tokenizer_path: Path, model_max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)
    which_tokenizer = _infer_which_tokenizer(tokenizer)
    # logger.info(f"Using tokenizer {which_tokenizer}", main_process_only=True)

    if which_tokenizer == "llama31":
        tokenizer = _update_llama31_tokenizer_(tokenizer)

    if which_tokenizer == "phi35":
        tokenizer = _update_phi35_tokenizer_(tokenizer, model_max_length)

    return tokenizer


def save_tokenizer(tokenizer, output_path: Path):
    tokenizer.save_pretrained(output_path)
