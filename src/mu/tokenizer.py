import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def get_tokenizer():
    tokenizer_name = "microsoft/Phi-3.5-mini-instruct"
    logger.info(f"Using tokenizer {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)

    tokenizer.padding_side = "right"

    tokenizer.bos_token_id = 1
    tokenizer.bos_token = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)

    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    return tokenizer
