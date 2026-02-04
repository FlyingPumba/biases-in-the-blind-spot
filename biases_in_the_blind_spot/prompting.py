import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase, prompt: str, response: str | None = None
) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=response is None, return_tensors="pt"
    )  # type: ignore
