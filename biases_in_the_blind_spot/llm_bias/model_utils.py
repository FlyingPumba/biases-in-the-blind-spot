from transformers import AutoTokenizer


def add_chat_template(
    prompts: list[str], model_name: str, task_prompt: str | None = None
) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    formatted_prompts = []
    for prompt in prompts:
        messages = []

        if "gemma-2" not in model_name and task_prompt is not None:
            # gemma-2 doesn't support system prompts
            messages.append({"role": "system", "content": task_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted_prompt)
    return formatted_prompts
