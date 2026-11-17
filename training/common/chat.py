"""Tokenizers and chat rendering shared by the prompt pool and trace generation."""
from __future__ import annotations

import json
from pathlib import Path

# Explicit system turn per family; other families keep their template default.
SYSTEM_PROMPTS = {
    "nanbeige": "You are Nanbeige, a professional large language model independently developed and trained by BOSS Zhipin.",
}
# Renders an assistant prefix without the template closing the turn (EvalPlus-style prompts).
SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def load_tokenizer(path):
    from transformers import AutoConfig, AutoTokenizer

    config = Path(path) / "config.json"
    if config.is_file() and json.loads(config.read_text()).get("model_type") == "dory":
        # transformers has no Dory; AutoTokenizer reads config.json through the vLLM fork's config class.
        from vllm.transformers_utils.configs.dory import DoryConfig

        try:
            AutoConfig.register("dory", DoryConfig)
        except ValueError:
            pass  # already registered in this process
    return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)


def system_messages(family):
    text = SYSTEM_PROMPTS.get(family)
    return [{"role": "system", "content": text}] if text else []


def eos_ids(tokenizer, checkpoint):
    """Every EOS id of the checkpoint's config/generation_config and tokenizer (never </think>)."""
    result = set()
    for filename in ("config.json", "generation_config.json"):
        path = Path(checkpoint) / filename
        if path.exists():
            value = json.loads(path.read_text()).get("eos_token_id")
            result.update(value if isinstance(value, list) else [value])
    result.add(tokenizer.eos_token_id)
    result = sorted(value for value in result if type(value) is int and value >= 0)
    close_think = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close_think) == 1 and close_think[0] in result:
        raise ValueError("The thinking-close token must never be configured as EOS")
    if not result:
        raise ValueError("The model's EOS token is missing")
    return result


def render_user_prompt(tokenizer, text):
    """Token IDs of one user turn with the generation prompt, thinking on (prompt pool rule)."""
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                        tokenize=True, enable_thinking=True)
    return ids if isinstance(ids, list) else ids["input_ids"]


def ouro_turns(ouro_tokenizer, prompt_ids):
    """(user turn, assistant prefix) of an Ouro-rendered prompt: one user turn and optionally an
    assistant prefix (EvalPlus code prompts)."""
    text = ouro_tokenizer.decode(prompt_ids)
    head, _, prefix = text.partition("<|im_end|>\n<|im_start|>assistant\n")
    return head.split("<|im_start|>user\n", 1)[1], prefix


def render_native(tokenizer, family, user, prefix=""):
    """Prompt token IDs with the family's system turn and native template, thinking on; a non-empty
    prefix is kept as the start of the assistant turn."""
    messages = [*system_messages(family), {"role": "user", "content": user}]
    if prefix:
        text = tokenizer.apply_chat_template([*messages, {"role": "assistant", "content": prefix + SPLITTER}],
                                             tokenize=False, enable_thinking=True).split(SPLITTER)[0]
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
    return list(tokenizer.encode(text, add_special_tokens=False))
