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
# Inkling2's chat template has no enable_thinking switch and no generation prompt: it prints a thinking
# effort in a system block before the first non-system message, and the model opens its own turn.
# Level names and default as in the checkpoint's chat_template.jinja.
INKLING2_EFFORTS = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
INKLING2_DEFAULT_EFFORT = 0.9
INKLING2_ROLES = ("system", "user", "assistant")
# What the template writes after the text of an assistant message.
INKLING2_TURN_END = "<|end_message|><|content_model_end_sampling|>"


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
    # Inkling2 (model_type inkling2_loop) needs no config class: transformers only warns about the type.
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


def inkling2_effort(value=None):
    """Inkling2 thinking effort as the number its chat template prints; None is the template default.

    Takes a level name or a number in [0, 0.99] with at most two decimals. The official renderer rounds
    the effort to two decimals and the template prints the number as given, so a finer value would
    render differently in the two.
    """
    if value is None:
        return INKLING2_DEFAULT_EFFORT
    if isinstance(value, str):
        if value in INKLING2_EFFORTS:
            return INKLING2_EFFORTS[value]
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"unknown reasoning effort {value!r}; use one of {', '.join(INKLING2_EFFORTS)} "
                             "or a number in [0, 0.99]") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"reasoning effort must be a level name or a number, not {value!r}")
    if not 0.0 <= value <= 0.99:  # before float(): an integer beyond the float range is out of range too
        raise ValueError(f"reasoning effort must be in [0, 0.99], not {value}")
    value = float(value)
    if round(value, 2) != value:
        raise ValueError(f"reasoning effort takes at most two decimals (the official renderer rounds it), not {value}")
    return value or 0.0  # never -0.0


def _placeholders(texts):
    """One marker per message text; the markers occur in none of the texts."""
    stem = "[[2dsd-text"
    while any(stem in text for text in texts):
        stem += "-"
    return [f"{stem}-{index}]]" for index in range(len(texts))]


def render_inkling2(tokenizer, messages, effort=None, continue_final=False):
    """(prompt text, token IDs) of system, user and assistant text messages in Inkling2's chat format.

    The checkpoint's chat template writes the framing, with the thinking effort and without a
    generation prefix. Each message text is then encoded on its own with special-token parsing off, as
    the official renderer encodes it (vLLM PR #69028 at b426b46e: vllm/renderers/inkling.py,
    `split_special_tokens=self._is_inkling2`), so a literal "<|end_message|>" in a message stays text.
    With `continue_final` the last message, an assistant text, is left open for the model to continue
    (the renderer's continue_final_message).
    """
    for message in messages:
        if (set(message) != {"role", "content"} or message["role"] not in INKLING2_ROLES
                or not isinstance(message["content"], str)):
            raise ValueError("Inkling2 prompts hold system, user and assistant text messages only")
    if continue_final and (not messages or messages[-1]["role"] != "assistant"):
        raise ValueError("Only a final assistant message can be left open")
    texts = [message["content"] for message in messages]
    marks = _placeholders(texts)
    rest = tokenizer.apply_chat_template(
        [{"role": message["role"], "content": mark} for message, mark in zip(messages, marks)],
        tokenize=False, reasoning_effort=inkling2_effort(effort))
    prompt, ids = "", []
    for mark, text in zip(marks, texts):
        framing, found, rest = rest.partition(mark)
        if not found or mark in rest:
            raise ValueError("The Inkling2 chat template must render each message text exactly once")
        prompt += framing + text
        ids += tokenizer.encode(framing, add_special_tokens=False)
        ids += tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
    if continue_final:
        if rest != INKLING2_TURN_END:
            raise ValueError("The Inkling2 chat template closes an assistant text in an unexpected way")
        return prompt, ids
    return prompt + rest, ids + list(tokenizer.encode(rest, add_special_tokens=False))


def render_native(tokenizer, family, user, prefix="", effort=None):
    """Prompt token IDs with the family's system turn and native template, thinking on; a non-empty
    prefix is kept as the start of the assistant turn. Inkling2 has no thinking switch: it takes a
    thinking `effort` (inkling2_effort) and, with a prefix, answers without a thinking block."""
    messages = [*system_messages(family), {"role": "user", "content": user}]
    if family == "inkling2":
        if prefix:
            messages.append({"role": "assistant", "content": prefix})
        return render_inkling2(tokenizer, messages, effort, continue_final=bool(prefix))[1]
    if effort is not None:
        raise ValueError("A thinking effort applies to Inkling2 only")
    if prefix:
        text = tokenizer.apply_chat_template([*messages, {"role": "assistant", "content": prefix + SPLITTER}],
                                             tokenize=False, enable_thinking=True).split(SPLITTER)[0]
    else:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True)
    return list(tokenizer.encode(text, add_special_tokens=False))
