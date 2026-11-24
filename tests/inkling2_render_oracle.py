# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""Test oracle: the official Inkling2 chat rendering of text conversations, without Jinja.

An independent restatement of the reference renderer of vLLM PR #69028 (head b426b46e), limited to
system, user and assistant messages whose content is one string (no tools, images, names or
reasoning). Sources:
  vllm/renderers/inkling_encoding.py  render_inkling2_messages, _append_message, _inkling2_blocks,
                                      _iter_inkling2_parts, continuation_prefix, _inkling2_effort_text
  vllm/renderers/inkling.py           _NAMED_REASONING_EFFORT, _resolve_inkling2_reasoning_effort,
                                      _HfBackedTmlTokenizer (`split_special_tokens=self._is_inkling2`)

The rule:
  - A message is one block: role token, <|content_text|>, the text, <|end_message|>. An empty text
    keeps its block.
  - Text is encoded with special-token parsing off, so it never yields a framing token.
  - The system block "Thinking effort level: <effort>" precedes the first non-system message, or ends
    a conversation that has none. The effort is rounded to two decimals and printed with %g.
  - <|content_model_end_sampling|> follows an assistant message.
  - There is no generation prefix: the model writes its own <|message_model|>.
  - With continue_final_message, a final assistant message is not rendered as a block: the ids end
    with <|message_model|>, <|content_text|> and its text, open.
"""
ROLE_TOKENS = {"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>"}
CONTENT_TEXT = "<|content_text|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"

NAMED_EFFORT = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99}
DEFAULT_EFFORT = 0.9
MAX_EFFORT = 0.99


def resolve_effort(value):
    """The effort number of a request value: None (the default), a level name, a number or a numeric string."""
    if value is None:
        return DEFAULT_EFFORT
    if isinstance(value, str):
        if value in NAMED_EFFORT:
            return NAMED_EFFORT[value]
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"invalid Inkling reasoning_effort: {value!r}") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid Inkling reasoning_effort: {value!r}")
    return float(value)


def effort_text(effort):
    """Text of the effort block: the effort in integer milli-effort, rounded to two decimals."""
    if not 0.0 <= effort <= MAX_EFFORT:
        raise ValueError(f"Inkling reasoning_effort must be in [0.0, 0.99], got {effort!r}")
    milli = round(effort * 1000)
    return f"Thinking effort level: {round(milli / 1000 * 100) / 100:g}"


def render(tokenizer, messages, reasoning_effort=None, continue_final_message=False):
    """Token ids of a text conversation under the official rule.

    `tokenizer` needs convert_tokens_to_ids(token) and
    encode(text, add_special_tokens=False, split_special_tokens=True).
    """
    def special(token):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"the tokenizer does not define {token}")
        return token_id

    def text_ids(text):
        return list(tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True))

    def block(role, text):
        return [special(ROLE_TOKENS[role]), special(CONTENT_TEXT), *text_ids(text), special(END_MESSAGE)]

    messages = list(messages)
    for message in messages:
        if set(message) != {"role", "content"} or message["role"] not in ROLE_TOKENS:
            raise ValueError("the oracle covers system, user and assistant messages with role and content only")
        if not isinstance(message["content"], str):
            raise ValueError("the oracle covers string contents only")
    prefix = None
    if continue_final_message and messages and messages[-1]["role"] == "assistant":
        prefix = messages.pop()["content"]

    ids = []
    effort = effort_text(resolve_effort(reasoning_effort))
    for message in messages:
        if effort is not None and message["role"] != "system":
            ids += block("system", effort)
            effort = None
        ids += block(message["role"], message["content"])
        if message["role"] == "assistant":
            ids.append(special(END_SAMPLING))
    if effort is not None:
        ids += block("system", effort)
    if prefix is not None:
        ids += [special(ROLE_TOKENS["assistant"]), special(CONTENT_TEXT), *text_ids(prefix)]
    return ids
