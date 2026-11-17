"""Checkpoint metadata of a shared-KV model.

``loopedtf_kv_policy`` / ``loopedtf_kv_window`` in config.json make the released vLLM fork apply the
sharing rule without hf_overrides. ``loopedtf_kd_policy`` records how the checkpoint was trained.
"""
from __future__ import annotations

POLICY_KEY, WINDOW_KEY = "loopedtf_kv_policy", "loopedtf_kv_window"
SHARED, NATIVE = "shared_last", "full"


def config_keys(window):
    """config.json keys that enable the shared rule with window W in the vLLM fork."""
    if type(window) is not int or window < 1:
        raise ValueError("the shared window W includes the query itself and must be an integer >= 1")
    return {POLICY_KEY: SHARED, WINDOW_KEY: window}


def kd_policy(window):
    """The training rule recorded as config.loopedtf_kd_policy."""
    return {"prompt": "native-one-pass", "generated_history": "last-loop-shared-outside-W",
            "train_window": window, "window_includes_self": True,
            "first_reader_native": True, "last_loop_current_causal": True,
            "student_pass1": "native-no-grad-recomputed-every-microbatch",
            "student_pass2": "shared-with-native-prompt-grad",
            "only_approximation": "old-response-final-KV-from-student-P1"}


def checkpoint_policy(config):
    """{policy[, window]} that a checkpoint config (dict) declares, or None if it declares none."""
    if POLICY_KEY not in config:
        return None
    if config[POLICY_KEY] == SHARED:
        return {POLICY_KEY: SHARED, WINDOW_KEY: config.get(WINDOW_KEY)}
    return {POLICY_KEY: config[POLICY_KEY]}
