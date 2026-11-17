"""FlashAttention version of the exact-replay flash path (CPU; the fork's support check and kernel are mocked)."""
import sys
import types
import unittest
from unittest.mock import patch

import torch

from training.hf_models import model_classes
from training.sharekv import exact_documents, flash


def fake_flash_attn(fa3_supported, calls):
    """vllm.vllm_flash_attn whose varlen kernel records fa_version and returns zeros of the right shapes."""
    module = types.ModuleType("vllm.vllm_flash_attn")
    module.is_fa_version_supported = lambda version: version == 2 or fa3_supported

    def varlen(q, k, v, **kwargs):
        calls.append(kwargs["fa_version"])
        return torch.zeros_like(q), q.new_zeros(q.shape[1], q.shape[0])

    module.flash_attn_varlen_func = varlen
    return {"vllm.vllm_flash_attn": module}


def flash_version(capability, fa3_supported):
    with (patch.dict(sys.modules, fake_flash_attn(fa3_supported, [])),
          patch.object(torch.cuda, "get_device_capability", return_value=capability)):
        return flash.flash_version(torch.device("cuda", 0))


class FlashVersion(unittest.TestCase):
    def test_fa3_on_hopper_and_fa2_on_ampere_and_ada(self):
        self.assertEqual(flash_version((9, 0), fa3_supported=True), 3)
        self.assertEqual(flash_version((9, 0), fa3_supported=False), 2)
        self.assertEqual(flash_version((8, 0), fa3_supported=True), 2)  # A100
        self.assertEqual(flash_version((8, 9), fa3_supported=True), 2)  # L40S

    def test_replay_passes_the_device_version_to_every_call(self):
        config_cls, model_cls = model_classes("ouro")
        torch.manual_seed(7)
        config = config_cls(vocab_size=96, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=2, head_dim=16, total_ut_steps=4,
                            layer_types=["full_attention"] * 2, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        model = model_cls(config).eval()
        documents = [{"input_ids": [1, 5, 9, 7, 8, 3, 6, 2], "prompt_length": 3},
                     {"input_ids": [1, 3, 4, 5, 2], "prompt_length": 2}]
        for capability, expected in (((8, 9), 2), ((9, 0), 3)):
            calls = []
            with (patch.dict(sys.modules, fake_flash_attn(True, calls)),
                  patch.object(torch.cuda, "get_device_capability", return_value=capability)):
                exact_documents(model, documents, 3, attention="flash", cuda_graph=False)
            self.assertTrue(calls)
            self.assertEqual(set(calls), {expected})


if __name__ == "__main__":
    unittest.main()
