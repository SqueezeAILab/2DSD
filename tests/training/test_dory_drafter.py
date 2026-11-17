"""Dory drafter: native teacher-forced features, training step and export (CPU, tiny random Dory)."""
import json
import math
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.attention.flex_attention import create_mask

from training.drafter import export
from training.drafter.blocks import Blocks, context_mask_mod
from training.drafter.data import build_packed
from training.drafter.model import DepthDFlash, drafter_config, drafter_forward
from training.drafter.targets import DoryTarget, native_target, target_feature_layers
from training.hf_models.dory.native import FEATURE_LAYERS, NativeDory, feature_layer_ids, unrolled_layers

H = 16


def tiny_config():
    """Dory1.0-4B's 16/40/16 sublayer layout (MLPs alternating with sliding and full attention) at width 16."""
    return dict(model_type="dory", vocab_size=96, hidden_size=H, intermediate_size=24, num_hidden_layers=72,
                n_input_layers=16, n_recurrent_layers=40, n_output_layers=16, n_recurrent_loops=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=4,
                initializer_range=0.0177, ngpt_alpha_init=0.05, attention_output_gate_logit_scale_init=None,
                hidden_act="silu", attention_bias=False, mlp_bias=False, tie_word_embeddings=False,
                max_position_embeddings=128,
                layer_types=["mlp" if i % 2 else "full_attention" if i % 8 == 6 else "sliding_attention"
                             for i in range(72)],
                rope_parameters={kind: dict(rope_type="default", rope_theta=theta, partial_rotary_factor=1.0)
                                 for kind, theta in (("sliding_attention", 1e4), ("full_attention", 1e7))})


def tiny_model():
    torch.manual_seed(1729)
    model = NativeDory(tiny_config())
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("weight"):
                param.normal_(std=0.12)
            elif name.endswith(("attn_alpha", "mlp_alpha", "sqk", "s_gate")):
                param.fill_(0.0177)
            elif name.endswith(("suv_gate", "suv_up")):
                param.fill_(1 / 4)
            elif name.endswith("logit_scale"):
                param.copy_(torch.linspace(0.01, 0.04, param.numel()))
    return model.eval()


def write_checkpoint(directory, model):
    """config.json, generation_config.json (Dory keeps bos/eos only there), tokenizer.json and weights."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(vars(model.config)))
    (directory / "generation_config.json").write_text(json.dumps({"bos_token_id": 1, "eos_token_id": [2, 11],
                                                                  "pad_token_id": 11}))
    added = [(0, "<unk>"), (1, "<s>"), (2, "</s>"), (10, "<|im_start|>"), (11, "<|im_end|>"), (12, "<think>"),
             (13, "</think>"), (89, "<SPECIAL_89>"), (90, "<SPECIAL_90>")]
    vocab = {f"t{i}": i for i in range(96) if i not in dict(added)}
    (directory / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": vocab}, "added_tokens": [{"id": i, "content": c} for i, c in added]}))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, directory / "model.safetensors")


class Reference:
    """Independent FP64 Dory from the checkpoint tensors (test oracle): explicit masked softmax, and
    every depth d evaluated as the d-loop model."""

    def __init__(self, model):
        self.c = model.config
        self.w = {k: v.double() for k, v in model.state_dict().items()}

    @staticmethod
    def unit(x):
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    def rope(self, x, kind):
        T, d = x.shape[0], x.shape[-1] // 2
        theta = self.c.rope_parameters[kind]["rope_theta"]
        angle = torch.arange(T, dtype=torch.float64)[:, None] * theta ** (-torch.arange(0, 2 * d, 2) / (2 * d))
        cos, sin = angle.cos()[:, None], angle.sin()[:, None]
        return torch.cat((x[..., :d] * cos - x[..., d:] * sin, x[..., d:] * cos + x[..., :d] * sin), -1)

    def attention(self, p, kind, h):
        c, w, T = self.c, self.w, h.shape[0]
        heads, kv, dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
        q = self.rope((h @ w[f"{p}.q_proj.weight"].T).view(T, heads, dim), kind)
        k = self.rope((h @ w[f"{p}.k_proj.weight"].T).view(T, kv, dim), kind)
        v = (h @ w[f"{p}.v_proj.weight"].T).view(T, kv, dim)
        q = self.unit(q) * (w[f"{p}.sqk"].view(heads, dim) * math.sqrt(dim) / c.initializer_range) ** 2
        k, v = self.unit(k).repeat_interleave(heads // kv, 1), v.repeat_interleave(heads // kv, 1)
        t = torch.arange(T)
        visible = t[None] <= t[:, None]
        if kind == "sliding_attention":
            visible &= t[None] > t[:, None] - c.sliding_window
        scores = torch.einsum("thd,jhd->htj", q, k) / math.sqrt(dim)
        out = torch.einsum("htj,jhd->thd", scores.masked_fill(~visible, -math.inf).softmax(-1), v).reshape(T, -1)
        base = c.attention_output_gate_logit_scale_init or math.sqrt(c.hidden_size)
        gate = (w[f"{p}.s_gate"] * math.sqrt(base) / c.initializer_range) ** 2
        out = out * torch.sigmoid((h @ w[f"{p}.g_proj.weight"].T) * gate)
        return self.unit(out) @ w[f"{p}.o_proj.weight"].T

    def mlp(self, p, h):
        w, factor = self.w, math.sqrt(self.c.hidden_size)
        g = (h @ w[f"{p}.gate_proj.weight"].T) * w[f"{p}.suv_gate"] * factor
        u = (h @ w[f"{p}.up_proj.weight"].T) * w[f"{p}.suv_up"] * factor
        return self.unit(F.silu(g) * u) @ w[f"{p}.down_proj.weight"].T

    def layer(self, i, h):
        kind, p = self.c.layer_types[i], f"backbone.layers.{i}"
        out = self.mlp(f"{p}.mixer", h) if kind == "mlp" else self.attention(f"{p}.mixer", kind, h)
        alpha = self.w[f"{p}.mlp_alpha" if kind == "mlp" else f"{p}.attn_alpha"]
        a, b = self.unit(h), self.unit(out)
        return self.unit(a + (alpha * self.c.ngpt_alpha_init / self.c.initializer_range).abs() * (b - a))

    def run(self, tokens, depth):
        """(taps [T, 5H], output-block state [T, H], logits [T, V]) of the depth-loop model."""
        c = self.c
        n_in, end = c.n_input_layers, c.n_input_layers + c.n_recurrent_layers
        taps, h = {}, self.w["backbone.embeddings.weight"][tokens]
        for i in range(n_in):
            taps[i] = h = self.layer(i, h)
        x = state = h
        for _ in range(depth):
            h = self.unit(state + x)
            for i in range(n_in, end):
                taps[i] = h = self.layer(i, h)
            state = self.unit(h)
        h = state
        for i in range(end, c.num_hidden_layers):
            taps[i] = h = self.layer(i, h)
        logits = (h @ self.w["lm_head.weight"].T) * self.w["lm_head.logit_scale"] / c.initializer_range
        return torch.cat([taps[i] for i in FEATURE_LAYERS], -1), h, logits


TOKENS = torch.tensor([1, 7, 9, 11, 14, 3, 8, 21, 5, 2])


class NativeFeatures(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_every_depth_is_the_d_loop_model_of_an_independent_reference(self):
        model = tiny_model()
        target = DoryTarget.from_model(model)
        self.assertEqual((target.R, target.H, target.attention), (4, H, "sdpa"))
        features, states = target.features(TOKENS)
        self.assertEqual(features.shape, (4, len(TOKENS), 5 * H))
        self.assertEqual(states.shape, (4, len(TOKENS), H))
        reference = Reference(model)
        for d in range(1, 5):
            taps, state, logits = reference.run(TOKENS, d)
            torch.testing.assert_close(features[d - 1].double(), taps, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(states[d - 1].double(), state, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(target.readout_logits(states[d - 1]).double(), logits, rtol=1e-4, atol=1e-5)
            self.assertTrue(torch.equal(target.readout_argmax(states[d - 1], chunk=3), logits.argmax(-1)))
        self.assertFalse(any(p.requires_grad for p in target.model.parameters()))
        self.assertFalse(features.requires_grad or states.requires_grad)

    def test_bf16_checkpoint_stays_at_the_rounding_floor(self):
        with tempfile.TemporaryDirectory() as root:
            write_checkpoint(Path(root), tiny_model().to(torch.bfloat16))
            loaded = NativeDory.from_pretrained(root, "cpu")
            reference = Reference(loaded)
            target = DoryTarget(root, FEATURE_LAYERS, "cpu")
        self.assertTrue(all(m.inv_freq.dtype == torch.float32 for m in loaded.modules() if hasattr(m, "inv_freq")))
        _, states = target.features(TOKENS)
        for d in range(1, 5):
            expected = reference.run(TOKENS, d)[2]
            logits = target.readout_logits(states[d - 1]).double()
            # BF16 rounding floor on logits of magnitude ~0.14 (measured: mean 0.002, max 0.013).
            self.assertLess((logits - expected).abs().mean().item(), 0.005)
            self.assertLess((logits - expected).abs().max().item(), 0.03)

    def test_features_are_causal(self):
        target = DoryTarget.from_model(tiny_model())
        changed = TOKENS.clone()
        changed[6:] = torch.tensor([55, 57, 59, 61])
        (fa, sa), (fb, sb) = target.features(TOKENS), target.features(changed)
        torch.testing.assert_close(fa[:, :6], fb[:, :6], rtol=0, atol=0)
        torch.testing.assert_close(sa[:, :6], sb[:, :6], rtol=0, atol=0)

    def test_streamed_loader_restores_every_weight(self):
        model = tiny_model().to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as root:
            write_checkpoint(Path(root), model)
            loaded = NativeDory.from_pretrained(root, "cpu")
            (Path(root) / "model.safetensors").unlink()
            save_file({"lm_head.weight": model.lm_head.weight.contiguous()}, Path(root) / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "Missing Dory weights"):
                NativeDory.from_pretrained(root, "cpu")
        for key, weight in model.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], weight, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))

    def test_scaled_head_keeps_the_fp32_scale_and_trains_only_its_input(self):
        target = DoryTarget.from_model(tiny_model())
        h = torch.randn(5, H, requires_grad=True)
        logits = target.lm_head(h)
        torch.testing.assert_close(logits, (h @ target.lm_head.weight.T) * target.lm_head.logit_scale / 0.0177)
        F.cross_entropy(logits, torch.tensor([1, 2, 3, 4, 5])).backward()
        self.assertTrue(h.grad.isfinite().all() and h.grad.abs().sum() > 0)
        self.assertIsNone(target.lm_head.weight.grad)

    def test_feature_taps_layout_and_factory(self):
        config = tiny_config()
        self.assertEqual(feature_layer_ids(config, 1), list(FEATURE_LAYERS))
        self.assertEqual(feature_layer_ids(config, 4), [3, 139, 155, 171, 187])
        self.assertEqual(unrolled_layers(config, 4), 192)
        self.assertEqual(target_feature_layers(config), [3, 19, 35, 51, 67])
        self.assertEqual(target_feature_layers({"model_type": "ouro", "num_hidden_layers": 24}), [1, 6, 11, 16, 21])
        with self.assertRaisesRegex(ValueError, "feature taps"):
            DoryTarget.from_model(tiny_model(), [1, 6, 11, 16, 21])
        with self.assertRaisesRegex(ValueError, "block layout"):
            feature_layer_ids({**config, "n_input_layers": 8, "n_recurrent_layers": 48})
        with patch("training.drafter.targets.DoryTarget") as cls:
            self.assertIs(native_target("dory1.0-4b", list(FEATURE_LAYERS), "cpu", path="/local/dory"),
                          cls.return_value)
            cls.assert_called_once_with("/local/dory", list(FEATURE_LAYERS), "cpu", torch.bfloat16, "flash")


class DrafterOnDory(unittest.TestCase):
    def test_drafter_config_rope_and_output_norm(self):
        cfg = drafter_config(5, tiny_model().config)
        self.assertEqual((cfg.hidden_size, cfg.head_dim, cfg.rms_norm_eps), (H, 8, 1e-6))
        self.assertEqual(cfg.rope_parameters["rope_theta"], 1e7)
        cfg._attn_implementation = "sdpa"
        torch.testing.assert_close(DepthDFlash(cfg, 5).norm.weight, torch.full((H,), H ** -0.5))

    def test_training_step_reaches_the_drafter_only(self):
        target = DoryTarget.from_model(tiny_model())
        ids = [1, 7]
        while len(ids) < 10:  # a greedy trace of the target, so that every drafted slot has a label
            ids.append(int(target.readout_argmax(target.features(torch.tensor(ids))[1][-1])[-1]))
        docs = [{"id": "d", "domain": "math", "source": "fixture", "resp_start": 2, "ids": ids}]
        p = build_packed(target, docs, [0], len(ids), "cpu")
        p.feats, p.loop_out = p.feats.float(), p.loop_out.float()
        cfg = drafter_config(2, target.model.config)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(0)
        drafter = DepthDFlash(cfg, 5)
        with torch.no_grad():
            drafter.mask_emb.copy_(target.embed.weight[90])

        def dense_mask(p, b, R, k):
            mask_mod, q_len, kv_len = context_mask_mod(p, b, R, k)
            return create_mask(mask_mod, None, None, q_len, kv_len, "cpu"), mask_mod

        # One block per depth; the last one is a verified (d = R) block.
        b = Blocks(torch.tensor([3] * 4), torch.tensor([1, 2, 3, 4]), torch.tensor([1, 1, 1, 0]),
                   torch.tensor([4] * 4), torch.tensor([0] * 4), torch.tensor([True] * 4))
        before = drafter.fc.weight.detach().clone()
        with patch("training.drafter.model.build_context_mask", dense_mask), \
                patch.object(torch, "autocast", lambda *a, **k: nullcontext()):
            loss, metrics = drafter_forward(drafter, target, p, b, 4, 4)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(metrics["n_valid"]), 12)
        loss.backward()
        self.assertTrue(drafter.fc.weight.grad.isfinite().all())
        self.assertGreater(drafter.mask_emb.grad.abs().sum().item(), 0)
        self.assertGreater(drafter.norm.weight.grad.abs().sum().item(), 0)
        torch.optim.AdamW(drafter.parameters(), lr=6e-4).step()
        self.assertFalse(torch.equal(before, drafter.fc.weight))
        self.assertTrue(all(p.grad is None for p in target.model.parameters()))


class DoryExport(unittest.TestCase):
    def checkpoint(self, root, target_dir, **overrides):
        cfg = drafter_config(1, tiny_model().config)
        ck = {"model": {"fc.weight": torch.randn(H, 5 * H), "mask_emb": torch.randn(H),
                        "rotary.inv_freq": torch.ones(4)},
              "cfg": cfg.to_dict(), "feat_layers": list(FEATURE_LAYERS), "d_weights": {1: .5, 2: .125, 3: .125, 4: .25},
              "learn_mask": True, "mask_token": 90, "input_format": "loop-context", "target": "dory1.0-4b",
              "target_variant": "original", "target_spec": str(target_dir), "target_loops": 4, "step": 3,
              **overrides}
        path = root / f"ck-{len(list(root.glob('ck-*')))}.pt"
        torch.save(ck, path)
        return path

    def test_unrolled_layout_head_scale_and_mask(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model = tiny_model().to(torch.bfloat16)
            write_checkpoint(root / "dory", model)
            out = root / "export"
            export.main(["--checkpoint", str(self.checkpoint(root, root / "dory")), "--out", str(out),
                         "--target-name", "org/dory@rev"])
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["num_target_layers"], 192)  # 16 + 4 x 40 + 16, not 4 x 72
            self.assertEqual(config["dflash_config"], {"mask_token_id": 90, "target_layer_ids": [3, 139, 155, 171, 187]})
            self.assertEqual((config["rope_theta"], config["bos_token_id"], config["eos_token_id"]), (1e7, 1, [2, 11]))
            drafter = config["loopedtf_drafter"]
            self.assertEqual((drafter["target_loops"], drafter["input_format"], drafter["target_checkpoint"]),
                             (4, "loop-context", "org/dory@rev"))
            self.assertEqual(drafter["feature_layers_per_loop"], list(FEATURE_LAYERS))
            self.assertEqual(drafter["mask_token_rule"], "last-reserved-added-token")
            scale = torch.load(out / "target_head_scale.pt", weights_only=True)
            torch.testing.assert_close(scale["logit_scale"], model.lm_head.logit_scale.detach(), rtol=0, atol=0)
            self.assertEqual(scale["initializer_range"], 0.0177)
            self.assertEqual(torch.load(out / "mask_embedding.pt", weights_only=True)["mask_token_id"], 90)
            with safe_open(out / "model.safetensors", "pt") as handle:
                self.assertEqual(sorted(handle.keys()), ["fc.weight", "lm_head.weight"])
                torch.testing.assert_close(handle.get_tensor("lm_head.weight"), model.lm_head.weight.detach())
            with self.assertRaises(SystemExit):  # the mask row it trained from is not the exported id
                export.main(["--checkpoint", str(self.checkpoint(root, root / "dory", mask_token=16)),
                             "--out", str(root / "bad"), "--target-name", "org/dory@rev"])
            with self.assertRaisesRegex(ValueError, "physical feature taps"):
                export.main(["--checkpoint", str(self.checkpoint(root, root / "dory", feat_layers=[1, 18, 35, 52, 69])),
                             "--out", str(root / "bad"), "--target-name", "org/dory@rev"])


if __name__ == "__main__":
    unittest.main()
