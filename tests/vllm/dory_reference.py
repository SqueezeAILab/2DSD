"""Independent FP32 reference of Dory for the tree-runner tests (test oracle only).

Plain PyTorch with exact masked softmax: no vLLM code and no fused kernels. The
math follows ``vllm/model_executor/models/dory.py``; ``bf`` marks where the vLLM
module rounds to BF16 (projections, RoPE, nGPT norms, attention output), and a
test that replaces it by the identity gets a pure FP32 model, so a comparison
isolates the KV read rule from rounding.

KV read rules (loop or readout depth r of R, query t, key j <= t):

- native: same-plane KV; sliding layers read j > t - sliding_window.
- shared (``window`` W): for response queries (t >= prompt length) of loops
  r < R in a shared recurrent layer, keys j <= t - W come from loop R and keys
  t - W < j <= t from loop r. Prompt queries are native. Replaying the response
  in chunks of at most W tokens is exact, because every final-loop key a chunk
  reads lies in an earlier chunk.
- shared output block (``output_window`` W_out): depth-r (< R) readouts of
  response queries read the depth-R output-block KV for j <= t - W_out.

The input block always reads one native cache. Every readout depth r runs the
output block on the loop-r state and then the scaled head; with per-depth
output-block KV (``output_window`` None) each depth-r readout is the r-loop
model.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-6


def bf(x):
    return x.to(torch.bfloat16).float()


def justnorm(x):
    """FP32 accumulation of an input already rounded to BF16, BF16 output (as vLLM)."""
    return bf(x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + EPS))


def justnorm_fp32(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + EPS)


class DoryReference:
    def __init__(self, model_dir, loops=4, threads=None):
        if threads:
            torch.set_num_threads(threads)
        c = self.config = json.loads((Path(model_dir) / "config.json").read_text())
        self.loops = loops
        self.layer_types = c["layer_types"]
        self.n_in, self.n_rec = c["n_input_layers"], c["n_recurrent_layers"]
        self.heads, self.kv_heads = c["num_attention_heads"], c["num_key_value_heads"]
        self.head_dim, self.window = c["head_dim"], c["sliding_window"]
        self.std = c["initializer_range"]
        self.alpha_scale = c["ngpt_alpha_init"] / self.std
        self.sqk_factor = self.head_dim**0.5 / self.std
        g0 = c["attention_output_gate_logit_scale_init"] or c["hidden_size"] ** 0.5
        self.gate_factor = g0**0.5 / self.std
        self.suv_factor = c["hidden_size"] ** 0.5
        with safe_open(str(Path(model_dir) / "model.safetensors"), "pt") as f:
            self.w = {name: f.get_tensor(name).float() for name in f.keys()}
        expected = self._expected_names()
        if set(self.w) != expected:
            raise ValueError(
                f"weight names differ: missing {sorted(expected - set(self.w))[:4]}, "
                f"unexpected {sorted(set(self.w) - expected)[:4]}"
            )
        self.rope = {}
        for kind, spec in c["rope_parameters"].items():
            exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
            self.rope[kind] = 1.0 / (spec["rope_theta"] ** exponent)

    def _expected_names(self):
        names = {"backbone.embeddings.weight", "lm_head.weight", "lm_head.logit_scale"}
        for i, kind in enumerate(self.layer_types):
            p = f"backbone.layers.{i}"
            if kind == "mlp":
                names |= {
                    f"{p}.mlp_alpha",
                    *(f"{p}.mixer.{n}_proj.weight" for n in ("gate", "up", "down")),
                    f"{p}.mixer.suv_gate",
                    f"{p}.mixer.suv_up",
                }
            else:
                names |= {
                    f"{p}.attn_alpha",
                    *(f"{p}.mixer.{n}_proj.weight" for n in "qkvgo"),
                    f"{p}.mixer.sqk",
                    f"{p}.mixer.s_gate",
                }
        return names

    # --- sublayers -------------------------------------------------------------
    def linear(self, x, name):
        return bf(x @ self.w[name].t())

    def rotary(self, x, positions, kind):
        """NeoX rotation with a BF16 cos/sin cache, FP32 math, BF16 output."""
        freqs = positions.float()[:, None] * self.rope[kind][None, :]
        cos, sin = bf(freqs.cos())[:, None, :], bf(freqs.sin())[:, None, :]
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        return bf(torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1))

    def slerp(self, residual, out, alpha):
        lr = (self.w[alpha] * self.alpha_scale).abs()
        a, b = justnorm(residual), justnorm(out)
        return bf(justnorm_fp32(a + lr * (b - a)))

    def mlp(self, i, h):
        p = f"backbone.layers.{i}.mixer"
        g = self.linear(h, f"{p}.gate_proj.weight")
        u = self.linear(h, f"{p}.up_proj.weight")
        g = bf(g * (self.w[f"{p}.suv_gate"] * self.suv_factor))
        u = bf(u * (self.w[f"{p}.suv_up"] * self.suv_factor))
        m = justnorm(bf(F.silu(g) * u))
        return self.linear(m, f"{p}.down_proj.weight")

    def qkv(self, i, h, positions):
        p = f"backbone.layers.{i}.mixer"
        n = h.shape[0]
        q = self.linear(h, f"{p}.q_proj.weight").view(n, self.heads, self.head_dim)
        k = self.linear(h, f"{p}.k_proj.weight").view(n, self.kv_heads, self.head_dim)
        v = self.linear(h, f"{p}.v_proj.weight").view(n, self.kv_heads, self.head_dim)
        gate = self.linear(h, f"{p}.g_proj.weight")
        kind = self.layer_types[i]
        q, k = self.rotary(q, positions, kind), self.rotary(k, positions, kind)
        sqk = (self.w[f"{p}.sqk"] * self.sqk_factor).view(1, self.heads, self.head_dim)
        q = bf(sqk.square() * justnorm(q))
        k = justnorm(k)
        return q, k, v, gate

    def finish_attention(self, i, attn, gate):
        p = f"backbone.layers.{i}.mixer"
        sg = (self.w[f"{p}.s_gate"] * self.gate_factor).square()
        out = bf(attn.reshape(attn.shape[0], -1) * torch.sigmoid(gate * sg))
        return self.linear(justnorm(out), f"{p}.o_proj.weight")

    def attend(self, q, sources):
        """q [C, H, D]; sources: list of (K [M, Hkv, D], V [M, Hkv, D], mask [C, M])."""
        groups = self.heads // self.kv_heads
        scores, values = [], []
        for k, v, mask in sources:
            k = k.repeat_interleave(groups, dim=1)
            s = torch.einsum("chd,mhd->hcm", q, k) / math.sqrt(self.head_dim)
            scores.append(s.masked_fill(~mask[None], float("-inf")))
            values.append(v.repeat_interleave(groups, dim=1))
        p = torch.softmax(torch.cat(scores, dim=-1), dim=-1)
        return bf(torch.einsum("hcm,mhd->chd", p, torch.cat(values, dim=0)))

    def head(self, rows):
        return bf(rows @ self.w["lm_head.weight"].t()) * (self.w["lm_head.logit_scale"] / self.std)

    # --- replay ----------------------------------------------------------------
    def replay_readouts(
        self,
        tokens,
        prompt_len,
        window=None,
        share_layers="all",
        output_window=None,
        output_share_layers="all",
    ):
        """Teacher-forced logits at every depth 1..R: {depth: [len - prompt_len, vocab]}.

        Row i predicts token prompt_len + i. ``window`` None keeps native
        recurrent KV, otherwise the shared rule in all recurrent attention
        layers or (``share_layers="full_attention"``) only the full-attention
        ones. ``output_window`` likewise for the output block.
        """
        tokens = torch.as_tensor(tokens)
        n = len(tokens)
        R, last = self.loops, self.loops - 1
        windows = [w for w in (window, output_window) if w is not None]
        chunk = min(windows) if windows else n
        caches = {}

        def cache(key):
            if key not in caches:
                caches[key] = (
                    torch.zeros(n, self.kv_heads, self.head_dim),
                    torch.zeros(n, self.kv_heads, self.head_dim),
                )
            return caches[key]

        rec = range(self.n_in, self.n_in + self.n_rec)
        out_layers = range(self.n_in + self.n_rec, len(self.layer_types))
        shared_layer = {
            i: window is not None
            and i in rec
            and self.layer_types[i] != "mlp"
            and (share_layers == "all" or self.layer_types[i] == "full_attention")
            for i in range(len(self.layer_types))
        }
        results = {d: [] for d in range(1, R + 1)}
        spans = [(0, prompt_len)] + [(a, min(a + chunk, n)) for a in range(prompt_len, n, chunk)]
        all_positions = torch.arange(n)
        for a, b in spans:
            positions = all_positions[a:b]
            t = positions[:, None]
            j = all_positions[None, :b]
            causal = j <= t
            response = a >= prompt_len

            def attention(i, h, key, final_key, w):
                q, k, v, gate = self.qkv(i, h, positions)
                K, V = cache(key)
                K[a:b], V[a:b] = k, v
                visible = causal
                if self.layer_types[i] == "sliding_attention":
                    visible = visible & (j > t - self.window)
                if final_key is not None and response:
                    Kf, Vf = cache(final_key)
                    old = j <= t - w
                    sources = [(K[:b], V[:b], visible & ~old), (Kf[:b], Vf[:b], visible & old)]
                else:
                    sources = [(K[:b], V[:b], visible)]
                return self.finish_attention(i, self.attend(q, sources), gate)

            def run(i, h, key, final_key=None, w=None):
                if self.layer_types[i] == "mlp":
                    return self.slerp(h, self.mlp(i, h), f"backbone.layers.{i}.mlp_alpha")
                out = attention(i, h, key, final_key, w)
                return self.slerp(h, out, f"backbone.layers.{i}.attn_alpha")

            h = bf(self.w["backbone.embeddings.weight"][tokens[a:b]])
            for i in range(self.n_in):
                h = run(i, h, (i, "in"))
            x = state = h
            states = []
            for r in range(R):
                h = justnorm(bf(state + x))
                for i in rec:
                    final = (i, "rec", last) if shared_layer[i] and r < last else None
                    h = run(i, h, (i, "rec", r), final, window)
                state = justnorm(h)
                states.append(state)
            # Depth R first so its output-block KV exists for shared depth-r readouts.
            for depth in [R, *range(1, R)]:
                h = states[depth - 1]
                for i in out_layers:
                    shared_out = (
                        output_window is not None
                        and depth < R
                        and (
                            output_share_layers == "all"
                            or self.layer_types[i] == "full_attention"
                        )
                    )
                    final = (i, "out", R) if shared_out else None
                    h = run(i, h, (i, "out", depth), final, output_window)
                first = max(a, prompt_len - 1)
                if first < b:
                    results[depth].append(self.head(h[first - a : b - a]))
        return {d: torch.cat(v)[: n - prompt_len] for d, v in results.items()}
