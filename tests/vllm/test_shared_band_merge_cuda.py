"""The fused decode band kernel matches exact shared-last attention (CUDA).

Query t reads final-loop KV at j <= t - W and its own loop at t - W < j <= t. The kernel
(band_merge) replaces the band FlashAttention call and merge_attn_states of the decode path;
it must be at least as close to an FP32 reference as that path.
"""

import unittest

import torch
from helpers import cuda_test


def paged(cache, table, length):
    """Positions 0..length-1 of a (B, H, N, 2D) paged cache as [length, H, 2D]."""
    block = cache.shape[2]
    positions = torch.arange(length, device=cache.device)
    return cache[table[positions // block], :, positions % block]


@cuda_test
class BandMergeTests(unittest.TestCase):
    def run_case(self, heads, kv_heads, seq_lens, query_lens, window=64, d=128, block=16):
        from vllm.model_executor.models.shared_attention import band_merge
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, get_flash_attn_version
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

        torch.manual_seed(len(seq_lens) * 1000 + heads)
        device, dtype = "cuda", torch.bfloat16
        pages = [-(-s // block) for s in seq_lens]
        blocks = sum(pages) * 2 + 4
        own = torch.randn(blocks, kv_heads, block, 2 * d, device=device, dtype=dtype)
        final = torch.randn(blocks, kv_heads, block, 2 * d, device=device, dtype=dtype)
        width = max(pages)
        own_table = torch.zeros(len(seq_lens), width, dtype=torch.int32, device=device)
        final_table = torch.zeros_like(own_table)
        order = torch.randperm(blocks, device=device).to(torch.int32)
        offset = 0
        for r, n in enumerate(pages):
            own_table[r, :n] = order[offset : offset + n]
            final_table[r, :n] = order[offset + n : offset + 2 * n]
            offset += 2 * n
        count = sum(query_lens)
        q = torch.randn(count, heads, d, device=device, dtype=dtype)
        qsl = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0)), dtype=torch.int32, device=device)
        lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        scale = d**-0.5
        version = get_flash_attn_version(head_size=d)
        own_k, own_v = own.transpose(1, 2).split(d, -1)
        final_k, final_v = final.transpose(1, 2).split(d, -1)
        memory_out, memory_lse = flash_attn_varlen_func(
            q=q, k=final_k, v=final_v, cu_seqlens_q=qsl, max_seqlen_q=max(query_lens),
            seqused_k=(lens - window).clamp_min(0), max_seqlen_k=max(seq_lens), block_table=final_table,
            softmax_scale=scale, causal=True, return_softmax_lse=True, fa_version=version)
        band_out, band_lse = flash_attn_varlen_func(
            q=q, k=own_k, v=own_v, cu_seqlens_q=qsl, max_seqlen_q=max(query_lens), seqused_k=lens,
            max_seqlen_k=max(seq_lens), block_table=own_table, softmax_scale=scale, causal=True,
            window_size=(window - 1, 0), return_softmax_lse=True, fa_version=version)
        old = torch.empty_like(q)
        merge_attn_states(old, memory_out, memory_lse.contiguous(), band_out, band_lse.contiguous())
        new = torch.empty_like(q)
        metadata = type("M", (), dict(block_table=own_table, seq_lens=lens, query_start_loc=qsl))
        band_merge(q, own, metadata, memory_out, memory_lse, new, window, scale)
        # The fused writer must read all new query rows from the input K/V,
        # including another program's earlier query, before those cache rows exist.
        slots, new_rows = [], []
        for r, (s, n) in enumerate(zip(seq_lens, query_lens)):
            positions = torch.arange(s - n, s, device=device)
            slot = own_table[r, positions // block].long() * block + positions % block
            slots.append(slot)
            new_rows.append(own[slot // block, :, slot % block])
        slots = torch.cat(slots)
        incoming = torch.cat(new_rows)
        key, value = incoming.split(d, -1)  # Interleaved heads, noncontiguous views.
        unwritten = own.clone()
        unwritten[slots // block, :, slots % block] = float("nan")
        metadata.slot_mapping = slots
        fused = torch.empty_like(q)
        band_merge(q, unwritten, metadata, memory_out, memory_lse, fused, window,
                   scale, key=key, value=value)
        torch.testing.assert_close(unwritten, own, atol=0, rtol=0)
        torch.testing.assert_close(fused, new, atol=0, rtol=0)
        exact = torch.empty(count, heads, d, device=device, dtype=torch.float32)
        group = heads // kv_heads
        for r, (s, n) in enumerate(zip(seq_lens, query_lens)):
            own_rows = paged(own, own_table[r].long(), s).float()
            final_rows = paged(final, final_table[r].long(), s).float()
            for i in range(n):
                token, t = int(qsl[r]) + i, s - n + i
                j = torch.arange(t + 1, device=device)
                rows = torch.where((j <= t - window)[:, None, None], final_rows[: t + 1], own_rows[: t + 1])
                k = rows[..., :d].repeat_interleave(group, 1)
                v = rows[..., d:].repeat_interleave(group, 1)
                weights = torch.softmax(torch.einsum("hd,jhd->hj", q[token].float(), k) * scale, -1)
                exact[token] = torch.einsum("hj,jhd->hd", weights, v)
        error_new = (new.float() - exact).abs().max().item()
        error_old = (old.float() - exact).abs().max().item()
        self.assertFalse(torch.isnan(new).any())
        self.assertLess(error_new, 2e-2)
        self.assertLessEqual(error_new, error_old + 4e-3)
        return error_new, error_old

    def test_decode_short_long_and_multi_query(self):
        for heads, kv_heads in ((16, 16), (32, 8)):
            with self.subTest(heads=heads, kv_heads=kv_heads):
                self.run_case(heads, kv_heads, [1], [1])
                self.run_case(heads, kv_heads, [40, 64, 65, 1000, 4097], [1, 1, 1, 1, 1])
                self.run_case(heads, kv_heads, [70, 2050], [16, 4])

    def test_graph_replay_with_fewer_live_requests_zeros_padding(self):
        """A 4-query graph must not read a fifth request when only 3 remain."""
        from types import SimpleNamespace

        from vllm.model_executor.models.shared_attention import band_merge

        torch.manual_seed(4321)
        q = torch.randn(4, 4, 32, device="cuda", dtype=torch.bfloat16)
        initial = torch.randn(8, 2, 16, 64, device="cuda", dtype=q.dtype)
        lengths = torch.tensor([32] * 4, device="cuda", dtype=torch.int32)
        starts = torch.arange(5, device="cuda", dtype=torch.int32)
        table = torch.arange(8, device="cuda", dtype=torch.int32).view(4, 2)
        slots = torch.tensor([31, 63, 95, 127], device="cuda")
        incoming = initial[slots // 16, :, slots % 16].clone()
        key, value = incoming.split(32, -1)
        metadata = SimpleNamespace(
            seq_lens=lengths, query_start_loc=starts, block_table=table,
            slot_mapping=slots,
        )
        memory_out = torch.zeros_like(q)
        memory_lse = torch.full((4, 4), -float("inf"), device="cuda")

        for write_kv in (False, True):
            with self.subTest(write_kv=write_kv):
                cache, output = initial.clone(), torch.empty_like(q)
                lengths.fill_(32)
                starts.copy_(torch.arange(5, device="cuda", dtype=torch.int32))
                slots[-1] = 127
                options = dict(key=key, value=value) if write_kv else {}

                def run():
                    band_merge(q, cache, metadata, memory_out, memory_lse,
                               output, 64, 32**-0.5, **options)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                expected = output.clone()
                # Same padded grid/metadata allocations; one request has ended.
                starts[-1] = 3
                lengths[-1] = 0
                # A stale positive slot and invalid input must still be ignored.
                slots[-1] = 127
                if write_kv:
                    key[-1].fill_(float("nan"))
                    value[-1].fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(output[:3], expected[:3], atol=0, rtol=0)
                self.assertEqual(torch.count_nonzero(output[3]).item(), 0)
                torch.testing.assert_close(cache, initial, atol=0, rtol=0)
                # A completely empty padded replay must also avoid metadata reads.
                starts.zero_()
                lengths.zero_()
                slots.fill_(-1)
                graph.replay()
                self.assertEqual(torch.count_nonzero(output).item(), 0)
                slots.copy_(torch.tensor([31, 63, 95, 127], device="cuda"))
        metadata.seq_lens = lengths[:0]
        metadata.query_start_loc = starts[:1]
        output.fill_(float("nan"))
        run()
        self.assertEqual(torch.count_nonzero(output).item(), 0)
        torch.testing.assert_close(cache, initial, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
