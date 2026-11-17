"""Shared-reader lengths follow the current forward and live graph inputs."""
import unittest
from types import SimpleNamespace

import torch
from helpers import cuda_test
from vllm.forward_context import ForwardContext
from vllm.model_executor.models.shared_attention import _shared_lengths


def context(decode_lengths=None):
    """A forward with a prompt prefill when per-request decode lengths are given."""
    return ForwardContext(
        no_compile_layers={}, attn_metadata=None, slot_mapping={},
        loop_kv_decode_lengths=decode_lengths,
        loop_kv_prefill_active=decode_lengths is not None,
    )


class SharedLengthTests(unittest.TestCase):
    def test_prompt_rows_do_not_read_shared_history_in_mixed_batch(self):
        metadata = SimpleNamespace(seq_lens=torch.tensor([5, 65, 100]))
        forward = context(torch.tensor([-1, 0, 0]))
        old, current, decode = _shared_lengths(forward, metadata, 64)
        self.assertEqual(old.tolist(), [0, 1, 36])
        self.assertEqual(current.tolist(), [0, 65, 100])
        self.assertEqual(decode.tolist(), [False, True, True])
        self.assertIs(_shared_lengths(forward, metadata, 64)[0], old)
        self.assertEqual(_shared_lengths(forward, metadata, 1)[0].tolist(),
                         [0, 64, 99])

    def test_reused_sequence_buffer_is_recomputed_on_next_forward(self):
        metadata = SimpleNamespace(seq_lens=torch.tensor([63, 64, 65]))
        old, _, _ = _shared_lengths(context(), metadata, 64)
        metadata.seq_lens.add_(16)
        updated, _, _ = _shared_lengths(context(), metadata, 64)
        self.assertEqual(old.tolist(), [0, 0, 1])
        self.assertEqual(updated.tolist(), [15, 16, 17])

    @cuda_test
    def test_graph_replay_updates_shared_lengths_after_ring_boundary(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            lengths = torch.tensor([63, 64, 65], device="cuda", dtype=torch.int32)
            metadata = SimpleNamespace(seq_lens=lengths)
            _shared_lengths(context(), metadata, 64)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward = context()
            old, current, _ = _shared_lengths(forward, metadata, 64)
            reused = _shared_lengths(forward, metadata, 64)[0]
        self.assertIs(reused, old)
        lengths.copy_(torch.tensor([127, 15, 0], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(old.tolist(), [63, 0, 0])
        self.assertEqual(current.tolist(), [127, 15, 0])


if __name__ == "__main__":
    unittest.main()
