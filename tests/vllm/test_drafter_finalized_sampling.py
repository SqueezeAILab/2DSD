"""Finalized sampling is lossless: committed tokens follow the loop-R distribution.

The fake target is a Markov chain: a node's loop-d readout depends only on its
token and d, so the exact output law is the loop-R chain. Shallow readouts and
the argmax drafter disagree with loop R often enough to exercise d1 rejection,
frozen commits, promotion with re-review, d2 forks and bonus restarts. With a
temperature and top-p the law is the tempered loop-R chain cut to its nucleus.
"""

import collections
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from helpers import FakeTreeRunner, cuda_test
from scipy.stats import chi2, kstest
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor

V = 5
torch.set_num_threads(1)


@dataclass
class Chain:
    readouts: dict  # loop d -> [V, V] next-token law given the current token
    draft: torch.Tensor  # argmax drafter's next token given the current token

    @property
    def final(self):
        return self.readouts[4]


def dirichlet(alpha, generator):
    sample = torch._standard_gamma(torch.full((V, V), alpha), generator=generator)
    return sample / sample.sum(-1, keepdim=True)


def mixed_chain():
    """Loop 1 mixes in an unrelated chain; loop 2 adds its own error, so a d2 fork
    is not exactly the residual p_R ⊖ p_d1. The drafter is right for odd tokens."""
    g = torch.Generator().manual_seed(20261030)
    final = dirichlet(0.7, g)
    shallow = 0.45 * final + 0.55 * dirichlet(1.0, g)
    middle = 0.4 * shallow + 0.4 * final + 0.2 * dirichlet(1.0, g)
    mode = final.argmax(-1)
    draft = torch.where(torch.arange(V) % 2 == 1, mode, (mode + 1) % V)
    return Chain({1: shallow, 2: middle, 3: 0.2 * shallow + 0.8 * final, 4: final}, draft)


def promotion_chain():
    """Loop 1 is mostly wrong and loop 2 nearly final, with an always-final drafter:
    d1 rejects often and d2 promotes often."""
    g = torch.Generator().manual_seed(7)
    final = dirichlet(0.5, g)
    shallow = 0.2 * final + 0.8 * dirichlet(0.5, g)
    readouts = {1: shallow, 2: 0.85 * final + 0.15 * shallow,
                3: 0.95 * final + 0.05 * shallow, 4: final}
    return Chain(readouts, final.argmax(-1))


class MarkovRunner(FakeTreeRunner):
    def __init__(self, chain):
        self.chain = chain
        super().__init__(8, causal=False)

    def reset(self):
        super().reset()
        if hasattr(self, "stats"):
            self.stats["calls"] = 0

    def ensure_capacity(self, required):
        while required > self.max_nodes:
            self.max_nodes *= 2
        return self.max_nodes

    def prefill(self, tokens):
        result = super().prefill(tokens)
        result.hidden = tokens.to(self.dtype).reshape(-1, 1) + 4000
        result.features = result.hidden.clone()
        return result

    def advance(self, spans):
        results = super().advance(spans)
        for span, result in zip(spans, results):
            value = self.nodes[span.node_id]["token"] + (span.loop + 1) * 1000
            result.hidden = torch.tensor([[value]], dtype=self.dtype)
            result.features = result.hidden.clone()
        return results

    def logits(self, hidden):
        values = hidden[:, 0].long()
        tokens, depths = values % 1000, values // 1000
        rows = [self.chain.readouts[int(d)][int(t)] for t, d in zip(tokens, depths)]
        return torch.stack(rows).log().to(self.dtype)


class MarkovDraft:
    """Argmax drafter of the chain; its proposals depend only on the head token."""

    input_format = "loop-context"
    stats = {}

    def __init__(self, chain):
        self.chain = chain

    def reset(self):
        pass

    def draft_context_many(self, requests):
        results = []
        for request in requests:
            last, slots = request["head"], []
            for _ in range(request["k"] - 1):
                last = int(self.chain.draft[last])
                slots.append(last)
            results.append(SimpleNamespace(token_ids=torch.tensor(slots, dtype=torch.long)))
        return results


def nucleus_law(law, temperature, top_p):
    """Rows of law at a temperature, cut to the smallest most-probable set holding top_p."""
    tempered = law ** (1 / temperature)
    tempered = tempered / tempered.sum(-1, keepdim=True)
    out = torch.zeros_like(tempered)
    for i, row in enumerate(tempered):
        mass = 0.0
        for token in row.argsort(descending=True).tolist():
            out[i, token] = row[token]
            mass += float(row[token])
            if mass >= top_p:
                break
    return out / out.sum(-1, keepdim=True)


def run(chain, stage, k, d1=1, runs=4000, length=4, patch=None, stage2_gate="argmax",
        temperature=1.0, top_p=1.0, proposal_top_p=None):
    executor = FinalizedTreeExecutor(
        MarkovRunner(chain), MarkovDraft(chain) if k > 1 else None, k, stage, d1=d1,
        temperature=temperature, stage2_gate=stage2_gate, top_p=top_p,
        proposal_top_p=proposal_top_p,
    )
    if patch:
        patch(executor)
    outputs, totals = [], collections.Counter()
    for seed in range(runs):
        result = executor.generate([0, 1], length, [], seed=seed)
        outputs.append(tuple(result["output_ids"]))
        stats = result["stats"]
        for name in ("promotions", "forks", "bonus_tokens", "frozen_created"):
            totals[name] += stats[name]
        totals["frozen_commits"] += stats["commit_sources"]["frozen"]
    return outputs, totals


def chi2_pvalue(observed, expected):
    """χ² over the support of expected; a draw outside the support fails outright."""
    support = [i for i, e in enumerate(expected) if e > 0]
    if any(observed[i] for i in range(len(expected)) if i not in support):
        return 0.0
    if len(support) == 1:  # a one-token nucleus: every draw inside it is the law
        return 1.0
    stat = sum((observed[i] - expected[i]) ** 2 / expected[i] for i in support)
    return float(chi2.sf(stat, len(support) - 1))


def pvalues(chain, outputs, law=None):
    """χ² p-values: each next-token law given the previous token (pooled over
    positions), and the joint law of the first two tokens. law defaults to loop R."""
    law = chain.final if law is None else law
    counts = collections.defaultdict(collections.Counter)
    for sequence in outputs:
        previous = 1
        for token in sequence:
            counts[previous][token] += 1
            previous = token
    result = []
    for previous, observed in counts.items():
        n = sum(observed.values())
        result.append(chi2_pvalue([observed[t] for t in range(V)],
                                  [float(law[previous][t]) * n for t in range(V)]))
    joint = collections.Counter(s[:2] for s in outputs)
    pairs = [(first, second) for first in range(V) for second in range(V)]
    result.append(chi2_pvalue([joint[pair] for pair in pairs],
                              [float(law[1][a] * law[a][b]) * len(outputs) for a, b in pairs]))
    return result


class FinalizedSamplingTests(unittest.TestCase):
    def assert_lossless(self, chain, outputs):
        self.assertGreater(min(pvalues(chain, outputs)), 1e-3)

    def assert_lossless_against(self, chain, outputs, law):
        self.assertGreater(min(pvalues(chain, outputs, law)), 1e-3)

    def test_every_stage_and_block_matches_the_loop_r_chain(self):
        chain = mixed_chain()
        for stage in ("stage1", "stage2a", "stage2b"):
            for k in (1, 4):
                with self.subTest(stage=stage, k=k):
                    outputs, totals = run(chain, stage, k)
                    self.assert_lossless(chain, outputs)
                    self.assertGreater(totals["bonus_tokens"], 0)
                    if k > 1:
                        self.assertGreater(totals["frozen_created"], 0)
                        self.assertGreater(totals["frozen_commits"], 0)
                    if stage != "stage1" and k > 1:
                        self.assertGreater(totals["promotions"], 0)
                    if stage == "stage2b":
                        self.assertGreater(totals["forks"], 0)

    def test_frequent_promotion_matches_the_loop_r_chain(self):
        chain = promotion_chain()
        for stage in ("stage2a", "stage2b"):
            with self.subTest(stage=stage):
                outputs, totals = run(chain, stage, 4)
                self.assert_lossless(chain, outputs)
                self.assertGreater(totals["promotions"], 1000)

    def test_stage2_gates_are_lossless(self):
        """The LoopSpec gate (0) and a thresholded gate change only which candidates
        exist; the default argmax gate runs in every other test."""
        for chain in (mixed_chain(), promotion_chain()):
            for gate in (0.0, 0.5):
                with self.subTest(gate=gate):
                    outputs, totals = run(chain, "stage2b", 4, stage2_gate=gate)
                    self.assert_lossless(chain, outputs)
                    self.assertGreater(totals["promotions"] + totals["forks"], 0)

    def test_temperature_and_top_p_follow_the_nucleus_of_loop_r(self):
        """T=0.6, top-p 0.8 cuts tokens from most rows; every readout gets the same
        mask, and commits follow the nucleus of the tempered loop-R chain."""
        for chain in (mixed_chain(), promotion_chain()):
            law = nucleus_law(chain.final, 0.6, 0.8)
            self.assertGreater(int((law == 0).sum()), V)
            for stage in ("stage1", "stage2a", "stage2b"):
                with self.subTest(stage=stage):
                    outputs, totals = run(chain, stage, 4, temperature=0.6, top_p=0.8)
                    self.assert_lossless_against(chain, outputs, law)
                    self.assertGreater(totals["frozen_commits"] + totals["bonus_tokens"], 0)
                    if stage == "stage2b":
                        self.assertGreater(totals["promotions"] + totals["forks"], 0)
                    # Power: the untruncated tempered law is rejected.
                    self.assertLess(min(pvalues(chain, outputs, nucleus_law(chain.final, 0.6, 1.0))), 1e-3)

    def test_untruncated_proposals_keep_the_nucleus_target(self):
        """proposal_top_p=1 leaves d1/d2 readouts uncut; only loop-R rows (and d1=R
        reviews) get the nucleus, and commits still follow the cut loop-R chain."""
        for chain in (mixed_chain(), promotion_chain()):
            law = nucleus_law(chain.final, 0.6, 0.8)
            for stage, k, d1 in (("stage1", 4, 1), ("stage2a", 4, 1), ("stage2b", 4, 1), ("stage1", 4, 4),
                                 ("stage1", 1, 1), ("stage2b", 1, 1)):  # k1: the LoopSpec d1 and (1,2) controls
                with self.subTest(stage=stage, k=k, d1=d1):
                    shallow = []

                    def patch(executor):
                        original = executor._readout_distributions

                        def recorded(logits, depths):
                            probs = original(logits, depths)
                            shallow.extend(int((row == 0).sum()) for row, d in zip(probs, depths) if d < 4)
                            return probs

                        executor._readout_distributions = recorded

                    outputs, totals = run(chain, stage, k, d1=d1, temperature=0.6, top_p=0.8,
                                          proposal_top_p=1.0, patch=patch)
                    self.assert_lossless_against(chain, outputs, law)
                    if d1 < 4:
                        self.assertTrue(shallow)
                        self.assertEqual(max(shallow), 0)  # proposals keep every token
                    if stage == "stage2b":
                        self.assertGreater(totals["promotions"] + totals["forks"], 0)

    def test_top_p_requires_sampling(self):
        chain = mixed_chain()
        for temperature, top_p, proposal in ((0.0, 0.95, None), (1.0, 0.0, None), (1.0, 1.5, None),
                                             (0.0, 1.0, 0.9), (1.0, 0.9, 0.0)):
            with self.subTest(temperature=temperature, top_p=top_p, proposal=proposal), \
                    self.assertRaises(ValueError):
                FinalizedTreeExecutor(MarkovRunner(chain), MarkovDraft(chain), 4, "stage2b",
                                             temperature=temperature, top_p=top_p, proposal_top_p=proposal)

    def test_final_depth_review_is_the_1d_boundary(self):
        chain = mixed_chain()
        outputs, totals = run(chain, "stage1", 4, d1=4)
        self.assert_lossless(chain, outputs)
        self.assertEqual(totals["frozen_created"], 0)

    def test_wrong_proposals_are_detected(self):
        """Power check: frozen verified against p_d1 instead of one-hot, or a fork
        verified against p_d2 instead of its residual, breaks the law."""
        chain = mixed_chain()
        wrong = {"frozen": lambda c: c.parent.dists[1], "fork": lambda c: c.parent.dists[2]}
        for role, proposal in wrong.items():
            with self.subTest(role=role):

                def patch(executor):
                    original = executor._proposal
                    executor._proposal = (
                        lambda c: proposal(c) if c.role == role else original(c)
                    )

                outputs, _ = run(chain, "stage2b", 4, patch=patch)
                self.assertLess(min(pvalues(chain, outputs)), 1e-3)

    def test_sampling_check_statistics_are_calibrated(self):
        """The recorded exactness check: mean p(x) - sum p^2 is zero and the
        probability-ranked PIT is uniform."""
        chain = mixed_chain()
        executor = FinalizedTreeExecutor(
            MarkovRunner(chain), MarkovDraft(chain), 4, "stage2b", temperature=1.0,
            record_sampling_check=True,
        )
        rows = collections.defaultdict(list)
        for seed in range(600):
            check = executor.generate([0, 1], 8, [], seed=seed)["sampling_check"]
            for name, values in check.items():
                rows[name] += values
        p, above, squares = (torch.tensor(rows[n], dtype=torch.float64) for n in
                             ("token_probability", "more_probable_mass", "sum_squares"))
        gap = p - squares
        self.assertLess(abs(float(gap.mean() / (gap.std() / len(gap) ** 0.5))), 4)
        pit = above + torch.rand(len(p), generator=torch.Generator().manual_seed(1),
                                 dtype=torch.float64) * p
        self.assertGreater(float(kstest(pit.numpy(), "uniform").pvalue), 1e-3)

    def test_seed_fixes_the_output(self):
        chain = mixed_chain()
        first, _ = run(chain, "stage2b", 4, runs=20)
        second, _ = run(chain, "stage2b", 4, runs=20)
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)


class ProbabilityReadback(unittest.TestCase):
    def check_draw(self, device):
        weights = torch.arange(200, device=device, dtype=torch.float32).view(4, 50)
        weights[0].zero_()
        weights[1].zero_()
        weights[1, 7] = 1
        saved = weights.clone()
        # Empty fork rows are discarded; removing their old one-hot placeholder
        # must preserve the returned token and the subsequent RNG state.
        placeholder = weights.clone()
        placeholder[0, 0] = 1
        generator = torch.Generator(device=device).manual_seed(271)
        reference = torch.Generator(device=device).manual_seed(271)
        for _ in range(10):
            expected = (placeholder / torch.empty_like(weights).exponential_(
                generator=reference)).argmax(-1)
            actual = FinalizedTreeExecutor._draw(
                SimpleNamespace(generator=generator), weights,
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(weights, saved, atol=0, rtol=0)
        torch.testing.assert_close(generator.get_state(), reference.get_state())

    def check_views(self, device):
        from vllm.v1.spec_decode.looped_sampling import gather_probabilities

        base = torch.arange(240, dtype=torch.float32, device=device).view(3, 4, 20)
        standalone = torch.rand(50, device=device)
        distributions = [base[1, 2, 1::2], standalone[3:25], base[0, 1], base[2, 3]]
        items = [(distributions[i % 4], i % 7) for i in range(259)]
        expected = [float(dist[token]) for dist, token in items]
        buffers = {}
        for size in (3, len(items), 1, 10, len(items)):
            self.assertEqual(gather_probabilities(items[:size], buffers), expected[:size])
        self.assertEqual(gather_probabilities([]), [])

    def test_cpu_distribution_views(self):
        self.check_views("cpu")
        self.check_draw("cpu")

    @cuda_test
    def test_cuda_distribution_offsets_strides_and_multiple_blocks(self):
        self.check_views("cuda")
        self.check_draw("cuda")


if __name__ == "__main__":
    unittest.main()
