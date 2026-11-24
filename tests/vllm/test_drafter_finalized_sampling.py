"""Finalized sampling is lossless: committed tokens follow the loop-R distribution.

The fake target is a Markov chain: a node's loop-d readout depends only on its
token and d, so the exact output law is the loop-R chain. Shallow readouts and
the argmax drafter disagree with loop R often enough to exercise d1 rejection,
frozen commits, promotion with re-review, d2 forks and bonus restarts. With a
temperature and top-p the law is the tempered loop-R chain cut to its nucleus.
"""

import collections
import itertools
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from helpers import FakeTreeRunner, FusedReadoutRunner, NodeStateRunner, cuda_test
from scipy.stats import chi2, kstest
from vllm.v1.spec_decode.looped_2d_finalized import FinalizedTreeExecutor

V = 5
torch.set_num_threads(1)


@dataclass
class Chain:
    readouts: dict  # loop d -> [V, V] next-token law given the current token
    draft: torch.Tensor  # argmax drafter's next token given the current token

    @property
    def loops(self):
        return max(self.readouts)

    @property
    def final(self):
        return self.readouts[self.loops]


def readouts(loops, shallow, middle, late, final):
    """Loop 1, loop 2, every later loop below the last, and the last loop."""
    early = (shallow, middle) + (late,) * (loops - 3)
    return {d: early[d - 1] for d in range(1, loops)} | {loops: final}


def dirichlet(alpha, generator):
    sample = torch._standard_gamma(torch.full((V, V), alpha), generator=generator)
    return sample / sample.sum(-1, keepdim=True)


def mixed_chain(loops=4):
    """Loop 1 mixes in an unrelated chain; loop 2 adds its own error, so a d2 fork
    is not exactly the residual p_R ⊖ p_d1. The drafter is right for odd tokens."""
    g = torch.Generator().manual_seed(20261030)
    final = dirichlet(0.7, g)
    shallow = 0.45 * final + 0.55 * dirichlet(1.0, g)
    middle = 0.4 * shallow + 0.4 * final + 0.2 * dirichlet(1.0, g)
    mode = final.argmax(-1)
    draft = torch.where(torch.arange(V) % 2 == 1, mode, (mode + 1) % V)
    return Chain(readouts(loops, shallow, middle, 0.2 * shallow + 0.8 * final, final), draft)


def promotion_chain(loops=4):
    """Loop 1 is mostly wrong and loop 2 nearly final, with an always-final drafter:
    d1 rejects often and d2 promotes often."""
    g = torch.Generator().manual_seed(7)
    final = dirichlet(0.5, g)
    shallow = 0.2 * final + 0.8 * dirichlet(0.5, g)
    laws = readouts(loops, shallow, 0.85 * final + 0.15 * shallow,
                    0.95 * final + 0.05 * shallow, final)
    return Chain(laws, final.argmax(-1))


def cycle_chain(loops=4):
    """The drafter follows the cycle t -> t + 1, the mode of the last loop. Loop 1
    underrates that step after even tokens, so d1 freezes the drafts and the nearly
    final loop 2 promotes them, and overrates it after odd tokens, so the token
    after a promoted head passes the re-review with p_d1 far above the final law:
    the law holds only if its proposal becomes p_d2."""
    step = torch.roll(torch.eye(V), 1, 1)
    uniform = torch.full((V, V), 1 / V)
    final = 0.5 * step + 0.5 * uniform
    weight = torch.where(torch.arange(V) % 2 == 0, 0.05, 0.9)[:, None]
    shallow = weight * step + (1 - weight) * uniform
    middle = 0.9 * final + 0.1 * shallow
    return Chain(readouts(loops, shallow, middle, middle, final), final.argmax(-1))


class MarkovRunner(FakeTreeRunner):
    def __init__(self, chain):
        self.chain = chain
        super().__init__(8, causal=False, loops=chain.loops)

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
        result.hidden = tokens.to(self.dtype).reshape(-1, 1) + 1000 * self.R
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
        temperature=1.0, top_p=1.0, proposal_top_p=None, protocol=None, tight=False, nucleus_width=None,
        prefetch=False, d2=2):
    runner = MarkovRunner(chain)
    if tight:  # the request ends at the last context position
        runner.max_len = 2 + length
    if protocol is not None:  # fused readouts, with or without node state
        runner = protocol.for_schedule(runner, stage, d1)
    executor = FinalizedTreeExecutor(
        runner, MarkovDraft(chain) if k > 1 else None, k, stage, d1=d1, d2=d2,
        temperature=temperature, stage2_gate=stage2_gate, top_p=top_p,
        proposal_top_p=proposal_top_p, nucleus_width=nucleus_width, prefetch=prefetch,
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
        for name in ("nucleus_calls", "nucleus_sorts"):
            totals[name] += stats.get(name, 0)
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

    def test_a_nucleus_read_from_the_most_probable_tokens_is_the_sorted_one(self):
        """nucleus_width reads the top-p cut of a target row from its most probable tokens. On rows whose
        two most probable tokens hold top_p the distribution is the sort's (same support, same values up
        to rounding), shallow rows keep every token, and a call with a row that needs a third token falls
        back to the sort, bit for bit. Commits then follow the cut loop-R chain as with the sort."""
        runner = MarkovRunner(mixed_chain())
        generator = torch.Generator().manual_seed(0)
        peaked = torch.randn(6, V, generator=generator)
        peaked[torch.arange(6), torch.randint(V, (6,), generator=generator)] += 4.0
        flat = torch.cat((peaked[:5], torch.zeros(1, V)))
        for proposal in (1.0, None):  # proposals uncut, and cut like the target (every row of a call then)
            sort, top = (FinalizedTreeExecutor(runner, None, 1, "stage1", d1=1, temperature=0.6, top_p=0.8,
                                               proposal_top_p=proposal, nucleus_width=width) for width in (None, 2))
            for logits, fast in ((peaked, True), (flat, False)):
                # Every row a target, proposals among them, and a call of proposals only.
                for depths in ([4] * 6, [1, 4, 1, 4, 4, 4], [1] * 6):
                    with self.subTest(proposal=proposal, fast=fast, depths=depths):
                        want, got = sort._readout_distributions(logits.clone(), depths), \
                            top._readout_distributions(logits.clone(), depths)
                        cut = [d == 4 or proposal is None for d in depths]
                        if fast:
                            self.assertTrue(torch.equal(want > 0, got > 0))
                            self.assertTrue(torch.allclose(want, got, atol=1e-6))
                            self.assertTrue(all(int((row > 0).sum()) <= 2 for row, c in zip(got, cut) if c))
                        else:
                            self.assertTrue(torch.equal(want, got))
                        self.assertTrue(all(int((row == 0).sum()) == 0 for row, c in zip(got, cut) if not c))
        chain = mixed_chain()
        law = nucleus_law(chain.final, 0.6, 0.8)
        self.assertEqual(sorted(set((law > 0).sum(-1).tolist())), [1, 2, 3])
        # Three tokens hold every row's cut; with two, plain decoding (a row per call) takes both paths.
        for stage, k, d1, width, proposal in (("stage1", 4, 1, 3, 1.0), ("stage2b", 4, 1, 3, 1.0), ("stage1", 4, 4, 3, 1.0),
                                              ("stage1", 1, 4, 2, 1.0), ("stage2b", 4, 1, 3, None)):
            with self.subTest(stage=stage, k=k, d1=d1, width=width, proposal=proposal):
                paths = collections.Counter()

                def patch(executor):
                    original = executor._nucleus

                    def counted(probs, rows):
                        done = original(probs, rows)
                        paths[done] += 1
                        return done

                    executor._nucleus = counted

                outputs, totals = run(chain, stage, k, d1=d1, temperature=0.6, top_p=0.8, proposal_top_p=proposal,
                                      nucleus_width=width, patch=patch)
                # A request reports its calls with a cut and those that went to the sort.
                self.assertEqual((totals["nucleus_calls"], totals["nucleus_sorts"]), (sum(paths.values()), paths[False]))
                self.assert_lossless_against(chain, outputs, law)
                self.assertGreater(paths[True], 0)
                if proposal == 1.0:  # a cut proposal row of loop 1 or 2 can need a fourth token
                    self.assertEqual(paths[False] > 0, width == 2)

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


class ThreeLoopSamplingTests(unittest.TestCase):
    """The same law at R = 3, the loop count of Inkling2: Stage 1 reviews at
    d1 = 1, 2 or 3 and Stage 2 has the single schedule (d1, d2) = (1, 2), whose
    promotions and forks are verified one call later.

    Each check draws half the runs of its R = 4 twin; the wrong-proposal test
    shows that this still exposes a broken rule. The cycle chain draws twice
    RUNS: a wrong proposal after a promotion re-review shifts the law the
    least (with these seeds p is 5e-5 at RUNS draws and 3e-17 at twice as many).
    """

    RUNS = 2000
    SCHEDULES = (
        ("stage1", 1), ("stage1", 2), ("stage1", 3), ("stage2a", 1), ("stage2b", 1),
    )

    def sample(self, chain, stage, k, runs=RUNS, **options):
        return run(chain, stage, k, runs=runs, **options)

    def assert_lossless(self, chain, outputs, law=None):
        self.assertGreater(min(pvalues(chain, outputs, law)), 1e-3)

    def test_every_schedule_and_block_matches_the_loop_3_chain(self):
        """Commits follow the loop-3 chain for every review depth, with block
        drafting (k = 4) and without (k = 1, LoopSpec). d1 = 3 is the 1D
        boundary: the final review is the only one and nothing is frozen."""
        chain = mixed_chain(3)
        for (stage, d1), k in itertools.product(self.SCHEDULES, (1, 4)):
            if k == 1 and (stage, d1) in (("stage1", 3), ("stage2a", 1)):
                continue  # plain AR, and Stage 1 again: no draft, no frozen suffix
            with self.subTest(stage=stage, d1=d1, k=k):
                outputs, totals = self.sample(chain, stage, k, d1=d1)
                self.assert_lossless(chain, outputs)
                self.assertGreater(totals["bonus_tokens"], 0)
                reviewed = k > 1 and d1 < 3
                self.assertEqual(totals["frozen_created"] > 0, reviewed)
                self.assertEqual(totals["frozen_commits"] > 0, reviewed)
                self.assertEqual(totals["promotions"] > 0, stage != "stage1" and k > 1)
                self.assertEqual(totals["forks"] > 0, stage == "stage2b")

    def test_prefetched_reviews_match_the_loop_3_chain(self):
        """prefetch reads, with a call's readout, the probabilities its reviews test and a drawn anchor
        per row below loop 3. Commits still follow the loop-3 chain: for every schedule, with the runner
        protocols, with top-p from the most probable tokens (proposals uncut and cut), and where most
        frozen heads are promoted. Stage 1 then never goes back to the GPU for a probability or an
        anchor; Stage 2 draws the anchors its readouts could not prepare (a row with two children)."""
        mixed, promotion = mixed_chain(3), promotion_chain(3)
        cut = dict(temperature=0.6, top_p=0.8, nucleus_width=3)
        cases = [(mixed, stage, d1, 4, {}) for stage, d1 in self.SCHEDULES]
        cases += [(mixed, "stage2b", 1, 1, {}), (promotion, "stage2b", 1, 4, {}),
                  (mixed, "stage1", 1, 4, dict(protocol=NodeStateRunner)),
                  (mixed, "stage2b", 1, 4, dict(protocol=FusedReadoutRunner)),
                  (mixed, "stage1", 1, 4, dict(cut, proposal_top_p=1.0)),
                  (mixed, "stage2b", 1, 4, dict(cut, proposal_top_p=1.0)),
                  (mixed, "stage2b", 1, 4, dict(cut)), (promotion, "stage2b", 1, 4, dict(cut))]
        for chain, stage, d1, k, options in cases:
            with self.subTest(stage=stage, d1=d1, k=k, options=sorted(options), promotion=chain is promotion):
                reads = collections.Counter()

                def patch(executor):
                    for name in ("_gather", "_draw_anchors", "_sample_anchors"):
                        def counted(*args, _original=getattr(executor, name), _name=name):
                            reads[_name] += 1
                            return _original(*args)
                        setattr(executor, name, counted)

                outputs, totals = self.sample(chain, stage, k, d1=d1, prefetch=True, patch=patch, **options)
                law = nucleus_law(chain.final, 0.6, 0.8) if "top_p" in options else None
                self.assert_lossless(chain, outputs, law)
                if chain is mixed:  # the promotion chain's cut proposals are nearly the target: none rejected
                    self.assertGreater(totals["bonus_tokens"], 0)
                if d1 < 3 and k > 1:
                    self.assertGreater(totals["frozen_commits"], 0)
                    self.assertGreater(reads["_sample_anchors"], 0)
                if stage == "stage1":
                    self.assertEqual((reads["_gather"], reads["_draw_anchors"]), (0, 0))
                else:
                    self.assertLess(reads["_draw_anchors"], reads["_sample_anchors"])
                if stage == "stage2b":
                    self.assertGreater(totals["promotions"] + totals["forks"], 0)

    def test_frequent_promotion_matches_the_loop_3_chain(self):
        """Loop 2 is nearly final, so most frozen heads are promoted at d2 = 2
        and they and their re-reviewed suffix are verified in the next call."""
        cases = (
            ("promotion", promotion_chain(3), self.RUNS),
            ("cycle", cycle_chain(3), 2 * self.RUNS),
        )
        for name, chain, runs in cases:
            with self.subTest(chain=name):
                outputs, totals = self.sample(chain, "stage2a", 4, runs=runs)
                self.assert_lossless(chain, outputs)
                self.assertGreater(totals["promotions"], runs)

    def test_stage2_gates_are_lossless(self):
        """The LoopSpec gate (0) and a thresholded gate only change which loop-2
        candidates exist."""
        chain = mixed_chain(3)
        for gate in (0.0, 0.5):
            with self.subTest(gate=gate):
                outputs, totals = self.sample(chain, "stage2b", 4, stage2_gate=gate)
                self.assert_lossless(chain, outputs)
                self.assertGreater(totals["promotions"] + totals["forks"], 0)

    def test_top_p_follows_the_nucleus_of_loop_3(self):
        """T=0.6, top-p 0.8: commits follow the nucleus of the tempered loop-3
        chain when every readout is cut, and when proposal_top_p=1 leaves loops 1
        and 2 whole so that only loop-3 rows (a d1 = 3 review included) are cut."""
        chain = mixed_chain(3)
        law = nucleus_law(chain.final, 0.6, 0.8)
        for stage, d1, proposal in (
            ("stage2b", 1, None), ("stage2b", 1, 1.0), ("stage1", 3, 1.0),
        ):
            with self.subTest(stage=stage, d1=d1, proposal=proposal):
                shallow = []

                def patch(executor):
                    original = executor._readout_distributions

                    def recorded(logits, depths):
                        probs = original(logits, depths)
                        shallow.extend(
                            int((row == 0).sum()) for row, d in zip(probs, depths) if d < 3
                        )
                        return probs

                    executor._readout_distributions = recorded

                outputs, totals = self.sample(
                    chain, stage, 4, d1=d1, temperature=0.6, top_p=0.8,
                    proposal_top_p=proposal, patch=patch,
                )
                self.assert_lossless(chain, outputs, law)
                # Power: the untruncated tempered law is rejected.
                uncut = nucleus_law(chain.final, 0.6, 1.0)
                self.assertLess(min(pvalues(chain, outputs, uncut)), 1e-3)
                if d1 == 3:
                    self.assertFalse(shallow)  # no readout below the last loop
                else:
                    self.assertEqual(max(shallow) == 0, proposal == 1.0)
                    self.assertGreater(totals["promotions"] + totals["forks"], 0)

    def test_wrong_proposals_are_detected(self):
        """Power of these checks at the draws they use. The law breaks when a
        frozen child is verified against p_1 instead of one-hot, a fork against
        p_2 instead of its residual, or a token accepted in a promotion
        re-review against p_1 instead of p_2."""
        accepted = "promotion_accepted"
        wrong = {
            "frozen": (mixed_chain(3), lambda c: c.role == "frozen", 1, self.RUNS),
            "fork": (mixed_chain(3), lambda c: c.role == "fork", 2, self.RUNS),
            "promoted suffix": (
                cycle_chain(3), lambda c: c.origin == accepted, 1, 2 * self.RUNS
            ),
        }
        for name, (chain, affected, depth, runs) in wrong.items():
            with self.subTest(wrong=name):

                def patch(executor):
                    original = executor._proposal
                    executor._proposal = (
                        lambda c: c.parent.dists[depth] if affected(c) else original(c)
                    )

                outputs, _ = self.sample(chain, "stage2b", 4, runs=runs, patch=patch)
                self.assertLess(min(pvalues(chain, outputs)), 1e-3)

    def test_sampling_check_compares_loops_1_and_2_with_loop_3(self):
        """The recorded check stays calibrated, and its tv_d1 and tv_d2 columns
        are the distances of the verified token's loop-1 and loop-2 laws from
        its loop-3 law: a readout kept under another depth would show here."""
        chain = mixed_chain(3)
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
        distances = {d: 0.5 * (chain.final - chain.readouts[d]).abs().sum(-1) for d in (1, 2)}
        # The two loops are told apart: no token has the same distance at both.
        self.assertGreater(float((distances[1][:, None] - distances[2]).abs().min()), 1e-3)
        for name, depth in (("tv_d1", 1), ("tv_d2", 2)):
            recorded = torch.tensor(rows[name])
            nearest = (recorded[:, None] - distances[depth]).abs().min(-1).values
            self.assertLess(float(nearest.max()), 1e-5)

    def test_runner_protocols_draw_the_same_tokens(self):
        """A seed fixes the output, and the laws read through fused readouts or
        node state (the protocols of an input/looped/output-block target) commit
        the same tokens as the span protocol checked above. The repeats end at
        the last context position: sampled anchors, forks and bonus tokens stay
        inside prompt + max_new, or the fake refuses them."""
        chain = mixed_chain(3)
        for stage, d1 in self.SCHEDULES:
            with self.subTest(stage=stage, d1=d1):
                first, _ = run(chain, stage, 4, d1=d1, runs=40)
                self.assertGreater(len(set(first)), 1)
                for protocol in (None, FusedReadoutRunner, NodeStateRunner):
                    again, _ = run(
                        chain, stage, 4, d1=d1, runs=40, protocol=protocol, tight=True
                    )
                    self.assertEqual(again, first)


class DeeperScheduleSamplingTests(unittest.TestCase):
    """Every review depth of R = 4 and R = 5, as a variable-loop Inkling2 checkpoint run with more passes
    schedules them: Stage 1 at each d1 and Stage 2b at each pair d1 < d2 < R, on the Inkling2 path
    (prefetched reviews, top-p read from the most probable tokens)."""

    RUNS = 2000

    def schedules(self, loops):
        return ([("stage1", d1, 2) for d1 in range(1, loops + 1)]
                + [("stage2b", d1, d2) for d1 in range(1, loops) for d2 in range(d1 + 1, loops)])

    def test_every_depth_matches_the_loop_r_chain(self):
        for loops in (4, 5):
            chain = mixed_chain(loops)
            for stage, d1, d2 in self.schedules(loops):
                with self.subTest(loops=loops, stage=stage, d1=d1, d2=d2):
                    outputs, totals = run(chain, stage, 4, d1=d1, d2=d2, runs=self.RUNS, prefetch=True)
                    self.assertGreater(min(pvalues(chain, outputs)), 1e-3)
                    self.assertEqual(totals["frozen_created"] > 0, d1 < loops)
                    # The chain's loops from 3 on below R share one law: a second review there agrees.
                    differ = not torch.equal(chain.readouts[d1], chain.readouts[d2])
                    self.assertEqual(totals["forks"] > 0, stage == "stage2b" and differ)

    def test_every_depth_follows_the_nucleus_when_the_proposals_are_cut_like_the_target(self):
        """T = 0.6, top-p 0.8 on every readout: commits follow the nucleus of the tempered loop-R chain and
        not the uncut one."""
        for loops in (4, 5):
            chain = mixed_chain(loops)
            law, uncut = nucleus_law(chain.final, 0.6, 0.8), nucleus_law(chain.final, 0.6, 1.0)
            for stage, d1, d2 in self.schedules(loops):
                with self.subTest(loops=loops, stage=stage, d1=d1, d2=d2):
                    outputs, totals = run(chain, stage, 4, d1=d1, d2=d2, runs=self.RUNS, temperature=0.6, top_p=0.8,
                                          proposal_top_p=0.8, nucleus_width=V, prefetch=True)
                    self.assertGreater(min(pvalues(chain, outputs, law)), 1e-3)
                    self.assertLess(min(pvalues(chain, outputs, uncut)), 1e-3)
                    self.assertEqual(totals["nucleus_sorts"], 0)


class TwoLoopSamplingTests(unittest.TestCase):
    """R = 2 (Nanbeige) on the shared fakes: Stage 1 at d1 = 1 is the method."""

    def test_stage_1_matches_the_loop_2_chain(self):
        """The loop-count parameter of the fakes keeps the R = 2 law: block
        drafting reviewed at loop 1, and its k = 1 control, sample loop 2."""
        chain = mixed_chain(2)
        for k in (1, 4):
            with self.subTest(k=k):
                outputs, totals = run(chain, "stage1", k, runs=2000)
                self.assertGreater(min(pvalues(chain, outputs)), 1e-3)
                self.assertGreater(totals["bonus_tokens"], 0)
                self.assertEqual(totals["frozen_commits"] > 0, k > 1)


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
