"""FP32 optimizer/state checks with meaningful resume equivalence."""
import copy
import unittest

import torch

from training.kd.optim import ShardedAdamW, token_lr


OPTIONS = {"beta1": .9, "beta2": .95, "eps": 1e-8, "weight_decay": .01, "max_grad_norm": 1.0}


class OptimizerTests(unittest.TestCase):
    def test_accumulation_and_resumed_update_match_uninterrupted(self):
        torch.manual_seed(10)
        model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))
        optimizer = ShardedAdamW(model, OPTIONS)
        x = torch.randn(6, 8)

        def update(m, o):
            for chunk in x.split(3):
                (m(chunk).square().sum() / 6).backward()
                o.accumulate()
            return o.step(1e-5)

        update(model, optimizer)
        state = copy.deepcopy(optimizer.state_dict())
        resumed = copy.deepcopy(model)
        resumed_opt = ShardedAdamW(resumed, OPTIONS)
        resumed_opt.load_state_dict(state)
        norm1 = update(model, optimizer)
        norm2 = update(resumed, resumed_opt)
        self.assertAlmostEqual(norm1, norm2, places=10)
        for original, restored in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(original, restored, atol=0, rtol=0)
        self.assertTrue(all(p.dtype == torch.float32 for p in resumed_opt.masters.values()))

    def test_token_schedule(self):
        schedule = {"peak_lr": 1e-5, "min_lr": 1e-6, "warmup_fraction": .05}
        self.assertEqual(token_lr(0, 128000000, schedule), 0)
        self.assertAlmostEqual(token_lr(6400000, 128000000, schedule), 1e-5)
        self.assertAlmostEqual(token_lr(128000000, 128000000, schedule), 1e-6)


if __name__ == "__main__":
    unittest.main()
