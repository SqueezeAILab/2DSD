"""Online temperature-1 full-vocabulary forward KL(teacher || student) from a frozen teacher."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def full_forward_kl(student_logits, teacher_logits):
    """Full-vocabulary forward KL with a frozen temperature-1 teacher."""
    log_p = teacher_logits.detach().float().log_softmax(-1)
    log_q = student_logits.float().log_softmax(-1)
    return (log_p.exp() * (log_p - log_q)).sum(-1)


def chunked_full_kd_sum(student_head, student_hidden, teacher_head, teacher_hidden,
                        chunk_size=128, checkpoint_head=True):
    """Sum of per-position KL. The teacher head runs without grad inside each checkpointed chunk of
    positions, which bounds the vocabulary-sized saved activations."""
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(student_hidden), chunk_size):
        stop = min(start + chunk_size, len(student_hidden))

        def project(student, teacher):
            with torch.no_grad():
                target = teacher_head(teacher)
            return full_forward_kl(student_head(student), target).sum()

        args = student_hidden[start:stop], teacher_hidden[start:stop]
        if checkpoint_head and torch.is_grad_enabled():
            value = checkpoint(project, *args, use_reentrant=False)
        else:
            value = project(*args)
        total = total + value
    return total

