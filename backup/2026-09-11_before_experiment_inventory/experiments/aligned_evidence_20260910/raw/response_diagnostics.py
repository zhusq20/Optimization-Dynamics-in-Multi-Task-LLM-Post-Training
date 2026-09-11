"""Observe stopping and the IFBench report's repetition heuristic without changing rewards."""

from __future__ import annotations

import zlib
from collections import Counter
from statistics import mean

from slime.utils.types import Sample


def response_diagnostics(samples, *, teacher_scores=False):
    if not samples:
        return {}
    completed = repetitive = completed_nonrepetitive = 0
    terminal_tokens = Counter()
    advantages = []
    for sample in samples:
        text = sample.response or ""
        tail = text[-10000:].encode("utf-8")
        compression = len(tail) / len(zlib.compress(tail, 9))
        repeated = len(text) > 10000 and compression > 10
        ended = sample.status == Sample.Status.COMPLETED
        completed += ended
        repetitive += repeated
        completed_nonrepetitive += ended and not repeated
        terminal = int(sample.tokens[-1]) if sample.tokens and sample.response_length > 0 else None
        record = {"compression_tail_ratio": compression, "repetitive": repeated,
                  "completed_nonrepetitive": ended and not repeated, "terminal_token_id": terminal}
        if ended and terminal is not None:
            terminal_tokens[terminal] += 1
            if (teacher_scores and not (sample.metadata or {}).get("mopd_failure_kind")
                    and sample.rollout_log_probs is not None
                    and (sample.loss_mask is None or sample.loss_mask[-1])):
                from slime_plugins.m2rl.opd import _teacher_log_probs

                teacher = float(_teacher_log_probs(sample.reward, sample.response_length)[-1])
                student = float(sample.rollout_log_probs[-1])
                record.update(terminal_teacher_logp=teacher, terminal_student_logp=student,
                              terminal_advantage=teacher - student)
                advantages.append(teacher - student)
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["mopd_response_diagnostics"] = record
    metrics = {"completion_rate": completed / len(samples), "repetition_rate": repetitive / len(samples),
               "completed_nonrepetitive_rate": completed_nonrepetitive / len(samples),
               "terminal_advantage_observed_responses": len(advantages)}
    for token, count in terminal_tokens.items():
        metrics[f"completed_terminal_token/{token}/response_fraction"] = count / len(samples)
    if advantages:
        metrics.update(terminal_advantage_mean=mean(advantages), terminal_advantage_min=min(advantages),
                       terminal_advantage_negative_fraction=sum(value < 0 for value in advantages) / len(advantages))
    return metrics
