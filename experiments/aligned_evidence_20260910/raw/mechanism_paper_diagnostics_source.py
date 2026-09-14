"""Exact local losses and parameter measurements for the three paper studies.

Every function operates on fixed response prefixes. Callers restore model and
optimizer state before each proposed update and stream the resulting records to
W&B and JSONL. Full-vocabulary measurements are local probes, not online runs.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
import numpy as np

from .loss import corrected_reverse_kl, student_topk_advantage
from .topk import STUDENT_TOPK
from .sampler import PAPER_REDUCTIONS


def prefix_weights(masks, domains, reduction="domain_response", domain_weights=None):
    """Return position weights implementing the paper's three exact reductions."""
    if reduction not in PAPER_REDUCTIONS or len(masks) != len(domains) or not masks:
        raise ValueError("A reduction needs aligned nonempty response masks and domains")
    tasks = tuple(dict.fromkeys(domains))
    weights = dict.fromkeys(tasks, 1.0 / len(tasks)) if domain_weights is None else dict(domain_weights)
    if set(weights) != set(tasks) or any(w < 0 for w in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Domain weights must match represented domains and sum to one")
    masks = [torch.as_tensor(mask, dtype=torch.float32) for mask in masks]
    lengths = [float(mask.sum()) for mask in masks]
    if any(length <= 0 for length in lengths):
        raise ValueError("Local probes require a nonempty valid response for every sampled prompt")
    domain_tokens = {task: sum(n for n, domain in zip(lengths, domains, strict=True) if domain == task) for task in tasks}
    domain_counts = {task: domains.count(task) for task in tasks}
    result = []
    for mask, length, task in zip(masks, lengths, domains, strict=True):
        if reduction == "global_token":
            factor = 1.0 / sum(lengths)
        elif reduction == "domain_token":
            factor = weights[task] / domain_tokens[task]
        else:
            factor = weights[task] / domain_counts[task] / length
        result.append(mask * factor)
    return torch.cat(result)


def local_distillation_loss(student_logits, teacher_log_probs, *, loss, weights, action_ids=None, advantage_clip=None, topk=None):
    """Differentiate one loss with frozen full-vocabulary teacher probabilities.

    PG actions must be freshly drawn by the caller at the fixed prefixes. Reuse
    the same action IDs for paired teacher comparisons. Advantages are detached;
    no PPO ratio or additional policy clipping is inserted into this local loss.
    """
    log_p = student_logits.float().log_softmax(dim=-1)
    log_q = teacher_log_probs.detach().to(log_p)
    if log_p.shape != log_q.shape or log_p.ndim != 2:
        raise ValueError("Local losses need aligned [prefixes, vocabulary] distributions")
    if loss in {"full_vocab", "full_reverse_kl"}:
        positions = corrected_reverse_kl(log_p, log_q)
    elif loss == "teacher_topk":
        teacher_selected, ids = log_q.topk(min(topk or 64, log_q.shape[-1]), dim=-1)
        positions = corrected_reverse_kl(log_p.gather(-1, ids), teacher_selected)
    elif loss in {"student_topk", "topk_intersection"}:
        ids = log_p.detach().topk(min(topk or STUDENT_TOPK, log_p.shape[-1]), dim=-1).indices
        selected_p, selected_q = log_p.gather(-1, ids), log_q.gather(-1, ids)
        advantage = student_topk_advantage(selected_p, selected_q)
        if loss == "topk_intersection":
            teacher_ids = log_q.topk(ids.shape[-1], dim=-1).indices.sort(-1).values
            matches = torch.searchsorted(teacher_ids, ids.contiguous()).clamp_max(ids.shape[-1] - 1)
            advantage = advantage.masked_fill(teacher_ids.gather(-1, matches) != ids, 0)
        positions = -(advantage * selected_p).sum(dim=-1)
    elif loss in {"sampled_pg", "sampled_reverse_kl"}:
        if action_ids is None:
            raise ValueError("Local PG requires fresh action IDs sampled at these fixed prefixes")
        ids = torch.as_tensor(action_ids, device=log_p.device, dtype=torch.long).reshape(-1, 1)
        selected_p, selected_q = log_p.gather(-1, ids).squeeze(-1), log_q.gather(-1, ids).squeeze(-1)
        advantage = (selected_q - selected_p).detach()
        if advantage_clip is not None and advantage_clip > 0:
            advantage = advantage.clamp(-advantage_clip, advantage_clip)
        positions = -advantage * selected_p
    else:
        raise ValueError(f"Unknown local distillation loss {loss!r}")
    weights = torch.as_tensor(weights, device=log_p.device, dtype=torch.float32)
    if weights.shape != positions.shape:
        raise ValueError("Position weights must align with the fixed prefix bank")
    return (positions * weights).sum()


def js_divergence(log_p, log_q, weights=None):
    """Full-vocabulary Jensen-Shannon divergence in natural logarithms."""
    p, q = log_p.double(), log_q.double()
    mixture = torch.logaddexp(p, q) - math.log(2.0)
    positions = 0.5 * ((p.exp() * (p - mixture)).sum(-1) + (q.exp() * (q - mixture)).sum(-1))
    if weights is None:
        return float(positions.mean())
    return float((positions * torch.as_tensor(weights, device=positions.device, dtype=positions.dtype)).sum())


def flat_values(values):
    """One CPU FP32 coordinate order, shared by gradients and master deltas."""
    if isinstance(values, Mapping):
        values = list(values.values())
    if torch.is_tensor(values):
        return values.detach().reshape(-1).float().cpu()
    return torch.cat([value.detach().reshape(-1).float().cpu() for value in values])


def _energy(values):
    # FP64 accumulation without allocating a model-sized squared FP64 tensor.
    return float(np.einsum("i,i->", values, values, dtype=np.float64))


def _energy_count(values, target):
    """Exact number of largest coordinates reaching target energy, via partition."""
    start, stop, selected = 0, len(values), 0
    while stop - start > 64:
        middle = (start + stop) // 2
        values[start:stop].partition(middle - start)
        upper_energy = _energy(values[middle:stop])
        if upper_energy >= target:
            start = middle
        else:
            selected += stop - middle
            target -= upper_energy
            stop = middle
    ranked = np.sort(values[start:stop])[::-1].astype(np.float64)
    count = int(np.searchsorted(np.cumsum(ranked * ranked), target, side="left")) + 1
    return selected + min(count, stop - start)


def tensor_metrics(values, *, thresholds=(0.0, 1e-8, 1e-7, 1e-6), fractions=(0.01, 0.05, 0.1)):
    value = flat_values(values)
    # A single writable FP32 scratch vector supports exact selection. Partition
    # has linear expected work; sorting billions of parameters at every milestone
    # unnecessarily dominated the measurement cost.
    magnitudes = np.abs(value.numpy())
    total = _energy(magnitudes)
    output = {"parameters": value.numel(), "l2": math.sqrt(total)}
    for threshold in thresholds:
        output[f"sparsity_at_{threshold:g}"] = float(np.count_nonzero(magnitudes <= threshold)) / value.numel()
    if total == 0.0:
        output["energy90_fraction"] = None
        output.update({f"energy_at_{fraction:g}": None for fraction in fractions})
        return output
    output["energy90_fraction"] = _energy_count(magnitudes, total * 0.9) / value.numel()
    counts = [max(1, math.ceil(value.numel() * fraction)) for fraction in fractions]
    magnitudes.partition([value.numel() - count for count in counts])
    for fraction, count in zip(fractions, counts, strict=True):
        output[f"energy_at_{fraction:g}"] = _energy(magnitudes[-count:]) / total
    return output


sparsity_metrics = tensor_metrics


def support_overlap(left, right, fraction=None, *, threshold=None):
    """Fixed-fraction or absolute-threshold support with deterministic tie rules.

    Stable descending order breaks ties by coordinate index. Zero vectors have
    empty support; both empty gives undefined overlap, as in the paper appendix.
    The random baseline is the expected intersection/union ratio for equal-size
    independent selections in this coordinate population.
    """
    a, b = flat_values(left), flat_values(right)
    if a.shape != b.shape or (fraction is None) == (threshold is None):
        raise ValueError("Compare aligned coordinates with one support selection rule")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("Support fraction must lie in (0, 1]")
    supports, ties = [], []
    for value in (a, b):
        magnitude = np.abs(value.numpy())
        if threshold is not None:
            selected = magnitude > threshold
            ties.append(False)
        else:
            selected = np.zeros(magnitude.shape, dtype=bool)
            count = max(1, math.ceil(value.numel() * fraction))
            if np.any(magnitude):
                scratch = magnitude.copy()
                boundary = value.numel() - count
                scratch.partition(boundary)
                cutoff = scratch[boundary]
                del scratch
                selected = magnitude > cutoff
                needed = count - int(np.count_nonzero(selected))
                equal = np.flatnonzero(magnitude == cutoff)
                selected[equal[:needed]] = True
                ties.append(len(equal) > needed)
            else:
                ties.append(True)
        supports.append(selected)
    count_a, count_b = (int(np.count_nonzero(value)) for value in supports)
    intersection = int(np.count_nonzero(supports[0] & supports[1]))
    union = count_a + count_b - intersection
    expected_intersection = count_a * count_b / a.numel()
    expected_union = count_a + count_b - expected_intersection
    norms = math.sqrt(_energy(a.numpy()) * _energy(b.numpy()))
    return {
        "jaccard": intersection / union if union else None,
        "cosine": float(np.einsum("i,i->", a.numpy(), b.numpy(), dtype=np.float64)) / norms if norms else None,
        "random_jaccard": expected_intersection / expected_union if expected_union else None,
        "selected_left": count_a, "selected_right": count_b,
        "tie_dominated": any(ties),
        "tie_rule": "stable_coordinate_index",
    }


def normalization_decomposition(response_gradients, lengths):
    """Exact within-domain length-gradient identity on fixed response gradients."""
    gradients = torch.stack([flat_values(value) for value in response_gradients]).double()
    lengths = torch.as_tensor(lengths, dtype=torch.float64)
    response_mean = gradients.mean(0)
    covariance = ((lengths - lengths.mean()).unsqueeze(-1) * (gradients - response_mean)).mean(0)
    token_mean = (gradients * lengths.unsqueeze(-1)).sum(0) / lengths.sum()
    residual = token_mean - response_mean - covariance / lengths.mean()
    return {"response_mean": response_mean, "token_mean": token_mean, "covariance": covariance,
            "residual_l2": float(residual.norm()), "mean_length": float(lengths.mean())}
