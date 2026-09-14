"""Student TopK distillation, sampled PG, and the legacy teacher Top64 loss."""

from __future__ import annotations

from typing import Any

import torch

from .topk import student_topk_size, teacher_on_student_topk, teacher_student_topk_intersection


def teacher_topk(response: dict[str, Any], response_length: int, k: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    rows = response.get("meta_info", {}).get("input_top_logprobs")
    if response_length <= 0 or not rows or len(rows) < response_length:
        raise ValueError("teacher top-k payload is missing response-aligned input_top_logprobs")
    rows = rows[-response_length:]
    if any(row is None or len(row) != k for row in rows):
        raise ValueError(f"teacher must return exactly {k} entries at every response position")
    ids = torch.tensor([[entry[1] for entry in row] for row in rows], dtype=torch.long)
    log_probs = torch.tensor([[entry[0] for entry in row] for row in rows], dtype=torch.float32)
    if not torch.isfinite(log_probs).all() or (log_probs > 0).any() or (ids < 0).any():
        raise ValueError("teacher top-k IDs and log-probabilities must be valid and finite")
    if (ids.sort(dim=-1).values.diff(dim=-1) == 0).any():
        raise ValueError("teacher top-k IDs must be unique at each position")
    return ids, log_probs


def corrected_reverse_kl(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    """Sum p(log p - log q) - p + q over the teacher support, without renormalizing."""
    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.ndim != 2:
        raise ValueError("corrected reverse KL requires aligned [positions, support] log-probabilities")
    log_p = student_log_probs.float()
    log_q = teacher_log_probs.detach().float()
    p, q = log_p.exp(), log_q.exp()
    return (p * (log_p - log_q) - p + q).sum(dim=-1)


def student_topk_advantage(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    """Open-MOPD's detached, per-prefix mass-normalized token advantages.

    Both inputs are full-vocabulary log-probabilities gathered at the saved
    student support. Only the weighting distribution is renormalized on that
    support; the teacher/student log-ratio retains full-vocabulary probabilities.
    """
    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.ndim != 2:
        raise ValueError("Student top-k requires aligned [positions, support] log-probabilities")
    with torch.no_grad():
        log_p, log_q = student_log_probs.float(), teacher_log_probs.float()
        return log_p.softmax(dim=-1) * (log_q - log_p)


class _VocabParallelSelectedLogProbs(torch.autograd.Function):
    """Full-vocabulary normalization with only the selected columns replicated across TP."""

    @staticmethod
    def forward(ctx, logits, ids, group, rank, vocab_size):
        width = logits.shape[-1]
        start = rank * width
        local_logits = logits.float().clone()
        valid_width = max(0, min(width, vocab_size - start))
        local_logits[..., valid_width:] = -torch.inf
        maximum = local_logits.max(dim=-1).values
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
        exp = (local_logits - maximum.unsqueeze(-1)).exp()
        denominator = exp.sum(dim=-1)
        torch.distributed.all_reduce(denominator, group=group)
        local_ids = ids - start
        owned = (local_ids >= 0) & (local_ids < valid_width)
        indices = local_ids.clamp(0, width - 1)
        selected = local_logits.gather(-1, indices).masked_fill(~owned, 0)
        torch.distributed.all_reduce(selected, group=group)
        ctx.save_for_backward(exp / denominator.unsqueeze(-1), indices, owned)
        return selected - maximum.unsqueeze(-1) - denominator.log().unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, indices, owned = ctx.saved_tensors
        gradient = -probabilities * grad_output.sum(dim=-1, keepdim=True)
        gradient.scatter_add_(-1, indices, grad_output * owned)
        return gradient, None, None, None, None


def selected_log_probs(logits: torch.Tensor, ids: torch.Tensor, *, vocab_size: int | None = None) -> torch.Tensor:
    from megatron.core import mpu

    size = mpu.get_tensor_model_parallel_world_size()
    vocab_size = vocab_size or logits.shape[-1] * size
    if size == 1:
        return logits[..., :vocab_size].float().log_softmax(dim=-1).gather(-1, ids)
    return _VocabParallelSelectedLogProbs.apply(
        logits, ids, mpu.get_tensor_model_parallel_group(), mpu.get_tensor_model_parallel_rank(), vocab_size
    )


def teacher_topk_loss(args, batch, logits, sum_of_sample_mean):
    from slime.backends.megatron_utils.loss import get_responses

    losses, student_masses, teacher_masses = [], [], []
    for (response_logits, _), metadata, mask in zip(
        get_responses(
            logits.float(),
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            apply_temperature=False,
        ),
        batch["metadata"],
        batch["loss_masks"],
        strict=True,
    ):
        ids = torch.as_tensor(metadata["teacher_topk_ids"], device=logits.device, dtype=torch.long)
        teacher = torch.as_tensor(metadata["teacher_topk_log_probs"], device=logits.device, dtype=torch.float32)
        student = selected_log_probs(response_logits, ids, vocab_size=args.vocab_size)
        positions = corrected_reverse_kl(student, teacher)
        valid = torch.as_tensor(mask, device=logits.device)
        metadata["mopd_teacher_loss"] = float((positions.detach() * valid).sum() / valid.sum().clamp_min(1))
        losses.append(positions)
        student_masses.append(student.detach().exp().sum(-1))
        teacher_masses.append(teacher.exp().sum(-1))
    loss = sum_of_sample_mean(torch.cat(losses))
    return loss, {"loss": loss.detach(), "teacher_top64_corrected_reverse_kl": loss.detach(),
                  "student_top64_retained_mass": sum_of_sample_mean(torch.cat(student_masses)).detach(),
                  "teacher_top64_retained_mass": sum_of_sample_mean(torch.cat(teacher_masses)).detach()}


def student_topk_loss(args, batch, logits, sum_of_sample_mean):
    """Student-selected TopK PG surrogate without PPO or domain reweighting.

    Rollout fixes the support and the teacher scores those same IDs. Intersection
    mode keeps only shared teacher/student TopK terms, without renormalizing
    the remaining weights. Each actor
    forward refreshes the student probabilities and the detached advantage.
    The caller's existing response/token reduction is applied unchanged.
    """
    from slime.backends.megatron_utils.loss import get_responses

    k = student_topk_size(args)
    intersection = args.mopd_loss == "topk_intersection"
    losses, log_ratios, student_masses, teacher_masses = [], [], [], []
    intersection_sizes, intersection_empty, intersection_weights, intersection_masses = [], [], [], []
    for (response_logits, _), metadata, mask in zip(
        get_responses(
            logits.float(), args=args, unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
            apply_temperature=False,
        ),
        batch["metadata"], batch["loss_masks"], strict=True,
    ):
        ids = torch.as_tensor(metadata["student_topk_ids"], device=logits.device, dtype=torch.long)
        teacher = torch.as_tensor(metadata["teacher_on_student_topk_log_probs"], device=logits.device, dtype=torch.float32)
        if metadata.get("student_topk_k", k) != k or ids.shape != (response_logits.shape[0], k):
            raise ValueError(f"Saved student Top{k} support must align with every response logit")
        student = selected_log_probs(response_logits, ids, vocab_size=args.vocab_size)
        advantage = student_topk_advantage(student, teacher)
        if intersection:
            shared = torch.as_tensor(metadata["student_teacher_topk_mask"], device=logits.device)
            if shared.dtype != torch.bool or shared.shape != ids.shape:
                raise ValueError("TopK intersection mask must be boolean and align with the saved student support")
            advantage = advantage.masked_fill(~shared, 0)
            intersection_sizes.append(shared.float().sum(-1))
            intersection_empty.append((~shared.any(-1)).float())
            intersection_weights.append((student.detach().softmax(-1) * shared).sum(-1))
            intersection_masses.append((student.detach().exp() * shared).sum(-1))
        losses.append(-(advantage * student).sum(dim=-1))
        log_ratio = -advantage.sum(dim=-1)
        log_ratios.append(log_ratio)
        valid = torch.as_tensor(mask, device=logits.device)
        metadata["mopd_teacher_loss"] = float((log_ratio * valid).sum() / valid.sum().clamp_min(1))
        student_masses.append(student.detach().exp().sum(dim=-1))
        teacher_masses.append((teacher.detach().exp() * shared).sum(-1) if intersection else teacher.detach().exp().sum(-1))
    loss = sum_of_sample_mean(torch.cat(losses))
    prefix = f"student_teacher_top{k}_intersection" if intersection else f"student_top{k}"
    teacher_mass_name = f"teacher_on_student_top{k}_{'intersection' if intersection else 'retained'}_mass"
    metrics = {
        "loss": loss.detach(),
        f"{prefix}_surrogate": loss.detach(),
        f"{prefix}_normalized_logratio": sum_of_sample_mean(torch.cat(log_ratios)).detach(),
        f"student_top{k}_retained_mass": sum_of_sample_mean(torch.cat(student_masses)).detach(),
        teacher_mass_name: sum_of_sample_mean(torch.cat(teacher_masses)).detach(),
    }
    if intersection:
        for name, values in (("size", intersection_sizes), ("empty_fraction", intersection_empty),
                             ("retained_weight", intersection_weights), ("student_mass", intersection_masses)):
            metrics[f"{prefix}_{name}"] = sum_of_sample_mean(torch.cat(values)).detach()
    return loss, metrics


def post_process_rewards(args, samples, **kwargs):
    del kwargs
    for sample in samples:
        metadata = dict(getattr(sample, "train_metadata", None) or {})
        # Legacy reference-bank callers may omit this field; CLI runs always
        # set it explicitly, with student_topk as the new command-line default.
        loss = getattr(args, "mopd_loss", "teacher_topk")
        if loss in {"student_topk", "topk_intersection"}:
            k = student_topk_size(args)
            if loss == "topk_intersection":
                ids, log_probs, shared = teacher_student_topk_intersection(sample, k=k)
                metadata["student_teacher_topk_mask"] = shared
            else:
                ids, log_probs = teacher_on_student_topk(sample, k=k)
            metadata.update(student_topk_k=k, student_topk_ids=ids, teacher_on_student_topk_log_probs=log_probs)
        elif loss == "teacher_topk":
            ids, log_probs = teacher_topk(sample.reward, sample.response_length)
            metadata.update(teacher_topk_ids=ids, teacher_topk_log_probs=log_probs)
        elif loss == "sampled_reverse_kl":
            from slime_plugins.m2rl.opd import _teacher_log_probs

            metadata["teacher_sampled_log_probs"] = _teacher_log_probs(sample.reward, sample.response_length)
        else:
            raise ValueError(f"Unknown MOPD loss {loss!r}")
        sample.train_metadata = metadata
    return [0.0] * len(samples), [0.0] * len(samples)


def sampled_pg_loss(args, batch, logits, sum_of_sample_mean):
    """The paper's detached sampled-token PG objective on fresh responses."""
    from slime.backends.megatron_utils.loss import get_responses

    losses, ratios, clipped = [], [], []
    for (response_logits, response_tokens), metadata, mask in zip(
        get_responses(logits.float(), args=args, unconcat_tokens=batch["unconcat_tokens"],
                      total_lengths=batch["total_lengths"], response_lengths=batch["response_lengths"],
                      apply_temperature=False), batch["metadata"], batch["loss_masks"], strict=True,
    ):
        student = selected_log_probs(response_logits, response_tokens.unsqueeze(-1), vocab_size=args.vocab_size).squeeze(-1)
        teacher = torch.as_tensor(metadata["teacher_sampled_log_probs"], device=logits.device, dtype=torch.float32)
        advantage = (teacher - student).detach()
        cap = float(getattr(args, "mopd_pg_advantage_clip", 0.0) or 0.0)
        bounded = advantage.clamp(-cap, cap) if cap > 0 else advantage
        losses.append(-bounded * student)
        ratios.append(-advantage)
        clipped.append((advantage != bounded).float())
        valid = torch.as_tensor(mask, device=logits.device)
        metadata["mopd_teacher_loss"] = float((-advantage * valid).sum() / valid.sum().clamp_min(1))
    loss = sum_of_sample_mean(torch.cat(losses))
    return loss, {"loss": loss.detach(), "sampled_reverse_kl_logratio": sum_of_sample_mean(torch.cat(ratios)).detach(),
                  "pg_advantage_clipped_fraction": sum_of_sample_mean(torch.cat(clipped)).detach()}


def paper_loss(args, batch, logits, sum_of_sample_mean):
    function = {"student_topk": student_topk_loss, "topk_intersection": student_topk_loss, "teacher_topk": teacher_topk_loss,
                "sampled_reverse_kl": sampled_pg_loss}[args.mopd_loss]
    return function(args, batch, logits, sum_of_sample_mean)


def score_topk(
    logits,
    *,
    args,
    unconcat_tokens,
    total_lengths,
    response_lengths,
    metadata,
    loss_masks,
    with_entropy=False,
    non_loss_data=True,
):
    del with_entropy, non_loss_data
    teacher_topk_loss(
        args,
        {
            "unconcat_tokens": unconcat_tokens,
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "metadata": metadata,
            "loss_masks": loss_masks,
        },
        logits,
        lambda values: values.sum(),
    )
    return torch.empty(0, device=logits.device), {
        "teacher_loss": [torch.tensor(record["mopd_teacher_loss"], device=logits.device) for record in metadata]
    }
