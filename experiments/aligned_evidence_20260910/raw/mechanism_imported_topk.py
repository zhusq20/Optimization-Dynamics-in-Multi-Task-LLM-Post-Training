"""Student-selected support and position-aligned SGLang teacher scores."""

from __future__ import annotations

import torch

STUDENT_TOPK = 16
# SGLang accepts one ID list per request, shared by all scored positions.
# Keep the same maximum ID-union size for both Top16 and Top64.
TEACHER_SCORE_CHUNK = 32
TEACHER_SCORE_MAX_IDS = TEACHER_SCORE_CHUNK * STUDENT_TOPK


def uses_student_topk(args):
    return bool(getattr(args, "mopd_enabled", False)) and getattr(args, "mopd_loss", None) in {
        "student_topk", "topk_intersection"
    }


def student_topk_size(args):
    k = getattr(args, "mopd_topk", STUDENT_TOPK)
    if k not in (16, 64):
        raise ValueError("Student TopK supports --mopd-topk 16 or 64")
    return k


def teacher_score_chunk_size(k):
    return min(TEACHER_SCORE_CHUNK, max(1, TEACHER_SCORE_MAX_IDS // k))


def read_topk_rows(rows, length, *, k=STUDENT_TOPK, source="student"):
    if length <= 0 or rows is None or len(rows) < length:
        raise ValueError(f"{source} top-k payload is missing response-aligned rows")
    rows = rows[-length:]
    if any(row is None or len(row) != k for row in rows):
        raise ValueError(f"{source} must return exactly {k} entries at every response position")
    ids = torch.tensor([[entry[1] for entry in row] for row in rows], dtype=torch.long)
    log_probs = torch.tensor([[entry[0] for entry in row] for row in rows], dtype=torch.float32)
    if (ids < 0).any() or not torch.isfinite(log_probs).all() or (log_probs > 0).any():
        raise ValueError(f"{source} top-k IDs and log-probabilities must be valid and finite")
    if (ids.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError(f"{source} top-k IDs must be unique at each position")
    return ids, log_probs


def append_student_topk(sample, meta_info, new_length, *, k=STUDENT_TOPK):
    if not new_length:
        return
    ids, log_probs = read_topk_rows(meta_info.get("output_top_logprobs"), new_length, k=k)
    metadata = dict(sample.train_metadata or {})
    previous_ids = metadata.get("student_topk_ids", [])
    previous_log_probs = metadata.get("student_topk_log_probs", [])
    previous_length = sample.response_length - new_length
    if metadata.get("student_topk_k", k) != k:
        raise ValueError("Cannot change student TopK size within a response")
    if len(previous_ids) != previous_length or len(previous_log_probs) != previous_length:
        raise ValueError(f"Student Top{k} support does not align with the previously generated response")
    metadata["student_topk_k"] = k
    metadata["student_topk_ids"] = list(previous_ids) + ids.tolist()
    metadata["student_topk_log_probs"] = list(previous_log_probs) + log_probs.tolist()
    sample.train_metadata = metadata


def student_support(sample, *, k=STUDENT_TOPK):
    metadata = sample.train_metadata or {}
    ids = torch.as_tensor(metadata.get("student_topk_ids", []), dtype=torch.long)
    if metadata.get("student_topk_k", k) != k or ids.shape != (sample.response_length, k) or sample.response_length <= 0:
        raise ValueError(f"Student Top{k} IDs must be collected at every rollout response position before teacher scoring")
    if (ids < 0).any() or (ids.sort(-1).values.diff(dim=-1) == 0).any():
        raise ValueError(f"Student Top{k} IDs must be nonnegative and unique at each position")
    return ids


def select_teacher_rows(response, ids, response_tokens, *, k=STUDENT_TOPK):
    """Gather each position's IDs from a chunk's union, independent of return order."""
    count = len(response_tokens)
    meta = response.get("meta_info", {})
    rows = meta.get("input_token_ids_logprobs")
    sampled = meta.get("input_token_logprobs")
    if not count or rows is None or len(rows) < count or sampled is None or len(sampled) < count:
        raise ValueError("Teacher is missing position-aligned input_token_ids_logprobs or input_token_logprobs")
    sampled = sampled[-count:]
    if [entry[1] for entry in sampled] != list(response_tokens):
        raise ValueError("Teacher selected-token scores do not align with the original response token IDs")
    selected = []
    for row, wanted in zip(rows[-count:], ids, strict=True):
        if row is None:
            raise ValueError("Teacher returned an empty selected-token scoring row")
        by_id = {entry[1]: entry for entry in row}
        if len(by_id) != len(row) or any(token not in by_id for token in wanted):
            raise ValueError("Teacher selected-token scoring row has duplicate or missing student IDs")
        selected.append([[by_id[token][0], token] for token in wanted])
    read_topk_rows(selected, count, k=k, source="teacher on student")
    return selected, sampled


def teacher_on_student_topk(sample, *, k=STUDENT_TOPK):
    ids = student_support(sample, k=k)
    scored_ids, log_probs = read_topk_rows(
        sample.reward.get("meta_info", {}).get("input_token_ids_logprobs"),
        sample.response_length,
        k=k,
        source="teacher on student",
    )
    if not torch.equal(ids, scored_ids):
        raise ValueError(f"Teacher targets must use exactly the saved student Top{k} IDs in the same order")
    return ids, log_probs


def teacher_student_topk_intersection(sample, *, k=STUDENT_TOPK):
    """Align native teacher TopK to frozen student IDs, with a mask for missing IDs.

    Non-intersecting entries carry finite zero placeholders, never teacher
    targets. The caller must mask their contributions after weighting on the
    original student support. An empty intersection is valid.
    """
    ids = student_support(sample, k=k)
    meta = sample.reward.get("meta_info", {})
    sampled = meta.get("input_token_logprobs")
    length = sample.response_length
    if sampled is None or len(sampled) < length or [entry[1] for entry in sampled[-length:]] != sample.tokens[-length:]:
        raise ValueError("Teacher TopK scores do not align with the original response token IDs")
    teacher_ids, teacher_log_probs = read_topk_rows(
        meta.get("input_top_logprobs"), length, k=k, source="teacher"
    )
    sorted_ids, order = teacher_ids.sort(dim=-1)
    indices = torch.searchsorted(sorted_ids, ids.contiguous()).clamp_max(k - 1)
    mask = sorted_ids.gather(-1, indices) == ids
    aligned = teacher_log_probs.gather(-1, order.gather(-1, indices)).masked_fill(~mask, 0)
    return ids, aligned, mask
