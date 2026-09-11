#!/usr/bin/env python3
"""Run a bounded fixed-prefix AdamW probe and measure the realized BF16 writeback."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
import yaml


CHUNK = 2_000_000
EXPECTED_PARAMETERS = 1_720_574_976
EXPECTED_TENSORS = 310
METRIC_THRESHOLDS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def status(output: Path, stage: str, **fields) -> None:
    atomic_json(output / "status.json", {"at_utc": utc(), "stage": stage, **fields})


def corrected_reverse_kl(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.ndim != 2:
        raise ValueError("Corrected reverse KL needs aligned [positions, support] tensors")
    log_p = student_log_probs.float()
    log_q = teacher_log_probs.detach().float()
    p, q = log_p.exp(), log_q.exp()
    return (p * (log_p - log_q) - p + q).sum(dim=-1)


def student_topk_advantage(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    if student_log_probs.shape != teacher_log_probs.shape or student_log_probs.ndim != 2:
        raise ValueError("Student TopK needs aligned [positions, support] tensors")
    with torch.no_grad():
        log_p, log_q = student_log_probs.float(), teacher_log_probs.float()
        return log_p.softmax(dim=-1) * (log_q - log_p)


def local_loss(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    loss: str,
    weight: float,
    action_id: int,
    topk: int,
    advantage_clip: float,
) -> torch.Tensor:
    log_p = student_logits.float().log_softmax(dim=-1)
    log_q = teacher_log_probs.detach().to(log_p)
    if log_p.shape != log_q.shape or log_p.shape[0] != 1:
        raise ValueError("One fixed prefix needs aligned [1, vocabulary] distributions")
    if loss == "full_vocab":
        value = corrected_reverse_kl(log_p, log_q)[0]
    elif loss == "student_topk":
        ids = log_p.detach().topk(min(topk, log_p.shape[-1]), dim=-1).indices
        selected_p, selected_q = log_p.gather(-1, ids), log_q.gather(-1, ids)
        value = -(student_topk_advantage(selected_p, selected_q) * selected_p).sum()
    elif loss == "sampled_pg":
        ids = torch.tensor([[action_id]], dtype=torch.long, device=log_p.device)
        selected_p = log_p.gather(-1, ids).squeeze()
        selected_q = log_q.gather(-1, ids).squeeze()
        advantage = (selected_q - selected_p).detach()
        if advantage_clip > 0:
            advantage = advantage.clamp(-advantage_clip, advantage_clip)
        value = -advantage * selected_p
    else:
        raise ValueError(f"Unknown local loss: {loss}")
    return value * weight


def js_divergence(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    p, q = log_p.double(), log_q.double()
    mixture = torch.logaddexp(p, q) - math.log(2.0)
    value = 0.5 * ((p.exp() * (p - mixture)).sum() + (q.exp() * (q - mixture)).sum())
    return float(value)


def new_stats() -> dict:
    return {
        "parameters": 0,
        "squared_l2": 0.0,
        "max_abs": 0.0,
        "above": {f"{threshold:g}": 0 for threshold in METRIC_THRESHOLDS},
    }


def add_stats(target: dict, values: torch.Tensor) -> None:
    array = values.detach().reshape(-1).float().cpu().numpy()
    magnitude = np.abs(array)
    target["parameters"] += array.size
    target["squared_l2"] += float(np.einsum("i,i->", array, array, dtype=np.float64))
    if array.size:
        target["max_abs"] = max(target["max_abs"], float(magnitude.max()))
    for threshold in METRIC_THRESHOLDS:
        target["above"][f"{threshold:g}"] += int(np.count_nonzero(magnitude > threshold))


def finish_stats(value: dict) -> dict:
    parameters = value["parameters"]
    return {
        "parameters": parameters,
        "l2": math.sqrt(value["squared_l2"]),
        "rms": math.sqrt(value["squared_l2"] / parameters) if parameters else 0.0,
        "max_abs": value["max_abs"],
        **{
            f"fraction_above_{key}": count / parameters if parameters else 0.0
            for key, count in value["above"].items()
        },
    }


def layer_name(name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return f"layer_{match.group(1)}" if match else "embedding_and_head"


def tensor_bytes(tensor: torch.Tensor):
    return memoryview(tensor.detach().contiguous().view(torch.uint8).numpy())


def gradient_l2(gradients: dict[str, torch.Tensor] | None) -> float:
    if gradients is None:
        return 0.0
    energy = 0.0
    for gradient in gradients.values():
        flat = gradient.reshape(-1)
        for start in range(0, flat.numel(), CHUNK):
            values = flat[start : start + CHUNK].numpy()
            energy += float(np.einsum("i,i->", values, values, dtype=np.float64))
    return math.sqrt(energy)


def adamw_bf16_writeback(
    parameters: dict,
    gradients: dict[str, torch.Tensor] | None,
    clip_grad: float,
    *,
    snapshot_digest: hashlib._Hash | None = None,
    progress=None,
):
    """Apply one virtual AdamW step and retain only the BF16 values actually written."""
    names = sorted(parameters)
    if gradients is not None and set(gradients) != set(names):
        raise ValueError("Gradient and optimizer coordinate populations differ")
    norm = gradient_l2(gradients)
    coefficient = min(1.0, clip_grad / (norm + 1e-6)) if clip_grad > 0 else 1.0
    totals = {name: new_stats() for name in (
        "raw_gradient", "clipped_gradient", "master_update", "bf16_writeback", "rounding_error"
    )}
    layer_totals = {}
    after = {}
    for ordinal, name in enumerate(names, 1):
        state = parameters[name]
        required = {"value", "model_value", "exp_avg", "exp_avg_sq", "lr", "betas", "eps", "weight_decay", "step"}
        if not required.issubset(state):
            raise ValueError(f"Incomplete AdamW state for {name}: {required - set(state)}")
        value = state["value"].reshape(-1)
        before_bf16 = state["model_value"].reshape(-1)
        first_moment = state["exp_avg"].reshape(-1)
        second_moment = state["exp_avg_sq"].reshape(-1)
        if value.dtype != torch.float32 or first_moment.dtype != torch.float32 or second_moment.dtype != torch.float32:
            raise ValueError(f"Optimizer tensors must be FP32: {name}")
        if before_bf16.dtype != torch.bfloat16 or not (
            value.shape == before_bf16.shape == first_moment.shape == second_moment.shape
        ):
            raise ValueError(f"Model/optimizer shape or precision mismatch: {name}")
        gradient = None if gradients is None else gradients[name].reshape(-1)
        if gradient is not None and (gradient.dtype != torch.float32 or gradient.shape != value.shape):
            raise ValueError(f"Gradient shape or precision mismatch: {name}")
        realized = torch.empty_like(before_bf16, device="cpu")
        beta1, beta2 = map(float, state["betas"])
        step = int(state["step"]) + 1
        correction1 = 1.0 - beta1 ** step if state.get("bias_correction", True) else 1.0
        correction2 = 1.0 - beta2 ** step if state.get("bias_correction", True) else 1.0
        lr = float(state["lr"])
        eps = float(state["eps"])
        decay = 1.0 - lr * float(state["weight_decay"])
        current_layer = layer_totals.setdefault(layer_name(name), new_stats())
        if snapshot_digest is not None:
            metadata = {key: state[key] for key in ("lr", "betas", "eps", "weight_decay", "step")}
            metadata["bias_correction"] = bool(state.get("bias_correction", True))
            snapshot_digest.update(name.encode() + b"\0" + json.dumps(metadata, sort_keys=True).encode() + b"\0")
        for start in range(0, value.numel(), CHUNK):
            stop = min(start + CHUNK, value.numel())
            old_master = value[start:stop]
            old_bf16 = before_bf16[start:stop]
            old_first = first_moment[start:stop]
            old_second = second_moment[start:stop]
            raw = torch.zeros_like(old_master) if gradient is None else gradient[start:stop]
            clipped = raw * coefficient
            new_first = old_first * beta1 + clipped * (1.0 - beta1)
            new_second = old_second * beta2 + clipped.square() * (1.0 - beta2)
            denominator = (new_second / correction2).sqrt() + eps
            new_master = old_master * decay - lr * (new_first / correction1) / denominator
            master_update = new_master - old_master
            new_bf16 = new_master.to(torch.bfloat16)
            bf16_update = new_bf16.float() - old_bf16.float()
            rounding_error = bf16_update - master_update
            realized[start:stop].copy_(new_bf16)
            for key, values in (
                ("raw_gradient", raw),
                ("clipped_gradient", clipped),
                ("master_update", master_update),
                ("bf16_writeback", bf16_update),
                ("rounding_error", rounding_error),
            ):
                add_stats(totals[key], values)
            add_stats(current_layer, bf16_update)
            if snapshot_digest is not None:
                for field, tensor in (
                    ("value", old_master),
                    ("model_value", old_bf16),
                    ("exp_avg", old_first),
                    ("exp_avg_sq", old_second),
                ):
                    snapshot_digest.update(field.encode() + b"\0")
                    snapshot_digest.update(tensor_bytes(tensor))
            del clipped, new_first, new_second, denominator, new_master, master_update, new_bf16, bf16_update, rounding_error
        after[name] = realized.reshape_as(state["model_value"])
        if progress is not None and (ordinal % 20 == 0 or ordinal == len(names)):
            progress(ordinal, len(names), name)
    metrics = {key: finish_stats(value) for key, value in totals.items()}
    metrics["bf16_writeback_by_layer"] = {
        name: finish_stats(value) for name, value in sorted(layer_totals.items())
    }
    metrics["gradient_l2_before_clipping"] = norm
    metrics["clip_coefficient"] = coefficient
    master_energy = totals["master_update"]["squared_l2"]
    bf16_energy = totals["bf16_writeback"]["squared_l2"]
    master_nonzero = totals["master_update"]["above"]["0"]
    bf16_nonzero = totals["bf16_writeback"]["above"]["0"]
    metrics["bf16_nonzero_survival_fraction"] = bf16_nonzero / master_nonzero if master_nonzero else None
    metrics["bf16_to_master_update_energy_ratio"] = bf16_energy / master_energy if master_energy else None
    return after, metrics


def compare_writebacks(parameters: dict, left: dict, right: dict) -> dict:
    thresholds = (0.0, 1e-5)
    accumulators = {
        f"{threshold:g}": {"left": 0, "right": 0, "intersection": 0, "union": 0}
        for threshold in thresholds
    }
    dot = left_energy = right_energy = difference_energy = 0.0
    parameters_seen = 0
    layer_supports = {}
    for name in sorted(parameters):
        layer = layer_supports.setdefault(layer_name(name), {
            "parameters": 0,
            "support": {str_key: {"left": 0, "right": 0, "intersection": 0, "union": 0} for str_key in accumulators},
        })
        before = parameters[name]["model_value"].reshape(-1)
        a_after, b_after = left[name].reshape(-1), right[name].reshape(-1)
        for start in range(0, before.numel(), CHUNK):
            stop = min(start + CHUNK, before.numel())
            origin = before[start:stop].float()
            a = a_after[start:stop].float() - origin
            b = b_after[start:stop].float() - origin
            a_np, b_np = a.numpy(), b.numpy()
            dot += float(np.einsum("i,i->", a_np, b_np, dtype=np.float64))
            left_energy += float(np.einsum("i,i->", a_np, a_np, dtype=np.float64))
            right_energy += float(np.einsum("i,i->", b_np, b_np, dtype=np.float64))
            difference = a_np - b_np
            difference_energy += float(np.einsum("i,i->", difference, difference, dtype=np.float64))
            parameters_seen += a.numel()
            layer["parameters"] += a.numel()
            for threshold in thresholds:
                left_support = np.abs(a_np) > threshold
                right_support = np.abs(b_np) > threshold
                target = accumulators[f"{threshold:g}"]
                counts = {
                    "left": int(np.count_nonzero(left_support)),
                    "right": int(np.count_nonzero(right_support)),
                    "intersection": int(np.count_nonzero(left_support & right_support)),
                    "union": int(np.count_nonzero(left_support | right_support)),
                }
                for key, count in counts.items():
                    target[key] += count
                    layer["support"][f"{threshold:g}"][key] += count
    output = {
        "parameters": parameters_seen,
        "cosine": dot / math.sqrt(left_energy * right_energy) if left_energy and right_energy else None,
        "l2_difference": math.sqrt(difference_energy),
        "support": {},
    }
    for key, value in accumulators.items():
        expected_intersection = value["left"] * value["right"] / parameters_seen
        expected_union = value["left"] + value["right"] - expected_intersection
        output["support"][key] = {
            **value,
            "jaccard": value["intersection"] / value["union"] if value["union"] else None,
            "random_jaccard": expected_intersection / expected_union if expected_union else None,
        }
    for key, value in output["support"].items():
        expected = sum(layer["support"][key]["left"] * layer["support"][key]["right"] / layer["parameters"] for layer in layer_supports.values())
        union = value["left"] + value["right"] - expected
        value["layer_random_expected_intersection"] = expected
        value["layer_random_jaccard_ratio_of_expectations"] = expected / union if union else None
    output["by_layer"] = layer_supports
    return output


def next_logits(model, tokens: list[int], device: str) -> torch.Tensor:
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[:, -1, :]
    return model.lm_head(hidden).float()


@torch.no_grad()
def next_log_probs(model, tokens: list[int], device: str) -> torch.Tensor:
    return next_logits(model, tokens, device).log_softmax(dim=-1).squeeze(0).cpu()


def load_prompt_rows(manifest_path: Path, tokenizer, tasks: tuple[str, ...], max_prompt_tokens: int):
    manifest = yaml.safe_load(manifest_path.read_text())
    sources = {row["name"]: row for row in manifest["sources"]}
    result = {}
    for task in tasks:
        path = Path(os.path.expandvars(sources[task]["path"]))
        if not path.is_absolute():
            path = manifest_path.parent / path
        candidates = []
        for index, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            prompt = json.loads(line)["prompt"]
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if 0 < len(tokens) <= max_prompt_tokens:
                candidates.append((index, prompt, tokens))
        if not candidates:
            raise ValueError(f"No {task} prompt fits the {max_prompt_tokens}-token bound")
        result[task] = candidates
    return result


def generate_bank(model, tokenizer, manifest_path: Path, tasks: tuple[str, ...], args):
    candidates = load_prompt_rows(manifest_path, tokenizer, tasks, args.max_prompt_tokens)
    rows = []
    for task_index, task in enumerate(tasks):
        rng = random.Random(args.seed + task_index * 100_003)
        index, prompt, prompt_ids = rng.choice(candidates[task])
        torch.manual_seed(args.seed + task_index * 1_000_003)
        torch.cuda.manual_seed_all(args.seed + task_index * 1_000_003)
        with torch.no_grad():
            generated = model.generate(
                torch.tensor([prompt_ids], dtype=torch.long, device=args.device),
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )[0].tolist()[len(prompt_ids) :]
        if not generated:
            raise ValueError(f"The {task} diagnostic prompt produced an empty response")
        positions = sorted(rng.sample(range(len(generated)), min(args.prefixes_per_response, len(generated))))
        student_scores = []
        actions = []
        action_rng = torch.Generator(device="cpu").manual_seed(args.seed + task_index * 10_000_019)
        for position in positions:
            score = next_log_probs(model, prompt_ids + generated[:position], args.device)
            student_scores.append(score)
            actions.append(int(torch.multinomial(score.exp(), 1, generator=action_rng)))
        rows.append({
            "task": task,
            "prompt_index": index,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_ids": prompt_ids,
            "response_ids": generated,
            "positions": positions,
            "actions": actions,
            "student_log_probs": student_scores,
        })
    return rows


def score_teachers(model_tokenizer, rows, tasks: tuple[str, ...], teachers_path: Path, device: str, output: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    routes = json.loads(teachers_path.read_text())
    scores = {}
    for ordinal, task in enumerate(tasks, 1):
        status(output, "loading_teacher", teacher=task, teacher_position=ordinal, teachers_total=len(tasks))
        route = routes[task]
        tokenizer = AutoTokenizer.from_pretrained(route["model_path"])
        if model_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError(f"Teacher {task} vocabulary differs from the student")
        suffix = tokenizer.encode(route.get("prompt_suffix", ""), add_special_tokens=False)
        teacher = AutoModelForCausalLM.from_pretrained(
            route["model_path"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to(device).eval()
        with torch.no_grad():
            scores[task] = [
                [
                    next_log_probs(
                        teacher,
                        row["prompt_ids"] + suffix + row["response_ids"][:position],
                        device,
                    )
                    for position in row["positions"]
                ]
                for row in rows
            ]
        del teacher, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    return scores, routes


def weighted_js(rows, left, right, domain_weights: dict[str, float]) -> float:
    counts = {task: sum(row["task"] == task for row in rows) for task in domain_weights}
    total = 0.0
    for row_index, row in enumerate(rows):
        weight = domain_weights[row["task"]] / counts[row["task"]] / len(row["positions"])
        for position_index in range(len(row["positions"])):
            total += weight * js_divergence(left[row_index][position_index], right[row_index][position_index])
    return total


def joint_gradient(model, rows, teacher_scores, domain_weights, loss, args, *, teacher=None, routed=False):
    model.zero_grad(set_to_none=True)
    counts = {task: sum(row["task"] == task for row in rows) for task in domain_weights}
    if routed and teacher not in domain_weights:
        raise ValueError("A routed branch must specify a teacher in the domain weights")
    objective_total = 0.0
    for row_index, row in enumerate(rows):
        if routed and row["task"] != teacher:
            continue
        domain_weight = 1.0 if routed else domain_weights[row["task"]]
        weight = domain_weight / counts[row["task"]] / len(row["positions"])
        for position_index, position in enumerate(row["positions"]):
            logits = next_logits(model, row["prompt_ids"] + row["response_ids"][:position], args.device)
            objective = local_loss(
                logits,
                teacher_scores[teacher or row["task"]][row_index][position_index].unsqueeze(0).to(args.device),
                loss=loss,
                weight=weight,
                action_id=row["actions"][position_index],
                topk=args.topk,
                advantage_clip=args.pg_advantage_clip,
            )
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite local objective")
            objective_total += float(objective.detach())
            objective.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            gradients[name] = torch.zeros(parameter.shape, dtype=torch.float32, device="cpu")
        else:
            gradients[name] = parameter.grad.detach().cpu().float()
    if not all(torch.isfinite(g).all() for g in gradients.values()):
        raise ValueError("Nonfinite local gradient")
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return gradients, objective_total


def public_bank(rows):
    return [
        {key: value for key, value in row.items() if key != "student_log_probs"}
        for row in rows
    ]


def flatten_scalars(prefix: str, value, output: dict) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            flatten_scalars(f"{prefix}/{key}" if prefix else str(key), item, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None:
        output[prefix] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--hf-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--domain-weights", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot-step", type=int, default=500)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prefixes-per-response", type=int, default=2)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--pg-advantage-clip", type=float, default=0.0)
    parser.add_argument("--losses", default="sampled_pg,student_topk,full_vocab")
    parser.add_argument("--wandb-project", default="iclr2027-mopd-dynamics")
    parser.add_argument("--wandb-entity", default="zsqzz")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tasks = tuple(item for item in args.tasks.split(",") if item)
    weights = tuple(float(item) for item in args.domain_weights.split(","))
    losses = tuple(item for item in args.losses.split(",") if item)
    if len(tasks) != len(weights) or any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("Domain weights must align with tasks and sum to one")
    if losses != ("sampled_pg", "student_topk", "full_vocab"):
        raise ValueError("This frozen pilot expects the three planned losses in a fixed order")
    domain_weights = dict(zip(tasks, weights, strict=True))
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": utc(),
        "job": args.job,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "precision": {
            "student_forward": "BF16",
            "teacher_forward": "BF16",
            "gradient_collection": "BF16 model gradients converted to CPU FP32",
            "optimizer_state": "saved FP32 master weights and Adam moments",
            "reported_model_update": "BF16(new master) minus saved BF16 model value",
        },
        "scope": "fixed-prefix local mechanism probe; no training checkpoint is mutated",
    }
    atomic_json(args.output / "probe_manifest.json", manifest)
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group="qwen3-bf16-local-update",
            name=args.job,
            mode=args.wandb_mode,
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        )
        (args.output / "wandb_run_id.txt").write_text(wandb_run.id + "\n")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        status(args.output, "loading_snapshot")
        snapshot_stat = args.snapshot.stat()
        snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True, mmap=True)
        if snapshot["step"] != args.snapshot_step or snapshot["model_precision"] != "torch.bfloat16":
            raise ValueError("Expected the requested-step BF16 online-model snapshot")
        if snapshot["optimizer_precision"] != "torch.float32" or len(snapshot["parameters"]) != EXPECTED_TENSORS:
            raise ValueError("Expected a complete FP32 AdamW snapshot")
        if not set(tasks).issubset(snapshot["tasks"]):
            raise ValueError(f"Requested tasks {tasks} are not contained in snapshot tasks {snapshot['tasks']}")

        status(args.output, "loading_student")
        tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_checkpoint, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model_parameters = dict(model.named_parameters())
        if set(model_parameters) != set(snapshot["parameters"]):
            raise ValueError("HF and exported optimizer coordinate populations differ")
        model_count = 0
        with torch.no_grad():
            for name, parameter in model_parameters.items():
                stored = snapshot["parameters"][name]["model_value"]
                if stored.dtype != torch.bfloat16 or stored.shape != parameter.shape:
                    raise ValueError(f"Saved model value differs from HF coordinate {name}")
                parameter.copy_(stored)
                model_count += parameter.numel()
        if model_count != EXPECTED_PARAMETERS:
            raise ValueError(f"Unexpected model parameter population: {model_count}")
        model.to(args.device).eval()
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        status(args.output, "generating_fixed_prefix_bank")
        rows = generate_bank(model, tokenizer, args.manifest, tasks, args)
        atomic_json(args.output / "prefix_bank.json", public_bank(rows))
        teacher_scores, teacher_routes = score_teachers(
            tokenizer, rows, tasks, args.teachers, args.device, args.output
        )
        student_scores = [row["student_log_probs"] for row in rows]
        distances = []
        for task in tasks:
            distances.append({
                "left": task,
                "right": "student",
                "full_vocab_js": weighted_js(rows, teacher_scores[task], student_scores, domain_weights),
            })
        for left, right in itertools.combinations(tasks, 2):
            distances.append({
                "left": left,
                "right": right,
                "full_vocab_js": weighted_js(rows, teacher_scores[left], teacher_scores[right], domain_weights),
            })
        del student_scores

        branches = {}
        after_states = {}
        snapshot_digest = hashlib.sha256()

        def update_progress(branch):
            return lambda ordinal, total, name: status(
                args.output, "computing_bf16_writeback", branch=branch,
                tensors_processed=ordinal, tensors_total=total, last_tensor=name,
            )

        status(args.output, "computing_bf16_writeback", branch="zero_gradient")
        after, metrics = adamw_bf16_writeback(
            snapshot["parameters"], None, float(snapshot["clip_grad"]),
            snapshot_digest=snapshot_digest, progress=update_progress("zero_gradient"),
        )
        after_states["zero_gradient"] = after
        branches["zero_gradient"] = {"objective": None, "metrics": metrics}

        for loss in losses:
            status(args.output, "computing_gradient", branch=loss)
            gradients, objective = joint_gradient(
                model, rows, teacher_scores, domain_weights, loss, args
            )
            status(args.output, "computing_bf16_writeback", branch=loss)
            after, metrics = adamw_bf16_writeback(
                snapshot["parameters"], gradients, float(snapshot["clip_grad"]),
                progress=update_progress(loss),
            )
            del gradients
            gc.collect()
            after_states[loss] = after
            branches[loss] = {"objective": objective, "metrics": metrics}

        # Each teacher starts at the same unchanged student and saved Adam state.
        # Routed branches are standalone domain means; common branches use the
        # exact same equally weighted four-domain prefix bank and sampled actions.
        for mode in ("routed", "common"):
            for teacher in tasks:
                branch = f"{mode}/{teacher}/sampled_pg"
                status(args.output, "computing_gradient", branch=branch)
                gradients, objective = joint_gradient(
                    model, rows, teacher_scores, domain_weights, "sampled_pg", args,
                    teacher=teacher, routed=(mode == "routed"),
                )
                after, metrics = adamw_bf16_writeback(
                    snapshot["parameters"], gradients, float(snapshot["clip_grad"]),
                    progress=update_progress(branch),
                )
                del gradients
                gc.collect()
                after_states[branch] = after
                branches[branch] = {"objective": objective, "metrics": metrics}
                atomic_json(args.output / "branches_partial.json", branches)

        pairs = list(itertools.combinations(("zero_gradient", *losses), 2))
        for mode in ("routed", "common"):
            names = [f"{mode}/{teacher}/sampled_pg" for teacher in tasks]
            pairs.extend(itertools.combinations(names, 2))
            pairs.extend(("zero_gradient", name) for name in names)
        comparisons = []
        for left, right in pairs:
            status(args.output, "comparing_bf16_writebacks", left=left, right=right)
            comparisons.append({
                "left": left,
                "right": right,
                "metrics": compare_writebacks(snapshot["parameters"], after_states[left], after_states[right]),
            })
        del after_states
        gc.collect()

        results = {
            "schema_version": 1,
            "status": "complete",
            "completed_at_utc": utc(),
            "job": args.job,
            "snapshot": {
                "path": str(args.snapshot),
                "bytes": snapshot_stat.st_size,
                "mtime_ns": snapshot_stat.st_mtime_ns,
                "ordered_state_sha256": snapshot_digest.hexdigest(),
                "step": snapshot["step"],
                "loss": snapshot["loss"],
                "reduction": snapshot["reduction"],
                "tasks": snapshot["tasks"],
                "domain_weights": list(snapshot["domain_weights"]),
            },
            "tasks": list(tasks),
            "domain_weights": domain_weights,
            "prefix_bank": public_bank(rows),
            "teacher_routes": {task: teacher_routes[task] for task in tasks},
            "teacher_distances": distances,
            "interpretation": {
                "teacher_update_loss": "sampled_pg",
                "routed": "Each teacher uses its own diagnostic domain; unit domain mean before clipping",
                "common": "All four teachers use identical four-domain prefixes and PG actions; equal domain weights",
                "supervision_controls": "Joint routed sampled PG, student Top64, and full vocabulary",
                "sampling": "One response per domain; selected prefix positions only; one independent bank per seed",
                "scope": "Local exploratory measurement, not full-response online gradient or causal attribution",
            },
            "branches": branches,
            "pairwise_bf16_writebacks": comparisons,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        atomic_json(args.output / "measurements.json", results)
        atomic_json(
            args.output / "run_complete.json",
            {"status": "complete", "at_utc": utc(), "job": args.job,
             "measurement": str(args.output / "measurements.json")},
        )
        manifest.update({
            "status": "complete",
            "completed_at_utc": utc(),
            "measurement": str(args.output / "measurements.json"),
            "snapshot_ordered_state_sha256": snapshot_digest.hexdigest(),
        })
        atomic_json(args.output / "probe_manifest.json", manifest)
        status(args.output, "complete")
        if wandb_run is not None:
            scalars = {}
            flatten_scalars("teacher_distance", {f"{row['left']}_vs_{row['right']}": row["full_vocab_js"] for row in distances}, scalars)
            for branch, record in branches.items():
                flatten_scalars(f"branch/{branch}", record, scalars)
            wandb_run.log(scalars)
            artifact = __import__("wandb").Artifact(args.job + "-measurements", type="paper-probe-data")
            for name in ("probe_manifest.json", "prefix_bank.json", "measurements.json", "run_complete.json"):
                artifact.add_file(str(args.output / name), name=name)
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        return results
    except Exception as exc:
        atomic_json(args.output / "run_failed.json", {"status": "failed", "at_utc": utc(), "error": repr(exc)})
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    run(parse_args())
