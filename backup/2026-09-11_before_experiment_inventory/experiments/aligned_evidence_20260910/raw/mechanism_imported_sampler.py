"""Four-task micro-batch allocation for the MOPD/GPAS experiment."""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

TASKS = ("math", "code", "if", "science")
PAPER_REDUCTIONS = ("global_token", "domain_token", "domain_response")


def active_tasks(args: Any) -> tuple[str, ...]:
    value = getattr(args, "mopd_tasks", None)
    if not value:
        return TASKS[:3] if getattr(args, "mopd_profile", None) == "smollm3" else TASKS
    tasks = tuple(item.strip() for item in value.split(",")) if isinstance(value, str) else tuple(value)
    if not tasks or len(set(tasks)) != len(tasks) or any(task not in TASKS for task in tasks):
        raise ValueError(f"MOPD tasks must be a unique nonempty subset of {TASKS}")
    return tasks


def active_domain_weights(args: Any) -> tuple[float, ...]:
    tasks = active_tasks(args)
    value = getattr(args, "mopd_domain_weights", None)
    if not value:
        return (1.0 / len(tasks),) * len(tasks)
    weights = tuple(map(float, value.split(",") if isinstance(value, str) else value))
    if len(weights) != len(tasks) or any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("MOPD domain weights must align with active tasks and sum to one")
    return weights
CORE_RUNS = {"uniform-s1": "uniform", "gpas-s1": "gpas", "gpas-raw-s1": "raw_noise", "d3-fixed-s1": "d3_fixed"}
CORE_ALLOCATIONS = tuple(CORE_RUNS.values())
ALLOCATIONS = (
    "uniform",
    "gpas",
    "cost_gpas",
    "raw_noise",
    "loss_gap",
    "std_mopd",
    "d3_mopd",
    "open_mopd",
    "d3_fixed",
)
MICROBATCHES_PER_STEP = 16
PROMPTS_PER_MICROBATCH = 4
RESPONSES_PER_PROMPT = 1
RESPONSES_PER_STEP = MICROBATCHES_PER_STEP * PROMPTS_PER_MICROBATCH
M_MIN = 2
M_MAX = 8
MAIN_STEPS = 500
MAIN_RESPONSE_BUDGET = MAIN_STEPS * RESPONSES_PER_STEP
MAIN_CHECKPOINT_STEPS = (100, 200, 250, 300, 400, 500)
MAIN_CHECKPOINT_RESPONSES = tuple(step * RESPONSES_PER_STEP for step in MAIN_CHECKPOINT_STEPS)
MAIN_EVAL_RESPONSES = tuple(step * RESPONSES_PER_STEP for step in range(100, MAIN_STEPS + 1, 100))
SMOKE_STEPS = 20
SMOKE_RESPONSE_BUDGET = SMOKE_STEPS * RESPONSES_PER_STEP
SMOKE_CHECKPOINT_STEPS = (10, 20)
SMOKE_EVAL_RESPONSES = tuple(step * RESPONSES_PER_STEP for step in SMOKE_CHECKPOINT_STEPS)
QUICK_SMOKE_STEPS = 1
QUICK_SMOKE_RESPONSE_BUDGET = QUICK_SMOKE_STEPS * RESPONSES_PER_STEP
QUICK_SMOKE_CHECKPOINT_STEPS = (1,)
QUICK_SMOKE_EVAL_RESPONSES: tuple[int, ...] = ()
QUICK_SMOKE_MAX_RESPONSE_LEN = 256
VARIANCE_CHECKPOINT_STEPS = (50, 250, 500)
VARIANCE_MICROBATCHES_PER_TASK = 32
VARIANCE_MICROBATCHES = len(TASKS) * VARIANCE_MICROBATCHES_PER_TASK
VARIANCE_RESPONSE_BUDGET = VARIANCE_MICROBATCHES * PROMPTS_PER_MICROBATCH

# D³-MOPD scheduler values from arXiv:2608.24987, Table 3. The watcher is
# evaluated synchronously at the same rollout-step frontier in this experiment;
# this is mathematically equivalent and makes controller checkpointing atomic.
D3_UPDATE_CADENCE = 10
D3_WINDOW = 10
D3_WINDOWS = 3
D3_INITIAL_STEPS = 5
D3_EMA_WINDOW = 10
D3_KL_FLOOR = 0.15
D3_TEMPERATURE = 0.5
D3_MIXTURE_FLOOR = 0.10
D3_JITTER = 0.30

# Open-MOPD values from arXiv:2608.19098, Eq. 9 / Table 8.  Four domains
# replace the paper's three, so the equal target is 1/4 rather than 1/3.
OPEN_MOPD_GAP_ALPHA = 1.0
OPEN_MOPD_GAP_FACTOR_MIN = 0.05
OPEN_MOPD_GAP_FACTOR_MAX = 20.0


def d3_scheduler_mixture(
    ema_histories: Sequence[Sequence[float]],
    initial_kl: Sequence[float],
    completed_steps: int,
) -> dict[str, Any]:
    """Apply the D³-MOPD gap-times-velocity rule at one watcher tick."""

    step = int(completed_steps)
    if step < 2 * D3_WINDOW:
        raise ValueError("D³-MOPD needs at least two complete KL windows")
    if len(ema_histories) != len(initial_kl) or not ema_histories:
        raise ValueError("D³-MOPD KL state must contain the same non-zero number of domains")
    if any(len(history) < step for history in ema_histories):
        raise ValueError("D³-MOPD KL history is shorter than the requested watcher step")

    available_windows = min(D3_WINDOWS, step // D3_WINDOW - 1)
    gaps: list[float] = []
    velocities: list[float] = []
    signals: list[float] = []
    for history, initial in zip(ema_histories, initial_kl, strict=True):
        gap = max(float(history[step - 1]), 0.0) / max(float(initial), D3_KL_FLOOR)
        changes = []
        for index in range(available_windows):
            current_step = step - index * D3_WINDOW
            previous_step = current_step - D3_WINDOW
            current = float(history[current_step - 1])
            previous = float(history[previous_step - 1])
            changes.append((current - previous) / max(previous, D3_KL_FLOOR))
        velocity = max(0.0, -sum(changes) / available_windows)
        gaps.append(gap)
        velocities.append(velocity)
        signals.append(gap * velocity)

    maximum = max(signals)
    normalized = [0.0] * len(signals) if maximum == 0.0 else [value / maximum for value in signals]
    logits = [value / D3_TEMPERATURE for value in normalized]
    logit_max = max(logits)
    exponentials = [math.exp(value - logit_max) for value in logits]
    exponential_sum = sum(exponentials)
    free_mass = 1.0 - len(signals) * D3_MIXTURE_FLOOR
    mixture = [D3_MIXTURE_FLOOR + free_mass * value / exponential_sum for value in exponentials]
    return {
        "watcher_step": step,
        "available_windows": available_windows,
        "remaining_gap": gaps,
        "descent_velocity": velocities,
        "composite_signal": signals,
        "normalized_signal": normalized,
        "mixture": mixture,
    }


def crossed_response_milestone(before: int, after: int, milestones: Sequence[int]) -> bool:
    return any(int(before) < int(value) <= int(after) for value in milestones)


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return result


def continuous_bounded_allocation(
    scores: Sequence[float],
    *,
    total: int = MICROBATCHES_PER_STEP,
    lower: int = M_MIN,
    upper: int = M_MAX,
) -> list[float]:
    """Return ``clip(scale * score_i, lower, upper)`` with the requested sum."""

    values = [_finite(value, "allocation score", 0.0) for value in scores]
    if len(values) * lower > total or len(values) * upper < total:
        raise ValueError("infeasible micro-batch bounds")
    if not any(values):
        return [float(total) / len(values)] * len(values)

    lo, hi = 0.0, 1.0
    while sum(min(max(hi * value, lower), upper) for value in values) < total:
        hi *= 2.0
    for _ in range(100):
        scale = (lo + hi) / 2.0
        allocated = sum(min(max(scale * value, lower), upper) for value in values)
        if allocated < total:
            lo = scale
        else:
            hi = scale
    result = [min(max(hi * value, lower), upper) for value in values]
    residual = total - sum(result)
    free = [index for index, value in enumerate(result) if lower < value < upper]
    if free:
        share = residual / len(free)
        for index in free:
            result[index] += share
    return result


def largest_remainder_allocation(
    scores: Sequence[float],
    *,
    total: int = MICROBATCHES_PER_STEP,
    lower: int = M_MIN,
    upper: int = M_MAX,
) -> list[int]:
    """Bound a proportional allocation and round it while preserving its sum."""

    continuous = continuous_bounded_allocation(scores, total=total, lower=lower, upper=upper)
    counts = [int(math.floor(value + 1e-12)) for value in continuous]
    remaining = total - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (continuous[index] - counts[index], -index),
        reverse=True,
    )
    for index in order:
        if remaining == 0:
            break
        if counts[index] < upper:
            counts[index] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError("largest-remainder allocation did not reach the requested total")
    return counts


def feasible_allocations(
    task_count: int = len(TASKS),
    *,
    total: int = MICROBATCHES_PER_STEP,
    lower: int = M_MIN,
    upper: int = M_MAX,
):
    for counts in itertools.product(range(lower, upper + 1), repeat=task_count):
        if sum(counts) == total:
            yield counts


def integer_allocation(
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    total: int = MICROBATCHES_PER_STEP,
    lower: int = M_MIN,
    upper: int = M_MAX,
) -> list[int]:
    """Enumerate variance minimization, or squared distance to a probability target.

    Equal objectives prefer squared distance to Uniform, then protocol task order.
    """
    values = [_finite(value, "allocation signal", 0.0) for value in values]
    if not values:
        raise ValueError("allocation requires at least one task")
    if weights is not None:
        weights = [_finite(value, "target weight", 0.0) for value in weights]
        if len(weights) != len(values):
            raise ValueError("noise and target weights must align")
    elif not math.isclose(sum(values), 1.0, abs_tol=1e-10):
        raise ValueError("allocation probabilities must sum to one")
    uniform = total / len(values)

    def key(counts):
        objective = (
            math.fsum(w * w * e / m for w, e, m in zip(weights, values, counts, strict=True))
            if weights is not None
            else math.fsum((m - total * p) ** 2 for m, p in zip(counts, values, strict=True))
        )
        return objective, sum((m - uniform) ** 2 for m in counts), counts

    candidates = list(feasible_allocations(len(values), total=total, lower=lower, upper=upper))
    if not candidates:
        raise ValueError("infeasible micro-batch bounds")
    return list(min(candidates, key=key))


def cost_gpas_allocation(
    weights: Sequence[float],
    noise: Sequence[float],
    task_seconds: Sequence[float],
    fixed_seconds: float,
    *,
    total: int = MICROBATCHES_PER_STEP,
    lower: int = M_MIN,
    upper: int = M_MAX,
) -> list[int]:
    """Enumerate the bounded integer minimizer of ``t(m) * V(m)``."""

    w = [_finite(value, "target weight", 0.0) for value in weights]
    e = [_finite(value, "scaled noise", 0.0) for value in noise]
    tau = [_finite(value, "task seconds", 0.0) for value in task_seconds]
    fixed = _finite(fixed_seconds, "fixed seconds", 0.0)
    if not any(e):
        return largest_remainder_allocation([1.0] * len(w), total=total, lower=lower, upper=upper)

    def objective(counts: Sequence[int]) -> float:
        step_seconds = fixed + sum(count * seconds for count, seconds in zip(counts, tau, strict=True))
        variance = sum(weight * weight * value / count for weight, value, count in zip(w, e, counts, strict=True))
        return step_seconds * variance

    return list(min(feasible_allocations(len(w), total=total, lower=lower, upper=upper), key=objective))


@dataclass(frozen=True)
class ControllerConfig:
    task_names: tuple[str, ...]
    target_weights: tuple[float, ...]
    allocation: str
    seed: int
    rollout_offset: int
    ema_decay: float
    total_steps: int
    checkpoint_steps: tuple[int, ...]
    microbatches_per_step: int
    prompts_per_microbatch: int
    min_microbatches: int
    max_microbatches: int
    reduction: str | None = None


class MOPDController:
    """Choose one bounded four-task allocation and checkpoint its online state."""

    SCHEMA_VERSION = 5

    def __init__(
        self,
        task_names: Sequence[str] = TASKS,
        *,
        target_weights: Sequence[float] | None = None,
        allocation: str = "uniform",
        seed: int = 42,
        rollout_offset: int = 0,
        ema_decay: float = 0.9,
        total_steps: int = MAIN_STEPS,
        checkpoint_steps: Sequence[int] = MAIN_CHECKPOINT_STEPS,
        microbatches_per_step: int = MICROBATCHES_PER_STEP,
        prompts_per_microbatch: int = PROMPTS_PER_MICROBATCH,
        min_microbatches: int = M_MIN,
        max_microbatches: int = M_MAX,
        reduction: str | None = None,
    ) -> None:
        names = tuple(map(str, task_names))
        if reduction is None and names != TASKS:
            raise ValueError(f"task order must be {TASKS}, got {names}")
        if reduction is not None and (reduction not in PAPER_REDUCTIONS or not names or len(set(names)) != len(names)):
            raise ValueError("paper MOPD requires a valid reduction and unique task names")
        weights = tuple(float(value) for value in (target_weights or [1.0 / len(names)] * len(names)))
        if (
            len(weights) != len(names)
            or any(not math.isfinite(value) or value < 0 for value in weights)
            or not math.isclose(sum(weights), 1.0, abs_tol=1e-12)
        ):
            raise ValueError("target weights must contain four positive values summing to one")
        if allocation not in ALLOCATIONS:
            raise ValueError(f"unknown MOPD allocation {allocation!r}")
        decay = _finite(ema_decay, "ema_decay", 0.0)
        if decay >= 1:
            raise ValueError("ema_decay must be below one")
        steps = int(total_steps)
        checkpoints = tuple(map(int, checkpoint_steps))
        if (
            steps <= 0
            or checkpoints != tuple(sorted(set(checkpoints)))
            or any(value <= 0 or value > steps for value in checkpoints)
        ):
            raise ValueError("checkpoint steps must be unique, increasing, and within the run")
        continuous_bounded_allocation(
            [1.0] * len(names),
            total=int(microbatches_per_step),
            lower=int(min_microbatches),
            upper=int(max_microbatches),
        )
        self.config = ControllerConfig(
            task_names=names,
            target_weights=weights,
            allocation=allocation,
            seed=int(seed),
            rollout_offset=int(rollout_offset),
            ema_decay=decay,
            total_steps=steps,
            checkpoint_steps=checkpoints,
            microbatches_per_step=int(microbatches_per_step),
            prompts_per_microbatch=int(prompts_per_microbatch),
            min_microbatches=int(min_microbatches),
            max_microbatches=int(max_microbatches),
            reduction=reduction,
        )
        n = len(names)
        self.scaled_noise: list[float | None] = [None] * n
        self.raw_noise: list[float | None] = [None] * n
        self.task_seconds: list[float | None] = [None] * n
        self.loss_ema: list[float | None] = [None] * n
        self.fixed_seconds: float | None = None
        self.d3_raw_kl_history: list[list[float]] = [[] for _ in names]
        self.d3_ema_kl_history: list[list[float]] = [[] for _ in names]
        self.d3_initial_kl: list[float | None] = [None] * n
        self.d3_mixture: list[float] = [1.0 / n] * n
        self.d3_last_watcher: dict[str, Any] | None = None
        self.completed_steps = 0
        self.attempted_responses = 0
        self.pending: dict[str, Any] | None = None

    @property
    def task_names(self) -> tuple[str, ...]:
        return self.config.task_names

    @property
    def optimizer_updates(self) -> int:
        return self.completed_steps

    @property
    def completed_operations(self) -> int:
        return self.completed_steps

    @property
    def budget_complete(self) -> bool:
        return self.completed_steps >= self.config.total_steps

    def _uniform_counts(self) -> list[int]:
        return largest_remainder_allocation(
            [1.0] * len(self.task_names),
            total=self.config.microbatches_per_step,
            lower=self.config.min_microbatches,
            upper=self.config.max_microbatches,
        )

    def _d3_counts(self) -> tuple[list[int], dict[str, Any]]:
        step = self.completed_steps
        watcher_updated = False
        if step >= 2 * D3_WINDOW and step % D3_UPDATE_CADENCE == 0:
            if any(value is None for value in self.d3_initial_kl):
                raise RuntimeError("D³-MOPD initial KL normalizers were not seeded")
            self.d3_last_watcher = d3_scheduler_mixture(
                self.d3_ema_kl_history,
                [float(value) for value in self.d3_initial_kl],
                step,
            )
            self.d3_mixture = list(self.d3_last_watcher["mixture"])
            watcher_updated = True

        # Stateless per-step randomness reproduces the same batch exactly after
        # resume without serializing a language-runtime RNG object.
        rng = random.Random(self.config.seed + step * 1_000_003)
        jitter = [rng.uniform(-D3_JITTER, D3_JITTER) for _ in self.task_names]
        perturbed = [probability * (1.0 + value) for probability, value in zip(self.d3_mixture, jitter, strict=True)]
        perturbed_total = sum(perturbed)
        jittered = [value / perturbed_total for value in perturbed]
        # The shared experiment freezes 2 <= m_i <= 8 so every method fits the
        # same finite, non-repeating prompt streams. This is the only projection
        # applied after the paper's probability and jitter equations.
        counts = integer_allocation(
            jittered,
            total=self.config.microbatches_per_step,
            lower=self.config.min_microbatches,
            upper=self.config.max_microbatches,
        )
        details = {
            "paper": "D³ signal, fixed weights" if self.config.allocation == "d3_fixed" else "D3-MOPD",
            "implementation_version": "d3-table3-synchronous-v1-integer-projection",
            "watcher_updated": watcher_updated,
            "watcher_step": None if self.d3_last_watcher is None else self.d3_last_watcher["watcher_step"],
            "base_probabilities": dict(zip(self.task_names, self.d3_mixture, strict=True)),
            "jitter": dict(zip(self.task_names, jitter, strict=True)),
            "jittered_probabilities": dict(zip(self.task_names, jittered, strict=True)),
            "scheduler": copy.deepcopy(self.d3_last_watcher),
            "scheduler_hyperparameters": {
                "update_cadence": D3_UPDATE_CADENCE,
                "window": D3_WINDOW,
                "windows": D3_WINDOWS,
                "initial_steps": D3_INITIAL_STEPS,
                "ema_window": D3_EMA_WINDOW,
                "kl_floor": D3_KL_FLOOR,
                "temperature": D3_TEMPERATURE,
                "mixture_floor": D3_MIXTURE_FLOOR,
                "jitter": D3_JITTER,
            },
        }
        return counts, details

    def _counts(self) -> list[int]:
        if self.completed_steps == 0 or self.config.allocation in {"uniform", "std_mopd", "open_mopd"}:
            return self._uniform_counts()
        weights = self.config.target_weights
        if self.config.allocation == "cost_gpas":
            return cost_gpas_allocation(
                weights,
                self.scaled_noise,
                self.task_seconds,
                self.fixed_seconds,
                total=self.config.microbatches_per_step,
                lower=self.config.min_microbatches,
                upper=self.config.max_microbatches,
            )
        if self.config.allocation in {"gpas", "raw_noise"}:
            return integer_allocation(
                self.scaled_noise if self.config.allocation == "gpas" else self.raw_noise,
                weights=weights,
                total=self.config.microbatches_per_step,
                lower=self.config.min_microbatches,
                upper=self.config.max_microbatches,
            )
        elif self.config.allocation == "loss_gap":
            scores = [weight * max(float(value), 0.0) for weight, value in zip(weights, self.loss_ema, strict=True)]
        else:
            raise RuntimeError(f"allocation {self.config.allocation!r} needs a dedicated planning branch")
        return largest_remainder_allocation(
            scores,
            total=self.config.microbatches_per_step,
            lower=self.config.min_microbatches,
            upper=self.config.max_microbatches,
        )

    def plan(self, rollout_id: int) -> dict[str, Any]:
        if self.pending is not None:
            if int(self.pending["rollout_id"]) == int(rollout_id):
                return copy.deepcopy(self.pending)
            raise RuntimeError("the previous MOPD step still awaits feedback")
        expected = self.config.rollout_offset + self.completed_steps
        if int(rollout_id) != expected:
            raise ValueError(f"expected rollout id {expected}, got {rollout_id}")
        if self.budget_complete:
            raise StopIteration("MOPD step budget is complete")

        allocation_details: dict[str, Any] | None = None
        if self.config.allocation in {"d3_mopd", "d3_fixed"}:
            counts, allocation_details = self._d3_counts()
        else:
            counts = self._counts()
            if self.config.allocation == "open_mopd":
                allocation_details = {
                    "paper": "Open-MOPD",
                    "variant": "K=1 sampled-token in-protocol adaptation",
                    "share_target": dict.fromkeys(self.task_names, 1.0 / len(self.task_names)),
                    "gap_alpha": OPEN_MOPD_GAP_ALPHA,
                    "gap_factor_bounds": [OPEN_MOPD_GAP_FACTOR_MIN, OPEN_MOPD_GAP_FACTOR_MAX],
                    "reward_refresh": "identity_at_K=1",
                }
        unit_weights = (
            [1.0 / len(self.task_names)] * len(self.task_names)
            if self.config.allocation == "open_mopd"
            else self.config.target_weights
        )
        units = [
            {
                "task_index": index,
                "task": task,
                "microbatches": counts[index],
                "target_weight": unit_weights[index],
            }
            for index, task in enumerate(self.task_names)
        ]
        prompt_count = self.config.microbatches_per_step * self.config.prompts_per_microbatch
        if any(value is None for value in self.scaled_noise):
            variance_gain = 1.0
        else:
            uniform = self._uniform_counts()
            uniform_variance = sum(
                weight * weight * float(noise) / count
                for weight, noise, count in zip(self.config.target_weights, self.scaled_noise, uniform, strict=True)
            )
            allocated_variance = sum(
                weight * weight * float(noise) / count
                for weight, noise, count in zip(self.config.target_weights, self.scaled_noise, counts, strict=True)
            )
            variance_gain = 1.0 if allocated_variance == 0.0 else uniform_variance / allocated_variance
        self.pending = {
            "schema_version": self.SCHEMA_VERSION,
            "rollout_id": int(rollout_id),
            "operation_index": self.completed_steps,
            "optimizer_step": self.completed_steps + 1,
            "operation": "train",
            "run_mode": "train",
            "allocation": self.config.allocation,
            "aggregation": (
                self.config.reduction if self.config.reduction is not None else "token_mean"
                if self.config.allocation in {"std_mopd", "d3_mopd"}
                else "open_mopd" if self.config.allocation == "open_mopd" else "fixed_objective"
            ),
            "counts": dict(zip(self.task_names, counts, strict=True)),
            "task_units": units,
            "allocation_details": allocation_details,
            "execution_order": list(self.task_names),
            "prompt_count": prompt_count,
            "responses_per_prompt": RESPONSES_PER_PROMPT,
            "attempted_responses": prompt_count,
            "attempted_responses_before": self.attempted_responses,
            "optimizer_updates_before": self.completed_steps,
            "step_global_batch_size": self.config.prompts_per_microbatch,
            "scaled_noise_before": dict(zip(self.task_names, self.scaled_noise, strict=True)),
            "raw_noise_before": dict(zip(self.task_names, self.raw_noise, strict=True)),
            "loss_ema_before": dict(zip(self.task_names, self.loss_ema, strict=True)),
            "task_seconds_before": dict(zip(self.task_names, self.task_seconds, strict=True)),
            "fixed_seconds_before": self.fixed_seconds,
            "H": variance_gain,
            "issued": 0,
        }
        return copy.deepcopy(self.pending)

    def complete(self, rollout_id: int, feedback: dict[str, Any]) -> dict[str, Any]:
        if self.pending is None or int(self.pending["rollout_id"]) != int(rollout_id):
            raise RuntimeError(f"no pending MOPD step for rollout {rollout_id}")
        pending = self.pending
        units = list(feedback.get("task_units") or [])
        if [unit["task"] for unit in units] != list(self.task_names):
            raise ValueError("trainer feedback must contain all four tasks in protocol order")
        if [int(unit["microbatches"]) for unit in units] != [pending["counts"][task] for task in self.task_names]:
            raise ValueError("trainer feedback micro-batch counts differ from the allocation")
        if int(feedback["attempted_responses"]) != int(pending["attempted_responses"]):
            raise ValueError("trainer feedback attempted-response count differs from the allocation")

        previous_scaled = list(self.scaled_noise)
        decay = self.config.ema_decay
        for index, unit in enumerate(units):
            for name in ("scaled_noise", "raw_noise"):
                if unit.get(name) is not None:
                    observed = _finite(unit[name], name, 0.0)
                    history = getattr(self, name)
                    history[index] = (
                        observed if history[index] is None else decay * history[index] + (1 - decay) * observed
                    )
            observed_seconds = _finite(unit["microbatch_seconds"], "microbatch_seconds", 0.0)
            observed_loss = _finite(unit["teacher_loss"], "teacher_loss")
            old_seconds = self.task_seconds[index]
            old_loss = self.loss_ema[index]
            self.task_seconds[index] = (
                observed_seconds if old_seconds is None else decay * old_seconds + (1 - decay) * observed_seconds
            )
            self.loss_ema[index] = (
                observed_loss if old_loss is None else decay * old_loss + (1 - decay) * observed_loss
            )
            if self.config.allocation in {"d3_mopd", "d3_fixed"}:
                self.d3_raw_kl_history[index].append(observed_loss)
                alpha = 2.0 / (D3_EMA_WINDOW + 1.0)
                history = self.d3_ema_kl_history[index]
                history.append(observed_loss if not history else alpha * observed_loss + (1.0 - alpha) * history[-1])
                if len(self.d3_raw_kl_history[index]) == D3_INITIAL_STEPS:
                    self.d3_initial_kl[index] = sum(self.d3_raw_kl_history[index]) / D3_INITIAL_STEPS
        observed_fixed = _finite(feedback["fixed_seconds"], "fixed_seconds", 0.0)
        self.fixed_seconds = (
            observed_fixed if self.fixed_seconds is None else decay * self.fixed_seconds + (1 - decay) * observed_fixed
        )

        self.completed_steps += 1
        self.attempted_responses += int(pending["attempted_responses"])
        record = copy.deepcopy(pending)
        record.pop("issued", None)
        record.update(
            {
                "feedback": copy.deepcopy(feedback),
                "attempted_responses_after": self.attempted_responses,
                "optimizer_updates_after": self.completed_steps,
                "scaled_noise_after": dict(zip(self.task_names, self.scaled_noise, strict=True)),
                "raw_noise_after": dict(zip(self.task_names, self.raw_noise, strict=True)),
                "loss_ema_after": dict(zip(self.task_names, self.loss_ema, strict=True)),
                "task_seconds_after": dict(zip(self.task_names, self.task_seconds, strict=True)),
                "fixed_seconds_after": self.fixed_seconds,
                "d3_initial_kl_after": (
                    dict(zip(self.task_names, self.d3_initial_kl, strict=True))
                    if self.config.allocation in {"d3_mopd", "d3_fixed"}
                    else None
                ),
                "d3_ema_kl_after": (
                    {
                        task: (None if not history else history[-1])
                        for task, history in zip(self.task_names, self.d3_ema_kl_history, strict=True)
                    }
                    if self.config.allocation in {"d3_mopd", "d3_fixed"}
                    else None
                ),
                "scaled_noise_relative_change": {
                    task: None if previous is None else (float(current) - previous) / max(abs(previous), 1e-12)
                    for task, previous, current in zip(
                        self.task_names, previous_scaled, self.scaled_noise, strict=True
                    )
                },
                "checkpoint_due": self.completed_steps in self.config.checkpoint_steps,
                "budget_complete": self.budget_complete,
            }
        )
        self.pending = None
        return record

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": asdict(self.config),
            "scaled_noise": self.scaled_noise,
            "raw_noise": self.raw_noise,
            "task_seconds": self.task_seconds,
            "loss_ema": self.loss_ema,
            "fixed_seconds": self.fixed_seconds,
            "d3_raw_kl_history": self.d3_raw_kl_history,
            "d3_ema_kl_history": self.d3_ema_kl_history,
            "d3_initial_kl": self.d3_initial_kl,
            "d3_mixture": self.d3_mixture,
            "d3_last_watcher": self.d3_last_watcher,
            "completed_steps": self.completed_steps,
            "attempted_responses": self.attempted_responses,
            "pending": self.pending,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported MOPD controller checkpoint schema")
        config = dict(state["config"])
        config.setdefault("reduction", None)
        for key in ("task_names", "target_weights", "checkpoint_steps"):
            config[key] = tuple(config[key])
        if config != asdict(self.config):
            raise ValueError("saved controller configuration differs from this run")
        for key in ("scaled_noise", "raw_noise", "task_seconds", "loss_ema"):
            setattr(self, key, copy.deepcopy(state[key]))
        self.fixed_seconds = state["fixed_seconds"]
        n = len(self.task_names)
        self.d3_raw_kl_history = copy.deepcopy(state.get("d3_raw_kl_history", [[] for _ in range(n)]))
        self.d3_ema_kl_history = copy.deepcopy(state.get("d3_ema_kl_history", [[] for _ in range(n)]))
        self.d3_initial_kl = copy.deepcopy(state.get("d3_initial_kl", [None] * n))
        self.d3_mixture = copy.deepcopy(state.get("d3_mixture", [1.0 / n] * n))
        self.d3_last_watcher = copy.deepcopy(state.get("d3_last_watcher"))
        self.completed_steps = int(state["completed_steps"])
        self.attempted_responses = int(state["attempted_responses"])
        self.pending = copy.deepcopy(state.get("pending"))


__all__ = [
    "ALLOCATIONS",
    "D3_EMA_WINDOW",
    "D3_INITIAL_STEPS",
    "D3_JITTER",
    "D3_KL_FLOOR",
    "D3_MIXTURE_FLOOR",
    "D3_TEMPERATURE",
    "D3_UPDATE_CADENCE",
    "D3_WINDOW",
    "D3_WINDOWS",
    "MAIN_CHECKPOINT_RESPONSES",
    "MAIN_CHECKPOINT_STEPS",
    "MAIN_EVAL_RESPONSES",
    "MAIN_RESPONSE_BUDGET",
    "MAIN_STEPS",
    "MICROBATCHES_PER_STEP",
    "MOPDController",
    "M_MAX",
    "M_MIN",
    "OPEN_MOPD_GAP_ALPHA",
    "OPEN_MOPD_GAP_FACTOR_MAX",
    "OPEN_MOPD_GAP_FACTOR_MIN",
    "PROMPTS_PER_MICROBATCH",
    "RESPONSES_PER_PROMPT",
    "RESPONSES_PER_STEP",
    "SMOKE_CHECKPOINT_STEPS",
    "SMOKE_EVAL_RESPONSES",
    "SMOKE_RESPONSE_BUDGET",
    "SMOKE_STEPS",
    "TASKS",
    "VARIANCE_CHECKPOINT_STEPS",
    "VARIANCE_MICROBATCHES",
    "VARIANCE_MICROBATCHES_PER_TASK",
    "VARIANCE_RESPONSE_BUDGET",
    "continuous_bounded_allocation",
    "cost_gpas_allocation",
    "crossed_response_milestone",
    "d3_scheduler_mixture",
    "feasible_allocations",
    "largest_remainder_allocation",
]
