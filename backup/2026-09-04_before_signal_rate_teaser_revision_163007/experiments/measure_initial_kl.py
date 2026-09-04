#!/usr/bin/env python3
"""Measure the initial sampled student--teacher log-ratio diagnostic.

For each held-out prompt, the initial student generates one response. The
diagnostic averages log p_student - log p_teacher over its tokens, then over
responses. It is not the teacher top-64 training loss. With non-unit sampling
temperature, it is also not a Monte Carlo estimate under the raw student
distribution. Objective weights are fixed and equal; this diagnostic does not
select weights, change teachers, or determine whether training proceeds.

Example (one GPU; vLLM for generation, HF for scoring):
  python experiments/measure_initial_kl.py \
    --student /ckpt/Qwen3-1.7B \
    --teacher math=/ckpt/math_teacher --teacher code=/ckpt/code_teacher \
    --teacher if=/ckpt/if_teacher --teacher science=/ckpt/science_teacher \
    --prompts math=heldout/math.parquet --prompts code=heldout/code.parquet \
    --prompts if=heldout/if.parquet --prompts science=heldout/science.parquet \
    --n-prompts 128 --max-tokens 4096 --temperature 1.0 --out initial_kl.json

Prompt files: parquet with a column (--prompt-key, default "prompt") holding either a
string or a list of chat messages. Student and teachers must share the tokenizer.
"""
import argparse, json, time
import pandas as pd
import torch


def parse_kv(items):
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_prompts(path, key, n):
    df = pd.read_parquet(path)
    vals = df[key].tolist()[:n]
    msgs = []
    for v in vals:
        if isinstance(v, str):
            msgs.append([{"role": "user", "content": v}])
        else:
            msgs.append([dict(m) for m in v])
    return msgs


@torch.no_grad()
def response_logprobs(model, prompt_ids, resp_ids, device):
    """Per-token log-prob of resp_ids given prompt_ids under model (one sequence)."""
    ids = torch.tensor([prompt_ids + resp_ids], device=device)
    logits = model(ids).logits[0]                       # [T, V]
    start = len(prompt_ids) - 1
    sl = logits[start:start + len(resp_ids)].float()    # predicts resp tokens
    lp = torch.log_softmax(sl, dim=-1)
    tgt = torch.tensor(resp_ids, device=device)
    return lp.gather(1, tgt[:, None]).squeeze(1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", action="append", required=True, help="task=path")
    ap.add_argument("--prompts", action="append", required=True, help="task=parquet")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--n-prompts", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.45, help="vLLM gpu_memory_utilization")
    ap.add_argument("--full-matrix", action="store_true", help="score every task's responses with every teacher")
    ap.add_argument("--out", default="initial_kl.json")
    args = ap.parse_args()

    teachers = parse_kv(args.teacher)
    prompt_files = parse_kv(args.prompts)
    tasks = list(teachers.keys())
    assert set(tasks) == set(prompt_files.keys()), "teacher and prompt task names must match"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.student)

    # 1) Generate one response per held-out prompt with the initial student.
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.student, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_tokens + 2048, seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
    gen = {}
    for task in tasks:
        msgs = load_prompts(prompt_files[task], args.prompt_key, args.n_prompts)
        prompt_ids = [tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False, tokenize=True)
                      for m in msgs]
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": p} for p in prompt_ids], sp)
        recs = []
        for p, o in zip(prompt_ids, outs):
            c = o.outputs[0]
            recs.append({"prompt_ids": p, "resp_ids": list(c.token_ids), "truncated": c.finish_reason == "length"})
        gen[task] = recs
        print(f"[gen] {task}: {len(recs)} responses in {time.time()-t0:.0f}s, "
              f"mean len {sum(len(r['resp_ids']) for r in recs)/len(recs):.0f}, "
              f"trunc {sum(r['truncated'] for r in recs)/len(recs):.3f}")
    del llm
    torch.cuda.empty_cache()

    # 2) Score sampled tokens under the student and the teachers (HF, one sequence at a time).
    device = "cuda"
    def load(path):
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()

    student = load(args.student)
    for task in tasks:
        for r in gen[task]:
            r["logq"] = response_logprobs(student, r["prompt_ids"], r["resp_ids"], device) if r["resp_ids"] else torch.zeros(0)
    del student
    torch.cuda.empty_cache()

    result = {"tasks": {}, "matrix": {}}
    for tname, tpath in teachers.items():
        teacher = load(tpath)
        score_tasks = tasks if args.full_matrix else [tname]
        for task in score_tasks:
            per_resp = []
            for r in gen[task]:
                if not r["resp_ids"]:
                    continue
                logt = response_logprobs(teacher, r["prompt_ids"], r["resp_ids"], device)
                per_resp.append(float((r["logq"] - logt).mean()))   # token mean, then response mean below
            ell = sum(per_resp) / len(per_resp)
            result["matrix"][f"{task}|{tname}"] = ell
            print(f"[score] responses of {task} under teacher {tname}: ell = {ell:.5f}")
        del teacher
        torch.cuda.empty_cache()

    # 3) Report the diagnostic and the already specified equal task weights.
    result["metric"] = "response_mean_sampled_log_ratio"
    result["objective_weights"] = "fixed_equal"
    for t in tasks:
        recs = gen[t]
        result["tasks"][t] = {
            "sampled_log_ratio": result["matrix"][f"{t}|{t}"],
            "objective_weight": 1.0 / len(tasks),
            "n_responses": len(recs),
            "mean_response_len": sum(len(r["resp_ids"]) for r in recs) / len(recs),
            "truncation_rate": sum(r["truncated"] for r in recs) / len(recs),
        }
    result["config"] = vars(args)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)

    print("\ntask      log_ratio   fixed_w   mean_len  trunc")
    for t in tasks:
        d = result["tasks"][t]
        print(f"{t:8s}  {d['sampled_log_ratio']:.5f}   {d['objective_weight']:.3f}"
              f"     {d['mean_response_len']:7.0f}  {d['truncation_rate']:.3f}")


if __name__ == "__main__":
    main()
