import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'config.json').read_text())
TASKS = ['math', 'code', 'if', 'science']


def write(name, value):
    path = ROOT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def status(stage, **kw):
    data = {'stage': stage, 'time': time.time(), 'pid': os.getpid(), 'host': socket.gethostname(), **kw}
    write('prepare_status.json', data)
    print(json.dumps(data), flush=True)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def prefix_logits(model, row):
    tokens = row['prompt_ids'] + row['response_ids'][:-1]
    h = model.model(input_ids=torch.tensor([tokens], device='cuda'), use_cache=False).last_hidden_state
    indices = torch.tensor([len(row['prompt_ids']) - 1 + p for p in row['positions']], device='cuda')
    return model.lm_head(h[0, indices]).float()


def main():
    started = time.time()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = (ROOT / 'gpu4.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (ROOT / 'prepare_complete.json').exists():
        raise RuntimeError('Already completed; refusing to overwrite frozen bank')
    generated = Path(CONFIG['generated'])
    protocol_path = generated / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['schema_version'] == 8
    assert protocol['initialization']['repository'] == 'Qwen/Qwen3-1.7B'
    assert protocol['prompt_format']['teacher_prompt_suffix'] == ''
    assert protocol['response_semantics']['stop_token_ids'] == [151643, 151645]
    routes = yaml.safe_load((generated / 'teacher_router.yaml').read_text())['teachers']
    sources = []
    for task in TASKS:
        item = protocol['splits'][task]['diagnostic']
        assert digest(item['path']) == item['sha256']
        sources.append(item)
        assert routes[task]['prompt_suffix'] == ''
    write('input_manifest.json', {'config': CONFIG, 'protocol_sha256': digest(protocol_path), 'sources': sources,
          'teachers': routes, 'precision': 'FP32 forward/backward at stored BF16 parameter values; TF32 disabled'})
    status('load_student')
    model = AutoModelForCausalLM.from_pretrained(CONFIG['hf_checkpoint'], dtype=torch.float32,
              attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
    tok = AutoTokenizer.from_pretrained(CONFIG['hf_checkpoint'], local_files_only=True)
    assert tok.eos_token_id == 151645
    snapshot = torch.load(CONFIG['snapshot'], mmap=True, weights_only=True, map_location='cpu')
    assert snapshot['step'] == 100 and snapshot['tasks'] == TASKS
    assert set(dict(model.named_parameters())) == set(snapshot['parameters'])
    checked = 0
    for name, parameter in model.named_parameters():
        expected = snapshot['parameters'][name]['model_value']
        assert torch.equal(parameter.detach().cpu(), expected.float()), name
        checked += parameter.numel()
    assert checked == 1720574976
    write('student_verified.json', {'parameters': checked, 'matches_saved_bf16_exactly': True, 'snapshot_step': 100})
    banks = []
    excluded = {t: set() for t in TASKS}
    for bank_id, seed in enumerate(CONFIG['bank_seeds']):
        rows = []
        for task_idx, task in enumerate(TASKS):
            candidates = []
            for idx, line in enumerate(Path(protocol['splits'][task]['diagnostic']['path']).read_text().splitlines()):
                prompt = json.loads(line)['prompt']
                ids = tok.encode(prompt, add_special_tokens=False)
                assert prompt.endswith('<think>\n\n</think>\n\n')
                if 0 < len(ids) <= CONFIG['max_prompt_tokens'] and idx not in excluded[task]:
                    candidates.append((idx, ids))
            rng = random.Random(seed + task_idx * 100003)
            idx, ids = rng.choice(candidates)
            excluded[task].add(idx)
            torch.manual_seed(seed + task_idx * 1000003)
            status('generate', bank=bank_id, task=task, prompt_index=idx)
            with torch.no_grad():
                response = model.generate(torch.tensor([ids], device='cuda'), do_sample=True,
                    temperature=1., top_p=1., top_k=0, max_new_tokens=CONFIG['max_new_tokens'],
                    eos_token_id=[151643, 151645], pad_token_id=151645, use_cache=True)[0].tolist()[len(ids):]
            assert response
            row = {'task': task, 'prompt_index': idx, 'prompt_ids': ids, 'response_ids': response,
                   'positions': sorted(rng.sample(range(len(response)), min(CONFIG['prefixes_per_response'],len(response)))),
                   'truncated': response[-1] not in [151643,151645], 'bank': bank_id, 'seed': seed}
            with torch.no_grad():
                logp = prefix_logits(model, row).log_softmax(-1).cpu()
            arng = torch.Generator().manual_seed(seed + task_idx * 10000019)
            row['actions'] = torch.multinomial(logp.exp(), 1, generator=arng).squeeze(-1).tolist()
            row['student_log_probs'] = logp
            rows.append(row)
            write('bank_partial.json', [{k:v for k,v in r.items() if k!='student_log_probs'} for b in banks+[rows] for r in b])
        banks.append(rows)
    torch.save(banks, ROOT / 'banks.pt')
    del model, snapshot
    torch.cuda.empty_cache()
    scores = {}
    weight_sources = []
    for task in TASKS:
        status('load_teacher', teacher=task)
        route = routes[task]
        teacher_tok = AutoTokenizer.from_pretrained(route['model_path'], local_files_only=True)
        assert teacher_tok.get_vocab() == tok.get_vocab()
        model = AutoModelForCausalLM.from_pretrained(route['model_path'], dtype=torch.float32,
              attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
        assert model.config.eos_token_id == 151645
        with torch.no_grad():
            scores[task] = [[prefix_logits(model, row).log_softmax(-1).cpu() for row in rows] for rows in banks]
        assert all(torch.isfinite(v).all() for rows in scores[task] for v in rows)
        del model
        torch.cuda.empty_cache()
        for path in sorted(Path(route['model_path']).glob('*.safetensors')):
            status('hash_teacher', teacher=task, file=path.name)
            weight_sources.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    torch.save(scores, ROOT / 'teacher_scores.pt')
    status('hash_snapshot')
    manifest = json.loads((ROOT / 'input_manifest.json').read_text())
    manifest.update({'teacher_weights':weight_sources, 'snapshot_sha256':digest(CONFIG['snapshot']),
        'banks_sha256':digest(ROOT/'banks.pt'), 'teacher_scores_sha256':digest(ROOT/'teacher_scores.pt')})
    write('input_manifest.json', manifest)
    write('prepare_complete.json', {'status':'complete','elapsed_seconds':time.time()-started,
        'responses':sum(map(len,banks)), 'prefixes':sum(len(r['positions']) for b in banks for r in b),
        'gpu': torch.cuda.get_device_name(), 'max_memory_bytes':torch.cuda.max_memory_allocated(),
        'torch':torch.__version__})
    status('complete')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=repr(exc))
        raise
