# 用于测试核心代码的正确

import os
import json
import time
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT / 'resources' / 'biorelex'

def backup_if_exists(fp: Path):
    if fp.exists():
        bak = fp.with_suffix('.bak.' + time.strftime('%Y%m%d%H%M%S'))
        shutil.copy2(fp, bak)
        print(f'Backed up {fp} -> {bak}')

def write_samples(train_n=20, dev_n=5):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    train_fp = RES_DIR / 'train.json'
    dev_fp = RES_DIR / 'dev.json'
    backup_if_exists(train_fp)
    backup_if_exists(dev_fp)

    def make_inst(i):
        return {
            'id': f'test_{i}',
            'text': f'这是测试文本 {i}。',
            'entities': [],
            'interactions': []
        }

    train_data = [make_inst(i) for i in range(train_n)]
    dev_data = [make_inst(i) for i in range(dev_n)]

    with open(train_fp, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(dev_fp, 'w', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False, indent=2)
    print(f'Wrote {train_n} train samples and {dev_n} dev samples to {RES_DIR}')

def run_trainer(config='debug'):
    cmd = [os.environ.get('PYTHON_EXE', 'python'), 'trainer.py', '-c', config, '-d', 'biorelex', '-s', '0']
    env = os.environ.copy()
    env['TORCH_COMPILE_DISABLE'] = '1'
    cwd = str(ROOT)
    print('Running trainer:', ' '.join(cmd), 'cwd=', cwd)
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    duration = time.time() - start
    print(f'Trainer exited with {proc.returncode} (elapsed {duration:.1f}s)')
    return proc.returncode, duration

if __name__ == '__main__':
    write_samples(train_n=20, dev_n=5)
    rc, dur = run_trainer('debug')
    print('Smoke test done. return_code=', rc, 'duration=', dur)
"""Quick smoke test: load minimal biorelex dataset and construct DataInstance objects.
This script purposefully avoids importing `models` to not require DGL / CUDA / local TCMroberta.
It uses a tiny `DummyTokenizer` implementing `convert_tokens_to_ids` used by data pipeline.
"""
import os
from data.biorelex import load_biorelex_dataset


class DummyTokenizer:
    def __init__(self):
        # simple mapping for special tokens
        self.special_map = {'[CLS]': 101, '[SEP]': 102, '[PAD]': 0}

    def convert_tokens_to_ids(self, tokens):
        # tokens is a list of tokens (strings). Return list of ints.
        ids = []
        for t in tokens:
            if t in self.special_map:
                ids.append(self.special_map[t])
            else:
                # map character to small integer deterministically
                ids.append((ord(t[0]) % 100) + 10)
        return ids


def run_smoke():
    base_path = os.path.join('resources', 'biorelex_min')
    print('Using base_path =', base_path)

    tokenizer = DummyTokenizer()
    train, dev = load_biorelex_dataset(base_path, tokenizer)

    print('Loaded train instances:', len(train))
    print('Loaded dev instances:', len(dev))

    if len(train) > 0:
        inst = train[0]
        print('Instance id:', inst.id)
        print('Text:', inst.text)
        print('Number of tokens:', len(inst.tokens))
        print('token_windows shape:', getattr(inst, 'input_ids').shape)
        print('input_masks shape:', getattr(inst, 'input_masks').shape)
        print('mask_windows shape:', getattr(inst, 'mask_windows').shape)
        print('example tuple length:', len(inst.example))


if __name__ == '__main__':
    run_smoke()
