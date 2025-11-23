import os
import json
import argparse
import torch

from transformers import AutoTokenizer

from utils import prepare_configs
from data.helpers import tokenize
from data.base import DataInstance
from data.biorelex import read_split as read_biorelex_split
from models import JointModel


def load_tokenizer(root_dir):
    local_path = os.path.join(root_dir, 'TCMroberta')
    if os.path.exists(local_path):
        return AutoTokenizer.from_pretrained(local_path)
    else:
        raise FileNotFoundError('TCMroberta folder not found under {}'.format(root_dir))


def predict_from_text(model, tokenizer, text, out_id='single_0'):
    tokenization = tokenize(tokenizer, list(text))
    inst = DataInstance({}, out_id, text, tokenization, True, entities=[], relations=[])
    with torch.no_grad():
        pred = model.predict(inst)
    return pred


def predict_from_file(model, tokenizer, input_fp):
    # read_split will convert JSON to DataInstance objects
    data_insts = read_biorelex_split(input_fp, tokenizer)
    results = {}
    with torch.no_grad():
        for inst in data_insts:
            results[inst.id] = model.predict(inst)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Single text to run prediction on')
    parser.add_argument('--input_json', type=str, help='Path to JSON input (biorelex format)')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to model checkpoint (.pt)')
    parser.add_argument('--config', type=str, default='basic', help='Config name in configs/')
    parser.add_argument('--dataset', type=str, default='biorelex', help='Dataset name')
    parser.add_argument('--split', type=int, default=0, help='Split number')
    parser.add_argument('--out', type=str, default='results/predictions.json', help='Output JSON path')
    args = parser.parse_args()

    root_dir = os.path.dirname(__file__)
    tokenizer = load_tokenizer(root_dir)

    configs = prepare_configs(args.config, args.dataset, args.split)
    # Auto-detect CUDA availability and ensure gid is valid. If CUDA not available or gid invalid,
    # fall back to CPU by setting configs['no_cuda']=True or resetting gid to 0.
    try:
        import torch
        if not configs.get('no_cuda', False):
            if not torch.cuda.is_available():
                print('CUDA not available -> forcing CPU')
                configs['no_cuda'] = True
            else:
                # check gid validity
                try:
                    gid = int(configs.get('gid', 0))
                except Exception:
                    gid = 0
                n_gpus = torch.cuda.device_count()
                if gid < 0 or gid >= n_gpus:
                    print(f'Invalid gid={gid} for available GPUs={n_gpus}. Using gid=0')
                    configs['gid'] = 0
    except Exception:
        # If anything goes wrong with torch, force CPU
        configs['no_cuda'] = True

    model = JointModel(configs)

    if args.ckpt:
        if os.path.exists(args.ckpt):
            ckpt = torch.load(args.ckpt, map_location=model.device)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
            print('Loaded checkpoint', args.ckpt)
        else:
            print('Checkpoint not found:', args.ckpt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = {}

    if args.text:
        pred = predict_from_text(model, tokenizer, args.text, out_id='single_0')
        results['single_0'] = pred
    elif args.input_json:
        if not os.path.exists(args.input_json):
            raise FileNotFoundError(args.input_json)
        results = predict_from_file(model, tokenizer, args.input_json)
    else:
        raise ValueError('Either --text or --input_json must be provided')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('Saved predictions to', args.out)


if __name__ == '__main__':
    main()
