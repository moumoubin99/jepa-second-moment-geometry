"""Verify archived artifacts, regenerate analyses, or explicitly rerun one experiment.

Default: analysis of existing logs. GPU training requires `rerun --execute`.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE_KEYS = ('exact_ref', 'tensor_ref', 'rms_ref_file')

def runs():
    return [json.loads(line) for line in (ROOT / 'manifest/experiments.jsonl').read_text().splitlines()]

def resolve_reference(original):
    index = json.loads((ROOT / 'manifest/reference_index.json').read_text())
    relative = index.get(original)
    if relative is None:
        # Basename resolution is permitted only when the archived name is unambiguous.
        matches = {v for k, v in index.items() if Path(k).name == Path(original).name}
        if len(matches) != 1:
            raise ValueError(f'Unknown or ambiguous reference: {original}')
        relative = matches.pop()
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def verify():
    inventory = json.loads((ROOT / 'manifest/source_files.json').read_text())
    for item in inventory:
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'], item['path']
        assert hashlib.sha256(data).hexdigest() == item['sha256'], item['path']
    manifest = runs()
    assert len(manifest) == 398
    assert len({r['run_id'] for r in manifest}) == 398
    references = set()
    for run in manifest:
        log = json.loads((ROOT / run['evidence']).read_text())
        assert log['args']['seed'] == run['config']['seed'], run['run_id']
        assert log['args']['steps'] == run['config']['steps'], run['run_id']
        assert log['args']['dataset'] == run['config']['dataset'], run['run_id']
        assert abs(log['knn_acc'] - run['metric']['value']) < 1e-12, run['run_id']
        for key in REFERENCE_KEYS:
            if log['args'].get(key):
                references.add(str(resolve_reference(log['args'][key]).relative_to(ROOT)))
    print(f'Verified {len(manifest)} experiment logs, {len(references)} argument references, {len(inventory)} source hashes.')

def tables():
    # Preserve the canonical byte layout while testing from current archived logs.
    expected = {p.name: p.read_bytes() for p in (ROOT / 'paper/tables').glob('tab_*.tex')}
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(ROOT / 'paper/figures/gen_tables.py'), run_name='__main__')
    regenerated = {p.name: p.read_bytes() for p in (ROOT / 'paper/tables').glob('tab_*.tex')}
    # Windows text I/O changes LF to CRLF; compare the logical byte stream consistently.
    for name, data in regenerated.items():
        lf = data.replace(b'\r\n', b'\n')
        (ROOT / 'paper/tables' / name).write_bytes(lf)
        assert lf == expected[name].replace(b'\r\n', b'\n'), f'Table mismatch: {name}'
    print('Regenerated all four LaTeX tables; bytes match the archived canonical LF files.')

def figures():
    runpy.run_path(str(ROOT / 'paper/figures/gen_figures.py'), run_name='__main__')
    print('Regenerated numeric Figures 2–5 from archived logs. Figure 1 is the supplied editable diagram.')

def rebuild_tensor_reference():
    import numpy as np
    paths = [ROOT / f'pilot-logs/r3/n2r3_fwd_600_s{seed}_dense.npz' for seed in range(5)]
    with np.load(paths[0], allow_pickle=False) as first:
        names = first['names'].copy()
    matrices = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['names'], names), path
            matrices.append(z['rms'])
    mean = np.mean(matrices, axis=0).astype(np.float32)
    existing = resolve_reference('pilot-logs/r3_tref_fwd600.npz')
    with np.load(existing, allow_pickle=False) as z:
        assert np.array_equal(z['names'], names)
        assert np.array_equal(z['rms'], mean), 'Tensor reference differs from archived source runs.'
    out = ROOT / 'results' / 'r3_tref_fwd600_rebuilt.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=names, rms=mean)
    print(f'Rebuilt and array-verified five-seed tensor reference: {out.relative_to(ROOT)}')

def rerun(args):
    selected = [r for r in runs() if r['run_id'] == args.run_id]
    if len(selected) != 1:
        raise ValueError('Select exactly one manifest run_id; inspect manifest/experiments.jsonl.')
    log = json.loads((ROOT / selected[0]['evidence']).read_text())
    configuration = dict(log['args'])
    out = ROOT / 'results/reproduced' / (args.run_id + '.json')
    if out.exists():
        raise FileExistsError(f'Preserve the existing reproduction output before repeating: {out}')
    configuration['out'] = str(out)
    for key in REFERENCE_KEYS:
        if configuration.get(key):
            configuration[key] = str(resolve_reference(configuration[key]))
    command = [sys.executable, str(ROOT / 'jepa_pilot/train.py')]
    for key, value in configuration.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                command.append('--' + key)
        else:
            command.extend(['--' + key, str(value)])
    print('Selected manifest run:', args.run_id)
    print('Configuration source:', selected[0]['evidence'])
    print('Command:', shlex.join(command))
    if not args.execute:
        print('Command preview only. Add --execute to explicitly launch this full GPU training run.')
        return
    dataset = configuration['dataset']
    necessary = {
        'cifar100': ['data/cifar-100-python/train', 'data/cifar-100-python/test'],
        'stl10': ['data/stl10_binary/' + n for n in ['unlabeled_X.bin', 'train_X.bin', 'train_y.bin', 'test_X.bin', 'test_y.bin']],
    }
    if dataset not in necessary:
        raise ValueError(f'Dataset not represented in the release manifest: {dataset}')
    for relative in necessary[dataset]:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f'Acquire the source dataset as described in README.md: {relative}')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('The archived training implementation requires a CUDA GPU.')
    out.parent.mkdir(parents=True, exist_ok=True)
    resolved = {'run_id': args.run_id, 'source': selected[0]['evidence'], 'resolved_args': configuration,
                'python': sys.version, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    (out.with_suffix('.resolved.json')).write_text(json.dumps(resolved, indent=2) + '\n', encoding='utf-8')
    subprocess.run(command, cwd=ROOT, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action')
    for name in ['analyze', 'verify', 'tables', 'figures', 'rebuild-tensor-ref']:
        sub.add_parser(name)
    training = sub.add_parser('rerun', help='Preview or explicitly execute one manifest training configuration.')
    training.add_argument('--run-id', required=True)
    training.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    action = args.action or 'analyze'
    if action == 'rerun':
        rerun(args)
    elif action == 'verify':
        verify()
    elif action == 'tables':
        tables()
    elif action == 'figures':
        figures()
    elif action == 'rebuild-tensor-ref':
        rebuild_tensor_reference()
    else:
        verify(); tables(); figures(); rebuild_tensor_reference()

if __name__ == '__main__':
    main()
