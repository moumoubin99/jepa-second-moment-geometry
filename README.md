# Second-moment geometry in I-JEPA

Research code and archived experimental outputs for **Which second moments does a joint-embedding predictive architecture need?**, prepared for submission to *Neurocomputing*.

This release contains the 398 manifest-listed experiment logs used for the study, the current training/optimizer implementation, update-RMS reference trajectories, and scripts to regenerate the four numeric tables and Figures 2–5. Figure 1 is supplied as an editable SVG diagram. The archive preserves the recorded outputs; the default reproduction command recomputes analyses from those outputs.

## Files

| Path | Contents |
| --- | --- |
| `jepa_pilot/` | Training, model, dataset loaders, metrics, optimizer implementation, optimizer tests, reference generator |
| `pilot-logs/r3`, `r4`, `r5` | 398 experiment JSONs in the paths recorded by the manifest; five dense tensor-reference source files and additional global-reference source logs in `r3` |
| `pilot-logs/refs/` | Global update-RMS references, reference-source JSONs, and `r3_tref_fwd600.npz` |
| `manifest/experiments.jsonl` | Original experiment manifest, one line per experiment |
| `manifest/reference_index.json` | Archived argument paths mapped to available reference files |
| `manifest/source_files.json` | SHA-256, byte length, and original source of each imported artifact |
| `manifest/release_provenance.json` | Archive hash, observed environments, extraction scope, code snapshot status |
| `paper/tables/`, `paper/figures/` | Canonical table snapshots, numeric generation scripts, editable Figure 1 |
| `reproduce.py` | Analysis-first verification and optional single-run training entrypoint |

## Recompute existing results

Use Python 3.10 or later. Python 3.12 was used for the release checks. The analysis dependencies are compatibility ranges because their historical versions were not recorded.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python reproduce.py
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead. The default command verifies the 398 manifest entries, all source hashes and 15 reference dependencies; regenerates and compares four LaTeX tables; renders numeric Figures 2–5; and rebuilds the tensor-reference arrays. It requires no source-image download or GPU.

Individual commands are also available:

```bash
python reproduce.py verify
python reproduce.py tables
python reproduce.py figures
python reproduce.py rebuild-tensor-ref
```

Tables are written to `paper/tables/`. Numeric figures are written to `paper/figures/`. The regenerated tensor reference is written to `results/r3_tref_fwd600_rebuilt.npz`; its parameter names and RMS values must match the supplied reference exactly. PDF metadata may vary across render environments, so regenerated figure PDFs are checked as outputs rather than compared byte-for-byte.

Direct scripts can be run from the repository root:

```bash
python paper/figures/gen_tables.py
python paper/figures/gen_figures.py
python jepa_pilot/make_rms_ref.py --tensor 'pilot-logs/r3/n2r3_fwd_600_s*.json' results/r3_tref_fwd600_rebuilt.npz
```

Create `results/` before the last direct command. The table script uses LF-normalized comparisons through `reproduce.py` for cross-platform consistency.

## Recorded environment and full training

All 398 logs record **PyTorch `2.8.0+cu128`** and an **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPU. Logs carry experiment labels `R3` (251 runs) or `R5` (147 runs); these labels do not identify version-control commits. The released source was byte-equal to the training/source scripts in the original supplementary archive when this package was prepared. Exact historical Python, NumPy, scikit-learn, and external optimizer versions were not recorded.

Training uses CUDA and bfloat16 autocast. It stores dataset arrays in GPU memory, so a suitable GPU and its memory capacity are required. Analysis reproduction was tested locally; a fresh training campaign was not run during preparation of this release. Different hardware or package versions may change training outcomes.

Install the recorded PyTorch CUDA build using the [official prior-version instructions](https://pytorch.org/get-started/previous-versions/):

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-training.txt
```

External optimizer runs additionally require the corresponding packages in `requirements-optional-optimizers.txt`: [bitsandbytes](https://pypi.org/project/bitsandbytes/), [GaLore](https://pypi.org/project/galore-torch/), and [Adam-mini](https://pypi.org/project/adam-mini/). Their constraints describe candidate compatible versions and do not reconstruct an observed environment lock. Controlled-family runs (`full`, `rowonly`, `dimbal`, `scalar`, `adafactor`, `q8v`, `rowperm`, `minipart`) use the included optimizer implementation.

## Acquire source datasets

Source-image binaries are obtained from their original providers and are excluded from this release.

* **CIFAR-100:** download the Python archive from the [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html), verify its published MD5 `eb9058c3a382ffc7106e4002c42a8d85`, and extract its `cifar-100-python/` folder into `data/`.
* **STL-10:** download the binary archive from the [official STL-10 page](https://cs.stanford.edu/~acoates/stl10/) and extract `stl10_binary/` into `data/`.

Expected layout:

```text
data/cifar-100-python/train
data/cifar-100-python/test
data/stl10_binary/unlabeled_X.bin
data/stl10_binary/train_X.bin
data/stl10_binary/train_y.bin
data/stl10_binary/test_X.bin
data/stl10_binary/test_y.bin
```

The included CIFAR loader uses the fine labels. SSL uses the training split, while kNN/probes use the first `bank_n` training examples and first `query_n` test examples, in archive order. Configurations with `val_query=true` evaluate on the final 5,000 training examples instead; their bank is restricted by its recorded `bank_n` setting. STL-10 uses the unlabeled split for SSL and the labeled train/test splits for evaluation. The study uses a frozen-embedding diagnostic protocol, whose recorded labeled-bank size differs from the provider's standard 10-fold supervised protocol. Follow each log's `args` for exact counts and settings. The 398 runs cover CIFAR-100 and STL-10 only.

## Preview or rerun one recorded experiment

Select a unique `run_id` from `manifest/experiments.jsonl`. A command without `--execute` prints the resolved configuration and starts no training:

```bash
python reproduce.py rerun --run-id p4_full_c0_s10
python reproduce.py rerun --run-id n2r3_pertensor_600_s0
```

To explicitly execute a complete selected configuration after dataset setup:

```bash
python reproduce.py rerun --run-id p4_full_c0_s10 --execute
```

The wrapper reads all recorded arguments, resolves moved `exact_ref`, `tensor_ref`, and `rms_ref_file` paths, preserves the archived experiment logs, and writes the fresh output to `results/reproduced/<run_id>.json`. It records the resolved arguments and current software/GPU environment alongside that output. Existing fresh outputs must be preserved before repeating a run. This command runs the archived step count, which ranges from 1,500 to 20,000 steps, rather than a shortened smoke test.

Optimizer implementation checks, after installing PyTorch:

```bash
python jepa_pilot/test_r3.py
```

## Definitions and matching protocols

`full`, `rowonly`, `dimbal`, and `scalar` correspond to full Adam, row-level, column-level, and matrix-level second-moment geometry. Other parameters and the first-moment state follow the included implementation. The stored field `opt_state_bytes / 2**20` is reported as **moment storage in MiB**, according to the included implementation's tensor accounting; index maps are excluded. “Trajectory mean” is the arithmetic mean of RankMe over the stored evaluation records, including the initial record; it is not a time-integrated AUC.

In the optimizer comparison table, **Scale** is the arithmetic mean of `dense.scale[200:]`: the first 200 warm-up steps are excluded. The dense record starts at training step 1, so the average covers steps 201 onward. Unmatched rows show no matching scale. The reported mean update-RMS follows the code's post-warm-up accumulation.

Reference protocols vary by panel. The long CIFAR-100 experiment uses a three-seed full-Adam reference. Short controlled comparisons use five-seed references. The geometry-switching experiments apply matching after the switch when configured; reverse switches use a column-level source reference. Per-tensor replay uses `r3_tref_fwd600.npz`, averaged from the five supplied forward-switch dense logs. The original arguments and reference metadata retain the source of each configuration.

Paired differences use seed intersections and two-sided Student-t 95% confidence intervals; `*` denotes intervals excluding zero. Table summaries use sample standard deviations where specified. Metric values and original log metadata are preserved; no estimated values replace missing observations.

## Provenance and reuse

`manifest/source_files.json` gives the exact import source and SHA-256 of each original artifact. `manifest/release_provenance.json` identifies the original archive by hash and records package scope. Additional generated/documentation files are recorded in the release file manifest. Raw datasets, private configuration, credentials, unrelated work, third-party paper PDFs, and obsolete remote execution wrappers are excluded.

No explicit source-project license was found. This release does not assign a code or data license. Dataset use follows the source providers' terms; author confirmation is needed before declaring a repository-wide reuse license.
