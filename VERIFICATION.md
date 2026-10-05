# Release verification — 2026-10-05

The analysis-only release checks completed using Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, and Matplotlib 3.11.1 on Windows. These versions describe the verification environment; they are not inferred historical training versions.

* All **398** unique manifest-listed experiments are present at their original `pilot-logs/{r3,r4,r5}` paths: 150 in `r3`, 40 in `r4`, and 208 in `r5`.
* All 398 log configurations match manifest seed, step count, and dataset; recorded kNN values match manifest values exactly to the verification tolerance of `1e-12`.
* All **15** nonempty archived reference arguments resolve to present files. Added to the previous ZIP: four projector/SIGReg/STL global-reference JSONs, `r3_tref_fwd600.npz`, and its five dense source NPZs. Ten full/column reference-source logs are also available under `r3`.
* All **455** imported source records passed SHA-256 and byte-length checks. The original experiment logs and manifest were preserved without modification.
* All four LaTeX tables regenerated from archived logs and matched the original canonical tables **byte-for-byte after LF normalization**. Numeric content and headers are unchanged; the manuscript captions define the storage unit as MiB.
* Numeric Figures 2–5 regenerated successfully from the archived logs. Their sizes were 27,954, 22,700, 25,766, and 18,419 bytes respectively in this verification environment. Rendered PDF bytes are environment-dependent. Figure 1 remains the supplied editable SVG diagram.
* The five supplied forward-switch dense NPZs regenerated the tensor-reference parameter names and RMS arrays **exactly**. The regenerated file was written to ignored `results/`.
* All **398** archived argument sets use option names recognized by the released trainer, verified against its argument definitions. A `rerun --run-id n2r3_pertensor_600_s0` command preview resolved the tensor-reference location without starting training.
* A scoped scan found no common GitHub, AWS, OpenAI-style token patterns or private-key headers in release text files. Only the explicitly inventoried project artifacts and authored release documentation are included.

No new GPU training experiment was run during release preparation. PyTorch is not installed in the analysis verification environment, so `test_r3.py` was preserved but not rerun here. Historical external optimizer/package versions were not recorded; the dependency constraints state this limitation explicitly.

Publication scope excludes source-dataset image binaries, credentials/private configuration, third-party paper PDFs, other projects, legacy backup code, and remote queue wrappers. Generated reproduction outputs and Python caches are ignored and excluded from the release manifest. No project-wide reuse license was present or assigned.
