# Paper draft

`main.tex` + `refs.bib`: the Jugalbandi paper, written to the structure and rules of
[Orchestra-Research/AI-Research-SKILLs `ml-paper-writing`](https://github.com/Orchestra-Research/AI-Research-SKILLs/tree/main/20-ml-paper-writing/ml-paper-writing).
Generic preprint layout (`article` + natbib); no venue chosen yet.

## Build

No LaTeX toolchain is installed locally. Either upload this folder to Overleaf, or:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs `pgfplots`, `booktabs`, `natbib`, `enumitem` (all in TeX Live / Overleaf defaults).

## Where every number comes from

| In the paper | Source | Regenerate |
|---|---|---|
| Table 2 (baselines) | `eval/results/*.json` | `venv/Scripts/python -m eval.evaluate --all` |
| Tables 3–4, Figure 1 | `paper/data/baseline_stats.json` | `venv/Scripts/python -m paper.scripts.baseline_stats` |
| Lint (cites, refs, braces) | — | `venv/Scripts/python -m paper.scripts.lint_tex` |

Each result file's fingerprint holds content hashes of every file its numbers depend on — those
are the staleness check. `git_sha` is the commit the tree was *based on* (`git_dirty: true` when the
result is committed together with its code, which is unavoidable).

## Citations

Every `refs.bib` entry came from an API (arXiv, CrossRef, Semantic Scholar, OpenAlex, GitHub), not
from memory. Two still need a human check before submission (marked `TODO` in the bib):

- `lan2019raveforce`: proceedings name. The APIs returned only the UiO archive record, pp. 217–222.
- `bhatkhande_kpm`: edition, publisher, and year.

Corrections relative to `docs/research.md`'s reference list:
- RaveForce is by **Lan, Tørresen & Jensenius (2019)**, not "Han & Lee".
- The ISMIR 2012 motif paper is by Ross, **Vinutha** & Rao.
- arXiv 1611.02796 is titled *Sequence Tutor*.
- The non-stationary RL survey (2005.10619) is by Padakandla alone.

## Open TODOs in `main.tex` (render in red)

1. §6.2 trained-policy results. All existing adapters predate the prompt-decoding fix and must be
   retrained (`docs/RETRAIN_PLAN.md`).
2. Exact TRL version and resolved GRPO defaults (KL coefficient, temperature, loss type) from the
   training log.
3. Final seed count per arm.
4. GPU-hours per run, and the Qwen2.5 model licence.
5. Final wording of the LLM-usage disclosure.
6. Trained-policy curves in Figure 1.
