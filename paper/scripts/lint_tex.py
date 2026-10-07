"""Mechanical pre-compile checks for paper/main.tex: every \\cite key exists in
refs.bib, every \\ref has a \\label, environments balance, braces balance.

    venv/Scripts/python -m paper.scripts.lint_tex
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent


def main() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    cited = {k.strip() for m in re.findall(r"\\cite[tp]?\{([^}]*)\}", tex) for k in m.split(",")}
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
    envs = set(re.findall(r"\\begin\{(\w+\*?)\}", tex))
    unbalanced = {
        e: (tex.count(f"\\begin{{{e}}}"), tex.count(f"\\end{{{e}}}"))
        for e in envs
        if tex.count(f"\\begin{{{e}}}") != tex.count(f"\\end{{{e}}}")
    }
    print("cited but missing from refs.bib:", sorted(cited - keys) or "none")
    print("in refs.bib but uncited:", sorted(keys - cited) or "none")
    print("undefined \\ref targets:", sorted(refs - labels) or "none")
    print("unbalanced environments:", unbalanced or "none")
    print("brace balance ({ minus }):", tex.count("{") - tex.count("}"))
    print("open TODOs:", len(re.findall(r"\\todo\{", tex)))


if __name__ == "__main__":
    main()
