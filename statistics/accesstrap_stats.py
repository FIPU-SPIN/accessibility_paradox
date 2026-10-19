#!/usr/bin/env python3
# accesstrap_stats.py
# Statistical analysis pipeline for the AccessTrap-CS benchmark.
#
# Usage:
#   python accesstrap_stats.py --datadir . --outdir results --only all
#   python accesstrap_stats.py --datadir . --outdir results --only tables
#   python accesstrap_stats.py --datadir . --outdir results --only figures
#
# Inputs (expected in --datadir or its data/ and annotation/ subdirs):
#   corpus.csv              -- 735 rows
#   evaluations.csv         -- 720 rows
#   annotations_final.csv   -- 270 rows (h1, h2, consensus)
#   validation_map.csv      -- 150 rows
#   ground_truth.csv        -- optional
#
# Outputs:
#   results/tables/*.tex
#   results/figures/*.pdf
#   results/accesstrap_stats.log

from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LOG = logging.getLogger("accesstrap")


def setup_logging(logfile: Path) -> None:
    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    LOG.handlers.clear()
    LOG.addHandler(fh)
    LOG.addHandler(sh)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITION_ORDER = ["A_standard", "B_simplified", "C_highly_simplified"]
CONDITION_LABEL = {
    "A_standard": "A (standard)",
    "B_simplified": "B (simplified)",
    "C_highly_simplified": "C (highly simplified)",
}
MODEL_ORDER = [
    "mistral:7b",
    "deepseek-r1:8b",
    "ministral-3:14b",
    "phi4:14b",
    "gpt-oss:20b",
    "gemma4:31b",
    "deepseek-r1:32b",
]
SUBFIELD_ORDER = [
    "Algorithms and Data Structures",
    "Programming Languages and Paradigms",
    "Databases",
    "Computer Networks",
    "Operating Systems",
]
SUBFIELD_SHORT = {
    "Algorithms and Data Structures": "Algorithms \\& Data Structures",
    "Programming Languages and Paradigms": "Programming Languages \\& Paradigms",
    "Databases": "Databases",
    "Computer Networks": "Computer Networks",
    "Operating Systems": "Operating Systems",
}

RANDOM_STATE = 20261013
N_BOOT = 5000
ALPHA = 0.05


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_csv_robust(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as fh:
        head = fh.readline()
    sep = "\t" if head.count("\t") > head.count(",") else ","
    return pd.read_csv(path, sep=sep)


def _find_file(datadir: Path, filename: str) -> Path:
    """Pronađi datoteku u datadir, data/ ili annotation/."""
    candidates = [
        datadir / filename,
        datadir / "data" / filename,
        datadir / "annotation" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Ne mogu pronaći {filename}. Tražio sam u: "
        + ", ".join(str(c) for c in candidates)
    )


def load_corpus(datadir: Path) -> pd.DataFrame:
    p = _find_file(datadir, "corpus.csv")
    df = _read_csv_robust(p)
    for col in ["truncated"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    LOG.info("Loaded corpus: %d rows from %s", len(df), p)
    return df


def load_evaluations(datadir: Path) -> pd.DataFrame:
    p = _find_file(datadir, "evaluations.csv")
    df = _read_csv_robust(p)
    for col in ["strategic_misconception", "dangerous_analogy",
                "oversimplification_error", "truncated"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    LOG.info("Loaded evaluations: %d rows from %s", len(df), p)
    return df


def load_annotations(datadir: Path) -> pd.DataFrame:
    p = _find_file(datadir, "annotations_final.csv")
    df = _read_csv_robust(p)
    LOG.info("Loaded annotations_final: %d rows (raters: %s) from %s",
             len(df), df["rater"].value_counts().to_dict(), p)
    return df


def load_validation_map(datadir: Path) -> pd.DataFrame:
    p = _find_file(datadir, "validation_map.csv")
    df = _read_csv_robust(p)
    LOG.info("Loaded validation_map: %d rows from %s", len(df), p)
    return df


# ---------------------------------------------------------------------------
# Table writer
# ---------------------------------------------------------------------------

@dataclass
class TableWriter:
    outdir: Path

    def __post_init__(self):
        (self.outdir / "tables").mkdir(parents=True, exist_ok=True)
        (self.outdir / "figures").mkdir(parents=True, exist_ok=True)

    def write(self, name: str, body: str, caption: str, label: str,
              notes: Optional[str] = None, colspec: str = "l") -> None:
        tex = [
            "\\begin{table}[H]",
            "\\caption{%s}" % caption,
            "\\label{%s}" % label,
            "\\centering",
            "\\begin{tabular}{%s}" % colspec,
            "\\toprule",
            body.rstrip(),
            "\\bottomrule",
            "\\end{tabular}",
        ]
        if notes:
            tex.append("\\par\\smallskip\\footnotesize %s" % notes)
        tex.append("\\end{table}")
        tex.append("")
        (self.outdir / "tables" / name).write_text("\n".join(tex), encoding="utf-8")
        LOG.info("Wrote tables/%s", name)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "$<0.001$"
    return "%.3f" % p


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return ("%%.%df" % digits) % x


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_str(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return "%.1f [%.1f, %.1f]" % (100.0 * k / n, 100.0 * lo, 100.0 * hi)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pa1 = float(np.mean(a == 1)); pb1 = float(np.mean(b == 1))
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def quadratic_weighted_kappa(a, b, labels) -> float:
    a = np.asarray(a); b = np.asarray(b)
    labels = np.asarray(labels)
    k = len(labels)
    idx = {v: i for i, v in enumerate(labels)}
    O = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum()
    if n == 0:
        return float("nan")
    W = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            W[i, j] = ((labels[i] - labels[j]) ** 2) / ((labels[-1] - labels[0]) ** 2)
    row = O.sum(axis=1, keepdims=True)
    col = O.sum(axis=0, keepdims=True)
    E = row @ col / n
    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return float("nan")
    return 1.0 - num / den


def cochrans_q(binary_matrix: np.ndarray) -> Tuple[float, float]:
    n, k = binary_matrix.shape
    col_sums = binary_matrix.sum(axis=0)
    row_sums = binary_matrix.sum(axis=1)
    T = col_sums.sum()
    num = (k - 1) * (k * (col_sums ** 2).sum() - T ** 2)
    den = k * T - (row_sums ** 2).sum()
    if den == 0:
        return float("nan"), float("nan")
    Q = num / den
    p = 1 - stats.chi2.cdf(Q, df=k - 1)
    return Q, p


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * stats.binom.cdf(k, n, 0.5)
    return min(1.0, p)


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        val = max(val, prev)
        adj[idx] = min(1.0, val)
        prev = adj[idx]
    return adj.tolist()


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        val = min(val, prev)
        adj[idx] = min(1.0, val)
        prev = adj[idx]
    return adj.tolist()


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = N_BOOT,
                      alpha: float = ALPHA) -> Tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = values[idx].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def smd_cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sa, sb = a.var(ddof=1), b.var(ddof=1)
    sp = math.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return (a.mean() - b.mean()) / sp


def eta_squared_from_f(F: float, df1: int, df2: int) -> float:
    if F is None or math.isnan(F) or df2 <= 0:
        return float("nan")
    return (F * df1) / (F * df1 + df2)


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    if n == 0:
        return float("nan")
    denom = n * (min(r, c) - 1)
    if denom <= 0:
        return float("nan")
    return math.sqrt(chi2 / denom)


# ---------------------------------------------------------------------------
# Robust boolean -> float
# ---------------------------------------------------------------------------

def as_float(s: pd.Series) -> pd.Series:
    """Convert a boolean/object column to float (0.0/1.0), robust to strings."""
    if s.dtype == bool:
        return s.astype(float)
    return s.astype(str).str.upper().isin(["TRUE", "1", "YES"]).astype(float)


# ---------------------------------------------------------------------------
# GEE, conditional logit, Bayes mixed
# ---------------------------------------------------------------------------

def gee_exchangeable_binary(df: pd.DataFrame, outcome: str,
                            reference: str = "A_standard") -> Dict[str, dict]:
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
    except Exception as exc:
        LOG.warning("statsmodels not available (%s); GEE skipped.", exc)
        return {}

    d = df[["concept_id", "model", "condition", outcome]].dropna().copy()
    d["y"] = as_float(d[outcome]).astype(int)
    for cond in CONDITION_ORDER[1:]:
        d["is_" + cond] = (d["condition"] == cond).astype(int)
    d["cluster"] = d["concept_id"].astype(str) + "|" + d["model"].astype(str)

    X = sm.add_constant(d[["is_" + c for c in CONDITION_ORDER[1:]]])
    try:
        res = sm.GEE(d["y"], X, groups=d["cluster"], family=Binomial(),
                     cov_struct=Exchangeable()).fit()
    except Exception as exc:
        LOG.warning("GEE failed for %s: %s", outcome, exc)
        return {}

    out = {}
    for cond in CONDITION_ORDER[1:]:
        beta = res.params["is_" + cond]
        se = res.bse["is_" + cond]
        p = res.pvalues["is_" + cond]
        out["%s_vs_%s" % (cond, reference)] = {
            "or": float(math.exp(beta)),
            "ci_lo": float(math.exp(beta - 1.96 * se)),
            "ci_hi": float(math.exp(beta + 1.96 * se)),
            "p": float(p),
        }
    return out


def conditional_logit_pair_fe(binary_df: pd.DataFrame, outcome: str) -> Dict[str, dict]:
    d = binary_df[["concept_id", "model", "condition", outcome]].dropna().copy()
    d["pair"] = d["concept_id"].astype(str) + "|" + d["model"].astype(str)
    d["y"] = as_float(d[outcome]).astype(int)
    grouped = d.groupby("pair")["y"].nunique()
    keep = grouped[grouped > 1].index
    dd = d[d["pair"].isin(keep)].copy()
    if len(dd) == 0:
        return {}
    for cond in CONDITION_ORDER[1:]:
        dd["is_" + cond] = (dd["condition"] == cond).astype(float)
    for col in ["is_" + c for c in CONDITION_ORDER[1:]] + ["y"]:
        dd[col + "_dm"] = dd[col] - dd.groupby("pair")[col].transform("mean")
    try:
        import statsmodels.api as sm
        X = dd[["is_" + c + "_dm" for c in CONDITION_ORDER[1:]]]
        y = dd["y_dm"]
        model = sm.OLS(y, X).fit(cov_type="cluster",
                                 cov_kwds={"groups": dd["pair"]})
    except Exception as exc:
        LOG.warning("Conditional logit failed for %s: %s", outcome, exc)
        return {}
    out = {}
    for cond in CONDITION_ORDER[1:]:
        beta = model.params["is_" + cond + "_dm"]
        se = model.bse["is_" + cond + "_dm"]
        p = model.pvalues["is_" + cond + "_dm"]
        out["%s_vs_A_standard" % cond] = {
            "or": float(math.exp(beta)),
            "ci_lo": float(math.exp(beta - 1.96 * se)),
            "ci_hi": float(math.exp(beta + 1.96 * se)),
            "p": float(p),
        }
    return out


def bayes_mixed_logistic_vb(binary_df: pd.DataFrame, outcome: str) -> Dict[str, dict]:
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except Exception as exc:
        LOG.warning("BinomialBayesMixedGLM not available (%s); skipped.", exc)
        return {}
    d = binary_df[["concept_id", "model", "condition", outcome]].dropna().copy()
    d["y"] = as_float(d[outcome]).astype(int)
    for cond in CONDITION_ORDER[1:]:
        d["is_" + cond] = (d["condition"] == cond).astype(float)
    try:
        vc = {"concept": "0 + C(concept_id)", "model": "0 + C(model)"}
        model = BinomialBayesMixedGLM.from_formula(
            "y ~ is_B_simplified + is_C_highly_simplified", vc, d)
        res = model.fit_vb()
    except Exception as exc:
        LOG.warning("BayesMixedGLM failed for %s: %s", outcome, exc)
        return {}
    out = {}
    for cond in CONDITION_ORDER[1:]:
        try:
            idx = list(res.model.exog_names).index("is_" + cond)
            beta = res.fe_mean[idx]
            se = res.fe_sd[idx]
            out["%s_vs_A_standard" % cond] = {
                "or": float(math.exp(beta)),
                "ci_lo": float(math.exp(beta - 1.96 * se)),
                "ci_hi": float(math.exp(beta + 1.96 * se)),
                "p": float("nan"),
            }
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def sample_table(eval_df: pd.DataFrame, tw: TableWriter) -> None:
    n_by_model = eval_df.groupby(["model", "condition"]).size().unstack(fill_value=0)
    lines = []
    for m in MODEL_ORDER:
        if m not in n_by_model.index:
            continue
        row = n_by_model.loc[m]
        lines.append("%s & %d & %d & %d & %d \\\\"
                     % (m, row.get("A_standard", 0), row.get("B_simplified", 0),
                        row.get("C_highly_simplified", 0), row.sum()))
    totals = n_by_model.sum(axis=0)
    lines.append("\\midrule")
    lines.append("Total & %d & %d & %d & %d \\\\"
                 % (totals.get("A_standard", 0), totals.get("B_simplified", 0),
                    totals.get("C_highly_simplified", 0), totals.sum()))
    body = ("Model & A & B & C & Total \\\\\n\\midrule\n" + "\n".join(lines))
    tw.write("tab_sample.tex", body,
             "Number of judged responses per model and condition.",
             "tab:sample", colspec="lrrrr")


def readability_table(corpus: pd.DataFrame, tw: TableWriter) -> None:
    try:
        import textstat
    except Exception as exc:
        LOG.warning("textstat not available (%s); readability skipped.", exc)
        return
    import re

    def strip_md(s: str) -> str:
        s = re.sub(r"```.*?```", " ", s, flags=re.S)
        s = re.sub(r"`[^`]*`", " ", s)
        s = re.sub(r"[#*_>~\-]+", " ", s)
        return s

    rows = []
    for _, r in corpus.iterrows():
        txt = strip_md(str(r["response"]))
        rows.append({
            "condition": r["condition"],
            "fre": textstat.flesch_reading_ease(txt),
            "fkgl": textstat.flesch_kincaid_grade(txt),
            "asl": textstat.words_per_sentence(txt),
            "words": r["word_count"],
        })
    df = pd.DataFrame(rows)

    desc = []
    for cond in CONDITION_ORDER:
        sub = df[df["condition"] == cond]
        desc.append((CONDITION_LABEL[cond], len(sub),
                     sub["fre"].mean(), sub["fre"].std(ddof=1),
                     sub["fkgl"].mean(), sub["fkgl"].std(ddof=1),
                     sub["asl"].mean(), sub["asl"].std(ddof=1),
                     sub["words"].mean(), sub["words"].std(ddof=1)))
    body_lines = []
    for d in desc:
        body_lines.append("%s & %d & %.1f (%.1f) & %.1f (%.1f) & %.1f (%.1f) & %.0f (%.0f) \\\\" % d)
    body = ("Condition & $n$ & FRE & FKGL & ASL & Words \\\\\n\\midrule\n"
            + "\n".join(body_lines))
    tw.write("tab_readability.tex", body,
             "Readability metrics by condition (mean (SD)).",
             "tab:readability", colspec="lrrrrr")

    anova_rows = []
    for metric in ["fre", "fkgl", "asl", "words"]:
        groups = [df[df["condition"] == c][metric].values for c in CONDITION_ORDER]
        F, p = stats.f_oneway(*groups)
        eta2 = eta_squared_from_f(F, 2, len(df) - 3)
        W, pW = stats.shapiro(df[metric].values)
        L, pL = stats.levene(*groups)
        anova_rows.append("%s & %.2f & 2, %d & %s & %.3f & %s & %s \\\\"
                          % (metric, F, len(df) - 3, fmt_p(p), eta2,
                             fmt_p(pW), fmt_p(pL)))
    body = ("Metric & $F$ & df & $p$ & $\\eta^2$ & Shapiro $p$ & Levene $p$ \\\\\n\\midrule\n"
            + "\n".join(anova_rows))
    tw.write("tab_readability_tukey.tex", body,
             "Omnibus ANOVA, normality and variance-homogeneity tests for readability metrics.",
             "tab:readability:tukey", colspec="lrrrrrr")


def condition_summary(ev: pd.DataFrame, outcome: str, tw: TableWriter,
                      tab_name: str, label: str, caption: str) -> None:
    lines = []
    for cond in CONDITION_ORDER:
        sub = ev[ev["condition"] == cond]
        n = len(sub)
        k = int(as_float(sub[outcome]).sum())
        lines.append("%s & %d & %d & %s \\\\"
                     % (CONDITION_LABEL[cond], k, n, wilson_str(k, n)))
    body = ("Condition & Count & $n$ & Rate (\\%) [95\\% CI] \\\\\n\\midrule\n"
            + "\n".join(lines))
    tw.write(tab_name, body, caption, label, colspec="lrrr")


def omnibus_tests(ev: pd.DataFrame, outcome: str, tw: TableWriter,
                  tab_name: str, label: str, caption: str) -> None:
    groups = [as_float(ev[ev["condition"] == c][outcome]).values
              for c in CONDITION_ORDER]
    F, p = stats.f_oneway(*groups)
    eta2 = eta_squared_from_f(F, 2, len(ev) - 3)
    ct = pd.crosstab(ev["condition"], ev[outcome])
    if ct.shape == (3, 2):
        chi2, pchi, dof, _ = stats.chi2_contingency(ct)
        V = cramers_v(chi2, len(ev), 3, 2)
    else:
        chi2, pchi, V = float("nan"), float("nan"), float("nan")
    pivot = ev.pivot_table(index=["concept_id", "model"],
                           columns="condition", values=outcome,
                           aggfunc="first").dropna()
    if pivot.shape[1] == 3:
        Q, pQ = cochrans_q(pivot[CONDITION_ORDER].apply(as_float).values.astype(int))
    else:
        Q, pQ = float("nan"), float("nan")
    body = ("Test & Statistic & df & $p$ & Effect size \\\\\n\\midrule\n"
            "One-way ANOVA & $F = %.2f$ & 2, %d & %s & $\\eta^2 = %.3f$ \\\\\n"
            "Chi-square & $\\chi^2 = %.2f$ & 2 & %s & $V = %.3f$ \\\\\n"
            "Cochran's $Q$ & $Q = %.2f$ & 2 & %s & -- \\\\"
            % (F, len(ev) - 3, fmt_p(p), eta2, chi2, fmt_p(pchi), V, Q, fmt_p(pQ)))
    tw.write(tab_name, body, caption, label, colspec="lrrrr")


def pairwise_mcnemar(ev: pd.DataFrame, outcome: str, tw: TableWriter,
                     tab_name: str, label: str, caption: str) -> None:
    pairs = [("B_simplified", "A_standard"), ("C_highly_simplified", "A_standard"),
             ("C_highly_simplified", "B_simplified")]
    pivot = ev.pivot_table(index=["concept_id", "model"],
                           columns="condition", values=outcome,
                           aggfunc="first").dropna()
    pivot = pivot[CONDITION_ORDER].apply(as_float)
    rows = []
    pvals = []
    for b, a in pairs:
        bb = int(((pivot[b] == 1) & (pivot[a] == 0)).sum())
        cc = int(((pivot[b] == 0) & (pivot[a] == 1)).sum())
        p = exact_mcnemar(bb, cc)
        pvals.append(p)
        rows.append((b, a, bb, cc, p))
    adj = holm_adjust(pvals)
    body_lines = []
    for (b, a, bb, cc, p), pa in zip(rows, adj):
        body_lines.append("%s vs %s & %d & %d & %s & %s \\\\"
                          % (CONDITION_LABEL[b], CONDITION_LABEL[a], bb, cc,
                             fmt_p(p), fmt_p(pa)))
    body = ("Contrast & $b$ & $c$ & $p$ & Holm $p$ \\\\\n\\midrule\n"
            + "\n".join(body_lines))
    tw.write(tab_name, body, caption, label, colspec="lrrrr")


def regression_tables(ev: pd.DataFrame, outcome: str, tw: TableWriter,
                      tab_name: str, label: str, caption: str) -> None:
    gee = gee_exchangeable_binary(ev, outcome)
    clog = conditional_logit_pair_fe(ev, outcome)
    bayes = bayes_mixed_logistic_vb(ev, outcome)
    lines = []
    for key in ["B_simplified_vs_A_standard", "C_highly_simplified_vs_A_standard"]:
        if key in gee:
            g = gee[key]
            lines.append("GEE & %s & %.2f & %.2f--%.2f & %s \\\\"
                         % (key, g["or"], g["ci_lo"], g["ci_hi"], fmt_p(g["p"])))
        if key in clog:
            g = clog[key]
            lines.append("Conditional logit & %s & %.2f & %.2f--%.2f & %s \\\\"
                         % (key, g["or"], g["ci_lo"], g["ci_hi"], fmt_p(g["p"])))
        if key in bayes:
            g = bayes[key]
            lines.append("Bayes mixed (VB) & %s & %.2f & %.2f--%.2f & -- \\\\"
                         % (key, g["or"], g["ci_lo"], g["ci_hi"]))
    if not lines:
        lines.append("-- & -- & -- & -- & -- \\\\")
    body = ("Model & Contrast & OR & 95\\% CI & $p$ \\\\\n\\midrule\n"
            + "\n".join(lines))
    tw.write(tab_name, body, caption, label, colspec="llrrr")


def accuracy_analysis(ev: pd.DataFrame, tw: TableWriter) -> None:
    lines = []
    for cond in CONDITION_ORDER:
        sub = ev[ev["condition"] == cond]
        n = len(sub)
        m = sub["accuracy_score"].mean()
        sd = sub["accuracy_score"].std(ddof=1)
        perfect = int((sub["accuracy_score"] == 5).sum())
        lines.append("%s & %d & %.2f (%.2f) & %d (%.1f\\%%) \\\\"
                     % (CONDITION_LABEL[cond], n, m, sd,
                        perfect, 100.0 * perfect / n))
    body = ("Condition & $n$ & Mean (SD) & Score 5 \\\\\n\\midrule\n"
            + "\n".join(lines))
    tw.write("tab_accuracy_by_condition.tex", body,
             "Accuracy scores (1--5) by condition.", "tab:accuracy:cond",
             colspec="lrrr")

    groups = [ev[ev["condition"] == c]["accuracy_score"].astype(float).values
              for c in CONDITION_ORDER]
    F, p = stats.f_oneway(*groups)
    eta2 = eta_squared_from_f(F, 2, len(ev) - 3)
    pivot = ev.pivot_table(index=["concept_id", "model"],
                           columns="condition", values="accuracy_score",
                           aggfunc="first").dropna()
    try:
        fr, pfr = stats.friedmanchisquare(*[pivot[c].values for c in CONDITION_ORDER])
        W = fr / (len(pivot) * (3 - 1))
    except Exception:
        fr, pfr, W = float("nan"), float("nan"), float("nan")
    pairs = [("B_simplified", "A_standard"), ("C_highly_simplified", "A_standard"),
             ("C_highly_simplified", "B_simplified")]
    wrows = []
    pvals = []
    for b, a in pairs:
        try:
            w, pw = stats.wilcoxon(pivot[b].values, pivot[a].values)
        except Exception:
            w, pw = float("nan"), float("nan")
        pvals.append(pw)
        wrows.append((b, a, w, pw))
    adj = holm_adjust(pvals)
    wlines = []
    for (b, a, w, pw), pa in zip(wrows, adj):
        d = smd_cohen_d(pivot[b].values, pivot[a].values)
        wlines.append("%s vs %s & $W = %.1f$ & %s & %s & $d = %.2f$ \\\\"
                      % (CONDITION_LABEL[b], CONDITION_LABEL[a], w,
                         fmt_p(pw), fmt_p(pa), d))
    body = ("Test & Statistic & df & $p$ & Effect size \\\\\n\\midrule\n"
            "One-way ANOVA & $F = %.2f$ & 2, %d & %s & $\\eta^2 = %.3f$ \\\\\n"
            "Friedman & $\\chi^2 = %.2f$ & 2 & %s & $W = %.3f$ \\\\"
            % (F, len(ev) - 3, fmt_p(p), eta2, fr, fmt_p(pfr), W))
    tw.write("tab_accuracy_tests.tex", body,
             "Omnibus and paired tests for accuracy scores.",
             "tab:accuracy:tests", colspec="lrrrr")
    if wlines:
        body = ("Contrast & Wilcoxon $W$ & $p$ & Holm $p$ & Cohen's $d$ \\\\\n\\midrule\n"
                + "\n".join(wlines))
        tw.write("tab_accuracy_pairwise.tex", body,
                 "Paired Wilcoxon tests for accuracy scores with Holm correction.",
                 "tab:accuracy:pairwise", colspec="lrrrr")
    lines = []
    for outcome, lab in [("strategic_misconception", "SM"),
                         ("dangerous_analogy", "DA"),
                         ("oversimplification_error", "OER")]:
        sub = ev.dropna(subset=["accuracy_score", outcome])
        if len(sub) < 3 or sub[outcome].nunique() < 2:
            lines.append("%s & -- & -- \\\\" % lab)
            continue
        rho, p = stats.spearmanr(sub["accuracy_score"], as_float(sub[outcome]).astype(int))
        lines.append("%s & %.2f & %s \\\\" % (lab, rho, fmt_p(p)))
    body = ("Indicator & Spearman $\\rho$ & $p$ \\\\\n\\midrule\n" + "\n".join(lines))
    tw.write("tab_accuracy_spearman.tex", body,
             "Spearman correlations between accuracy scores and the binary error indicators.",
             "tab:accuracy:spearman", colspec="lrr")


def subfield_analysis(ev: pd.DataFrame, corpus: pd.DataFrame, tw: TableWriter) -> None:
    domain_map = corpus[["concept_id", "domain"]].drop_duplicates()
    ev2 = ev.drop(columns=["domain"], errors="ignore").merge(
        domain_map, on="concept_id", how="left")
    rows = []
    for dom in SUBFIELD_ORDER:
        sub_dom = ev2[ev2["domain"] == dom]
        for cond in CONDITION_ORDER:
            sub = sub_dom[sub_dom["condition"] == cond]
            n = len(sub)
            if n == 0:
                continue
            smr = as_float(sub["strategic_misconception"]).mean()
            dar = as_float(sub["dangerous_analogy"]).mean()
            oer = (as_float(sub["oversimplification_error"]).mean()
                   if cond != "A_standard" else float("nan"))
            acc = sub["accuracy_score"].mean()
            rows.append("%s & %s & %.1f & %.1f & %s & %.2f \\\\"
                        % (SUBFIELD_SHORT[dom], CONDITION_LABEL[cond],
                           100 * smr, 100 * dar,
                           ("%.1f" % (100 * oer)) if not math.isnan(oer) else "--",
                           acc))
    body = ("Subfield & Condition & SMR (\\%) & DAR (\\%) & OER (\\%) & Accuracy \\\\\n"
            "\\midrule\n" + "\n".join(rows))
    tw.write("tab_subfield.tex", body,
             "Error rates and accuracy by CS subfield and condition.",
             "tab:subfield", colspec="llrrrr")
    anova_rows = []
    raw_ps = []
    for outcome, lab in [("strategic_misconception", "SM"),
                         ("dangerous_analogy", "DA"),
                         ("oversimplification_error", "OER"),
                         ("accuracy_score", "Accuracy")]:
        for cond in CONDITION_ORDER:
            sub = ev2[(ev2["condition"] == cond)].dropna(subset=[outcome])
            groups = [as_float(sub[sub["domain"] == d][outcome]).values
                      for d in SUBFIELD_ORDER]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) < 2:
                continue
            F, p = stats.f_oneway(*groups)
            raw_ps.append(p)
            anova_rows.append((outcome, lab, cond, F, p))
    adj = benjamini_hochberg(raw_ps)
    body_lines = []
    for (outcome, lab, cond, F, p), pa in zip(anova_rows, adj):
        body_lines.append("%s & %s & %.2f & %s & %s \\\\"
                          % (lab, CONDITION_LABEL[cond], F, fmt_p(p), fmt_p(pa)))
    body = ("Outcome & Condition & $F$ & $p$ & BH $p$ \\\\\n\\midrule\n"
            + "\n".join(body_lines))
    tw.write("tab_subfield_tests.tex", body,
             "One-way ANOVAs of each outcome across subfields, within each condition.",
             "tab:subfield:tests", colspec="llrrr")


def concept_top10(ev: pd.DataFrame, corpus: pd.DataFrame, tw: TableWriter) -> None:
    pivot = ev.pivot_table(index=["concept_id", "model"],
                           columns="condition",
                           values="strategic_misconception",
                           aggfunc="first").dropna().reset_index()
    pivot["A_standard"] = as_float(pivot["A_standard"])
    pivot["B_simplified"] = as_float(pivot["B_simplified"])
    pivot["C_highly_simplified"] = as_float(pivot["C_highly_simplified"])
    pivot["mean_BC"] = (pivot["B_simplified"] + pivot["C_highly_simplified"]) / 2.0
    pivot["delta"] = pivot["mean_BC"] - pivot["A_standard"]
    agg = pivot.groupby("concept_id")["delta"].mean().reset_index()
    name_map = corpus[["concept_id", "concept"]].drop_duplicates().set_index("concept_id")["concept"]
    agg["concept"] = agg["concept_id"].map(name_map)
    agg = agg.sort_values("delta", ascending=False).head(10)
    lines = []
    for _, r in agg.iterrows():
        lines.append("%s & %s & %+.1f \\\\"
                     % (r["concept_id"], r["concept"], 100 * r["delta"]))
    body = ("ID & Concept & $\\Delta$ SMR (pp) \\\\\n\\midrule\n" + "\n".join(lines))
    tw.write("tab_concepts_top10.tex", body,
             "Concepts with the largest mean increase in SMR from A to mean(B, C).",
             "tab:concepts:top10", colspec="llr")


def concept_all(corpus: pd.DataFrame, tw: TableWriter) -> None:
    df = corpus[["concept_id", "domain", "concept"]].drop_duplicates()
    df["domain_short"] = df["domain"].map(SUBFIELD_SHORT).fillna(df["domain"])
    df = df.sort_values(["domain", "concept_id"])
    lines = []
    for _, r in df.iterrows():
        lines.append("%s & %s & %s \\\\"
                     % (r["concept_id"], r["domain_short"], r["concept"]))
    body = ("ID & Subfield & Concept \\\\\n\\midrule\n" + "\n".join(lines))
    tw.write("tab_concepts_all.tex", body,
             "The 35 CS concepts in AccessTrap-CS, grouped by subfield.",
             "tab:concepts:all", colspec="lll")


def model_analysis(ev: pd.DataFrame, tw: TableWriter) -> None:
    rows = []
    for m in MODEL_ORDER:
        sub = ev[ev["model"] == m]
        if len(sub) == 0:
            continue
        smr = as_float(sub["strategic_misconception"]).mean()
        dar = as_float(sub["dangerous_analogy"]).mean()
        oer = as_float(sub[sub["condition"].isin(["B_simplified", "C_highly_simplified"])][
            "oversimplification_error"]).mean()
        acc = sub["accuracy_score"].mean()
        rows.append("%s & %.1f & %.1f & %.1f & %.2f \\\\"
                    % (m, 100 * smr, 100 * dar, 100 * oer, acc))
    body = ("Model & SMR (\\%) & DAR (\\%) & OER (\\%) & Accuracy \\\\\n\\midrule\n"
            + "\n".join(rows))
    tw.write("tab_models.tex", body,
             "Error rates and accuracy by model, pooled over conditions.",
             "tab:models", colspec="lrrrr")

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except Exception as exc:
        LOG.warning("statsmodels not available for two-way ANOVA: %s", exc)
        return
    all_lines = []
    for outcome, lab in [("strategic_misconception", "SM"),
                         ("accuracy_score", "Accuracy")]:
        d = ev[["model", "condition", outcome]].dropna().copy()
        d[outcome] = as_float(d[outcome])
        d["condition"] = pd.Categorical(d["condition"], categories=CONDITION_ORDER)
        d["model"] = pd.Categorical(d["model"], categories=MODEL_ORDER)
        formula = "%s ~ C(model) + C(condition) + C(model):C(condition)" % outcome
        try:
            model = ols(formula, data=d).fit()
            aov = sm.stats.anova_lm(model, typ=2)
        except Exception:
            continue
        for term in aov.index:
            row = aov.loc[term]
            all_lines.append("%s & %s & %.2f & %d & %s \\\\"
                             % (lab, term, row["F"], int(row["df"]),
                                fmt_p(row["PR(>F)"])))
    if all_lines:
        body = ("Outcome & Term & $F$ & df & $p$ \\\\\n\\midrule\n" + "\n".join(all_lines))
        tw.write("tab_twoway_anova.tex", body,
                 "Two-way ANOVA (Type II) for SM and accuracy.",
                 "tab:twoway:anova", colspec="llrrr")


def position_bias(ev: pd.DataFrame, tw: TableWriter) -> None:
    rows = []
    for pos in [1, 2, 3]:
        sub = ev[ev["position_shown"] == pos]
        n = len(sub)
        oer_sub = sub[sub["condition"].isin(["B_simplified", "C_highly_simplified"])]
        rows.append("%d & %d & %.1f & %.1f & %.1f & %.2f \\\\"
                    % (pos, n,
                       100 * as_float(sub["strategic_misconception"]).mean(),
                       100 * as_float(sub["dangerous_analogy"]).mean(),
                       100 * as_float(oer_sub["oversimplification_error"]).mean(),
                       sub["accuracy_score"].mean()))
    body = ("Position & $n$ & SMR (\\%) & DAR (\\%) & OER (\\%) & Accuracy \\\\\n"
            "\\midrule\n" + "\n".join(rows))
    tw.write("tab_position_bias.tex", body,
             "Judge decisions by the position in which the response was shown.",
             "tab:position:bias", colspec="lrrrrr")
    ct = pd.crosstab(ev["position_shown"], ev["condition"])
    lines = []
    for pos in ct.index:
        lines.append("%d & %d & %d & %d \\\\"
                     % (pos, ct.loc[pos].get("A_standard", 0),
                        ct.loc[pos].get("B_simplified", 0),
                        ct.loc[pos].get("C_highly_simplified", 0)))
    body = ("Position & A & B & C \\\\\n\\midrule\n" + "\n".join(lines))
    tw.write("tab_position_by_condition.tex", body,
             "Cross-tabulation of position shown by condition.",
             "tab:position:bycond", colspec="lrrr")


def first_shown(ev: pd.DataFrame, tw: TableWriter) -> None:
    sub = ev[ev["position_shown"] == 1]
    rows = []
    for cond in CONDITION_ORDER:
        s = sub[sub["condition"] == cond]
        if len(s) == 0:
            continue
        oer_sub = s[s["condition"].isin(["B_simplified", "C_highly_simplified"])]
        rows.append("%s & %d & %.1f & %.1f & %.1f & %.2f \\\\"
                    % (CONDITION_LABEL[cond], len(s),
                       100 * as_float(s["strategic_misconception"]).mean(),
                       100 * as_float(s["dangerous_analogy"]).mean(),
                       100 * as_float(oer_sub["oversimplification_error"]).mean(),
                       s["accuracy_score"].mean()))
    body = ("Condition & $n$ & SMR (\\%) & DAR (\\%) & OER (\\%) & Accuracy \\\\\n"
            "\\midrule\n" + "\n".join(rows))
    tw.write("tab_sens_first_shown.tex", body,
             "Rates for responses shown first in each group.",
             "tab:sens:first", colspec="lrrrrr")


def baseline_before(ev: pd.DataFrame, tw: TableWriter) -> None:
    pos_A = (ev[ev["condition"] == "A_standard"]
             .set_index(["concept_id", "model"])["position_shown"])
    simp = ev[ev["condition"].isin(["B_simplified", "C_highly_simplified"])].copy()
    simp = simp.set_index(["concept_id", "model"])
    simp["pos_A"] = pos_A
    simp = simp.reset_index()
    simp["A_before"] = simp["pos_A"] < simp["position_shown"]
    rows = []
    for label, mask in [("A before", simp["A_before"]),
                        ("A after", ~simp["A_before"])]:
        sub = simp[mask]
        rows.append("%s & %d & %.1f & %.1f & %.1f & %.2f \\\\"
                    % (label, len(sub),
                       100 * as_float(sub["strategic_misconception"]).mean(),
                       100 * as_float(sub["dangerous_analogy"]).mean(),
                       100 * as_float(sub["oversimplification_error"]).mean(),
                       sub["accuracy_score"].mean()))
    body = ("A position & $n$ & SMR (\\%) & DAR (\\%) & OER (\\%) & Accuracy \\\\\n"
            "\\midrule\n" + "\n".join(rows))
    tw.write("tab_sens_A_before.tex", body,
             "Simplified responses judged after vs before the baseline was shown.",
             "tab:sens:Abefore", colspec="lrrrrr")


def position_adjusted_gee(ev: pd.DataFrame, tw: TableWriter) -> None:
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
    except Exception as exc:
        LOG.warning("statsmodels not available for position-adjusted GEE: %s", exc)
        return
    lines = []
    for outcome, lab in [("strategic_misconception", "SM"),
                         ("dangerous_analogy", "DA")]:
        d = ev[["concept_id", "model", "condition", "position_shown", outcome]].dropna().copy()
        d["y"] = as_float(d[outcome]).astype(int)
        for cond in CONDITION_ORDER[1:]:
            d["is_" + cond] = (d["condition"] == cond).astype(int)
        d["cluster"] = d["concept_id"].astype(str) + "|" + d["model"].astype(str)
        X = sm.add_constant(d[["is_B_simplified", "is_C_highly_simplified", "position_shown"]])
        try:
            res = sm.GEE(d["y"], X, groups=d["cluster"], family=Binomial(),
                         cov_struct=Exchangeable()).fit()
        except Exception as exc:
            LOG.warning("Position-adjusted GEE failed for %s: %s", outcome, exc)
            continue
        for cond in CONDITION_ORDER[1:]:
            key = "is_" + cond
            beta = res.params[key]; se = res.bse[key]; p = res.pvalues[key]
            lines.append("%s & %s vs A & %.2f & %.2f--%.2f & %s \\\\"
                         % (lab, cond, math.exp(beta),
                            math.exp(beta - 1.96 * se),
                            math.exp(beta + 1.96 * se), fmt_p(p)))
    if not lines:
        lines.append("-- & -- & -- & -- & -- \\\\")
    body = ("Outcome & Contrast & OR & 95\\% CI & $p$ \\\\\n\\midrule\n" + "\n".join(lines))
    tw.write("tab_position_adjusted.tex", body,
             "GEE models for SM and DA adjusted for position.",
             "tab:position:adjusted", colspec="llrrr")


# ---------------------------------------------------------------------------
# Human consensus analysis
# ---------------------------------------------------------------------------

def human_consensus_analysis(annotations_final: pd.DataFrame, tw: TableWriter) -> None:
    """Compute human consensus rates by condition and write a new table."""
    cons = annotations_final[annotations_final["rater"] == "consensus"].copy()
    if len(cons) == 0:
        LOG.warning("No consensus rows found; skipping human consensus analysis.")
        return

    for col in ["SM", "DA", "OER"]:
        cons[col] = pd.to_numeric(cons[col], errors="coerce")
    cons["accuracy"] = pd.to_numeric(cons["accuracy"], errors="coerce")

    lines = []
    for ind, label in [("SM", "SM"), ("DA", "DA"), ("OER", "OER")]:
        for cond in CONDITION_ORDER:
            sub = cons[cons["condition"] == cond]
            vals = sub[ind].dropna()
            if len(vals) == 0:
                continue
            n = len(vals)
            k = int(vals.sum())
            lo, hi = wilson_ci(k, n)
            lines.append("%s & %s & %d/%d & %.1f [%.1f, %.1f] \\\\"
                         % (label, CONDITION_LABEL[cond], k, n,
                            100 * k / n, 100 * lo, 100 * hi))

    body = ("Indicator & Condition & Count & Rate (\\%) [95\\% CI] \\\\\n\\midrule\n"
            + "\n".join(lines))
    tw.write("tab_reviewer_consensus.tex", body,
             "Human consensus rates by condition on the validation subsample. "
             "SM, DA, OER are binary indicators; Wilson 95\\% confidence intervals "
             "are shown. OER is undefined in condition A by construction. "
             "These rates represent the adjudicated agreement of two domain experts "
             "and serve as the human reference standard.",
             "tab:reviewer:consensus", colspec="llrrr")

    # Accuracy table (separate)
    acc_lines = []
    for cond in CONDITION_ORDER:
        sub = cons[cons["condition"] == cond]
        vals = sub["accuracy"].dropna()
        if len(vals) == 0:
            continue
        acc_lines.append("%s & %d & %.2f (%.2f) \\\\"
                         % (CONDITION_LABEL[cond], len(vals),
                            vals.mean(), vals.std(ddof=1)))
    body_acc = ("Condition & $n$ & Mean (SD) \\\\\n\\midrule\n" + "\n".join(acc_lines))
    tw.write("tab_reviewer_consensus_accuracy.tex", body_acc,
             "Human consensus accuracy scores by condition on the validation subsample.",
             "tab:reviewer:consensus:accuracy", colspec="lrr")


# ---------------------------------------------------------------------------
# Judge validation
# ---------------------------------------------------------------------------

def judge_validation(eval_df: pd.DataFrame, annotations_final: pd.DataFrame,
                     validation_map: pd.DataFrame, tw: TableWriter) -> None:
    """Validate the LLM judge against the two human reviewers."""
    keys = ["concept_id", "model", "condition"]

    # Filtriraj samo na 150 validacijskih uzoraka (ispravak propusta)
    val_keys = validation_map[keys].drop_duplicates()
    
    eval_df_val = eval_df.merge(val_keys, on=keys, how="inner")
    raters = annotations_final[annotations_final["rater"].isin(["h1", "h2"])].copy()
    raters = raters.merge(val_keys, on=keys, how="inner")
    consensus = annotations_final[annotations_final["rater"] == "consensus"].copy()
    consensus = consensus.merge(val_keys, on=keys, how="inner")

    # ----- Reviewer agreement (h1 vs h2) -----
    long = raters.melt(id_vars=["rater"] + keys,
                       value_vars=["SM", "DA", "OER", "accuracy"],
                       var_name="indicator", value_name="value")
    wide = long.pivot_table(index=keys + ["indicator"],
                            columns="rater", values="value",
                            aggfunc="first").reset_index()

    agreement_rows = []
    for ind in ["SM", "DA", "OER"]:
        sub = wide[wide["indicator"] == ind].dropna(subset=["h1", "h2"])
        if len(sub) == 0:
            continue
        k = cohen_kappa(sub["h1"].astype(int), sub["h2"].astype(int))
        po = (sub["h1"] == sub["h2"]).mean()
        agreement_rows.append("%s & %.3f & %.1f \\\\" % (ind, k, 100 * po))
    sub_acc = wide[wide["indicator"] == "accuracy"].dropna(subset=["h1", "h2"])
    if len(sub_acc) > 0:
        qwk = quadratic_weighted_kappa(sub_acc["h1"].astype(int),
                                       sub_acc["h2"].astype(int),
                                       labels=[1, 2, 3, 4, 5])
        rho, _ = stats.spearmanr(sub_acc["h1"], sub_acc["h2"])
        agreement_rows.append("Accuracy & %.3f & %.3f \\\\" % (qwk, rho))

    body = ("Indicator & Agreement & Value \\\\\n\\midrule\n" + "\n".join(agreement_rows))
    tw.write("tab_reviewer_agreement.tex", body,
             "Agreement between the two reviewers. For SM, DA, OER: Cohen's "
             "$\\kappa$ and raw agreement (\\%). For accuracy: quadratic-weighted "
             "$\\kappa$ and Spearman's $\\rho$.",
             "tab:reviewer:agreement", colspec="llr")

    # ----- Judge vs consensus (with h1 fallback for accuracy and missing) -----
    judge_cols = keys + ["strategic_misconception", "dangerous_analogy",
                         "oversimplification_error", "accuracy_score"]
    judge = eval_df_val[judge_cols].copy()
    for c in ["strategic_misconception", "dangerous_analogy", "oversimplification_error"]:
        judge[c] = as_float(judge[c])
    judge = judge.rename(columns={
        "strategic_misconception": "SM_judge",
        "dangerous_analogy": "DA_judge",
        "oversimplification_error": "OER_judge",
        "accuracy_score": "ACC_judge",
    })

    wide = wide.rename(columns={"h1": "h1_orig", "h2": "h2_orig"})

    if len(consensus) > 0:
        cons_long = consensus.melt(id_vars=["rater"] + keys,
                                   value_vars=["SM", "DA", "OER", "accuracy"],
                                   var_name="indicator", value_name="consensus")
        h1_long = raters[raters["rater"] == "h1"].melt(
            id_vars=["rater"] + keys,
            value_vars=["SM", "DA", "OER", "accuracy"],
            var_name="indicator", value_name="h1_fallback")
        merged = wide.merge(cons_long[keys + ["indicator", "consensus"]],
                            on=keys + ["indicator"], how="left")
        merged = merged.merge(h1_long[keys + ["indicator", "h1_fallback"]],
                              on=keys + ["indicator"], how="left")
        merged["consensus"] = merged["consensus"].fillna(merged["h1_fallback"])
    else:
        merged = wide.copy()
        merged["consensus"] = merged["h1_orig"]

    merged = merged.merge(judge, on=keys, how="inner")

    judge_rows = []
    for ind, judge_col in [("SM", "SM_judge"), ("DA", "DA_judge"), ("OER", "OER_judge")]:
        sub = merged[merged["indicator"] == ind].dropna(subset=["consensus", judge_col])
        if len(sub) == 0:
            continue
        ref = sub["consensus"].astype(int)
        jd = sub[judge_col].astype(int)
        k = cohen_kappa(ref, jd)
        tp = int(((jd == 1) & (ref == 1)).sum())
        fp = int(((jd == 1) & (ref == 0)).sum())
        fn = int(((jd == 0) & (ref == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        prev_ref = ref.mean()
        prev_judge = jd.mean()
        judge_rows.append("%s & %.3f & %.3f & %.3f & %.3f & %.3f \\\\"
                          % (ind, k, prec, rec, prev_ref, prev_judge))

    sub_acc = merged[merged["indicator"] == "accuracy"].dropna(subset=["consensus", "ACC_judge"])
    if len(sub_acc) > 0:
        qwk = quadratic_weighted_kappa(sub_acc["consensus"].astype(int),
                                       sub_acc["ACC_judge"].astype(int),
                                       labels=[1, 2, 3, 4, 5])
        rho, _ = stats.spearmanr(sub_acc["consensus"], sub_acc["ACC_judge"])
        judge_rows.append("Accuracy & %.3f & -- & -- & %.2f & %.2f \\\\"
                          % (qwk, sub_acc["consensus"].mean(), sub_acc["ACC_judge"].mean()))

    body = ("Indicator & $\\kappa$ & Precision & Recall & Reviewer prev. & Judge prev. \\\\\n"
            "\\midrule\n" + "\n".join(judge_rows))
    tw.write("tab_judge_validation.tex", body,
             "Agreement between the LLM judge and the reviewer consensus. Precision is "
             "the proportion of judge-flagged responses also flagged by reviewers; recall "
             "is the proportion of reviewer-flagged responses also flagged by the judge. "
             "For accuracy, quadratic-weighted $\\kappa$ is shown and prevalence is the mean score.",
             "tab:judge:validation", colspec="lrrrrr")

    # ----- Reviewer rates by condition -----
    rows = []
    for rater in ["h1", "h2"]:
        sub_r = raters[raters["rater"] == rater]
        for cond in CONDITION_ORDER:
            sub = sub_r[sub_r["condition"] == cond]
            if len(sub) == 0:
                continue
            sm = sub["SM"].mean()
            da = sub["DA"].mean()
            oer_vals = sub["OER"].dropna()
            oer = oer_vals.mean() if len(oer_vals) > 0 else float("nan")
            acc = sub["accuracy"].mean()
            rows.append("%s & %s & %.1f & %.1f & %s & %.2f \\\\"
                        % (rater, CONDITION_LABEL[cond], 100 * sm, 100 * da,
                           ("%.1f" % (100 * oer)) if not math.isnan(oer) else "--",
                           acc))
    body = ("Reviewer & Condition & SM (\\%) & DA (\\%) & OER (\\%) & Accuracy \\\\\n"
            "\\midrule\n" + "\n".join(rows))
    tw.write("tab_reviewer_rates.tex", body,
             "Reviewer rates by condition on the validation subsample.",
             "tab:reviewer:rates", colspec="llrrrr")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figures(eval_df: pd.DataFrame, corpus: pd.DataFrame, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#E3E3E3",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    })

    COND_COL = {"A_standard": "#9DB9D6",
                "B_simplified": "#4A7FB0",
                "C_highly_simplified": "#1F3F66"}
    COND_TXT = {"A_standard": "#222222",
                "B_simplified": "white",
                "C_highly_simplified": "white"}
    METRIC_COL = {"SMR": "#2F7F6F", "DAR": "#C0763A", "OER": "#7A5CA8"}
    MEAN_COL = "#D1495B"
    TXT, REF = "#222222", "#666666"

    XLAB = ["A\n(standard)", "B\n(simplified)", "C\n(highly simplified)"]
    ERR = dict(ecolor="#333333", elinewidth=0.8, capthick=0, capsize=0)

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        fig.savefig(figdir / f"{name}.pdf")
        fig.savefig(figdir / f"{name}.png", dpi=600)
        plt.close(fig)

    def _fmt(v):
        return "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "−")

    def _wrap(name):
        # prelomi dugačko ime u dva reda na razmaku najbližem sredini
        if len(name) <= 11 or " " not in name:
            return name
        spaces = [i for i, ch in enumerate(name) if ch == " "]
        k = min(spaces, key=lambda i: abs(i - len(name) / 2))
        return name[:k] + "\n" + name[k + 1:]

    def _bar(ax, xpos, m, w, color, lo, hi, label_fs=8.5):
        """Bar s error barom i vrijednošću iznad; ako je 0, nacrta crtu na 0."""
        ax.bar(xpos, m, w, color=color, edgecolor="white", linewidth=0.5,
               zorder=3, yerr=[[max(m - lo, 0)], [max(hi - m, 0)]],
               error_kw=ERR)
        if m == 0:
            ax.hlines(0, xpos - w / 2, xpos + w / 2, color=color,
                      linewidth=2.2, zorder=4)
        ax.text(xpos, hi + 0.4, f"{m:.1f}", ha="center", va="bottom",
                fontsize=label_fs, color=TXT)

    models = [m for m in MODEL_ORDER if m in eval_df["model"].unique()]

    # Wilson statistika po (model, uvjet) — koristi se u više grafova
    stats_dict = {}
    for m in models:
        for cond in CONDITION_ORDER:
            sub = eval_df[(eval_df["model"] == m) & (eval_df["condition"] == cond)]
            nn = len(sub)
            if nn == 0:
                continue
            k = int(as_float(sub["strategic_misconception"]).sum())
            lo, hi = wilson_ci(k, nn)
            stats_dict[(m, cond)] = (100 * k / nn, 100 * lo, 100 * hi)

    def _rates(df, outcome):
        out = []
        for cond in CONDITION_ORDER:
            sub = df[df["condition"] == cond]
            n = len(sub)
            k = int(as_float(sub[outcome]).sum())
            lo, hi = wilson_ci(k, n)
            out.append((100 * k / n, 100 * lo, 100 * hi))
        return np.array(out)

    # =====================================================================
    # Figure 1: accuracy po uvjetu — bar chart, os od 2,
    # "overall mean" linija IZNAD barova, brojevi na dnu barova
    # =====================================================================
    YMIN = 2.0
    fig, ax = plt.subplots(figsize=(5.6, 3.5), layout="constrained")
    means, errs = [], []
    for cond in CONDITION_ORDER:
        sub = eval_df[eval_df["condition"] == cond]["accuracy_score"]
        means.append(sub.mean())
        errs.append(1.96 * sub.std(ddof=1) / math.sqrt(len(sub)))
    grand = float(eval_df["accuracy_score"].mean())
    x = np.arange(3)
    ax.bar(x, means, 0.55, color=[COND_COL[c] for c in CONDITION_ORDER],
           edgecolor="white", linewidth=0.6, zorder=3,
           yerr=errs, error_kw=ERR)
    ax.axhline(grand, color=MEAN_COL, linewidth=1.4, linestyle="--", zorder=5)
    for xi, cond, m in zip(x, CONDITION_ORDER, means):
        ax.text(xi, YMIN + 0.12, f"{m:.2f}", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=COND_TXT[cond], zorder=6)
    ax.set_ylim(YMIN, 5.0)
    ax.set_yticks(np.arange(YMIN, 5.01, 0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(XLAB)
    ax.set_ylabel("Mean accuracy (1–5), 95% CI")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([0], [0], color=MEAN_COL, linestyle="--",
                              linewidth=1.4,
                              label=f"Overall mean = {grand:.2f}")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, "fig_accuracy_by_condition")

    # =====================================================================
    # Figure 2: SMR po modelu, A vs C — grupirani barovi
    # Δ (C − A) piše se ispod imena modela.
    # =====================================================================
    order = sorted(models, key=lambda m: stats_dict[(m, "C_highly_simplified")][0],
                   reverse=True)
    w = 0.34
    fig, ax = plt.subplots(figsize=(8.0, 3.9), layout="constrained")
    top = 0
    xt = []
    for i, m in enumerate(order):
        for off, cond in ((-w / 2, "A_standard"), (w / 2, "C_highly_simplified")):
            p, lo, hi = stats_dict[(m, cond)]
            _bar(ax, i + off, p, w, COND_COL[cond], lo, hi)
            top = max(top, hi)
        d = stats_dict[(m, "C_highly_simplified")][0] - stats_dict[(m, "A_standard")][0]
        xt.append(f"{_wrap(m)}\n$\\Delta$ = {_fmt(d)}")
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xt, fontsize=8.5)
    ax.set_ylabel("SMR (%), Wilson 95% CI")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=COND_COL[c], label=CONDITION_LABEL[c])
                       for c in ("A_standard", "C_highly_simplified")],
              loc="upper right", ncol=2, columnspacing=1.0)
    _save(fig, "fig_model_delta_smr")

    # =====================================================================
    # Figure 3: SMR, DAR, OER po uvjetu — nule su prikazane crtom i "0.0"
    # =====================================================================
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    series = [("strategic_misconception", "SMR"),
              ("dangerous_analogy", "DAR"),
              ("oversimplification_error", "OER")]
    rates = {lab: _rates(eval_df, col) for col, lab in series}
    width = 0.24
    offsets = [-width, 0.0, width]
    top = 0
    for i, cond in enumerate(CONDITION_ORDER):
        for (_, lab), off in zip(series, offsets):
            if lab == "OER" and cond == "A_standard":
                m, lo, hi = 0.0, 0.0, 0.0      # OER u A = 0
            else:
                m, lo, hi = rates[lab][i]
            _bar(ax, i + off, m, width, METRIC_COL[lab], lo, hi)
            top = max(top, hi)
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(range(3))
    ax.set_xticklabels(XLAB)
    ax.set_ylabel("Rate (%), Wilson 95% CI")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=METRIC_COL[l], label=l) for _, l in series],
              loc="upper left", ncol=3, columnspacing=1.0)
    _save(fig, "fig_rates_by_condition")

    # =====================================================================
    # Figure 4: SMR po modelu i uvjetu — grupirani barovi (A/B/C)
    # Modeli sortirani po prosječnom SMR-u, vrijednosti iznad barova.
    # =====================================================================
    order4 = sorted(models, reverse=True,
                    key=lambda m: np.mean([stats_dict[(m, c)][0]
                                           for c in CONDITION_ORDER
                                           if (m, c) in stats_dict]))
    w4 = 0.26
    offs4 = [-w4, 0.0, w4]
    fig, ax = plt.subplots(figsize=(9.0, 4.0), layout="constrained")
    top = 0
    for i, m in enumerate(order4):
        for off, cond in zip(offs4, CONDITION_ORDER):
            if (m, cond) not in stats_dict:
                continue
            p, lo, hi = stats_dict[(m, cond)]
            _bar(ax, i + off, p, w4, COND_COL[cond], lo, hi, label_fs=7.5)
            top = max(top, hi)
    ax.set_ylim(0, top * 1.18)
    ax.set_xticks(range(len(order4)))
    ax.set_xticklabels([_wrap(m) for m in order4], fontsize=8.5)
    ax.set_ylabel("SMR (%), Wilson 95% CI")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(color=COND_COL[c], label=CONDITION_LABEL[c])
                       for c in CONDITION_ORDER],
              loc="upper right", ncol=3, columnspacing=1.0)
    _save(fig, "fig_smr_by_model")

    # =====================================================================
    # Figure 5: heatmap po poddomeni — NEPROMIJENJENO
    # =====================================================================
    domain_map = corpus[["concept_id", "domain"]].drop_duplicates()
    ev2 = eval_df.drop(columns=["domain"], errors="ignore").merge(
        domain_map, on="concept_id", how="left")
    mat = np.zeros((len(SUBFIELD_ORDER), 3))
    for i, dom in enumerate(SUBFIELD_ORDER):
        for j, cond in enumerate(CONDITION_ORDER):
            sub = ev2[(ev2["domain"] == dom) & (ev2["condition"] == cond)]
            if len(sub):
                mat[i, j] = 100 * as_float(sub["strategic_misconception"]).mean()
    vmax = max(10, np.ceil(mat.max() / 5) * 5)
    cmap = LinearSegmentedColormap.from_list("smr_seq", ["#F4F8FB", "#1F3F66"])

    fig, ax = plt.subplots(figsize=(5.8, 3.2), layout="constrained")
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_xticklabels(XLAB)
    ax.set_yticks(range(len(SUBFIELD_ORDER)))
    ax.set_yticklabels([SUBFIELD_SHORT[d].replace("\\\\", "").replace("\\", "")
                        for d in SUBFIELD_ORDER], fontsize=8.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9.5,
                    fontweight="bold",
                    color="white" if v > vmax * 0.5 else TXT)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SUBFIELD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("SMR (%)", fontsize=8.5)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=0)
    _save(fig, "fig_subfield_heatmap")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default=".", type=str,
                    help="Root repozitorija (sadrži data/ i annotation/).")
    ap.add_argument("--outdir", default="results", type=str,
                    help="Gdje pisati tables/ i figures/ (default: results).")
    ap.add_argument("--only", default="all", choices=["all", "tables", "figures"])
    args = ap.parse_args()

    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir / "accesstrap_stats.log")

    LOG.info("AccessTrap-CS statistics pipeline")
    LOG.info("Data directory:  %s", datadir.resolve())
    LOG.info("Output directory: %s", outdir.resolve())

    # Provjeri da su svi ulazi dostupni prije početka
    required = {
        "corpus.csv": ["", "data"],
        "evaluations.csv": ["", "data"],
        "annotations_final.csv": ["", "annotation"],
        "validation_map.csv": ["", "annotation"],
    }
    missing = []
    for fname, subs in required.items():
        if not any((datadir / s / fname).exists() for s in subs):
            missing.append(fname)
    if missing:
        LOG.error("Nedostaju ulazne datoteke: %s", missing)
        sys.exit(1)

    tw = TableWriter(outdir=outdir)
    corpus = load_corpus(datadir)
    eval_df = load_evaluations(datadir)
    annotations_final = load_annotations(datadir)
    validation_map = load_validation_map(datadir)

    if args.only in ("all", "tables"):
        LOG.info("--- Tables ---")
        sample_table(eval_df, tw)
        readability_table(corpus, tw)

        condition_summary(eval_df, "strategic_misconception", tw,
                          "tab_sm_by_condition.tex", "tab:sm:cond",
                          "Strategic misconception rate (SMR) by condition.")
        omnibus_tests(eval_df, "strategic_misconception", tw,
                      "tab_sm_omnibus.tex", "tab:sm:omnibus",
                      "Omnibus tests for SMR.")
        pairwise_mcnemar(eval_df, "strategic_misconception", tw,
                         "tab_sm_pairwise.tex", "tab:sm:pairwise",
                         "Exact McNemar tests for SMR with Holm correction.")
        regression_tables(eval_df, "strategic_misconception", tw,
                          "tab_sm_regression.tex", "tab:sm:regression",
                          "GEE, conditional logit and Bayes mixed models for SMR.")

        condition_summary(eval_df, "dangerous_analogy", tw,
                          "tab_da_by_condition.tex", "tab:da:cond",
                          "Dangerous analogy rate (DAR) by condition.")
        omnibus_tests(eval_df, "dangerous_analogy", tw,
                      "tab_da_omnibus.tex", "tab:da:omnibus",
                      "Omnibus tests for DAR.")
        pairwise_mcnemar(eval_df, "dangerous_analogy", tw,
                         "tab_da_pairwise.tex", "tab:da:pairwise",
                         "Exact McNemar tests for DAR with Holm correction.")
        regression_tables(eval_df, "dangerous_analogy", tw,
                          "tab_da_regression.tex", "tab:da:regression",
                          "GEE, conditional logit and Bayes mixed models for DAR.")

        condition_summary(eval_df, "oversimplification_error", tw,
                          "tab_oer_by_condition.tex", "tab:oer:cond",
                          "Oversimplification error rate (OER) by condition.")

        pivot = eval_df.pivot_table(
            index=["concept_id", "model"], columns="condition",
            values="strategic_misconception", aggfunc="first").dropna()
        pivot = pivot[CONDITION_ORDER].apply(as_float)
        derived = []
        for cond in ["B_simplified", "C_highly_simplified"]:
            mask = (pivot[cond] == 1) & (pivot["A_standard"] == 0)
            k = int(mask.sum()); n = len(pivot)
            lo, hi = wilson_ci(k, n)
            derived.append("%s & %d & %d & %.1f [%.1f, %.1f] \\\\"
                           % (CONDITION_LABEL[cond], k, n, 100 * k / n, 100 * lo, 100 * hi))
        body = ("Condition & Count & $n$ & Rate (\\%) [95\\% CI] \\\\\n\\midrule\n"
                + "\n".join(derived))
        tw.write("tab_oer_derived.tex", body,
                 "Derived OER: proportion of simplified responses with SM = 1 whose standard "
                 "response has SM = 0.",
                 "tab:oer:derived", colspec="lrrr")

        rows = []
        for m in MODEL_ORDER:
            sub = eval_df[(eval_df["model"] == m) &
                          (eval_df["condition"].isin(["B_simplified", "C_highly_simplified"]))]
            if len(sub) == 0:
                continue
            rows.append("%s & %.1f \\\\" % (m, 100 * as_float(sub["oversimplification_error"]).mean()))
        body = ("Model & OER (\\%) \\\\\n\\midrule\n" + "\n".join(rows))
        tw.write("tab_oer_by_model.tex", body,
                 "Oversimplification error rate by model (conditions B and C only).",
                 "tab:oer:model", colspec="lr")

        accuracy_analysis(eval_df, tw)
        subfield_analysis(eval_df, corpus, tw)
        concept_top10(eval_df, corpus, tw)
        concept_all(corpus, tw)
        model_analysis(eval_df, tw)
        position_bias(eval_df, tw)
        first_shown(eval_df, tw)
        baseline_before(eval_df, tw)
        position_adjusted_gee(eval_df, tw)

        judge_validation(eval_df, annotations_final, validation_map, tw)
        human_consensus_analysis(annotations_final, tw)

    if args.only in ("all", "figures"):
        LOG.info("--- Figures ---")
        figures(eval_df, corpus, outdir)

    LOG.info("Done.")


if __name__ == "__main__":
    main()