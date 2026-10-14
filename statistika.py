# ============================================================
# AccessTrap-CS — Statistical analysis script (Python)
# Opis: Kompletna statistika za rad "The Accessibility Paradox"
# Ulaz: corpus.csv, evaluations.csv
# Izlaz: output/tables/*.csv, output/tables/*.tex, output/figures/*
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import textstat

warnings.filterwarnings("ignore")
np.random.seed(20260101)
os.makedirs("output/tables", exist_ok=True)
os.makedirs("output/figures", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------- 1. Učitavanje ----------
corpus = pd.read_csv("corpus.csv")
eval_df = pd.read_csv("evaluations.csv")

def to_bool(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().upper()
    if s == "TRUE":  return True
    if s == "FALSE": return False
    return np.nan

eval_df["SM"]  = eval_df["strategic_misconception"].apply(to_bool)
eval_df["DA"]  = eval_df["dangerous_analogy"].apply(to_bool)
eval_df["OER"] = eval_df["oversimplification_error"].apply(to_bool)
eval_df["accuracy_score"] = pd.to_numeric(eval_df["accuracy_score"], errors="coerce")

cond_order = ["A_standard", "B_simplified", "C_highly_simplified"]
eval_df["condition"]  = pd.Categorical(eval_df["condition"], categories=cond_order, ordered=True)
eval_df["model"]      = eval_df["model"].astype("category")
eval_df["domain"]     = eval_df["domain"].astype("category")
eval_df["concept_id"] = eval_df["concept_id"].astype("category")

print("\n===== STRUKTURA PODATAKA =====")
print(f"Koncepata: {eval_df['concept_id'].nunique()}")
print(f"Modela:    {eval_df['model'].nunique()} -> {list(eval_df['model'].cat.categories)}")
print(f"Uvjeta:    {eval_df['condition'].nunique()} -> {list(eval_df['condition'].cat.categories)}")
print(f"Redaka:    {len(eval_df)}")
print(f"NA u SM/DA/OER/accuracy: "
      f"{eval_df['SM'].isna().sum()} / {eval_df['DA'].isna().sum()} / "
      f"{eval_df['OER'].isna().sum()} / {eval_df['accuracy_score'].isna().sum()}")
print(f"Sudac (judge_model): {eval_df['judge_model'].unique()}\n")

# ---------- 2. Readability (manipulation check) ----------
corpus["condition"] = pd.Categorical(corpus["condition"], categories=cond_order, ordered=True)
corpus["n_words"]  = corpus["response"].astype(str).str.split().str.len()
corpus["n_sents"]  = corpus["response"].astype(str).str.count(r"[.!?]+").clip(lower=1)
corpus["avg_sent"] = corpus["n_words"] / corpus["n_sents"]
corpus["fre"]      = corpus["response"].astype(str).apply(textstat.flesch_reading_ease)
corpus["fkgl"]     = corpus["response"].astype(str).apply(textstat.flesch_kincaid_grade)

read_desc = corpus.groupby("condition", observed=True).agg(
    N=("response", "size"),
    FRE_mean=("fre", "mean"),   FRE_sd=("fre", "std"),
    FKGL_mean=("fkgl", "mean"), FKGL_sd=("fkgl", "std"),
    Sent_mean=("avg_sent", "mean"), Sent_sd=("avg_sent", "std"),
    Words_mean=("n_words", "mean"), Words_sd=("n_words", "std"),
).reset_index()
read_desc.to_csv("output/tables/T1_readability_desc.csv", index=False)

def one_way_anova(df, dv):
    groups = [g[dv].dropna().values for _, g in df.groupby("condition", observed=True)]
    F, p = stats.f_oneway(*groups)
    return F, p

anova_lines = []
for dv in ["fre", "fkgl", "avg_sent"]:
    F, p = one_way_anova(corpus, dv)
    anova_lines.append(f"ANOVA {dv} ~ condition: F = {F:.3f}, p = {p:.4g}")
    # Tukey
    sub = corpus[["condition", dv]].dropna()
    tuk = pairwise_tukeyhsd(sub[dv], sub["condition"])
    anova_lines.append(str(tuk))
    anova_lines.append("-" * 60)
with open("output/tables/T2_readability_anova_tukey.txt", "w") as f:
    f.write("\n".join(anova_lines))

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=corpus, x="condition", y="fre", ax=ax, palette="Set2")
sns.stripplot(data=corpus, x="condition", y="fre", ax=ax, color="k", alpha=.15, size=2)
ax.set_xlabel("Condition"); ax.set_ylabel("Flesch Reading Ease (higher = easier)")
ax.set_title("Readability by condition")
plt.tight_layout(); plt.savefig("output/figures/F1_readability.pdf"); plt.close()

# ---------- 3. Deskriptivna statistika ----------
def desc_by(df, groupvars):
    g = df.groupby(groupvars, observed=True)
    out = g.agg(
        N=("SM", "size"),
        SMR=("SM", "mean"), DAR=("DA", "mean"), OER=("OER", "mean"),
        Acc_mean=("accuracy_score", "mean"), Acc_sd=("accuracy_score", "std"),
    ).reset_index()
    for c in ["SMR", "DAR", "OER"]:
        out[c + "_pct"] = out[c] * 100
    return out

d_cond  = desc_by(eval_df, ["condition"])
d_model = desc_by(eval_df, ["model"])
d_dom   = desc_by(eval_df, ["domain"])
d_cm    = desc_by(eval_df, ["condition", "model"])

d_cond.to_csv("output/tables/T3_desc_by_condition.csv", index=False)
d_model.to_csv("output/tables/T4_desc_by_model.csv", index=False)
d_dom.to_csv("output/tables/T5_desc_by_domain.csv", index=False)
d_cm.to_csv("output/tables/T6_desc_by_cond_model.csv", index=False)

# ---------- 4. H1: SMR ~ condition ----------
eval_df["SM_num"] = eval_df["SM"].astype(float)
model_smr = smf.ols("SM_num ~ C(condition)", data=eval_df).fit()
aov_smr   = sm.stats.anova_lm(model_smr, typ=2)

# Tukey
sub = eval_df[["condition", "SM_num"]].dropna()
tuk_smr = pairwise_tukeyhsd(sub["SM_num"], sub["condition"])

# Cochran's Q (po concept-model paru)
sm_wide = eval_df.pivot_table(index=["concept_id","model"], columns="condition",
                              values="SM", observed=True).dropna()
cq_sm = None
if sm_wide.shape[1] == 3:
    try:
        cq_sm = cochrans_q(sm_wide.values.astype(int))
    except Exception as e:
        cq_sm = f"Cochran Q failed: {e}"

# McNemar (Holm)
def mcnemar_pair(a, b):
    a = a.astype(int); b = b.astype(int)
    tb = pd.crosstab(a, b)
    if tb.shape != (2, 2): return np.nan
    return mcnemar(tb.values, exact=True).pvalue

p_AB = mcnemar_pair(sm_wide["A_standard"], sm_wide["B_simplified"])
p_AC = mcnemar_pair(sm_wide["A_standard"], sm_wide["C_highly_simplified"])
p_BC = mcnemar_pair(sm_wide["B_simplified"], sm_wide["C_highly_simplified"])
p_holm = multipletests([p_AB, p_AC, p_BC], method="holm")[1]

# GLMM (binarni outcome) — BinomialBayesMixedGLM
glmm_sm = None
try:
    glmm_sm = BinomialBayesMixedGLM.from_formula(
        "SM ~ C(condition)",
        {"concept": "0 + C(concept_id)", "model": "0 + C(model)"},
        data=eval_df.dropna(subset=["SM"]))
    glmm_fit = glmm_sm.fit_vb()
    glmm_summary = glmm_fit.summary()
except Exception as e:
    glmm_summary = f"GLMM failed: {e}"

with open("output/tables/T8_H1_SMR_full.txt", "w") as f:
    f.write("=== ANOVA SMR ~ condition ===\n");  f.write(str(aov_smr) + "\n\n")
    f.write("=== Tukey HSD ===\n");              f.write(str(tuk_smr) + "\n\n")
    f.write("=== Cochran's Q ===\n");            f.write(str(cq_sm) + "\n\n")
    f.write("=== McNemar (Holm) ===\n")
    f.write(f"AB: {p_AB:.4g}, AC: {p_AC:.4g}, BC: {p_BC:.4g}\n")
    f.write(f"Holm-adjusted: {p_holm}\n\n")
    f.write("=== GLMM (Binomial Bayes Mixed) ===\n")
    f.write(str(glmm_summary) + "\n")

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(d_cond))
se = np.sqrt(d_cond["SMR"] * (1 - d_cond["SMR"]) / d_cond["N"]) * 100
ax.bar(x, d_cond["SMR_pct"], yerr=1.96*se, capsize=5, color=sns.color_palette("Set2", len(x)))
ax.set_xticks(x); ax.set_xticklabels(d_cond["condition"])
ax.set_ylabel("SMR (%)"); ax.set_title("Strategic Misconception Rate")
plt.tight_layout(); plt.savefig("output/figures/F2_SMR_by_condition.pdf"); plt.close()

# ---------- 5. H2: DAR ~ condition (chi-square) ----------
tb_dar = pd.crosstab(eval_df["condition"], eval_df["DA"])
chi2, p_dar, dof, exp = stats.chi2_contingency(tb_dar)
n = tb_dar.values.sum()
min_dim = min(tb_dar.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim))
with open("output/tables/T9_H2_DAR_chisq.txt", "w") as f:
    f.write(str(tb_dar) + "\n\n")
    f.write(f"chi2 = {chi2:.3f}, df = {dof}, p = {p_dar:.4g}\n")
    f.write(f"Cramer's V = {cramers_v:.3f}\n")

fig, ax = plt.subplots(figsize=(6, 4))
se = np.sqrt(d_cond["DAR"] * (1 - d_cond["DAR"]) / d_cond["N"]) * 100
ax.bar(x, d_cond["DAR_pct"], yerr=1.96*se, capsize=5, color=sns.color_palette("Set2", len(x)))
ax.set_xticks(x); ax.set_xticklabels(d_cond["condition"])
ax.set_ylabel("DAR (%)"); ax.set_title("Dangerous Analogy Rate")
plt.tight_layout(); plt.savefig("output/figures/F3_DAR_by_condition.pdf"); plt.close()

# ---------- 6. H3: OER (samo B i C) ----------
oer_bc = eval_df[eval_df["condition"].isin(["B_simplified","C_highly_simplified"])].copy()
d_oer = oer_bc.groupby("condition", observed=True).agg(
    N=("OER","size"), OER=("OER","mean")).reset_index()
d_oer["OER_pct"] = d_oer["OER"] * 100
d_oer.to_csv("output/tables/T10_H3_OER_BC.csv", index=False)

oer_wide = oer_bc.pivot_table(index=["concept_id","model"], columns="condition",
                              values="OER", observed=True).dropna()
p_oer_bc = mcnemar_pair(oer_wide["B_simplified"], oer_wide["C_highly_simplified"])
with open("output/tables/T11_H3_OER_mcnemar.txt", "w") as f:
    f.write(f"McNemar B vs C: p = {p_oer_bc:.4g}\n")

fig, ax = plt.subplots(figsize=(6, 4))
x2 = np.arange(len(d_oer))
se = np.sqrt(d_oer["OER"] * (1 - d_oer["OER"]) / d_oer["N"]) * 100
ax.bar(x2, d_oer["OER_pct"], yerr=1.96*se, capsize=5,
       color=sns.color_palette("Set2", len(x2)))
ax.set_xticks(x2); ax.set_xticklabels(d_oer["condition"])
ax.set_ylabel("OER (%)"); ax.set_title("Oversimplification Error Rate (B, C)")
plt.tight_layout(); plt.savefig("output/figures/F4_OER_BC.pdf"); plt.close()

# ---------- 7. H4: model × condition (eksploratorno) ----------
model_smr2 = smf.ols("SM_num ~ C(model) * C(condition)", data=eval_df).fit()
aov_smr2 = sm.stats.anova_lm(model_smr2, typ=2)
with open("output/tables/T12_H4_model_condition_anova.txt", "w") as f:
    f.write(str(aov_smr2) + "\n")

# Domena unutar svakog uvjeta + BH
rows = []
for cond, g in eval_df.groupby("condition", observed=True):
    m = smf.ols("SM_num ~ C(domain)", data=g).fit()
    a = sm.stats.anova_lm(m, typ=2)
    a = a.loc[["C(domain)"]].copy()
    a["condition"] = cond
    rows.append(a)
dom_aov = pd.concat(rows).reset_index()
dom_aov["p_BH"] = multipletests(dom_aov["PR(>F)"], method="BH")[1]
dom_aov.to_csv("output/tables/T13_H4_domain_within_condition.csv", index=False)

# Heatmap
pivot_heat = d_cm.pivot(index="model", columns="condition", values="SMR_pct")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(pivot_heat, annot=True, fmt=".1f", cmap="viridis", ax=ax)
ax.set_title("SMR (%) by model and condition")
plt.tight_layout(); plt.savefig("output/figures/F5_heatmap_SMR_model_condition.pdf"); plt.close()

# Grouped bar po modelu
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=d_cm, y="model", x="SMR_pct", hue="condition", ax=ax)
ax.set_xlabel("SMR (%)"); ax.set_title("SMR by model and condition")
plt.tight_layout(); plt.savefig("output/figures/F6_SMR_by_model.pdf"); plt.close()

# ---------- 8. Accuracy ----------
model_acc = smf.ols("accuracy_score ~ C(condition)", data=eval_df).fit()
aov_acc   = sm.stats.anova_lm(model_acc, typ=2)
sub = eval_df[["condition","accuracy_score"]].dropna()
tuk_acc = pairwise_tukeyhsd(sub["accuracy_score"], sub["condition"])
with open("output/tables/T14_accuracy_anova_tukey.txt", "w") as f:
    f.write(str(aov_acc) + "\n\n" + str(tuk_acc) + "\n")

# Friedman
acc_wide = eval_df.pivot_table(index=["concept_id","model"], columns="condition",
                               values="accuracy_score", observed=True).dropna()
fr = stats.friedmanchisquare(acc_wide["A_standard"], acc_wide["B_simplified"],
                             acc_wide["C_highly_simplified"])
with open("output/tables/T15_accuracy_friedman.txt", "w") as f:
    f.write(f"Friedman chi2 = {fr.statistic:.3f}, p = {fr.pvalue:.4g}\n")

# Spearman
def sp(x, y):
    m = pd.concat([x, y], axis=1).dropna()
    return stats.spearmanr(m.iloc[:,0], m.iloc[:,1])[0]
cor_tab = pd.DataFrame({
    "pair": ["accuracy~SMR","accuracy~DAR","accuracy~OER"],
    "rho": [sp(eval_df["accuracy_score"], eval_df["SM"].astype(float)),
            sp(eval_df["accuracy_score"], eval_df["DA"].astype(float)),
            sp(eval_df["accuracy_score"], eval_df["OER"].astype(float))]
})
cor_tab.to_csv("output/tables/T16_accuracy_spearman.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=eval_df, x="condition", y="accuracy_score", ax=ax, palette="Set2")
sns.stripplot(data=eval_df, x="condition", y="accuracy_score", ax=ax,
              color="k", alpha=.15, size=2)
ax.set_xlabel("Condition"); ax.set_ylabel("Accuracy score (1–5)")
ax.set_title("Accuracy by condition")
plt.tight_layout(); plt.savefig("output/figures/F7_accuracy.pdf"); plt.close()

# ---------- 9. LaTeX izvoz glavnih tablica ----------
def to_latex(df, path, caption, label):
    with open(path, "w") as f:
        f.write(df.to_latex(index=False, caption=caption, label=label,
                            escape=False, float_format="%.3f"))

to_latex(d_cond,  "output/tables/T3_desc_by_condition.tex",
         "Descriptive statistics by condition.", "tab:desc_cond")
to_latex(d_model, "output/tables/T4_desc_by_model.tex",
         "Descriptive statistics by model.",     "tab:desc_model")
to_latex(d_dom,   "output/tables/T5_desc_by_domain.tex",
         "Descriptive statistics by domain.",    "tab:desc_dom")
to_latex(d_oer,   "output/tables/T10_H3_OER_BC.tex",
         "OER in conditions B and C.",           "tab:oer_bc")

print("\n✅ Gotovo. Pogledaj output/tables/ i output/figures/.\n")