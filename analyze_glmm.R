# ============================================================
# AccessTrap-CS: Generalised Linear Mixed Models (GLMM)
# with crossed random effects for concept and model
# ============================================================
#
# This script fits three GLMMs on the AccessTrap-CS corpus:
#   1. SM  ~ condition + (1|concept_id) + (1|model)
#   2. DA  ~ condition + (1|concept_id) + (1|model)
#   3. OER ~ condition + (1|concept_id) + (1|model)  [B and C only]
#
# The models are reported in the manuscript as the mixed-effects
# sensitivity analysis (Section 3.9.3) and confirm the GEE results.
#
# Requirements:
#   R >= 4.3.0
#   lme4, broom.mixed, dplyr
#
# Input:  evaluations.csv (in working directory)
# Output: results/table_glmm.csv
#         environment/sessionInfo.txt
#
# Author: Marijela Milicevic
# ============================================================

# ============================================================
# 0. SETUP
# ============================================================

library(lme4)
library(broom.mixed)
library(dplyr)

# Make sure we are in the repo root
if (!file.exists("evaluations.csv")) {
  stop("evaluations.csv not found in working directory. ",
       "Set the working directory to the folder that contains evaluations.csv.")
}

# ============================================================
# 1. LOAD DATA
# ============================================================

df <- read.csv("evaluations.csv", stringsAsFactors = FALSE)

cat("Loaded:", nrow(df), "rows\n")
cat("Conditions:", paste(unique(df$condition), collapse = ", "), "\n")
cat("Models:", paste(unique(df$model), collapse = ", "), "\n")

# ============================================================
# 2. PREPARE VARIABLES
# ============================================================

# Condition as ordered factor
df$condition <- factor(df$condition,
                       levels = c("A_standard", "B_simplified", "C_highly_simplified"))

# Concept and model as factors
df$concept_id <- as.factor(df$concept_id)
df$model      <- as.factor(df$model)

# Convert "True"/"False" to 0/1
df$strategic_misconception  <- ifelse(df$strategic_misconception  == "True", 1, 0)
df$dangerous_analogy        <- ifelse(df$dangerous_analogy        == "True", 1, 0)
df$oversimplification_error <- ifelse(df$oversimplification_error == "True", 1, 0)

# Verify no NAs
stopifnot(
  sum(is.na(df$strategic_misconception))  == 0,
  sum(is.na(df$dangerous_analogy))        == 0,
  sum(is.na(df$oversimplification_error)) == 0,
  sum(is.na(df$concept_id))               == 0,
  sum(is.na(df$model))                    == 0,
  sum(is.na(df$condition))                == 0
)

cat("\nData prepared successfully.\n")
cat("SM prevalence:", round(mean(df$strategic_misconception) * 100, 1), "%\n")
cat("DA prevalence:", round(mean(df$dangerous_analogy) * 100, 1), "%\n")
cat("OER prevalence (B+C only):",
    round(mean(df$oversimplification_error[df$condition != "A_standard"]) * 100, 1), "%\n")

# ============================================================
# 3. MODEL 1: STRATEGIC MISCONCEPTION (SM)
# ============================================================

cat("\n========================================\n")
cat("MODEL 1: Strategic Misconception (SM)\n")
cat("========================================\n")

model_sm <- glmer(
  strategic_misconception ~ condition + (1|concept_id) + (1|model),
  data = df,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

print(summary(model_sm))

cat("\n--- Odds Ratios (SM) ---\n")
print(exp(lme4::fixef(model_sm)))

cat("\n--- 95% CI for Odds Ratios (SM) ---\n")
print(exp(confint(model_sm, method = "Wald")))

cat("\n--- Random Effects (SM) ---\n")
print(VarCorr(model_sm))

# ============================================================
# 4. MODEL 2: DANGEROUS ANALOGY (DA)
# ============================================================

cat("\n========================================\n")
cat("MODEL 2: Dangerous Analogy (DA)\n")
cat("========================================\n")

model_da <- glmer(
  dangerous_analogy ~ condition + (1|concept_id) + (1|model),
  data = df,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

print(summary(model_da))

cat("\n--- Odds Ratios (DA) ---\n")
print(exp(lme4::fixef(model_da)))

cat("\n--- 95% CI for Odds Ratios (DA) ---\n")
print(exp(confint(model_da, method = "Wald")))

cat("\n--- Random Effects (DA) ---\n")
print(VarCorr(model_da))

# ============================================================
# 5. MODEL 3: OVERSIMPLIFICATION ERROR (OER)
# ============================================================

cat("\n========================================\n")
cat("MODEL 3: Oversimplification Error (OER)\n")
cat("========================================\n")

df_oer <- df[df$condition != "A_standard", ]
df_oer$condition <- droplevels(df_oer$condition)

cat("Rows for OER:", nrow(df_oer), "\n")
cat("Conditions in OER:", paste(unique(df_oer$condition), collapse = ", "), "\n")

model_oer <- glmer(
  oversimplification_error ~ condition + (1|concept_id) + (1|model),
  data = df_oer,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
)

print(summary(model_oer))

cat("\n--- Odds Ratios (OER) ---\n")
print(exp(lme4::fixef(model_oer)))

cat("\n--- 95% CI for Odds Ratios (OER) ---\n")
print(exp(confint(model_oer, method = "Wald")))

cat("\n--- Random Effects (OER) ---\n")
print(VarCorr(model_oer))

# ============================================================
# 6. EXPORT RESULTS FOR MANUSCRIPT
# ============================================================

cat("\n========================================\n")
cat("EXPORTING RESULTS\n")
cat("========================================\n")

# --- Extract ORs (fixed effects only) ---
sm_or  <- exp(lme4::fixef(model_sm))
da_or  <- exp(lme4::fixef(model_da))
oer_or <- exp(lme4::fixef(model_oer))

# --- Extract 95% CIs (Wald) ---
sm_ci  <- exp(confint(model_sm,  method = "Wald"))
da_ci  <- exp(confint(model_da,  method = "Wald"))
oer_ci <- exp(confint(model_oer, method = "Wald"))

# Helper: get CI for a specific contrast name (by row name, not index)
get_ci <- function(ci_matrix, contrast) {
  ci_matrix[contrast, c("2.5 %", "97.5 %")]
}

# --- Random effects SD ---
re_sm  <- as.data.frame(VarCorr(model_sm))$sdcor
re_da  <- as.data.frame(VarCorr(model_da))$sdcor
re_oer <- as.data.frame(VarCorr(model_oer))$sdcor

# --- Contrast names ---
sm_b  <- "conditionB_simplified"
sm_c  <- "conditionC_highly_simplified"
da_b  <- "conditionB_simplified"
da_c  <- "conditionC_highly_simplified"
oer_c <- "conditionC_highly_simplified"

# --- Build results table ---
glmm_results <- data.frame(
  Outcome  = c("SM", "SM", "DA", "DA", "OER"),
  Contrast = c("B vs A", "C vs A", "B vs A", "C vs A", "C vs B"),
  OR       = round(c(sm_or[sm_b], sm_or[sm_c],
                     da_or[da_b], da_or[da_c],
                     oer_or[oer_c]), 2),
  CI_lower = round(c(get_ci(sm_ci,  sm_b)[1],
                     get_ci(sm_ci,  sm_c)[1],
                     get_ci(da_ci,  da_b)[1],
                     get_ci(da_ci,  da_c)[1],
                     get_ci(oer_ci, oer_c)[1]), 2),
  CI_upper = round(c(get_ci(sm_ci,  sm_b)[2],
                     get_ci(sm_ci,  sm_c)[2],
                     get_ci(da_ci,  da_b)[2],
                     get_ci(da_ci,  da_c)[2],
                     get_ci(oer_ci, oer_c)[2]), 2),
  p_value  = round(c(
    summary(model_sm)$coefficients[sm_b, 4],
    summary(model_sm)$coefficients[sm_c, 4],
    summary(model_da)$coefficients[da_b, 4],
    summary(model_da)$coefficients[da_c, 4],
    summary(model_oer)$coefficients[oer_c, 4]
  ), 3)
)

cat("\n=== Fixed effects (odds ratios) ===\n")
print(glmm_results)

cat("\n=== Random effects (SD) ===\n")
cat("SM:  concept =", round(re_sm[1], 2),  ", model =", round(re_sm[2], 2),  "\n")
cat("DA:  concept =", round(re_da[1], 2),  ", model =", round(re_da[2], 2),  "\n")
cat("OER: concept =", round(re_oer[1], 2), ", model =", round(re_oer[2], 2), "\n")

# --- Write to CSV ---
dir.create("results", showWarnings = FALSE)
write.csv(glmm_results, "results/table_glmm.csv", row.names = FALSE)
cat("\nResults saved to results/table_glmm.csv\n")

# ============================================================
# 7. SAVE SESSION INFO FOR REPRODUCIBILITY
# ============================================================

dir.create("environment", showWarnings = FALSE)
sink("environment/sessionInfo.txt")
cat("AccessTrap-CS: R session information for GLMM analysis\n")
cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("============================================================\n\n")
sessionInfo()
sink()
cat("Session info saved to environment/sessionInfo.txt\n")

cat("\n========================================\n")
cat("GOTOVO\n")
cat("========================================\n")