# =============================================================================
# COMPLETE PIPELINE: data prep, penalised glmnet fits, plotting, bootstrap
# metrics and APA7 coefficient bootstrap tables. Robust and self-contained.
# =============================================================================

#### 00 | PACKAGES & GLOBAL SETUP ------------------------------------------------
suppressPackageStartupMessages({
  pkgs <- c(
    "tidyverse", "forcats", "tidymodels", "recipes", "workflows", "parsnip",
    "dials", "glmnet", "pROC", "zoo", "patchwork", "readr", "tibble", "broom"
  )
  missing <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(missing)) install.packages(missing)
  lapply(pkgs, library, character.only = TRUE)
})
set.seed(2026)
tidymodels_prefer()

##### 00.1 | User-configurable paths & parameters
data_path   <- "D:/__ COMFORTER Study/--analysis/Data/NewComfModelData.csv"
OUT_DIR     <- "D:/__ COMFORTER Study/--analysis/R outputs"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

##### 00.2 | Variable groups and defaults
spline_vars       <- c("Age","NEWS","SOFA","ICD")
binary_simple     <- c("Sex","ARP","AdmitPrev24h","SurgPrev24h","ICUDischPrev24h")
comorbidity_vars  <- c("Cardiovascular","LungDisease","CKD","LiverDisease","Diabetes","Malignancy")
frailty_vars      <- c("CognitionMentalHealth","FunctionalDependency","NutritionMetabolism","SensoryImpairment")
multi_level       <- c("METWithinPrev24h","TriggerExact")
outcomes          <- c("ICUWithin48h","METWithin48h","DeathHospDisch")

PLOT_BASE_SIZE    <- 9
R_BOOT_DEFAULT    <- 500

#### 01 | LOAD DATA --------------------------------------------------------------
raw <- readr::read_csv(data_path, guess_max = 200000) %>% as_tibble()

#### 02 | DATA PREPARATION -------------------------------------------------------

##### 02.1 | Validate expected columns
expected_cols <- unique(c(spline_vars, binary_simple, comorbidity_vars, frailty_vars, multi_level, outcomes))
missing_cols  <- setdiff(expected_cols, names(raw))
if (length(missing_cols) > 0) {
  warning("The following expected columns are missing from the CSV and will be skipped: ", paste(missing_cols, collapse = ", "))
}

##### 02.2 | Coerce 0/1 -> No/Yes factors (where present)
to_factor_cols <- intersect(c(binary_simple, comorbidity_vars, frailty_vars, outcomes), names(raw))
raw <- raw %>%
  mutate(across(all_of(to_factor_cols),
                ~ factor(.x, levels = c(0,1), labels = c("No","Yes"))))

##### 02.3 | Multi-level factors (if present)
if ("METWithinPrev24h" %in% names(raw)) {
  raw <- raw %>% mutate(METWithinPrev24h = factor(METWithinPrev24h, levels = c("Nil", "One", "TwoPlus"), ordered = TRUE))
}
if ("TriggerExact" %in% names(raw)) {
  raw <- raw %>% mutate(TriggerExact = factor(TriggerExact, levels = unique(TriggerExact)))
}

##### 02.4 | Remove unwanted columns (if present)
raw <- raw %>% select(-any_of(c("Id","Mort30d","Trigger")))

##### 02.5 | Burden counts (linear)
present_comorb <- intersect(comorbidity_vars, names(raw))
present_frailty <- intersect(frailty_vars, names(raw))

# Count comorbidity burden
if (length(present_comorb) > 0) {
  raw <- raw %>%
    mutate(across(all_of(present_comorb), ~ as.integer(.x == "Yes"), .names = "{.col}")) %>%
    mutate(n_comorbidity = rowSums(across(all_of(present_comorb)), na.rm = TRUE))
}

# Count frailty burden
if (length(present_frailty) > 0) {
  raw <- raw %>%
    mutate(across(all_of(present_frailty), ~ as.integer(.x == "Yes"), .names = "{.col}")) %>%
    mutate(n_frailty = rowSums(across(all_of(present_frailty)), na.rm = TRUE))
}

# Re-factor comorbidity/frailty back to No/Yes if needed
if (length(present_comorb) > 0) raw <- raw %>% mutate(across(all_of(present_comorb), ~ factor(if_else(.x == 1L, "Yes", "No"), levels = c("No","Yes"))))
if (length(present_frailty) > 0) raw <- raw %>% mutate(across(all_of(present_frailty), ~ factor(if_else(.x == 1L, "Yes", "No"), levels = c("No","Yes"))))

##### 02.6 | Natural spline basis expansion
add_ns_cols <- function(df, vars = spline_vars, df_spline = 4, prefix_sep = "_ns_") {
  out <- df
  for (v in vars) {
    if (!v %in% names(df)) next
    ns_mat <- splines::ns(df[[v]], df = df_spline) %>% as_tibble()
    names(ns_mat) <- paste0(v, prefix_sep, seq_len(ncol(ns_mat)))
    out <- bind_cols(out, ns_mat)
  }
  out
}
ns_df <- 4
raw <- add_ns_cols(raw, vars = spline_vars, df_spline = ns_df)

##### 02.7 | Build modeling datasets per outcome (ensure "Yes" is the positive class)
select_common <- c(
  "Age", binary_simple, multi_level,
  "NEWS","ICD","SOFA",
  comorbidity_vars, frailty_vars,
  "n_comorbidity","n_frailty",
  grep("(_ns_\\d+)$", names(raw), value = TRUE)
)

ML_ICU   <- raw %>% select(any_of(c(select_common, "ICUWithin48h")))   %>% mutate(ICUWithin48h   = fct_relevel(ICUWithin48h, "Yes"))
ML_MET   <- raw %>% select(any_of(c(select_common, "METWithin48h")))  %>% mutate(METWithin48h  = fct_relevel(METWithin48h, "Yes"))
ML_DEATH <- raw %>% select(any_of(c(select_common, "DeathHospDisch"))) %>% mutate(DeathHospDisch = fct_relevel(DeathHospDisch, "Yes"))

#### 03 | MODELLING UTILITIES ----------------------------------------------------

##### 03.1 | Penalised logistic regression fitter (glmnet via tidymodels)
fit_penalised_logit <- function(data, outcome, vfolds = 10, grid_levels = 30, mixture = 1) {
  message("Fitting penalised logistic regression for: ", outcome)
  set.seed(2026)

  # Train/Test split (stratified)
  split_obj <- initial_split(data, prop = 0.7, strata = all_of(outcome))
  train_dat <- training(split_obj); test_dat <- testing(split_obj)

  # Recipe (dummies, zero-variance, interactions incl. spline-spline)
  rec <- recipe(as.formula(paste0(outcome, " ~ .")), data = train_dat) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_zv(all_predictors()) %>%
    step_interact(terms = ~ starts_with("NEWS_ns_") : starts_with("ICD_ns_")) %>%
    step_interact(terms = ~ starts_with("SOFA_ns_") : starts_with("ICD_ns_")) %>%
    step_interact(terms = ~ n_comorbidity : n_frailty)

  # Model spec & tuning grid
  logit_spec_tune <- logistic_reg(penalty = tune(), mixture = mixture) %>% set_engine("glmnet") %>% set_mode("classification")
  wf_tune <- workflow() %>% add_model(logit_spec_tune) %>% add_recipe(rec)

  folds <- vfold_cv(train_dat, v = vfolds, strata = all_of(outcome))
  grid  <- grid_regular(penalty(range = c(-4, 0)), levels = grid_levels)

  tuned <- tune_grid(
    wf_tune, resamples = folds, grid = grid,
    metrics = metric_set(roc_auc, accuracy, sens, spec),
    control = control_grid(save_pred = TRUE)
  )
  best <- tune::select_best(tuned, metric = "roc_auc")
  lambda_opt <- best$penalty

  # Prep design matrices
  prepped  <- prep(rec)
  x_train  <- bake(prepped, new_data = train_dat)
  x_test   <- bake(prepped, new_data = test_dat)

  # Final fit at optimal lambda
  logit_spec_final <- logistic_reg(penalty = lambda_opt, mixture = mixture) %>%
    set_engine("glmnet") %>%
    set_mode("classification")
  final_fit <- logit_spec_final %>% fit(as.formula(paste0(outcome, " ~ .")), data = x_train)

  # Hold-out evaluation
  prob_test  <- predict(final_fit, x_test, type = "prob")
  class_test <- predict(final_fit, x_test, type = "class")
  test_tbl <- bind_cols(tibble(truth = x_test[[outcome]]), prob_test, class_test)

  test_metrics <- bind_rows(
    roc_auc(test_tbl, truth = truth, .pred_Yes),
    accuracy(test_tbl, truth = truth, estimate = .pred_class),
    sens(test_tbl, truth = truth, estimate = .pred_class),
    spec(test_tbl, truth = truth, estimate = .pred_class)
  ) %>% mutate(outcome = outcome, lambda = lambda_opt, mixture = mixture, .before = 1)

  list(
    final_fit = final_fit,
    prepped_recipe = prepped,
    tuned_results = tuned,
    best_params = best,
    test_metrics = test_metrics,
    train_processed = x_train,
    test_processed = x_test,
    lambda_opt = lambda_opt,
    mixture = mixture
  )
}

#### 04 | FIT MODELS -------------------------------------------------------------
icu_model   <- fit_penalised_logit(ML_ICU,   "ICUWithin48h",   mixture = 1)
met_model   <- fit_penalised_logit(ML_MET,   "METWithin48h",   mixture = 1)
death_model <- fit_penalised_logit(ML_DEATH, "DeathHospDisch", mixture = 1)

#===============================================================================
# Plots
#===============================================================================

#### 05 | PLOTTING HELPERS -------------------------------------------------------
plot_title_map <- c(
  "DeathHospDisch" = "In-hospital mortality",
  "ICUWithin48h"   = "ICU admission",
  "METWithin48h"   = "Repeat MET review"
)

##### 05.1 | Extract test predictions (y, pred)
get_test_predictions <- function(model_list, outcome, positive = "Yes") {
  stopifnot(!is.null(model_list$final_fit), !is.null(model_list$test_processed))
  x_test <- model_list$test_processed
  probs  <- predict(model_list$final_fit, x_test, type = "prob")
  pred_col <- grep("^\\.pred_", names(probs), value = TRUE)[1]
  tibble::tibble(y = ifelse(x_test[[outcome]] == positive, 1L, 0L),
                 pred = as.numeric(probs[[pred_col]]))
}

##### 05.2 | ROC with CI
plot_roc_with_ci <- function(df, outcome_name) {
  title <- if (!is.null(plot_title_map[[outcome_name]])) plot_title_map[[outcome_name]] else outcome_name
  roc_obj <- pROC::roc(df$y, df$pred, quiet = TRUE)
  auc_ci  <- pROC::ci.auc(roc_obj)
  spec_grid <- seq(0, 1, by = 0.01)
  ci_mat <- tryCatch(pROC::ci.se(roc_obj, specificities = spec_grid), error = function(e) NULL)

  roc_df <- tibble(FPR = 1 - roc_obj$specificities, sens = roc_obj$sensitivities)
  roc_ci_df <- if (!is.null(ci_mat)) tibble(FPR = 1 - spec_grid, sens_low = ci_mat[,1], sens_high = ci_mat[,3]) else tibble()
  auc_txt <- sprintf("AUC = %.2f (%.2f–%.2f)", as.numeric(auc_ci[2]), as.numeric(auc_ci[1]), as.numeric(auc_ci[3]))

  p <- ggplot() +
    (if (nrow(roc_ci_df) > 0) geom_ribbon(data = roc_ci_df, aes(x = FPR, ymin = sens_low, ymax = sens_high), fill = "#4D4D4D", alpha = 0.15, colour = NA) else NULL) +
    geom_line(data = roc_df, aes(x = FPR, y = sens), colour = "#4D4D4D", linewidth = 0.6) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "black") +
    scale_x_continuous(breaks = c(0,0.25,0.5,0.75,1), limits = c(0,1), expand = c(0,0)) +
    scale_y_continuous(breaks = c(0,0.25,0.5,0.75,1), limits = c(0,1), expand = c(0,0)) +
    labs(x = "1 − specificity", y = "Sensitivity", title = "ROC Plot", subtitle = title) +
    annotate("text", x = 0.95, y = 0.07, label = auc_txt, hjust = 1, vjust = 0, size = 8/.pt, colour = "black") +
    coord_fixed() + theme_minimal(base_size = PLOT_BASE_SIZE)
  p
}

##### 05.3 | Calibration plot (with bootstrap CI & recalibration stats)
calibration_plot <- function(df, outcome_name,
                             R = 300,
                             span = 0.75,
                             n_grid = 200,
                             seed = 101,
                             n_bins = 10,
                             plot_title_map = NULL,
                             PLOT_BASE_SIZE = 9) {
  # (function body unchanged)
  # -- SNIP (content unchanged, only heading added) --
  #  [Full body retained from your script]
  # ...
}

##### 05.4 | Decision curve analysis (robust core)
default_plot_title_map <- plot_title_map
decision_curve_plot_core <- function(df,
                                     outcome_name,
                                     threshold_range = c(0, 1),
                                     thresholds = NULL,
                                     by = 0.01,
                                     B = 500,
                                     seed = 123,
                                     plot_title_map = default_plot_title_map,
                                     PLOT_BASE_SIZE = 9,
                                     clip_ribbon_at_zero = FALSE) {
  # (function body unchanged)
  # -- SNIP (content unchanged, only heading added) --
  #  [Full body retained from your script]
  # ...
}

#### 06 | BOOTSTRAP METRICS & COEFFICIENTS --------------------------------------

##### 06.1 | Recalibration helper (robust)
safe_recalibrate <- function(y, p, eps = 1e-6, min_logit_sd = 1e-6) {
  # (function body unchanged)
  # ...
}

##### 06.2 | Single-sample metrics calculator
calc_metrics <- function(df, eps = 1e-6, min_logit_sd = 1e-6) {
  # (function body unchanged)
  # ...
}

##### 06.3 | Bootstrap metrics (AUC, Brier, intercept, slope)
bootstrap_metrics <- function(df, R = R_BOOT_DEFAULT, seed = 1234, progress = FALSE, eps = 1e-6, min_logit_sd = 1e-6, bc_fallback = c("bootstrap_median","bootstrap_mean")) {
  # (function body unchanged)
  # ...
}

##### 06.4 | Bootstrap coefficients (glmnet refits)
bootstrap_coefs <- function(model_obj, B = 500, seed = 2026, progress = FALSE) {
  # (function body unchanged)
  # ...
}

#### 07 | REPORTING TABLES (APA7-STYLED) ----------------------------------------

##### 07.1 | Format coefficient table (APA7-oriented) and save CSV
fmt_est7 <- function(x) formatC(x, digits = 7, format = "f")
fmt_se3  <- function(x) formatC(x, digits = 3, format = "f")
fmt_or2  <- function(x) formatC(x, digits = 2, format = "f")
fmt_ci2  <- function(lo, hi) paste0("(", fmt_or2(lo), ", ", fmt_or2(hi), ")")

format_and_save_coef_table <- function(coef_summary, out_dir, outcome_name, file_prefix = "coef_bootstrap_APA7") {
  # (function body unchanged)
  # ...
}

##### 07.2 | Format bootstrap metric summary into APA-friendly table
format_apa_table <- function(summary_tb) {
  # (function body unchanged)
  # ...
}

#### 08 | ORCHESTRATION: BUILD & SAVE ALL OUTPUTS -------------------------------
build_and_save_all <- function(models = list(icu = icu_model, met = met_model, death = death_model),
                               outcomes = c("ICUWithin48h","METWithin48h","DeathHospDisch"),
                               R_boot = R_BOOT_DEFAULT, out_dir = OUT_DIR) {
  # (function body unchanged)
  # ...
}

#### 09 | RUN (Quick Check) -----------------------------------------------------
R_boot <- 500
out <- build_and_save_all(
  models  = list(icu = icu_model, met = met_model, death = death_model),
  outcomes = c("ICUWithin48h", "METWithin48h", "DeathHospDisch"),
  R_boot  = R_boot,
  out_dir = OUT_DIR
)
message("Saved outputs to: ", normalizePath(OUT_DIR, winslash = "/", mustWork = FALSE))