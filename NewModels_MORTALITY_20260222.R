# ============================================================
# COMFORTER: Group ridge (grpreg "gel", gamma = 1e6)
# Enhanced + corrected pipeline (predict_clean, safe_roc, progress bars)
# ============================================================

# ---------------------------
# Packages
# ---------------------------
pkgs <- c(
  "dplyr","tidyr","purrr","pROC","splines","grpreg","tibble","ggplot2",
  "stringr","patchwork","forcats","readr","metafor","logistf","viridis",
  "scales","pbapply"
)
invisible(lapply(pkgs, function(p) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)))
lapply(pkgs, library, character.only = TRUE)

# ---------------------------
# Global options and seeds
# ---------------------------
set.seed(12345)
options(stringsAsFactors = FALSE, scipen = 999)

# Timing helper
time_block <- function(label, expr) {
  message(">>> Starting: ", label)
  t0 <- Sys.time()
  out <- eval.parent(substitute(expr))
  t1 <- Sys.time()
  message("<<< Completed: ", label, " in ", round(as.numeric(difftime(t1, t0, units = "secs")), 1), " sec")
  out
}

# ---------------------------
# Config
# ---------------------------
B_boot_metrics <- 1000
B_iecv          <- 500
nfolds_cv       <- 10

# ---------------------------
# Read data
# ---------------------------
dat <- readr::read_csv(
  "C:/Users/Administrator/OneDrive/Documents/__COMFORTER Study/__Analysis/Data/NewModelV3.csv",
  show_col_types = FALSE
)

# ---------------------------
# Harmonise factors + centre continuous variables
# ---------------------------
dat <- dat %>%
  mutate(
    Facility = factor(Facility, levels = c("Site_1","Site_2","Site_3","Site_4")),
    Sex      = factor(Sex,      levels = c("Female","Male")),
    ARP      = factor(ARP,      levels = c("Resuscitate","DNAR")),
    AdmitPrev24h     = factor(AdmitPrev24h,     levels = c("No","Yes")),
    SurgPrev24h      = factor(SurgPrev24h,      levels = c("No","Yes")),
    ICUDischPrev24h  = factor(ICUDischPrev24h,  levels = c("No","Yes")),
    METWithinPrev24h = factor(METWithinPrev24h, levels = c("Nil","One","TwoPlus"))
  )

map_to_binary <- function(x) {
  if (is.numeric(x)) {
    ux <- sort(unique(x))
    if (all(ux %in% c(0,1))) return(as.integer(x))
    return(as.integer(x != 0))
  }
  as.integer(tolower(as.character(x)) %in% c("1","yes","y","true","t"))
}

dat <- dat %>%
  mutate(
    DeathHospDisch = map_to_binary(DeathHospDisch)
  ) %>%
  mutate(
    Age_c  = Age  - mean(Age,  na.rm = TRUE),
    NEWS_c = NEWS - mean(NEWS, na.rm = TRUE),
    ICD_c  = ICD  - mean(ICD,  na.rm = TRUE),
    SOFA_c = SOFA - mean(SOFA, na.rm = TRUE)
  )

y <- dat$DeathHospDisch

# ---------------------------
# Palettes
# ---------------------------
model_levels <- c("Model 1: NEWS", "Model 2: NEWS+SOFA+ICD", "Model 3: Full")
okabe_cols <- c(
  "Model 1: NEWS"            = "#E69F00",
  "Model 2: NEWS+SOFA+ICD"   = "#0072B2",
  "Model 3: Full"            = "#009E73"
)

# ============================================================
# SAFE HELPERS
# ============================================================

safe_roc <- function(y, p) {
  out <- try(pROC::roc(y, p, quiet = TRUE), silent = TRUE)
  if (inherits(out, "try-error") || is.null(out$auc)) return(NULL)
  out
}

predict_clean <- function(fit, X, lambda) {
  p <- try(predict(fit, X, type = "response", lambda = lambda), silent = TRUE)
  if (inherits(p, "try-error")) stop("Prediction failed in predict_clean().")
  
  # --- FIX: flatten list output from grpreg ---
  if (is.list(p)) {
    p <- unlist(p, use.names = FALSE)
  }
  
  p <- as.numeric(p)
  
  if (!is.numeric(p) || anyNA(p)) {
    stop("Prediction failed: non-numeric or NA values detected.")
  }
  
  p
}

compute_metrics_full <- function(y, p) {
  eps <- 1e-6
  p <- pmin(pmax(p, eps), 1 - eps)
  roc_obj <- safe_roc(y, p)
  auc_val <- if (is.null(roc_obj)) NA_real_ else as.numeric(pROC::auc(roc_obj))
  brier <- mean((y - p)^2)
  logit_p <- qlogis(p)
  cal_fit <- try(glm(y ~ logit_p, family = binomial), silent = TRUE)
  if (inherits(cal_fit, "try-error")) {
    slope <- NA_real_; citl <- NA_real_
  } else {
    coefv <- coef(cal_fit)
    slope <- if (length(coefv) >= 2) coefv[2] else NA_real_
    citl  <- if (length(coefv) >= 1) coefv[1] else NA_real_
  }
  tibble::tibble(AUC = auc_val, Brier = brier, CalSlope = slope, CITL = citl)
}

# ============================================================
# DESIGN MATRIX BUILDERS
# ============================================================

build_terms_and_groups <- function(data, include_ICUDisch = TRUE) {
  age_s   <- splines::ns(data$Age_c,  df = 4); colnames(age_s)   <- paste0("Age_s",1:4)
  news_s  <- splines::ns(data$NEWS_c, df = 4); colnames(news_s)  <- paste0("NEWS_s",1:4)
  icd_s   <- splines::ns(data$ICD_c,  df = 4); colnames(icd_s)   <- paste0("ICD_s",1:4)
  sofa_s  <- splines::ns(data$SOFA_c, df = 4); colnames(sofa_s)  <- paste0("SOFA_s",1:4)
  
  base_terms <- c(
    "Sex","ARP","AdmitPrev24h","SurgPrev24h",
    if (include_ICUDisch) "ICUDischPrev24h" else NULL,
    "METWithinPrev24h"
  )
  
  base_mm <- model.matrix(
    as.formula(paste("~", paste(base_terms, collapse="+"))),
    data = data
  )[,-1,drop=FALSE]
  
  int_mat <- cbind(
    INT_AGE_NEWS  = data$Age_c  * data$NEWS_c,
    INT_AGE_ICD   = data$Age_c  * data$ICD_c,
    INT_AGE_SOFA  = data$Age_c  * data$SOFA_c,
    INT_NEWS_ICD  = data$NEWS_c * data$ICD_c,
    INT_NEWS_SOFA = data$NEWS_c * data$SOFA_c,
    INT_ICD_SOFA  = data$ICD_c  * data$SOFA_c
  )
  
  X <- cbind(age_s, news_s, icd_s, sofa_s, base_mm, int_mat)
  
  zero_cols <- which(colSums(abs(X)) == 0)
  if (length(zero_cols) > 0) X <- X[, -zero_cols, drop = FALSE]
  
  vars <- colnames(X)
  g <- integer(length(vars))
  gid <- 1
  
  assign_block <- function(pattern) {
    idx <- grepl(pattern, vars)
    if (any(idx)) { g[idx] <<- gid; gid <<- gid + 1 }
  }
  
  assign_block("^Age_s")
  assign_block("^NEWS_s")
  assign_block("^ICD_s")
  assign_block("^SOFA_s")
  assign_block("^Sex")
  assign_block("^ARP")
  assign_block("^AdmitPrev24h")
  assign_block("^SurgPrev24h")
  assign_block("^ICUDischPrev24h")
  assign_block("^METWithinPrev24h")
  assign_block("^INT_")
  
  remaining <- which(g == 0)
  if (length(remaining) > 0)
    g[remaining] <- seq(from = max(g) + 1, length.out = length(remaining))
  
  list(X = X, group = g)
}

build_X_model1 <- function(data) {
  news_s <- splines::ns(data$NEWS_c, df = 4)
  colnames(news_s) <- paste0("NEWS_s", 1:4)
  list(X = as.matrix(news_s), group = rep(1, ncol(news_s)))
}

build_X_model2 <- function(data) {
  news_s <- splines::ns(data$NEWS_c, df = 4); colnames(news_s) <- paste0("NEWS_s",1:4)
  sofa_s <- splines::ns(data$SOFA_c, df = 4); colnames(sofa_s) <- paste0("SOFA_s",1:4)
  icd_s  <- splines::ns(data$ICD_c,  df = 4); colnames(icd_s)  <- paste0("ICD_s", 1:4)
  X <- cbind(news_s, sofa_s, icd_s)
  g <- rep(1:3, each = 4)
  list(X = as.matrix(X), group = g)
}

build_X_model3 <- function(data) {
  build_terms_and_groups(data, include_ICUDisch = TRUE)
}

# ============================================================
# CV WRAPPER
# ============================================================

fit_cv_grpen <- function(X, y, groups,
                         penalty = c("gel","grLasso","grMCP","grSCAD"),
                         gamma = 1e6,
                         nfolds = 10,
                         lambda_rule = c("1se","min")) {
  penalty <- match.arg(penalty)
  lambda_rule <- match.arg(lambda_rule)
  
  cvfit <- try(
    cv.grpreg(X, y,
              group  = groups,
              family = "binomial",
              penalty = penalty,
              gamma   = gamma,
              nfolds  = nfolds),
    silent = TRUE
  )
  if (inherits(cvfit, "try-error")) stop("cv.grpreg() failed")
  
  lambda_min <- if (!is.null(cvfit$lambda.min)) cvfit$lambda.min else cvfit$lambda[which.min(cvfit$cve)]
  idx_min <- which.min(cvfit$cve)
  se_min  <- cvfit$cvse[idx_min]
  lambda_1se <- if (!is.na(se_min)) cvfit$lambda[max(which(cvfit$cve <= cvfit$cve[idx_min] + se_min))] else NA_real_
  lambda_chosen <- if (lambda_rule == "1se" && !is.na(lambda_1se)) lambda_1se else lambda_min
  
  list(
    cv            = cvfit,
    fit           = cvfit$fit,
    lambda_chosen = lambda_chosen,
    lambda_min    = lambda_min,
    lambda_1se    = lambda_1se
  )
}

# ============================================================
# FIT FINAL MODELS + BOOTSTRAP METRICS
# ============================================================

fit_model_and_bootstrap_metrics <- function(build_fun, data, y, model_label,
                                            B = B_boot_metrics) {
  
  Xg <- build_fun(data)
  X  <- Xg$X
  g  <- Xg$group
  
  cvres <- fit_cv_grpen(X, y, g, penalty = "gel", gamma = 1e6,
                        nfolds = nfolds_cv, lambda_rule = "1se")
  lam <- cvres$lambda_chosen
  fit <- cvres$fit
  
  p_hat <- predict_clean(fit, X, lam)
  base_metrics <- compute_metrics_full(y, p_hat)
  
  n <- length(y)
  boot_mat <- matrix(NA_real_, nrow = B, ncol = 4)
  colnames(boot_mat) <- c("AUC","Brier","CalSlope","CITL")
  
  message("Bootstrapping overall metrics for ", model_label, " (B = ", B, ")")
  pb <- txtProgressBar(min = 0, max = B, style = 3)
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    mb  <- compute_metrics_full(y[idx], p_hat[idx])
    boot_mat[b,] <- as.numeric(mb[1,])
    setTxtProgressBar(pb, b)
  }
  close(pb)
  gc()
  
  ci_l <- apply(boot_mat, 2, quantile, probs = 0.025, na.rm = TRUE)
  ci_u <- apply(boot_mat, 2, quantile, probs = 0.975, na.rm = TRUE)
  
  metrics_df <- tibble::tibble(
    Model   = model_label,
    AUC     = base_metrics$AUC,
    AUC_l   = ci_l["AUC"],
    AUC_u   = ci_u["AUC"],
    Brier   = base_metrics$Brier,
    Brier_l = ci_l["Brier"],
    Brier_u = ci_u["Brier"],
    CalSlope   = base_metrics$CalSlope,
    CalSlope_l = ci_l["CalSlope"],
    CalSlope_u = ci_u["CalSlope"],
    CITL       = base_metrics$CITL,
    CITL_l     = ci_l["CITL"],
    CITL_u     = ci_u["CITL"]
  )
  
  coef_vec <- as.matrix(coef(fit, lambda = lam))
  coef_df <- tibble::tibble(
    term     = rownames(coef_vec),
    estimate = round(as.numeric(coef_vec), 7)
  )
  
  list(
    model_label = model_label,
    X = X,
    g = g,
    fit = fit,
    lambda = lam,
    p_hat = p_hat,
    metrics = metrics_df,
    coef = coef_df
  )
}

# ---------------------------
# RUN THREE MODELS
# ---------------------------
res_m1 <- time_block("Fit Model 1 + bootstrap metrics", {
  fit_model_and_bootstrap_metrics(build_X_model1, dat, y, "Model 1: NEWS")
})

res_m2 <- time_block("Fit Model 2 + bootstrap metrics", {
  fit_model_and_bootstrap_metrics(build_X_model2, dat, y, "Model 2: NEWS+SOFA+ICD")
})

res_m3 <- time_block("Fit Model 3 + bootstrap metrics", {
  fit_model_and_bootstrap_metrics(build_X_model3, dat, y, "Model 3: Full")
})

metrics_overall <- bind_rows(res_m1$metrics, res_m2$metrics, res_m3$metrics)
metrics_overall_2dp <- metrics_overall %>%
  mutate(across(where(is.numeric), ~ round(.x, 2)))
print(metrics_overall_2dp)

# Save coefficients
readr::write_csv(res_m1$coef, "coefficients_model1_news.csv")
readr::write_csv(res_m2$coef, "coefficients_model2_news_sofa_icd.csv")
readr::write_csv(res_m3$coef, "coefficients_model3_full.csv")

# ============================================================
# IECV: per-site metrics with bootstrap CIs + pooled (REML)
# ============================================================

bootstrap_iecv_for_model <- function(data, outcome_var, build_fun, model_label,
                                     B = B_iecv, seed = 12345) {
  set.seed(seed)
  data <- droplevels(data)
  sites <- levels(data$Facility)
  out_list <- vector("list", length(sites))
  
  for (i in seq_along(sites)) {
    s <- sites[i]
    message("IECV for ", model_label, " — held-out site: ", s)
    
    train <- data %>% filter(Facility != s)
    test  <- data %>% filter(Facility == s)
    if (nrow(test) == 0) next
    
    # Build design matrices
    train_Xg <- build_fun(train)
    X_train  <- train_Xg$X
    g_train  <- train_Xg$group
    
    test_Xg <- build_fun(test)
    X_test  <- test_Xg$X
    
    # Align columns
    missing_cols <- setdiff(colnames(X_train), colnames(X_test))
    if (length(missing_cols) > 0) {
      X_test <- cbind(
        X_test,
        matrix(0, nrow = nrow(X_test), ncol = length(missing_cols),
               dimnames = list(NULL, missing_cols))
      )
    }
    X_test <- X_test[, colnames(X_train), drop = FALSE]
    
    y_train <- train[[outcome_var]]
    y_test  <- test[[outcome_var]]
    
    cvres <- fit_cv_grpen(X_train, y_train, g_train,
                          penalty = "gel", gamma = 1e6,
                          nfolds = nfolds_cv, lambda_rule = "1se")
    lam <- cvres$lambda_chosen
    fit <- cvres$fit
    
    # Clean prediction
    p_test <- predict_clean(fit, X_test, lam)
    point  <- compute_metrics_full(y_test, p_test)
    
    # Bootstrap CIs
    n_test <- length(y_test)
    boot_mat <- matrix(NA_real_, nrow = B, ncol = 4)
    colnames(boot_mat) <- c("AUC","Brier","CalSlope","CITL")
    
    message("  Bootstrapping IECV metrics for ", model_label, " at ", s, " (B = ", B, ")")
    pb <- txtProgressBar(min = 0, max = B, style = 3)
    for (b in seq_len(B)) {
      idx <- sample.int(n_test, n_test, replace = TRUE)
      mb  <- compute_metrics_full(y_test[idx], p_test[idx])
      boot_mat[b,] <- as.numeric(mb[1,])
      setTxtProgressBar(pb, b)
    }
    close(pb)
    gc()
    
    ci_l <- apply(boot_mat, 2, quantile, probs = 0.025, na.rm = TRUE)
    ci_u <- apply(boot_mat, 2, quantile, probs = 0.975, na.rm = TRUE)
    
    out_list[[i]] <- tibble::tibble(
      Model = model_label,
      HeldOutSite = s,
      AUC     = point$AUC,
      AUC_l   = ci_l["AUC"],
      AUC_u   = ci_u["AUC"],
      Brier   = point$Brier,
      Brier_l = ci_l["Brier"],
      Brier_u = ci_u["Brier"],
      CalSlope   = point$CalSlope,
      CalSlope_l = ci_l["CalSlope"],
      CalSlope_u = ci_u["CalSlope"],
      CITL       = point$CITL,
      CITL_l     = ci_l["CITL"],
      CITL_u     = ci_u["CITL"]
    )
  }
  
  bind_rows(out_list)
}

# ---------------------------
# Run IECV for all models
# ---------------------------
iecv_m1 <- time_block("IECV Model 1", {
  bootstrap_iecv_for_model(dat, "DeathHospDisch", build_X_model1, "Model 1: NEWS")
})
iecv_m2 <- time_block("IECV Model 2", {
  bootstrap_iecv_for_model(dat, "DeathHospDisch", build_X_model2, "Model 2: NEWS+SOFA+ICD")
})
iecv_m3 <- time_block("IECV Model 3", {
  bootstrap_iecv_for_model(dat, "DeathHospDisch", build_X_model3, "Model 3: Full")
})

iecv_all <- bind_rows(iecv_m1, iecv_m2, iecv_m3)

# Save IECV per-site metrics
iecv_all_2dp <- iecv_all %>% mutate(across(where(is.numeric), ~ round(.x, 2)))
readr::write_csv(iecv_all_2dp, "iecv_site_metrics_all_models.csv")

# ============================================================
# Pooled metrics via REML
# ============================================================

pool_metrics_re <- function(df) {
  approx_se <- function(l, u) (u - l) / 3.92
  metrics <- c("AUC","Brier","CalSlope","CITL")
  
  purrr::map_dfr(metrics, function(m) {
    yi <- df[[m]]
    li <- df[[paste0(m, "_l")]]
    ui <- df[[paste0(m, "_u")]]
    sei <- approx_se(li, ui)
    ok  <- !is.na(yi) & !is.na(sei)
    
    if (sum(ok) < 2) {
      tibble::tibble(Metric = m, pooled = NA_real_, pooled_l = NA_real_, pooled_u = NA_real_, k = sum(ok))
    } else {
      r <- metafor::rma.uni(yi = yi[ok], sei = sei[ok], method = "REML")
      tibble::tibble(Metric = m, pooled = as.numeric(r$b), pooled_l = r$ci.lb, pooled_u = r$ci.ub, k = sum(ok))
    }
  })
}

pooled_m1 <- pool_metrics_re(iecv_m1)
pooled_m2 <- pool_metrics_re(iecv_m2)
pooled_m3 <- pool_metrics_re(iecv_m3)

pooled_all <- bind_rows(
  pooled_m1 %>% mutate(Model = "Model 1: NEWS"),
  pooled_m2 %>% mutate(Model = "Model 2: NEWS+SOFA+ICD"),
  pooled_m3 %>% mutate(Model = "Model 3: Full")
) %>% relocate(Model, Metric)

pooled_all_2dp <- pooled_all %>% mutate(across(where(is.numeric), ~ round(.x, 2)))
print(pooled_all_2dp)
readr::write_csv(pooled_all_2dp, "iecv_pooled_metrics_all_models.csv")

# ============================================================
# LABEL HELPERS FOR PLOTS
# ============================================================

fmt_ci <- function(est, lo, hi, digits = 2) {
  ifelse(is.na(est), "NA",
         sprintf(paste0("%.",digits,"f (%.",digits,"f–%.",digits,"f)"),
                 est, lo, hi))
}

metrics_overall_lbl <- metrics_overall %>%
  mutate(across(where(is.numeric), ~ round(.x, 2))) %>%
  mutate(
    AUC_txt      = fmt_ci(AUC, AUC_l, AUC_u, 2),
    Brier_txt    = fmt_ci(Brier, Brier_l, Brier_u, 2),
    CalSlope_txt = fmt_ci(CalSlope, CalSlope_l, CalSlope_u, 2),
    CITL_txt     = fmt_ci(CITL, CITL_l, CITL_u, 2)
  )

get_metric_label_block <- function(model_label) {
  row <- metrics_overall_lbl %>% filter(Model == model_label)
  if (nrow(row) == 0) return("")
  paste0(
    model_label, "\n",
    "AUC: ",      row$AUC_txt, "\n",
    "Brier: ",    row$Brier_txt, "\n",
    "Slope: ",    row$CalSlope_txt, "\n",
    "CITL: ",     row$CITL_txt
  )
}

# ============================================================
# ROC CURVES WITH 95% CI RIBBONS
# ============================================================

build_roc_with_ci <- function(y, p, model_label,
                              B = B_boot_metrics,
                              grid = seq(0, 1, length.out = 101)) {
  
  p <- as.numeric(p)
  roc_full <- safe_roc(y, p)
  if (is.null(roc_full)) stop("ROC failed for ", model_label)
  
  tpr_full <- unlist(
    pROC::coords(roc_full, x = grid, input = "specificity",
                 ret = "sensitivity", transpose = FALSE),
    use.names = FALSE
  )
  
  n <- length(y)
  boot_mat <- matrix(NA_real_, nrow = length(grid), ncol = B)
  
  message("Bootstrapping ROC for ", model_label, " (B = ", B, ")")
  pb <- txtProgressBar(min = 0, max = B, style = 3)
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    roc_b <- safe_roc(y[idx], p[idx])
    if (is.null(roc_b)) next
    tpr_b <- unlist(
      pROC::coords(roc_b, x = grid, input = "specificity",
                   ret = "sensitivity", transpose = FALSE),
      use.names = FALSE
    )
    boot_mat[, b] <- tpr_b
    setTxtProgressBar(pb, b)
  }
  close(pb)
  gc()
  
  tpr_l <- apply(boot_mat, 1, quantile, probs = 0.025, na.rm = TRUE)
  tpr_u <- apply(boot_mat, 1, quantile, probs = 0.975, na.rm = TRUE)
  
  tibble::tibble(
    FPR   = 1 - grid,
    TPR   = tpr_full,
    TPR_l = tpr_l,
    TPR_u = tpr_u,
    Model = model_label
  )
}

roc_m1 <- time_block("ROC Model 1", {
  build_roc_with_ci(y, res_m1$p_hat, "Model 1: NEWS")
})
roc_m2 <- time_block("ROC Model 2", {
  build_roc_with_ci(y, res_m2$p_hat, "Model 2: NEWS+SOFA+ICD")
})
roc_m3 <- time_block("ROC Model 3", {
  build_roc_with_ci(y, res_m3$p_hat, "Model 3: Full")
})

roc_df_all <- bind_rows(roc_m1, roc_m2, roc_m3)

plot_roc_overlay_ci <- function(df, title = "Receiver Operator Curves") {
  df$Model <- factor(df$Model, levels = model_levels)
  
  # Extract AUC text only
  auc_labels <- metrics_overall %>%
    mutate(across(where(is.numeric), ~ round(.x, 2))) %>%
    transmute(
      Model,
      txt = paste0("AUC: ", AUC, " (", AUC_l, "–", AUC_u, ")")
    )
  
  auc_block <- paste(
    auc_labels$Model,
    auc_labels$txt,
    sep = ": ",
    collapse = "\n"
  )
  
  ggplot(df, aes(x = FPR, y = TPR, colour = Model, fill = Model)) +
    geom_ribbon(aes(ymin = TPR_l, ymax = TPR_u), alpha = 0.2, colour = NA) +
    geom_line(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey60") +
    scale_colour_manual(values = okabe_cols) +
    scale_fill_manual(values = okabe_cols) +
    coord_equal() +
    labs(
      title = title,
      x = "1 - Specificity",
      y = "Sensitivity",
      colour = "Model",
      fill   = "Model"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = c(0.85, 0.50),   # right middle
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA),
      legend.text = element_text(size = rel(0.8)),
      legend.title = element_text(size = rel(0.8)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey90"),
      plot.title = element_text(face = "bold")
    ) +
    annotate(
      "text",
      x = 0.98, y = 0.02, hjust = 1, vjust = 0,
      label = auc_block,
      size = 3
    )
}

roc_plot <- plot_roc_overlay_ci(roc_df_all)
print(roc_plot)
ggsave("roc_group_ridge_models.png", roc_plot, width = 7, height = 7, dpi = 600)

# ============================================================
# CALIBRATION CURVES WITH 95% CI RIBBONS
# ============================================================

build_calibration_with_ci <- function(y, p, model_label,
                                      grid = seq(0.01, 0.99, by = 0.01),
                                      span = 0.75,
                                      B = B_boot_metrics) {
  
  p <- as.numeric(p)
  df <- tibble::tibble(y = y, p = p) %>% filter(!is.na(p), !is.na(y))
  
  lo_full <- stats::loess(y ~ p, data = df, span = span)
  obs_full <- pmin(pmax(predict(lo_full, newdata = data.frame(p = grid)), 0), 1)
  
  n <- nrow(df)
  boot_mat <- matrix(NA_real_, nrow = length(grid), ncol = B)
  
  message("Bootstrapping calibration for ", model_label, " (B = ", B, ")")
  pb <- txtProgressBar(min = 0, max = B, style = 3)
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    df_b <- df[idx, , drop = FALSE]
    lo_b <- try(stats::loess(y ~ p, data = df_b, span = span), silent = TRUE)
    if (inherits(lo_b, "try-error")) next
    pred_b <- try(pmin(pmax(predict(lo_b, newdata = data.frame(p = grid)), 0), 1),
                  silent = TRUE)
    if (inherits(pred_b, "try-error")) next
    if (length(pred_b) == length(grid)) boot_mat[, b] <- pred_b
    setTxtProgressBar(pb, b)
  }
  close(pb)
  gc()
  
  obs_l <- apply(boot_mat, 1, quantile, probs = 0.025, na.rm = TRUE)
  obs_u <- apply(boot_mat, 1, quantile, probs = 0.975, na.rm = TRUE)
  
  tibble::tibble(
    p_hat = grid,
    obs   = obs_full,
    obs_l = obs_l,
    obs_u = obs_u,
    Model = model_label
  )
}

cal_m1 <- time_block("Calibration Model 1", {
  build_calibration_with_ci(y, res_m1$p_hat, "Model 1: NEWS")
})
cal_m2 <- time_block("Calibration Model 2", {
  build_calibration_with_ci(y, res_m2$p_hat, "Model 2: NEWS+SOFA+ICD")
})
cal_m3 <- time_block("Calibration Model 3", {
  build_calibration_with_ci(y, res_m3$p_hat, "Model 3: Full")
})

cal_df_all <- bind_rows(cal_m1, cal_m2, cal_m3)

plot_calibration_overlay_ci <- function(cal_df,
                                        title = "Calibration plots") {
  cal_df$Model <- factor(cal_df$Model, levels = model_levels)
  
  # Extract slope + intercept only
  cal_labels <- metrics_overall %>%
    mutate(across(where(is.numeric), ~ round(.x, 2))) %>%
    transmute(
      Model,
      txt = paste0("Slope: ", CalSlope, " (", CalSlope_l, "–", CalSlope_u, ")\n",
                   "CITL: ", CITL, " (", CITL_l, "–", CITL_u, ")")
    )
  
  cal_block <- paste(
    cal_labels$Model,
    cal_labels$txt,
    sep = ":\n",
    collapse = "\n\n"
  )
  
  ggplot(cal_df, aes(x = p_hat, y = obs, colour = Model, fill = Model)) +
    geom_ribbon(aes(ymin = obs_l, ymax = obs_u), alpha = 0.2, colour = NA) +
    geom_line(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted", colour = "grey50") +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2)) +
    scale_colour_manual(values = okabe_cols) +
    scale_fill_manual(values = okabe_cols) +
    coord_equal() +
    labs(
      title = title,
      x = "Predicted probability",
      y = "Observed probability",
      colour = "Model",
      fill   = "Model"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = c(0.85, 0.20),   # bottom right
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA),
      legend.text = element_text(size = rel(0.8)),
      legend.title = element_text(size = rel(0.8)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey90"),
      plot.title = element_text(face = "bold")
    ) +
    annotate(
      "text",
      x = 0.02, y = 0.98, hjust = 0, vjust = 1,
      label = cal_block,
      size = 3
    )
}

cal_plot <- plot_calibration_overlay_ci(cal_df_all)
print(cal_plot)
ggsave("calibration_group_ridge_models.png", cal_plot, width = 7, height = 7, dpi = 600)

# ============================================================
# DCA WITH 95% CI RIBBONS
# ============================================================

compute_dca_single <- function(y, p, thresholds = seq(0.01, 0.99, by = 0.01)) {
  y <- as.numeric(y)
  p <- as.numeric(p)
  n <- length(y)
  prev <- mean(y == 1, na.rm = TRUE)
  nb_all <- prev - (1 - prev) * thresholds / (1 - thresholds)
  
  nb_model <- sapply(thresholds, function(t) {
    treat <- p >= t
    tp <- sum(treat & y == 1, na.rm = TRUE) / n
    fp <- sum(treat & y == 0, na.rm = TRUE) / n
    tp - fp * t / (1 - t)
  })
  
  list(threshold = thresholds, NB = nb_model, NB_all = nb_all)
}

build_dca_with_ci <- function(y, p, model_label,
                              thresholds = seq(0.01, 0.99, by = 0.01),
                              B = B_boot_metrics) {
  
  p <- as.numeric(p)
  base <- compute_dca_single(y, p, thresholds)
  n <- length(y)
  nb_mat <- matrix(NA_real_, nrow = length(thresholds), ncol = B)
  
  message("Bootstrapping DCA for ", model_label, " (B = ", B, ")")
  pb <- txtProgressBar(min = 0, max = B, style = 3)
  for (b in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    d_b <- compute_dca_single(y[idx], p[idx], thresholds)
    nb_mat[, b] <- d_b$NB
    setTxtProgressBar(pb, b)
  }
  close(pb)
  gc()
  
  nb_l <- apply(nb_mat, 1, quantile, probs = 0.025, na.rm = TRUE)
  nb_u <- apply(nb_mat, 1, quantile, probs = 0.975, na.rm = TRUE)
  
  tibble::tibble(
    threshold = thresholds,
    NB        = base$NB,
    NB_l      = nb_l,
    NB_u      = nb_u,
    NB_all    = base$NB_all,
    Model     = model_label
  )
}

dca_m1 <- time_block("DCA Model 1", {
  build_dca_with_ci(y, res_m1$p_hat, "Model 1: NEWS")
})
dca_m2 <- time_block("DCA Model 2", {
  build_dca_with_ci(y, res_m2$p_hat, "Model 2: NEWS+SOFA+ICD")
})
dca_m3 <- time_block("DCA Model 3", {
  build_dca_with_ci(y, res_m3$p_hat, "Model 3: Full")
})

dca_df_all <- bind_rows(dca_m1, dca_m2, dca_m3)

plot_dca_overlay_ci <- function(dca_df,
                                title = "Decision curve analyses") {
  dca_df$Model <- factor(dca_df$Model, levels = model_levels)
  
  ref_all <- dca_df %>%
    group_by(threshold) %>%
    summarise(NB_all = unique(NB_all)[1], .groups = "drop")
  
  ggplot() +
    geom_ribbon(
      data = dca_df,
      aes(x = threshold, ymin = NB_l, ymax = NB_u, fill = Model),
      alpha = 0.2, colour = NA
    ) +
    geom_line(
      data = dca_df,
      aes(x = threshold, y = NB, colour = Model),
      linewidth = 1
    ) +
    geom_line(
      data = ref_all,
      aes(x = threshold, y = NB_all),
      linetype = "dashed", colour = "grey40"
    ) +
    geom_hline(yintercept = 0, linetype = "dotted", colour = "black") +
    scale_colour_manual(values = okabe_cols) +
    scale_fill_manual(values = okabe_cols) +
    coord_cartesian(ylim = c(0, 0.25)) +
    labs(
      title = title,
      x = "Threshold probability",
      y = "Net benefit",
      colour = "Model",
      fill   = "Model"
    ) +
    theme(
      # Legend placement and sizing
      legend.position = c(0.85, 0.85),
      legend.background = element_rect(fill = alpha("white", 0.7), colour = NA),
      legend.text = element_text(size = 9),      # match axis text
      legend.title = element_text(size = 11),     # match axis text
      
      # Axis text and titles
      axis.text = element_text(size = 11),
      axis.title = element_text(size = 11),
      
      # White background
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      
      # Gridlines
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "grey90"),
      
      # Title
      plot.title = element_text(face = "bold")
    )
}

dca_plot <- plot_dca_overlay_ci(dca_df_all)
print(dca_plot)
ggsave("dca_group_ridge_models.png", dca_plot, width = 7, height = 7, dpi = 600)

# ============================================================
# END OF PIPELINE
# ============================================================
