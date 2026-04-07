## ---- Problem 1: Lasso Regression via Coordinate Descent ----

# ============================================================
# (1) Pseudocode for Coordinate Descent
# ============================================================
#
# INPUT : X (n x p), y (n x 1), lambda >= 0, tol, max_iter
# OUTPUT: beta (p x 1) lasso coefficient estimates
#
# 1. Standardize each column of X to mean 0, sd 1.
#    Center y to mean 0.
# 2. Initialize beta = OLS estimate (or zeros).
# 3. Repeat until convergence:
#      For j = 1, ..., p:
#        Compute partial residual: r_j = y - X_{-j} %*% beta_{-j}
#        Compute z_j = (1/n) * X_j' r_j   (OLS update ignoring penalty)
#        Apply soft-thresholding:
#            beta_j = S(z_j, lambda) / (1/n * ||X_j||^2)
#        where S(z, lam) = sign(z) * max(|z| - lam, 0)
#      Check convergence: max |beta_new - beta_old| < tol
# 4. Un-standardize coefficients; compute intercept.
#
# NOTE: The homework says "lasso (L2)" but lasso uses L1 penalty.
#       We implement L1 (standard lasso) via coordinate descent.

# ============================================================
# (2) Implementation
# ============================================================

# --- Soft-thresholding operator ---
soft_threshold <- function(z, lam) {
  sign(z) * max(abs(z) - lam, 0)
}

# --- Coordinate descent for lasso ---
lasso_cd <- function(X, y, lambda, tol = 1e-8, max_iter = 10000) {
  n <- nrow(X)
  p <- ncol(X)
  
  # Standardize X and center y
  X_means <- colMeans(X)
  X_sds   <- apply(X, 2, sd)
  X_std   <- scale(X)                    # standardized to mean 0, sd 1
  y_mean  <- mean(y)
  y_c     <- y - y_mean                  # centered y
  
  # Precompute column norms squared (all equal n after standardization)
  col_norm_sq <- colSums(X_std^2)        # each ≈ n
  
  # Starting values: zero (cold start)
  beta <- rep(0, p)
  
  for (iter in 1:max_iter) {
    beta_old <- beta
    
    for (j in 1:p) {
      # Partial residual excluding predictor j
      r_j <- y_c - X_std[, -j, drop = FALSE] %*% beta[-j]
      
      # Unnormalized OLS coefficient for j
      z_j <- sum(X_std[, j] * r_j) / n
      
      # Soft-thresholding
      beta[j] <- soft_threshold(z_j, lambda) / (col_norm_sq[j] / n)
    }
    
    # Check convergence
    if (max(abs(beta - beta_old)) < tol) {
      cat("Converged at iteration", iter, "\n")
      break
    }
    if (iter == max_iter) {
      cat("Warning: did not converge within", max_iter, "iterations\n")
    }
  }
  
  # Un-standardize coefficients: beta_orig_j = beta_std_j / sd(X_j)
  beta_orig <- beta / X_sds
  intercept <- y_mean - sum(X_means * beta_orig)
  
  list(intercept  = intercept,
       beta       = beta_orig,
       beta_std   = beta,
       iterations = min(iter, max_iter),
       lambda     = lambda)
}

# ============================================================
# (3) Load data and fit
# ============================================================

# Read peru.txt — adjust path / separator as needed
peru <- read.table("peru.txt", header = TRUE)
# If no header: peru <- read.table("peru.txt", header = FALSE)
# Then assign names:
# names(peru) <- c("SBP","age","years","fraclife","weight","height",
#                   "chin","forearm","calf","pulse")

str(peru)

# Adjust column names to match your file
# Y = systolic blood pressure (column 1 typically)
# X1..X9 = remaining columns
y <- peru[, 1]                            # systolic BP
X <- as.matrix(peru[, 2:10])              # 9 predictors

cat("n =", nrow(X), ", p =", ncol(X), "\n")

# --- Choose lambda via cross-validation (using glmnet for reference) ---
library(glmnet)
set.seed(42)
cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = 10)  # alpha=1 is lasso
lambda_best <- cv_fit$lambda.min
cat("CV-selected lambda (on raw scale):", lambda_best, "\n")

# Our coordinate descent uses standardized X, so we need lambda
# on the same scale as glmnet (which standardizes internally).
# glmnet divides the objective by n, so our lambda matches directly.

# --- Fit our coordinate descent ---
fit <- lasso_cd(X, y, lambda = lambda_best)

# --- Report parameter estimates ---
pred_names <- colnames(X)
if (is.null(pred_names)) {
  pred_names <- paste0("X", 1:9)
}

cat("\n========================================\n")
cat("Lasso Parameter Estimates (lambda =", round(lambda_best, 4), ")\n")
cat("========================================\n")
cat(sprintf("  %-12s  %10.5f\n", "Intercept", fit$intercept))
for (j in seq_along(fit$beta)) {
  cat(sprintf("  %-12s  %10.5f\n", pred_names[j], fit$beta[j]))
}

# --- Verify against glmnet ---
glmnet_coef <- coef(cv_fit, s = "lambda.min")
cat("\n--- Comparison with glmnet ---\n")
print(cbind(ours = c(fit$intercept, fit$beta),
            glmnet = as.numeric(glmnet_coef)))

# --- Solution path (optional plot) ---
fit_path <- glmnet(X, y, alpha = 1)
plot(fit_path, xvar = "lambda", label = TRUE,
     main = "Lasso solution path (glmnet reference)")
abline(v = log(lambda_best), lty = 2, col = "red")
legend("topright", legend = "CV lambda.min", lty = 2, col = "red", bty = "n")
