library(NeuralEstimators)
# Sys.setenv(JULIA_BINDIR = "/path/to/Julia/bin")
library(JuliaConnectoR)
## For the first time, run the following:
# JuliaConnectoR::juliaSetupOk()
# juliaEval('
# import Pkg
# Pkg.add("NeuralEstimators")
# Pkg.add("Flux")
# Pkg.precompile()
# ')

juliaEval('using NeuralEstimators, Flux')

#----------------------------
# 1. Prior sampler for theta
#----------------------------
theta_sampler <- function(K) {
  # Must keep theta >= 0 for this rejection sampler
  theta <- runif(K, min = 0, max = 2)
  matrix(theta, nrow = 1, ncol = K)
}

#------------------------------------------
# 2. Single draw from exp-tilted base model
#------------------------------------------
your_sampler <- function(theta) {
  ......
}

#----------------------------------------------------
# 3. Simulate replicated data for each theta column
#----------------------------------------------------
simulate <- function(Theta, m) {
  apply(
    Theta, 2,
    function(theta) {
      x <- replicate(m, your_sampler(theta[1]))
      matrix(x, nrow = 1)   # d = 1
    },
    simplify = FALSE
  )
}

#----------------------------
# 4. Training/validation data
#----------------------------
K <- 10000
m <- 50

theta_train <- theta_sampler(K)
theta_val   <- theta_sampler(K / 10)

Z_train <- simulate(theta_train, m)
Z_val   <- simulate(theta_val, m)

#----------------------------
# 5. Neural estimator
#----------------------------
estimator <- juliaEval('
  d = 1    # dimension of each replicate
  p = 1    # only one parameter: theta
  w = 32

  # enforce theta >= 0
  final_layer = Dense(w, p, softplus)

  psi = Chain(
    Dense(d, w, relu),
    Dense(w, w, relu),
    Dense(w, w, relu)
  )

  phi = Chain(
    Dense(w, w, relu),
    Dense(w, w, relu),
    final_layer
  )

  deepset = DeepSet(psi, phi)
  estimator = PointEstimator(deepset)
')

#----------------------------
# 6. Train
#----------------------------
estimator <- train(
  estimator,
  theta_train = theta_train,
  theta_val   = theta_val,
  Z_train     = Z_train,
  Z_val       = Z_val,
  epochs      = 50
)

#----------------------------
# 7. Assess
#----------------------------
theta_test <- theta_sampler(1000)
Z_test     <- simulate(theta_test, m)

assessment <- assess(
  estimator,
  theta_test,
  Z_test,
  estimator_names = "NBE",
  parameter_names = "theta"
)

mean((assessment$estimates$estimate-assessment$estimates$truth)^2)
plotestimates(
  assessment,
  parameter_labels = c("θ1" = expression(theta))
)