library("rstan")
setwd("~/Projects/dependent-observations/stan_hierarchical")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

N <- 1000			# Number of people
i <- 10		# Number of observations pp
#N <- i*J		# Total number

# Grand mean, and individual level mean
p.j <- rbeta(N, 1,1)

# Create observations
y <- as.vector(sapply(p.j, function (x) { sum(rbinom(1, i, x)) } ) )

# Create vector with number of total interactions per user
K <- rep(i,N)

M <- 100

# Re-use previous model to create starting values, not sure if this works properly yet
init_val <- list(list(theta = fit_summary.theta, kappa = fit_summary.kappa, phi = fit_summary.phi))
init_values <- list(init_val, init_val, init_val, init_val)
                   
fit_hier <- stan("tryout.stan", data=c("N", "K", "y"), iter=20, warmup=10, init=init_val, chains=1, seed=1234)#, control=list(stepsize=0.01, adapt_delta=0.99))

fit_hier <- stan("tryout.stan", data=c("N", "K", "y"), iter=20, warmup=10, chains=1, seed=1234)
#fit_hier <- stan("tryout.stan", data=c("N", "K", "y"), chains=4, seed=1234)

# Extract parameters to use in later reruns
fit_summary.theta <- summary(fit_hier, pars=c("theta"))$summary[,"mean"]
fit_summary.kappa <- summary(fit_hier, pars=c("kappa"))$summary[,"mean"]
fit_summary.phi <- summary(fit_hier, pars=c("phi"))$summary[,"mean"]
# Get any theta
#fit_summary[J] #Where J is the user id

# Check divergences, you want 0 divergences
check_divergences(fit_hier)

# Extract samples for theta

theta_samples <- extract(fit_hier, pars=c("theta"))$theta


#########################################################################

# Example of a Policy

#########################################################################


######### Initialization ###########

# Set up the following:

# We have J is the number of subjects:

J <- n_subjects # Given by user/parameter

## N (total interactions for population per arm)

N_a <- 0
N_b <- 0

# For each subject, we keep track of:

## n (total interactions for that subject per arm)

n_a <- rep(0,N)
n_b <- rep(0,N)

## l (number of successes for that subject)

l_a <- rep(0,N)
l_b <- rep(0,N)

# Also we need to keep track of total parameters

## theta, phi, kappa (for computing posterior)
## For this we just run Stan once so that we can use the parameters in a loop later

fit_a <- stan("beta_binom_hier.stan", data=c("J", "n_a", "l_a"), iter=20, warmup=10, chains=1, seed=1234)
theta_a <- summary(fit, pars=c("theta"))$summary[,"mean"]
kappa_a <- summary(fit, pars=c("kappa"))$summary[,"mean"]
phi_a <- summary(fit, pars=c("phi"))$summary[,"mean"]

fit_b <- stan("beta_binom_hier.stan", data=c("J", "n_b", "l_b"), iter=20, warmup=10, chains=1, seed=1234)
theta_b <- summary(fit, pars=c("theta"))$summary[,"mean"]
kappa_b <- summary(fit, pars=c("kappa"))$summary[,"mean"]
phi_b <- summary(fit, pars=c("phi"))$summary[,"mean"]

# Furthermore, we need to keep a list theta_samples

theta_samples_a <- extract(fit_a, pars=c("theta"))$theta
phi_samples_a <- extract(fit_a, pars=c("phi"))$phi
kappa_samples_a <- extract(fit_a, pars=c("kappa"))$kappa

theta_samples_b <- extract(fit_b, pars=c("theta"))$theta
phi_samples_b <- extract(fit_b, pars=c("phi"))$phi
kappa_samples_b <- extract(fit_b, pars=c("kappa"))$kappa

######## Getting an Action #########

# Get Theta from posterior of arm A and arm B

# Using n_a %% 10 and n_b %% 10 we select the current theta

# If we haven't seen a user yet, we just use Theta from Stan that is just random theta, is that okay?

# Or if n_a == 0, then do random sample from beta(kappa*phi,kappa*(1-phi))?

# Select arm with highest theta_sample(K_a%%10) theta_samples(K_b%%10)

if (theta_samples_a[n_a%%10+1,userid] > theta_samples_b[n_b%%10+1,userid]){
  return arm_a
} else {
  return arm_b
}

######## Setting a Reward ##########

# After 10 (total, so over all users) rewards for an arm, update it

# Stan code that updates posterior

# Retrieve theta, kappa and phi to use in next update

# And retrieve the theta's from 10 samples, to use for each sample