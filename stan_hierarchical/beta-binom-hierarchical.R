library("rstan")
setwd("~/Projects/dependent-observations/stan_hierarchical")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

N <- 1000			# Number of people
i <- 10		# Number of observations pp
#N <- i*J		# Total number

# Grand mean, and individual level mean
p.j <- rbeta(N, 1,1)

y <- as.vector(sapply(p.j, function (x) { sum(rbinom(1, i, x)) } ) )

K <- rep(i,N)

M <- 5000

fit_hier <- stan("tryout.stan", data=c("N", "K", "y"),
                 iter=M, chains=4,
                 seed=1234)

fit_summary <- summary(fit_hier, pars=c("theta"))$summary[,"mean"]
# Get any theta
fit_summary[J] #Where J is the user id