library("rstan")
setwd("~/Projects/dependent-observations/stan_hierarchical")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

N <- 10			# Number of people
i <- 100		# Number of observations pp
#N <- i*J		# Total number

# Grand mean, and individual level mean
p.j <- rbeta(J, 1,1)

y <- as.vector(sapply(p.j, function (x) { sum(rbinom(1, i, x)) } ) )

K <- rep(i,J)

M <- 10000

fit_hier <- stan("tryout.stan", data=c("N", "K", "y"),
                 iter=(M / 2), chains=4,
                 seed=1234)
