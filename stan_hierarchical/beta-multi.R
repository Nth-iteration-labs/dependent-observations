## Try out for Beta-binomial with Gamma hyperprior model:
library("rstan")
setwd("~/Projects/dependent-observations/stan_hierarchical")

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

J <- 10			# Number of people
i <- 100		# Number of observations pp
N <- i*J		# Total number

# id vector
id <- sapply(1:(N), function(x) { ( (x-1) %/% i)  + 1 } )

# Grand mean, and individual level mean
p.j <- rbeta(J, 1,1)

# Simulate i binary observations w. probablity p.j
y <- as.vector(sapply(p.j, function (x) { rbinom(i, 1, x) } ) )

dat <- list(y = y, id = id, J = J, N = N)

fit <- stan(file = 'beta-multi.stan', data = dat, 
            iter = 1000, chains = 4)