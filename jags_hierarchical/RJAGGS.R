

########################################################################
########################################################################
 
### First simple first example using RJAGS
# See: http://www.johnmyleswhite.com/notebook/2010/08/20/using-jags-in-r-with-the-rjags-package/
# Have Jags installed, and rjags package:
library('rjags')

# Generate some data
N <- 1000
x <- rnorm(N, 0, 5)

# specify the model (and do a few samples)
jags <- jags.model('example.jags',
			       data = list('x' = x,'N' = N),
			       n.chains = 4,
			       n.adapt = 100)

# Do 1000 samples and plot
samples <- coda.samples(jags, c('mu', 'tau'), 1000);
plot(samples)
head(samples)


########################################################################
########################################################################

## Multi-level normal-normal example using jags
## See, e.g., Gelman Bayesian analysis book


J <- 10			# Number of people
i <- 10			# Number of observations pp
N <- i*j		# Total number

# id vector
id <- sapply(1:(N), function(x) { ( (x-1) %/% i)  + 1 } )

# Grand mean, and individual level mean
mu <- 2;  					# grand mean of 2
mu.j <- rnorm(J, 0, 2)		# individual level means from normal

# Simulate data with some noise:
y <- mu + as.vector(sapply(mu.j, function (x) { rep(x, i) } ) ) + rnorm(N, 0, 1)

## The Buggs model
jags <- jags.model(
                   'multilevel.jags',
                   data = list('y' = y, 'J'=J, 'id'=id, 'N'=N),
                   n.chains = 4,
                   n.adapt = 100)

samples <- coda.samples(jags, c('mu', 'sigma.y', 'sigma.a', 'a'), 10000);
plot(samples);


########################################################################
########################################################################

## Try out for Beta-beta model:
## This is my own little attempt.... :S


J <- 10			# Number of people
i <- 100		# Number of observations pp
N <- i*J		# Total number

# id vector
id <- sapply(1:(N), function(x) { ( (x-1) %/% i)  + 1 } )

# Grand mean, and individual level mean
p.j <- rbeta(J, 1,1)

# Simulate i binary observations w. probablity p.j
y <- as.vector(sapply(p.j, function (x) { rbinom(i, 1, x) } ) ) 

ptm <- proc.time()
## The Buggs model
jags <- jags.model(
                   'beta-multi2.jags',
                   data = list('y' = y, 'id'=id, 'N'=N),
                   n.chains = 4,
                   n.adapt = 100)

# Get posterior samples for a[j] and b[j]
#samples <- coda.samples(jags, c('a', 'b'), 10000);
#plot(samples)

# For checking, get posterior p only and compute mean
samples2 <- coda.samples(jags, c('p'), 1000);
proc.time() - ptm

# Check difference for first chain:
colMeans(samples2[[1]]) - p.j

# So, this model is odd, but it does seem to work... 


########################################################################
########################################################################


# I guess the most traigtforward model however would be a hierarchical logit:

J <- 10			# Number of people
i <- 100		# Number of observations pp
N <- i*J		# Total number

# id vector
id <- sapply(1:(N), function(x) { ( (x-1) %/% i)  + 1 } )

# Grand mean, and individual level mean
p.j <- rbeta(J, 1,1)

# Simulate i binary observations w. probablity p.j
y <- as.vector(sapply(p.j, function (x) { rbinom(i, 1, x) } ) )
ptm = proc.time()
jags <- jags.model(
                   'logit-multi.jags',
                   data = list('y' = y, 'J'=J, 'id'=id, 'N'=N),
                   n.chains = 4,
                   n.adapt = 100)

p.j
samples <- coda.samples(jags, c('mu', 'u', 'z'), 1000);
proc.time() - ptm
#plot(samples)

# Check the z's (which are the p.j draws)
draws <- samples[[1]][,(12:21)]
colMeans(draws)

#### This works just fine!