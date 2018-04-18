data {
  int<lower=0> J; // Number of subjects
  int<lower=0> N; // Number of data points
  int<lower=0> id[N]; // Identifier vector
  int<lower=0> y[N]; // Observations
}
parameters {
  real<lower=0,upper=1> pi[J];
  real<lower=0.0001> alpha;
  real<lower=0.0001> beta;
}
model {
  alpha ~ gamma(0.0001,10);
  beta ~ gamma(0.0001,10);
  
  for (n in 1:N){
    y[n] ~ bernoulli(pi[id[n]]);
    pi[id[n]] ~ beta(alpha,beta);
  }
}
