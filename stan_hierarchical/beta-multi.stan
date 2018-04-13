data {
  int<lower=0> J; // Number of subjects
  int<lower=0> N; // Number of data points
  int<lower=0> id[N]; // Identifier vector
  int<lower=0> y[N]; // Observations
}
parameters {
  real<lower=0,upper=1> pi[N];
  real<lower=1> alpha[J];
  real<lower=1> beta[J];
}
model {
  for (j in 1:J){
    alpha[j] ~ gamma(0.0001,10);
    beta[j] ~ gamma(0.0001,10);
  }
  
  for (n in 1:N){
    y[n] ~ bernoulli(pi[n]);
    pi[n] ~ beta(alpha[id[n]]+1,beta[id[n]]+1);
  }
}
