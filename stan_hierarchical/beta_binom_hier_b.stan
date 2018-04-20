data {
  int<lower=0> J;           // items/subjects
  int<lower=0> n_b[J];        // total trials
  int<lower=0> l_b[J];        // total successes
}

parameters {
  real<lower=0, upper=1> phi;         // population chance of success
  real<lower=1> kappa;                // population concentration
  vector<lower=0, upper=1>[J] theta;  // chance of success 
}

model {
  kappa ~ pareto(1, 1.5);                        // hyperprior
  theta ~ beta(phi * kappa, (1 - phi) * kappa);  // prior
  l_b ~ binomial(n_b, theta);                        // likelihood
}
