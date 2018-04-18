library('rstan')
df <- read.csv("EMBB.tsv", sep="\t")
df <- with(df, data.frame(FirstName, LastName, 
                          Hits, At.Bats, 
                          RemainingAt.Bats,
                          RemainingHits = SeasonHits - Hits))
print(df)

N <- dim(df)[1]
K <- df$At.Bats
y <- df$Hits

M <- 10000

fit_hier <- stan("tryout.stan", data=c("N", "K", "y"),
                 iter=(M / 2), chains=4,
                 seed=1234)