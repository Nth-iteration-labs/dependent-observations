import numpy as np
import policies.efirst as ef
import policies.efirst_person as efp
import policies.egreedy as eg
import policies.ucb as ucb
import policies.thompson as thompson
import bandits.bernoullibandit as bb
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set_style("white")
sns.set_color_codes("colorblind")

def sim_run(bandit, policy, N, n_subjects):
    
    cum_reward = []
    cum_reward_oracle = []
    
    for i in range(N):
        # Set up context with subjects between 1 and n_subjects
        context = {}
        #context['subject'] = int(np.floor(np.random.pareto(1,1)))
        #context['subject'] = np.random.poisson(3.37754)
        context['subject'] = np.random.randint(0, n_subjects)
        #print(context['subject'])
        
        # Sample action
        action = policy.action(context)
        
        # Sample reward
        reward = bandit.sample(action, context)
        opt_reward = bandit.sample_optimum(action, context)
        
        if len(cum_reward_oracle) == 0:
            cum_reward_oracle.append(opt_reward)
        else:
            cum_reward_oracle.append(cum_reward_oracle[-1] + opt_reward)
        
        if len(cum_reward) == 0: 
            cum_reward.append(reward)
        else:
            cum_reward.append(cum_reward[-1] + reward)
        
        # Set the reward
        update = policy.reward(action, context, reward)
        # Repeat
        
    return cum_reward, cum_reward_oracle
    
# Number of interactions
N = 10000
# Number of expected iterations per subject
# Do runs for {5, 10, 50, 100}
n_j = 100
# Number of subjects
n_subjects = int(N / n_j)
# For each policy, run the sim_run function 1000 times and plot it.
iterations = 10

# List of policie to loop over
#policies = [ef.EFirstPooled, ef.EFirstUnpooled, ef.EFirstPartially]
#policies = [efp.EFirstPooled, efp.EFirstUnpooled, efp.EFirstPartially]
policies = [eg.EGreedyPooled, eg.EGreedyUnpooled, eg.EGreedyPartially, eg.EGreedyPartiallyBB]
#policies = [ucb.UCBPooled, ucb.UCBUnpooled, ucb.UCBPartially, ucb.UCBPartiallyBB]
#policies = [ucb.UCBPartiallyBB, ucb.UCBPartially]
#policies = [thompson.ThompPooled, thompson.ThompUnpooled]
#policies = [eg.EGreedyPartially, eg.EGreedyPartiallyBB]
# Bandit
# Do with alpha=beta=5, and alpha=beta=10
theta_a = [1.5,1.5]#[2,5]
theta_b = [1.5,1.5] #[3,2]
#theta_a = [10,10]#[2,5]
#theta_b = [10,10] #[3,2]
#bandit = bb.BernoulliBandit(n_subjects, theta_a, theta_b)

# Initialize plotting
fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(9,3))
#ax = fig.add_subplot(111)

time1 = time.time()

# Loop over list of policies
for p in policies:
    cum_reward_list = []
    cum_regret_list = []
    # For each policy, loop a number of iterations
    for i in range(iterations):
        bandit = bb.BernoulliBandit(n_subjects, theta_a, theta_b)
        #print("Iteration: {}, Policy: {}".format(i,p))
        # In each loop, initialize the policy, such that the parameters are reset
        #policy = p(n_subjects=n_subjects,epsilon=500) #For EFirst
        #policy = p(n_subjects=n_subjects,epsilon=(n_j / 5)) #For EFirst Personal
        policy = p(n_subjects=n_subjects,epsilon=0.1) #For Egreedy
        #policy = p(n_subjects=n_subjects) # For UCB and Thompson sampling
        # Retrieve the cum_reward
        cum_reward, cum_reward_oracle = sim_run(bandit=bandit,policy=policy,N=N,n_subjects=n_subjects)
        # Put it in a list
        cum_reward_list.append(cum_reward)
        cum_regret_list.append(np.array(cum_reward_oracle) - np.array(cum_reward))
    #policy.print_coefs()
    # Average over the list
    cum_regret_list = np.array(cum_regret_list)
    mean_cum_regret = np.mean(cum_regret_list, axis = 0)
    cum_reward_list = np.array(cum_reward_list)
    mean_cum_reward = np.mean(cum_reward_list, axis = 0)
    T = np.arange(1,N+1)
    mean_cum_reward_per_t = mean_cum_reward / T
    print("Mean cumulative reward: " + str(float(mean_cum_reward[-1])))
    #print(bandit.probs_a)
    #print(bandit.probs_b)
    # Plot the average
    ax[0].plot(mean_cum_reward, label=policy.type)
    ax[1].plot(mean_cum_reward_per_t)#, label=policy.type)
    ax[2].plot(mean_cum_regret)

time2 = time.time() - time1
print(time2)
    

#box = ax[1].get_position()
#ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])

#plt.title('Average regret over time for offline evaluation with delta: {}'.format(delta))
#plt.xlabel('Time')
#plt.ylabel('Cumulative reward')
ax[0].set_ylabel('Cumulative reward')
ax[0].set_xlabel('Time')
ax[1].set_ylabel('Mean reward per t')
ax[1].set_xlabel('Time')
ax[2].set_ylabel('Regret')
ax[2].set_xlabel('Time')
plt.suptitle(policy.name + ' n_j = {}'.format(n_j))
lgd = ax[0].legend(loc='upper left', title='Pooling type', frameon=True)
#lgd = ax[1].legend(loc='center left', title='Pooling type', bbox_to_anchor=(1, 0.5))
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#fig.savefig('cumulative_reward_egreedy_njs_{}_N_{}_iters_{}.eps'.format(n_j,N,iterations), format='eps', bbox_extra_artists=(lgd,), bbox_inches = 'tight')
#fig.savefig('ucbtestfullc_nj_{}.png'.format(n_j), format='png')#, bbox_inches='tight')#, bbox_extra_artists=(lgd,))
#fig.savefig('test_high_beta.png'.format(n_j), format='png')#, bbox_inches='tight')#, bbox_extra_artists=(lgd,))
