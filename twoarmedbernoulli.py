import numpy as np
import policies.efirst as ef
import policies.egreedy as eg
import bandits.bernoullibandit as bb
import matplotlib.pyplot as plt
import seaborn as sns
import time
sns.set_style("white")
sns.set_color_codes("colorblind")

def sim_run(bandit, policy, N):
    
    cum_reward = [0]
    
    for i in range(N):
        # Set up context with subjects between 1 and n_subjects
        context = {}
        context['subject'] = int(np.floor(np.random.pareto(1,1)))
        if context['subject'] >= 50:
            context['subject'] = 49
        #context['subject'] = np.random.randint(0,n_subjects)
        
        # Sample action
        action = policy.action(context)
        
        # Sample reward
        reward = bandit.sample(action,context)
        
        cum_reward.append(cum_reward[-1]+reward)
        
        # Set the reward
        update = policy.reward(action,context,reward)
        # Repeat
        
    return cum_reward
    
# Number of runs
N = 500
# Number of expected iterations per subject
# Do runs for {5, 10, 50, 100}
n_j = 10
# Number of subjects
n_subjects = int(N / n_j)
# For each policy, run the sim_run function 1000 times and plot it.
iterations = 1000


# List of policie to loop over
policies = [ef.EFirstPooled, ef.EFirstUnpooled, ef.EFirstPartially]
#policies = [eg.EGreedyPooled, eg.EGreedyUnpooled, eg.EGreedyPartially]
# Bandit
# Do with alpha=beta=5, and alpha=beta=10
theta_a = [2,5]
theta_b = [3,2]
bandit = bb.BernoulliBandit(n_subjects, theta_a, theta_b)

# Initialize plotting
fig = plt.figure(1)
ax = fig.add_subplot(111)

time1 = time.time()

# Loop over list of policies
for p in policies:
    cum_reward_list = []
    # For each policy, loop a number of iterations
    for i in range(iterations):
        #print("Iteration: {}, Policy: {}".format(i,p))
        # In each loop, initialize the policy, such that the parameters are reset
        policy = p(n_subjects=n_subjects,epsilon=50) #For EFirst
        #policy = p(n_subjects=n_subjects,epsilon=0.1) #For Egreedy
        # Retrieve the cum_reward
        cum_reward = sim_run(bandit=bandit,policy=policy,N=N)
        # Put it in a list
        cum_reward_list.append(cum_reward)
    #policy.print_coefs()
    # Average over the list
    cum_reward_list = np.array(cum_reward_list)
    mean_cum_reward = np.mean(cum_reward_list,axis=0)
    print("Cumulative reward: " + str(float(cum_reward[-1])))
    # Plot the average
    ax.plot(mean_cum_reward, label=policy.type)

time2 = time.time() - time1
print(time2)
    

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

#plt.title('Average regret over time for offline evaluation with delta: {}'.format(delta))
plt.xlabel('Time')
plt.ylabel('Cumulative reward')
plt.title(policy.name)
lgd = ax.legend(loc='center left', title='Pooling type', bbox_to_anchor=(1, 0.5))
plt.show()

fig.savefig('cumulative_reward_egreedy_njs_{}_N_{}_iters_{}.eps'.format(n_j,N,iterations), format='eps', bbox_extra_artists=(lgd,), bbox_inches = 'tight')
