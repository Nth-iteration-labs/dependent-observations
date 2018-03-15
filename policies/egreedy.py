import numpy as np

class EGreedyUnpooled(object):
    """ Completely unpooled EGreedy
    
    """
    def __init__(self, n_subjects, epsilon = .1):
        self.name = "E-Greedy"
        self.type = "None"
        self.n_subjects = n_subjects
        self.epsilon = epsilon
        self.count = 0
        self.probs = [{'A' : 0, 'B' : 0, 'n' : {'A' : 0, 'B' : 0}} for i in range(self.n_subjects)]
        self.choices = ['A','B']

    def count_update(self):
        self.count += 1
        
    def print_coefs(self):
        print(self.probs)
        
    def action(self, context):
        subject = context['subject']
        # If random sample is smaller or equal to epsilon, we do a random sample.
        if np.random.uniform(0,1) <= self.epsilon:
            # Random sample
            return self.choices[np.random.binomial(1,0.5)]
        # Else we take a maximum
        else:
            # Take maximum
            if self.probs[subject]['A'] > self.probs[subject]['B']:
                return 'A'
            else: #if probs_b > probs_a
                return 'B'
    
    def reward(self, action, context, reward):
        # Update proportion
        self.count_update()
        subject = context['subject']
        self.probs[subject]['n'][action] = int(self.probs[subject]['n'][action]) + 1
        self.probs[subject][action] = float(self.probs[subject][action]) + ((reward - float(self.probs[subject][action])) / int(self.probs[subject]['n'][action]))
        #print("Subject: {} data: {}".format(subject,self.probs[subject]))

class EGreedyPooled(object):
    """ Completely pooled EGreedy
    
    """
    def __init__(self, n_subjects, epsilon = .1):
        self.name = "E-Greedy"
        self.type = "Complete"
        self.n_subjects = n_subjects
        self.epsilon = epsilon
        self.probs = {'A' : 0, 'B' : 0, 'n' : {'A' : 0, 'B' : 0}}
        self.choices = ['A','B']
        
    def print_coefs(self):
        print(self.probs)
        
    def action(self, context):
        # If random sample is smaller or equal to epsilon, we do a random sample.
        if np.random.uniform(0,1) <= self.epsilon:
            # Random sample
            return self.choices[np.random.binomial(1,0.5)]
        # Else we take a maximum
        else:
            # Take maximum
            if self.probs['A'] > self.probs['B']:
                return 'A'
            else: #if probs_b > probs_a
                return 'B'
    
    def reward(self, action, context, reward):
        # Update proportion
        self.probs['n'][action] = int(self.probs['n'][action]) + 1
        self.probs[action] = float(self.probs[action]) + ((reward - float(self.probs[action])) / int(self.probs['n'][action]))
        #print(self.probs)
        
class EGreedyPartially(object):
    """ Partially (un)pooled EGreedy
    
    """
    def __init__(self, n_subjects, epsilon = .1):
        self.name = "E-Greedy"
        self.type = "Partial"
        self.n_subjects = n_subjects
        self.epsilon = epsilon
        self.count = 0
        self.probs = [{'A' : 0, 'B' : 0, 'n' : {'A' : 1, 'B' : 1}} for i in range(self.n_subjects)]
        self.probs_mean = {'A' : 0, 'B' : 0, 'n' : {'A' : 0, 'B' : 0}}
        self.choices = ['A','B']

    def count_update(self):
        self.count += 1
        
    def print_coefs(self):
        print(self.probs)
        
    def action(self, context):
        subject = context['subject']
        # If random sample is smaller or equal to epsilon, we do a random sample.
        if np.random.uniform(0,1) <= self.epsilon:
            # Random sample
            return self.choices[np.random.binomial(1,0.5)]
        # Else we take a maximum
        else:
            # Calculate p_a_i_hat and p_b_i_hat
            beta = 1/np.sqrt(self.probs[subject]['n']['A'] + self.probs[subject]['n']['B'])
            p_a_mean = self.probs_mean['A'] #sum(d['A'] for d in self.probs) / len(self.probs)
            p_b_mean = self.probs_mean['B'] #sum(d['B'] for d in self.probs) / len(self.probs)
            p_a_hat = beta * p_a_mean + (1-beta) * self.probs[subject]['A']
            p_b_hat = beta * p_b_mean + (1-beta) * self.probs[subject]['B']
            # Take maximum of the two
            if p_a_hat > p_b_hat:
                return 'A'
            else: #if probs_b > probs_a
                return 'B'
    
    def reward(self, action, context, reward):
        # Update proportion
        self.count_update()
        subject = context['subject']
        self.probs[subject]['n'][action] = int(self.probs[subject]['n'][action]) + 1
        self.probs[subject][action] = float(self.probs[subject][action]) + ((reward - float(self.probs[subject][action])) / int(self.probs[subject]['n'][action]))
        self.probs_mean['n'][action] = int(self.probs_mean['n'][action]) + 1
        self.probs_mean[action] = float(self.probs_mean[action]) + ((reward - float(self.probs_mean[action])) / int(self.probs_mean['n'][action]))
    
