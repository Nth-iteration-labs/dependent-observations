import numpy as np

"""
NOTE: Make sure that you use n_a's and n_b's when updating the proportions!
"""

class EFirstUnpooled(object):
    """ Completely unpooled EFirst
    
    """
    def __init__(self, n_subjects, epsilon = 100):
        self.name = "E-First"
        self.type = "None"
        self.n_subjects = n_subjects
        self.epsilon = epsilon
        self.count = 0
        self.probs = [{'A' : 0, 'B' : 0, 'n' : 0} for i in range(self.n_subjects)]
        self.choices = ['A','B']

    def count_update(self):
        self.count += 1
        
    def print_coefs(self):
        print(self.probs)
        
    def action(self, context):
        subject = context['subject']
        if self.count <= self.epsilon:
            # Random sample
            return self.choices[np.random.binomial(1,0.5)]
        else:
            # Take maximum
            if self.probs[subject]['A'] > self.probs[subject]['B']:
                return 'A'
            else: #if probs_b > probs_a
                return 'B'
    
    def reward(self, action, context, reward):
        # Update proportion
        if self.count <= self.epsilon:
            self.count_update()
            subject = context['subject']
            self.probs[subject]['n'] = int(self.probs[subject]['n']) + 1
            self.probs[subject][action] = float(self.probs[subject][action]) + ((reward - float(self.probs[subject][action])) / int(self.probs[subject]['n']))

class EFirstPooled(object):
    """ Completely pooled EFirst
    
    """
    def __init__(self, n_subjects, epsilon = 100):
        self.name = "E-First"
        self.type = "Complete"
        self.n_subjects = n_subjects
        self.epsilon = epsilon
        self.probs = {'A' : 0, 'B' : 0, 'n' : 0}
        self.choices = ['A','B']
        
    def print_coefs(self):
        print(self.probs)

    def action(self, context):
        if self.probs['n'] <= self.epsilon:
            # Random sample
            return self.choices[np.random.binomial(1,0.5)]
        else:
            # Take maximum
            if self.probs['A'] > self.probs['B']:
                return 'A'
            else: #if probs_b > probs_a
                return 'B'
    
    def reward(self, action, context, reward):
        # Update proportion
        if self.probs['n'] <= self.epsilon:
            self.probs['n'] = int(self.probs['n']) + 1
            self.probs[action] = float(self.probs[action]) + ((reward - float(self.probs[action])) / int(self.probs['n']))

# TODO Partially pooled

class EFirstPartially(object):
    """ Partially pooled EFirst
    
    The idea is that we have some p_ai_hat that combines p_ai and p_a with a beta = 1/sqrt(n_i)
    What we need to keep track of is: all individual p_a_i, and we need an easy way to combine them (mean)
    
    """
    def __init__(self, n_subjects, epsilon = 100):        
        self.name = "E-First"
        self.type = "Partial"
        self.n_subjects = n_subjects
        self.epsilon = epsilon
        self.count = 0
        self.probs = [{'A' : 0, 'B' : 0, 'n' : 1} for i in range(self.n_subjects)]
        self.probs_mean = {'A' : 0, 'B' : 0, 'n' : 0}
        self.choices = ['A','B']

    def count_update(self):
        self.count += 1
        
    def print_coefs(self):
        print(self.probs)
        
    def action(self, context):
        subject = context['subject']
        if self.count <= self.epsilon:
            # Random sample
            return self.choices[np.random.binomial(1,0.5)]
        else:
            # Calculate p_a_i_hat and p_b_i_hat
            beta = 1/np.sqrt(self.probs[subject]['n'])
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
        # Update proportion of subject and mean
        if self.count <= self.epsilon:
            self.count_update()
            subject = context['subject']
            self.probs[subject]['n'] = int(self.probs[subject]['n']) + 1
            self.probs[subject][action] = float(self.probs[subject][action]) + ((reward - float(self.probs[subject][action])) / int(self.probs[subject]['n']))
            self.probs_mean['n'] = int(self.probs_mean['n']) + 1
            self.probs_mean[action] = float(self.probs_mean[action]) + ((reward - float(self.probs_mean[action])) / int(self.probs_mean['n']))
