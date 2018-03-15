import numpy as np

class BernoulliBandit(object):
    
    def __init__(self, n_subjects, theta_a, theta_b):
        self.n_subjects = n_subjects
        #self.probs_a = [0.7,0.3,0.7,0.3]
        #self.probs_b = [0.3,0.7,0.3,0.7]
        self.probs_a = np.random.beta(theta_a[0], theta_a[1], self.n_subjects)
        self.probs_b = np.random.beta(theta_b[0], theta_b[1], self.n_subjects)
            
    def sample(self, action, context):
        subject = context['subject']
        if action == 'A':
            return np.random.binomial(1, self.probs_a[subject])
        elif action == 'B':
            return np.random.binomial(1, self.probs_b[subject])

    def sample_optimum(self, action, context):
        subject = context['subject']
        # Now return sample with highest of prob_a or prob_b
        return np.random.binomial(1, max(self.probs_a[subject], self.probs_b[subject]))