import numpy as np

"""
NOTE: Make sure you use n_a's and n_b's when updating the proportions!
"""

class UCBUnpooled(object):
    """ Completely unpooled UCB

    """
    def __init__(self, n_subjects):
        self.name = "UCB"
        self.type = "None"
        self.n_subjects = n_subjects
        self.probs = [{'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}, 'N' : 0} for i in range(self.n_subjects)]
        self.choices = ['A','B']
        # Put in other needed parameters

    def action(self, context):
        subject = context['subject']
        subject_value = self.probs[subject]
        if subject_value['N'] < len(self.choices):
            for choice in self.choices:
                if subject_value[choice]['n'] == 0:
                    return choice
        max_val = 0
        max_key = ""
        for choice in self.choices:
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            value = subject_value[choice]
            if (value['p'] + np.sqrt(2*np.log(subject_value['N'])/value['n'])) > max_val:
                max_key = choice
                max_val = value['p'] + np.sqrt(2*np.log(subject_value['N'])/value['n'])
        return max_key

    def reward(self, action, context, reward):
        #Observe the reward X_{j,t} and update the empirical mean for the chosen action.
        subject = context['subject']
        self.probs[subject]['N'] = int(self.probs[subject]['N']) + 1
        self.probs[subject][action]['n'] = int(self.probs[subject][action]['n']) + 1
        self.probs[subject][action]['p'] = float(self.probs[subject][action]['p']) + ((reward - float(self.probs[subject][action]['p'])) / int(self.probs[subject][action]['n']))

class UCBPooled(object):

    """ Completely pooled UCB

    """
    def __init__(self, n_subjects):
        self.name = "UCB"
        self.type = "Complete"
        self.n_subjects = n_subjects
        self.probs = {'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}, 'N' : 0}
        self.choices = ['A','B']

    def action(self, context):
        #Play each of the K actions once, giving initial values for empirical mean payoffs \overline{x}_i of each action i.
        if self.probs['N'] < len(self.choices):
            for choice in self.choices:
                if self.probs[choice]['n'] == 0:
                    return choice
        max_val = 0
        max_key = ""
        #For each round t = K, K+1, \dots:
        for choice in self.choices:
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            value = self.probs[choice]
            if (value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])) > max_val:
                max_key = choice
                max_val = value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])
        return max_key

    def reward(self, action, context, reward):
        #Observe the reward X_{j,t} and update the empirical mean for the chosen action.
        self.probs['N'] = int(self.probs['N']) + 1
        self.probs[action]['n'] = int(self.probs[action]['n']) + 1
        self.probs[action]['p'] = float(self.probs[action]['p']) + ((reward - float(self.probs[action]['p'])) / int(self.probs[action]['n']))

class UCBPartially(object):
    """ Partially pooled UCB

    """
    def __init__(self, n_subjects):
        self.name = "UCB"
        self.type = "Partial"
        self.n_subjects = n_subjects
        self.probs = [{'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}, 'N' : 0} for i in range(self.n_subjects)]
        self.probs_mean = {'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}, 'N' : 0} 
        self.choices = ['A','B']

    def action(self, context):
        #######################
        # NEEDS SHRINKAGE     #
        #######################
        subject = context['subject']
        subject_value = self.probs[subject]
        if subject_value['N'] < len(self.choices):
            for choice in self.choices:
                if subject_value[choice]['n'] == 0:
                    return choice
        max_val = 0
        max_key = ""
        beta = 1/np.sqrt(self.probs[subject]['N'])
        for choice in self.choices:
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            value_mean = self.probs_mean[choice]
            value = subject_value[choice]
            p_mean = value_mean['p'] #+ np.sqrt(2*np.log(self.probs_mean['N'])/value_mean['n'])
            p_choice = value['p'] #+ np.sqrt(2*np.log(subject_value['N'])/value['n'])
            p_hat = (beta * p_mean + (1-beta) * p_choice) + np.sqrt(2*np.log(self.probs_mean['N'])/value_mean['n'])

            if p_hat > max_val:
                max_key = choice
                max_val = p_hat
        return max_key

    def reward(self, action, context, reward):
        # Update subject and mean proportions
        subject = context['subject']
        self.probs[subject]['N'] = int(self.probs[subject]['N']) + 1
        self.probs[subject][action]['n'] = int(self.probs[subject][action]['n']) + 1
        self.probs[subject][action]['p'] = float(self.probs[subject][action]['p']) + ((reward - float(self.probs[subject][action]['p'])) / int(self.probs[subject][action]['n']))
        # Mean proportions
        self.probs_mean['N'] = int(self.probs_mean['N']) + 1
        self.probs_mean[action]['n'] = int(self.probs_mean[action]['n']) + 1
        self.probs_mean[action]['p'] = float(self.probs_mean[action]['p']) + ((reward - float(self.probs_mean[action]['p'])) / int(self.probs_mean[action]['n']))
