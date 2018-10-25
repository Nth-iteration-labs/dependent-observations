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
        beta = 1/np.sqrt(subject_value['N'])
        for choice in self.choices:
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            #beta = 1/np.sqrt(subject_value[choice]['n'])
            value_mean = self.probs_mean[choice]
            value = subject_value[choice]
            p_mean = value_mean['p'] + np.sqrt(2*np.log(self.probs_mean['N'])/value_mean['n'])
            p_choice = value['p'] + np.sqrt(2*np.log(subject_value['N'])/value['n'])
            p_hat = (beta * p_mean + (1-beta) * p_choice) 
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

class UCBPartiallyBB(object):
    """ Partially pooled UCB with BB shrinkage with BB shrinkage with BB shrinkage with BB shrinkage

    """
    def __init__(self, n_subjects):
        self.name = "UCB"
        self.type = "Partial BB"
        self.n_subjects = n_subjects
        self.probs = [{'p_i' : {'A' : 0, 'B' : 0}, 'n' : {'A' : 0, 'B' : 0}, 'SS_i' : {'A' : 0, 'B' : 0}, 'c_i' : {'A' : 0, 'B' : 0}, 'N' : 0} for i in range(self.n_subjects)]
        self.probs_mean = {'p' : {'A' : 0, 'B' : 0}, 'n' : {'A' : 0, 'B' : 0}, 'SS' : {'A' : 0, 'B' : 0}, 'c' : {'A' : 0, 'B' : 0}, 'N' : 0}
        self.choices = ['A','B']

    def action(self, context):
        #######################
        # NEEDS SHRINKAGE     #
        #######################
        subject = context['subject']
        subject_value = self.probs[subject]
        if subject_value['N'] < len(self.choices):
            for choice in self.choices:
                if subject_value['n'][choice] == 0:
                    return choice
        max_val = 0
        max_key = ""
        for choice in self.choices:
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            p = self.probs_mean['p'][choice]
            sigmasq = (self.n_subjects * self.probs_mean['SS'][choice]) / ((self.n_subjects - 1) * self.probs_mean['n'][choice])
            if sigmasq > 0:
                M = max((p * (1 - p) - sigmasq) / (sigmasq - ((p* (1 - p)) / self.n_subjects) * self.probs_mean['c'][choice]), 0)
                beta = M / (M + (self.probs[subject]['N']))
            else:
                beta = 1
            p_mean = self.probs_mean['p'][choice] + np.sqrt(2*np.log(self.probs_mean['N'])/self.probs_mean['n'][choice])
            p_choice = subject_value['p_i'][choice] + np.sqrt(2*np.log(subject_value['N'])/subject_value['n'][choice])
            p_hat = (beta * p_mean + (1-beta) * p_choice) 
            if p_hat > max_val:
                max_key = choice
                max_val = p_hat
        return max_key

    def reward(self, action, context, reward):
        # Update subject and mean proportions
        subject = context['subject']
        self.probs[subject]['N'] = int(self.probs[subject]['N']) + 1
        self.probs_mean['N'] = int(self.probs_mean['N']) + 1

        self.probs[subject]['n'][action] = int(self.probs[subject]['n'][action]) + 1
        self.probs[subject]['p_i'][action] = float(self.probs[subject]['p_i'][action]) + ((reward - float(self.probs[subject]['p_i'][action])) / int(self.probs[subject]['n'][action]))

        # Update overall parameters
        self.probs_mean['n'][action] = int(self.probs_mean['n'][action]) + 1
        self.probs_mean['p'][action] = float(self.probs_mean['p'][action]) + ((reward - float(self.probs_mean['p'][action])) / int(self.probs_mean['n'][action]))
        self.probs_mean['SS'][action] = self.probs_mean['SS'][action] - self.probs[subject]['SS_i'][action] + self.probs[subject]['n'][action] * (self.probs[subject]['p_i'][action] - self.probs_mean['p'][action])**2
        self.probs_mean['c'][action] = self.probs_mean['c'][action] - self.probs[subject]['c_i'][action] + 1/self.probs[subject]['n'][action]
    
        # Update rest of individual parameters, these have to be last!
        self.probs[subject]['c_i'][action] = 1/self.probs[subject]['n'][action]
        self.probs[subject]['SS_i'][action] = self.probs[subject]['n'][action] * (self.probs[subject]['p_i'][action] - self.probs_mean['p'][action])**2
