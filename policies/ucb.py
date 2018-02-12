import numpy as np

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
        if self.subject_value['N'] < len(self.choices):
            for key, value in subject_value.items():
                if value['n'] == 0:
                    return key
        max_val = 0
        max_key = ""
        for key, value in subject_value.items():
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            if (value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])) > max_val:
                max_key = key
                max_val = value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])
        return max_key

    def reward(self, action, context, reward):
        #Observe the reward X_{j,t} and update the empirical mean for the chosen action.
        subject = context['subject']
        subject_value = self.probs[subject]
        subject_value['N'] = int(subject_value['N']) + 1
        value = subject_value[action]
        value['n'] = int(value['n']) + 1
        value['p'] = float(value['p']) + ((reward - float(value['p'])) / int(value['n']))

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
            for key, value in self.probs.items():
                if value['n'] == 0:
                    return key
        max_val = 0
        max_key = ""
        #For each round t = K, K+1, \dots:
        for key, value in self.probs.items():
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            if (value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])) > max_val:
                max_key = key
                max_val = value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])
        return max_key

    def reward(self, action, context, reward):
        #Observe the reward X_{j,t} and update the empirical mean for the chosen action.
        self.probs['N'] = int(self.probs['N']) + 1
        value = self.probs[action]
        value['n'] = int(value['n']) + 1
        value['p'] = float(value['p']) + ((reward - float(value['p'])) / int(value['n']))

class UCBPartially(object):
    """ Partially pooled UCB

    """
    def __init__(self, n_subjects):
        self.name = "UCB"
        self.type = "Partial"
        self.n_subjects = n_subjects
        self.count = 0
        self.probs = [{'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}, 'N' : 0} for i in range(self.n_subjects)]
        self.probs_mean = {'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}, 'N' : 0} 
        self.choices = ['A','B']

    def count_update(self):
        self.count += 1

    def action(self, context):
        #######################
        # NEEDS SHRINKAGE     #
        #######################
        subject = context['subject']
        subject_value = self.probs[subject]
        if self.subject_value['N'] < len(self.choices):
            for key, value in subject_value.items():
                if value['n'] == 0:
                    return key
        max_val = 0
        max_key = ""
        for key, value in subject_value.items():
            #Play the action j maximizing \overline{x}_j + \sqrt{2 \log t / n_j}.
            if (value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])) > max_val:
                max_key = key
                max_val = value['p'] + np.sqrt(2*np.log(self.probs['N'])/value['n'])
        return max_key
        return True

    def reward(self, action, context, reward):
        # Update subject and mean proportions
        self.count_update()
        subject = context['subject']
        self.probs[subject]['n'] = int(self.probs[subject]['n']) + 1
        self.probs[subject][action] = float(self.probs[subject][action]) + ((reward - float(self.probs[subject][action])) / int(self.probs[subject]['n']))
        self.probs_mean['n'] = int(self.probs_mean['n']) + 1
        self.probs_mean[action] = float(self.probs_mean[action]) + ((reward - float(self.probs_mean[action])) / int(self.probs_mean['n']))
        return True
