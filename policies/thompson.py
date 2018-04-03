import numpy as np

class ThompUnpooled(object):
    """ Completely unpooled Thompson sampling

    """
    def __init__(self, n_subjects):
        self.name = "Thompson sampling"
        self.type = "None"
        self.n_subjects = n_subjects
        self.probs = [{'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}} for i in range(self.n_subjects)]
        self.choices = ['A', 'B']

    def action(self, context):
        subject = context['subject']
        sub_probs = self.probs[subject]
        max_val = 0
        max_key = ""
        for choice in self.choices:
            a = sub_probs[choice]['p'] * sub_probs[choice]['n']
            b = sub_probs[choice]['n'] - a
            if (a == 0) or (b == 0):
                draw = np.random.beta(1,1)
            else:
                draw = np.random.beta(a,b)
            if draw > max_val:
                max_val = draw
                max_key = choice
        return max_key

    def reward(self, action, context, reward):
        subject = context['subject']
        sub_probs = self.probs[subject][action]
        sub_probs['n'] = int(sub_probs['n']) + 1
        sub_probs['p'] = float(sub_probs['p']) + ((reward - float(sub_probs['p'])) / int(sub_probs['n']))

class ThompPooled(object):
    """ Thompson sampling Completely Pooled
    
    """
    def __init__(self, n_subjects):
        self.name = "Thompson sampling"
        self.type = "Complete"
        self.n_subjects = n_subjects
        self.probs = {'A' : {'p' : 0, 'n' : 0}, 'B' : {'p' : 0, 'n' : 0}}
        self.choices = ['A', 'B']

    def action(self, context):
        max_val = 0
        max_key = ""
        for choice in self.choices:
            a = self.probs[choice]['p'] * self.probs[choice]['n']
            b = self.probs[choice]['n'] - a
            if (a == 0) or (b == 0):
                draw = np.random.beta(1,1)
            else:
                draw = np.random.beta(a,b)
            if draw > max_val:
                max_val = draw
                max_key = choice
        return max_key

    def reward(self, action, context, reward):
        self.probs[action]['n'] = int(self.probs[action]['n']) + 1
        self.probs[action]['p'] = float(self.probs[action]['p']) + ((reward - float(self.probs[action]['p'])) / int(self.probs[action]['n']))
