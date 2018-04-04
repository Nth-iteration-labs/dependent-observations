from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("white")
a = 5
b = 5
x = np.arange (0.01, 1, 0.01)
y = beta.pdf(x,a,b)
plt.plot(x,y)
#sns.despine(offset=10,trim=True)
plt.xlim(0,1)
a = 10
b = 10
y = beta.pdf(x,a,b)
plt.plot(x,y)
plt.show()
