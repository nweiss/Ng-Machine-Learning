import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import scipy
from scipy.stats import spearmanr

import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import preprocessing

from collections import Counter
sb.set(style="ticks", color_codes=True)


data = np.loadtxt("ex2data2.txt", delimiter=',')
m = data.shape[0]
n = data.shape[1]

X = data[:,0:2]
y = data[:,2]

X_df = pd.DataFrame(X, columns=['test1','test2'])

# Check for independance between features
# Scatterplot
sb.regplot(x='test1',y='test2', data=X_df, scatter=True)
spearmanr_coeff, p_value = spearmanr(data[:,0],data[:,1])

test1_sq = X[:,0]**2
test1_sq = np.expand_dims(test1_sq,1)
test2_sq = X[:,1]**2
test2_sq = np.expand_dims(test2_sq,1)
X_expanded = np.concatenate((X,test1_sq,test2_sq), axis=1)
X_scaled = scale(X_expanded)

LogReg = LogisticRegression(C=1)
LogReg.fit(X_scaled, y)
print(LogReg.score(X_scaled,y))
iterations = LogReg.n_iter_

pred_training = LogReg.predict(X_scaled)
accept_ind = y == 1
reject_ind = y == 0

plt.scatter(X[accept_ind,0],X[accept_ind,1], c='b')
plt.scatter(X[reject_ind,0],X[reject_ind,1], c='r')
plt.show()