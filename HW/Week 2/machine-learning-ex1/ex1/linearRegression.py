import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
from collections import Counter
sb.set(style="ticks", color_codes=True)


data = np.loadtxt("ex1data2.txt", delimiter=',')
m = data.shape[0]
n = data.shape[1]

X = data[:,0:2]
X = np.concatenate((np.ones((m,1)),X),axis=1)
y = data[:,2]

X_df = pd.DataFrame(X, columns=['offset','sqft','rooms'])
sb.pairplot(X_df)

a = X_df.corr()

LinReg = LinearRegression(normalize=False)
LinReg.fit(X,y)
print("score: ")
print(LinReg.score(X,y))
print("params: ")
b = LinReg.get_params()
theta = LinReg.coef_

X_test = np.array([1, 1650, 3])
pred_cost = LinReg.predict(X_test)