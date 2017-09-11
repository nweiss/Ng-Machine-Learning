import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb

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
