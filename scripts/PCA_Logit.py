
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

# #

X_train = pd.read_csv('/home/orangebacked/Documents/applications/ibm-auto-robot/data/X_train.csv')
pca = PCA(.95)

pca.fit(X_train)

# X_train_PCA = pca.transform(X_train)

# X_test_PCA = pca.transform(X_test)

# X_test_PCA
