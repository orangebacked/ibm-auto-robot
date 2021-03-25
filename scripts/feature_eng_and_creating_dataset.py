import pandas as pd
import random 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('/home/orangebacked/Documents/applications/ibm-auto-robot/data/trainingset_devices11.csv')
df['ones'] = 1 


list_categorical_variables = list(df.columns)[4:8] + list(df.columns)[9:12]
# attributes 1-9 are bytes so I will read them as strings
for bytelist in list_categorical_variables:
    df[bytelist] = df[bytelist].astype(str)

# transform date into date object
df['date'] = pd.to_datetime(df.date)

# failed and unfailed devices
df_failed_dev = df[df['device'].isin(list(df[df['failure'] == 1]['device']))]
df_failed_dev_no_fail = df[df['failure'] != 1]

# creating the balanced set
df_failed_dev_no_fail['rand'] =  df_failed_dev_no_fail.ones.apply(lambda x: random.random())
random_sample_non_failed_dev = df_failed_dev_no_fail[df_failed_dev_no_fail['rand'] <= 0.08612567128661928]
balanced_set = pd.concat([df_failed_dev, random_sample_non_failed_dev])

print(balanced_set)


# creating the dummies 

for col in list(df.columns[3:12]):
    dum1 = pd.get_dummies(df_failed_dev[col])
    balanced_set = balanced_set.merge(dum1, left_index=True, right_index=True)

#print(balanced_set)

# dropping old variables
balanced_set = balanced_set.drop(['rand', 'attribute2', 'attribute3', 'attribute4', 'attribute5', 'attribute7', 'attribute8', 'attribute9', 'date', 'device'], axis=1)


# creating the data_set
y = balanced_set.failure.to_numpy()
X = balanced_set.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


pd.DataFrame(X_train).to_csv("/home/orangebacked/Documents/applications/ibm-auto-robot/data/X_train.csv", index=False)

pd.DataFrame(X_test).to_csv("/home/orangebacked/Documents/applications/ibm-auto-robot/data/X_test.csv", index=False)

pd.DataFrame(y_train).to_csv("/home/orangebacked/Documents/applications/ibm-auto-robot/data/y_train.csv", index=False)

pd.DataFrame(y_test).to_csv("/home/orangebacked/Documents/applications/ibm-auto-robot/data/y_test.csv", index=False)






