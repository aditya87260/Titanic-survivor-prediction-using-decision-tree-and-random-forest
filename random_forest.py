import pandas as pd
import numpy as np

df=pd.read_csv("Train.csv")
df_test=pd.read_csv("Test.csv")
print(df_test.shape)
#print(df.head(n=10))
#print(df.info())
columns_to_drop = ['name','ticket','cabin','embarked','boat','body','home.dest']
data_clean = df.drop(columns_to_drop,axis=1)
data_clean_test = df_test.drop(columns_to_drop,axis=1)
print(data_clean_test.shape)
#print(data_clean.head(n=5))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_clean["sex"] = le.fit_transform(data_clean["sex"])
data_clean_test["sex"] = le.fit_transform(data_clean_test["sex"])
#print(data_clean.head())
#print(data_clean.info())
avg_age = data_clean["age"].mean()
avg_age = data_clean_test["age"].mean()
#print(avg_age)
data_clean = data_clean.fillna(avg_age)
data_clean_test = data_clean_test.fillna(avg_age)
#print(data_clean.info())
#print("test",data_clean_test.info())
input_cols = ['pclass',"sex","age","sibsp","parch","fare"]
output_cols = ["survived"]
data_clean_test=data_clean_test[input_cols]

#print("test",data_clean_test.shape)
X = data_clean[input_cols]
Y = data_clean[output_cols]


#print(X.shape,Y.shape)
#print(type(X))

#Train-Validation-Test Set Split
split = int(0.7*data_clean.shape[0])
train_data = data_clean[:split]
test_data = data_clean[split:]
test_data = test_data.reset_index(drop=True)
print(train_data.shape,test_data.shape)

from sklearn.tree import DecisionTreeClassifier

#Random Forest
X_train = train_data[input_cols]
Y_train = np.array(train_data[output_cols]).reshape((-1,))
print(Y_train)
print("Y",Y_train.shape)
X_test = test_data[input_cols]
Y_test = np.array(test_data[output_cols]).reshape((-1,))
sk_tree = DecisionTreeClassifier(criterion='entropy',max_depth=5)
sk_tree.fit(X_train,Y_train)
print(sk_tree.score(X_train,Y_train))
print(sk_tree.score(X_test,Y_test))
from sklearn.ensemble import RandomForestClassifier
"""rf = RandomForestClassifier(n_estimators=10,criterion='entropy',max_depth=5)
#rf.fit(X_train,Y_train)
#print(rf.score(X_train,Y_train))
print(rf.score(X_test,Y_test))"""

from sklearn.model_selection import cross_val_score
#acc = cross_val_score(RandomForestClassifier(n_estimators=40,max_depth=5,criterion='entropy'),X_train,Y_train,cv=5).mean()
#print(acc)
acc_list = []
for i in range(1,50):
    acc = cross_val_score(RandomForestClassifier(n_estimators=i,max_depth=5),X_train,Y_train,cv=5).mean()
    acc_list.append(acc)

print(acc_list)

import matplotlib.pyplot as plt
plt.style.use("seaborn")
plt.plot(acc_list)
plt.show()

print(np.argmax(acc_list))

rf = RandomForestClassifier(n_estimators=44,max_depth=5,criterion='entropy')
rf.fit(X_train,Y_train)           
print(rf.score(X_train,Y_train))
print(rf.score(X_test,Y_test))
pred=rf.predict(data_clean_test)
pred=pred.reshape((300,1))
Y_test=pd.DataFrame(pred,columns=['survived'])
Y_test.to_csv('titanic_predictions1.csv')