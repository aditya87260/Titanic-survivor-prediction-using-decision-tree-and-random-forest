import pandas as pd
import numpy as np

df=pd.read_csv("Train.csv")
df_test=pd.read_csv("Test.csv")
#print(df.head(n=10))
#print(df.info())
columns_to_drop = ['name','ticket','cabin','embarked','boat','body','home.dest']
data_clean = df.drop(columns_to_drop,axis=1)
data_clean_test = df.drop(columns_to_drop,axis=1)
#print(data_clean.head(n=5))
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_clean["sex"] = le.fit_transform(data_clean["sex"])
data_clean_test["sex"] = le.transform(data_clean_test["sex"])
#print(data_clean.head())
#print(data_clean.info())
avg_age = data_clean["age"].mean()
avg_age = data_clean_test["age"].mean()
#print(avg_age)
data_clean = data_clean.fillna(avg_age)
data_clean_test = data_clean_test.fillna(avg_age)
#print(data_clean.info())
print("test",data_clean_test.info())
input_cols = ['pclass',"sex","age","sibsp","parch","fare"]
output_cols = ["survived"]
data_clean_test=data_clean_test[input_cols]
X = data_clean[input_cols]
Y = data_clean[output_cols]
print("test",data_clean_test.shape)

#print(X.shape,Y.shape)
#print(type(X))

#Train-Validation-Test Set Split
split = int(0.7*data_clean.shape[0])
train_data = data_clean[:split]
test_data = data_clean[split:]
test_data = test_data.reset_index(drop=True)
print(train_data.shape,test_data.shape)

from sklearn.tree import DecisionTreeClassifier
sk_tree = DecisionTreeClassifier(criterion='entropy',max_depth=5)
sk_tree.fit(train_data[input_cols],train_data[output_cols])
y_pred=sk_tree.predict(test_data[input_cols])
print(sk_tree.score(test_data[input_cols],test_data[output_cols]))

#Visualise a Decision Tree

import pydotplus

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
dot_data = StringIO()
export_graphviz(sk_tree,out_file=dot_data,filled=True,rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_pdf("Image.pdf")



