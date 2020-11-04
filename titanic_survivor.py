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

print(X.shape,Y.shape)
print(type(X))

# Define Entropy and Information Gain
def entropy(col):
    
    counts = np.unique(col,return_counts=True)
    N = float(col.shape[0])
    
    ent = 0.0
    
    for ix in counts[1]:
        p  = ix/N
        ent += (-1.0*p*np.log2(p))
    
    return ent

def divide_data(x_data,fkey,fval):
    #Work with Pandas Data Frames
    x_right = pd.DataFrame([],columns=x_data.columns)
    x_left = pd.DataFrame([],columns=x_data.columns)
    
    for ix in range(x_data.shape[0]):
        val = x_data[fkey].loc[ix]
        
        if val > fval:
            x_right = x_right.append(x_data.loc[ix])
        else:
            x_left = x_left.append(x_data.loc[ix])
            
    return x_left,x_right

#x_left,x_right = divide_data(data_clean[:10],'Sex',0.5)
#print(x_left)
#print(x_right)

def information_gain(x_data,fkey,fval):
    
    left,right = divide_data(x_data,fkey,fval)
    
    #% of total samples are on left and right
    l = float(left.shape[0])/x_data.shape[0]
    r = float(right.shape[0])/x_data.shape[0]
    
    #All examples come to one side!
    if left.shape[0] == 0 or right.shape[0] ==0:
        return -1000000 #Min Information Gain
    
    i_gain = entropy(x_data.survived) - (l*entropy(left.survived)+r*entropy(right.survived))
    return i_gain

# Test our function
for fx in X.columns:
    print(fx)
    print(information_gain(data_clean,fx,data_clean[fx].mean()))


class DecisionTree:
    
    #Constructor
    def __init__(self,depth=0,max_depth=5):
        self.left = None
        self.right = None
        self.fkey = None
        self.fval = None
        self.max_depth = max_depth
        self.depth = depth
        self.target = None
        
    def train(self,X_train):
        
        features = ['pclass',"sex","age","sibsp","parch","fare"]
        info_gains = []
        
        for ix in features:
            i_gain = information_gain(X_train,ix,X_train[ix].mean())
            info_gains.append(i_gain)
            
        self.fkey = features[np.argmax(info_gains)]
        self.fval = X_train[self.fkey].mean()
        print("Making Tree Features is",self.fkey)
        
        #Split Data
        data_left,data_right = divide_data(X_train,self.fkey,self.fval)
        data_left = data_left.reset_index(drop=True)
        data_right = data_right.reset_index(drop=True)
         
        #Truly a left node
        if data_left.shape[0]  == 0 or data_right.shape[0] ==0:
            if X_train.survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        #Stop earyly when depth >=max depth
        if(self.depth>=self.max_depth):
            if X_train.survived.mean() >= 0.5:
                self.target = "Survive"
            else:
                self.target = "Dead"
            return
        
        #Recursive Case
        self.left = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.left.train(data_left)
        
        self.right = DecisionTree(depth=self.depth+1,max_depth=self.max_depth)
        self.right.train(data_right)
        
        #You can set the target at every node
        if X_train.survived.mean() >= 0.5:
            self.target = "Survive"
        else:
            self.target = "Dead"
        return
    
    def predict(self,test):
        if test[self.fkey]>self.fval:
            #go to right
            if self.right is None:
                return self.target
            return self.right.predict(test)
        else:
            if self.left is None:
                return self.target
            return self.left.predict(test)

#Train-Validation-Test Set Split
split = int(0.7*data_clean.shape[0])
train_data = data_clean[:split]
test_data = data_clean[split:]
test_data = test_data.reset_index(drop=True)
print(train_data.shape,test_data.shape)

dt = DecisionTree()
print(dt.train(train_data))
print(dt.fkey)
print(dt.fval)
print(dt.left.fkey)
print(dt.right.fkey)
y_pred = []
for ix in range(data_clean_test.shape[0]):
    y_pred.append(dt.predict(data_clean_test.loc[ix]))

#y_actual = test_data[output_cols]
le = LabelEncoder()
y_pred = le.fit_transform(y_pred)
#print(y_pred)
"""
y_pred = np.array(y_pred).reshape((-1,1))
print(y_pred.shape)
acc = np.sum(y_pred==y_actual)/y_pred.shape[0]
acc = np.sum(np.array(y_pred)==np.array(y_actual))/y_pred.shape[0]
print(acc)
f = pd.DataFrame(y_pred,columns = ["survived"])
Id = np.arange(1009)
f['Id'] = np.arange(1009)
f = f[['Id','survived']]
print(f.head())
f.to_csv('sample.csv',index = False)"""