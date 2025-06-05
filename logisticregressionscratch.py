import math
import numpy as np
import pandas as pd

class LogisticRegression:


    def __init__(self,lr= 0.01,n_iters = 1000):

        self.lr = lr
        self.n_iters = n_iters
        self.weights = None #coefficients
        self.bias = None  #intercept

    def sigmoid(self,x):

        return 1/(1+np.exp(-x))

    def fit(self,X,y):

        total_samples, total_features = X.shape
        self.weights = np.zeros(total_features)
        self.bias = 0

        #gradient descent

        for i in range(total_features):


            log_model = np.dot(X,self.weights) + self.bias

            y_pred = self.sigmoid(log_model)

            #partial derivates

            dw = (1/total_samples)*(np.dot(X.T,(y_pred - y)))

            db = (1/total_samples)*(sum(y_pred - y))

            #update weights and bias

            self.weights -= (self.lr) *(dw)

            self.bias -= (self.lr)*(db)

    def predict(self,X):

        linear_model = np.dot(X,self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]

        return  y_predicted_class

#train test split

def train_test_split(data,split):

    data_shuffled = data.sample(frac=1)

    X,y = data_shuffled.iloc[:,:-1],data_shuffled.iloc[:,-1]
    total_records = X.shape[0]
    

    X_train,y_train,X_test,y_test = X.iloc[:(int(split*(total_records))),:],y.iloc[:int((split*(total_records)))],X.iloc[int((split*(total_records))):],y.iloc[int((split*(total_records))):]

    print("X_train.shape:",X_train.shape)
    print("y_train.shape:",y_train.shape)
    print("X_test.shape:",X_test.shape)
    print("y_test.shape:",y_test.shape)

    return X_train,y_train,X_test,y_test


#Accuracy

def accuracy(y_pred_cls, y):

    accuracy = np.sum(y == y_pred_cls)/len(y)

    return accuracy
    
        

data = pd.read_csv("/Users/dikshapaliwal/ML-Algorithms/diebetes.csv")
data.columns = ["I","II","III","IV","V","VI","VII","VIII","IX"]


X_train,y_train,X_test,y_test = train_test_split(data,0.75)

# print(X_train.head())
regression_model = LogisticRegression(lr = 0.01,n_iters= 2000)
regression_model.fit(X_train,y_train)

predictions = regression_model.predict(X_test)

print("Test Accuracy: ", accuracy(predictions,y_test))