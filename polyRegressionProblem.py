"""
    
Problem: 
Charlie wants to purchase office-space. He does a detailed survey of the offices and corporate complexes in the area, and tries to quantify a lot of factors, such as the distance of the offices from residential and other commercial areas, schools and workplaces; the reputation of the construction companies and builders involved in constructing the apartments; the distance of the offices from highways, freeways and important roads; the facilities around the office space and so on.

Each of these factors are quantified, normalized and mapped to values on a scale of 0 to 1. Charlie then makes a table. Each row in the table corresponds to Charlie's observations for a particular house. If Charlie has observed and noted F features, the row contains F values separated by a single space, followed by the office-space price in dollars/square-foot. If Charlie makes observations for H houses, his observation table has (F+1) columns and H rows, and a total of (F+1) * H entries.

Charlie does several such surveys and provides you with the tabulated data. At the end of these tables are some rows which have just F columns (the price per square foot is missing). Your task is to predict these prices. F can be any integer number between 1 and 5, both inclusive.

There is one important observation which Charlie has made.

The prices per square foot, are (approximately) a polynomial function of the features in the observation table. This polynomial always has an order less than 4
Input Format

The first line contains two space separated integers, F and N. Over here, F is the number of observed features. N is the number of rows for which features as well as price per square-foot have been noted.
This is followed by a table having F+1 columns and N rows with each row in a new line and each column separated by a single space. The last column is the price per square foot.

The table is immediately followed by integer T followed by T rows containing F columns.

Constraints

1 <= F <= 5
5 <= N <= 100
1 <= T <= 100
0 <= Price Per Square Foot <= 10^6 0 <= Factor Values <= 1

Output Format

T lines. Each line 'i' contains the predicted price for the 'i'th test case.

"""


import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

train_cols, train_rows = map(int, input().split())

x_train, y_train = [], []
for i in range(train_rows):
    row = list(map(float, input().split(" ")))
    x_train.append(row[:-1])
    y_train.append([row[-1]])

test_rows = int(input())
x_test = []
for i in range(test_rows):
    row = list(map(float, input().split(" ")))

    x_test.append(row)

# print(f"x_train: ",x_train)
# print(f"y_train: ",y_train)
# print(f"x_test: ",x_test)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x_train)
# print(X_poly)
# poly.fit(x_train,y_train)

poly_reg = LinearRegression()
poly_reg.fit(x_train, y_train)
# poly_reg.fit(X)

y_test_predict = poly_reg.predict(x_test)
# print(y_test_predict)
for value in y_test_predict:
    print(f"{value[0]:.2f}")
