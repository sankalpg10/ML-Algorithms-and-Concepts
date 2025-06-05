#to calculate probability of data by the class they belong to 

import pandas as pd
import numpy as np
import collections


iris = pd.read_csv("/Users/dikshapaliwal/ML-Algorithms/iris.csv")

# print(iris_data.head())
iris_data = iris.values
# print(iris_data)


def SeparateByClass(dataset):

    "separata the data wrt class, assuming last col is the class"

    class_data = {}

    # for i in range(dataset.shape[0]):
    for i in range(len(dataset)):

        # row = list(dataset.iloc[i])  #if using pandas dataframe
        row = dataset[i]

        print(f"row[{i}] is {row}")

        if row[-1] not in class_data.keys():
            class_data[row[-1]] = list()

        class_data[row[-1]].append(row[:-1])


    return class_data

dataset = [[3.393533211,2.331273381,0],
 [3.110073483,1.781539638,0],
 [1.343808831,3.368360954,0],
 [3.582294042,4.67917911,0],
 [2.280362439,2.866990263,0],
 [7.423436942,4.696522875,1],
 [5.745051997,3.533989803,1],
 [9.172168622,2.511101045,1],
 [7.792783481,3.424088941,1],
 [7.939820817,0.791637231,1]]
separated = SeparateByClass(iris_data)
print(separated)


def mean(nums):

    return sum(nums)/len(nums)

def stddev(nums):


    return (1/len(nums))*((mean(nums))**2 - mean([num**2 for num in nums]))