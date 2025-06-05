import numpy as np
# import collections


# class FeatureEngineering():

#     def __init__(self):
#         pass

def mean_encoding(feature_vector,label_data):

    """ For all unique values in the feature replace them with the sample mean 
    value of the label
    """

    uniques = list(np.unique(feature_vector))
    print(uniques)
    # print(f"uniques: " ,{uniques})

    cat_means = {}

    for category in uniques:

        cat_means[category] = sum([label_data[i] if feature_vector[i] == category else 0 for i in range(len(label_data))])/len(label_data)

    # print(cat_means)
    for i in range(len(feature_vector)):

        if feature_vector[i] in cat_means:

            feature_vector[i] = cat_means[feature_vector[i]]

    return feature_vector



x = ['a','b','c','d','e','e','a','b','d','d','e','e','a','b','d']
y = [1  ,2   ,3  ,4  ,4  ,3  ,2  ,4  ,3  ,2  ,1  ,1  ,2 , 3  ,4]


print(list(np.unique(x)))


x_ = mean_encoding(x,y)
print(x_)
