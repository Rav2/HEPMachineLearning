#This module uses user_defined_function.py to create dataset with 
# module io_TFRecords.py

import tensorflow as tf
import numpy as np
import user_defined_function as user 
import io_TFRecords as zap 

# user.get_data() is an user defined function. This function get_data() has to 
#return the folowing objects
# - is_this_classification that is true if this problem is classification
#     and false if it is regression problem. 
# - numerical_features is an dictionary that to each 
#       name of numerical feature binds 2d array with 
#       first index as example number and second as length of the feature.
# - cathegorical_features is dictionary that to each name of cathegorical feature
#       binds 1d array of ints.
# - ig_numerical_features that is like numerical_features but for ignored 
#       numerical features. 
# - ig_cathegorica_featuers which is like cathegorical_features but for
#       ignored cathegorical features. 
# - label is 1d numpy array. if if_this_classification is true then 
#       this should be ints (0 and 1), if it is false this is float. 

def create_tfr_record(nazwa_tfr_folderu = "tfr_folder"):

    is_this_classification, num_features, cat_features, ig_num_features, \
                ig_cat_features, labels = user.get_data()

    #shuffles data before it will be stored. 
    n = len(labels) 
    permutation = np.random.permutation(n)
    def permutate_dict(d):
        for k in d.keys():
            d[k] = d[k][permutation]

    labels = labels[permutation]
    permutate_dict(num_features)
    permutate_dict(cat_features)
    permutate_dict(ig_num_features)
    permutate_dict(ig_cat_features)


    #function useful to communicate with io_TFRecords.py module. 
    def create_type_numerical(data_num):
        res = {}
        for k in data_num.keys():
            res[k] = len(data_num[k][0])
        return res

    def create_type_categorical(data_cat):
        res = {}
        for k in data_cat.keys():
            res[k] = \
                list[set[list[data_cat[k].reshape((-1))]]]
        return res 

    #used to take i-th example
    def take_ith_numerical(data_num, i):
        res = {}
        for k in data_num.keys():
            res[k] = data_num[k][i]
        return res

    def take_ith_cathegorical(data_cat, i):
        res = {}
        for k in data_cat.keys():
            res[k] = int(data_cat[k][i])
        return res

    def take_ith_label(labels, i):
        if is_this_classification:
            return int(labels[i])
        else:
            return float(labels[i])

    lnt, lkt, ig_nt, ig_kt, writer = \
    zap.create_empty_data_folder(nazwa_tfr_folderu, 
        create_type_numerical(num_features), 
        create_type_categorical(cat_features), 
        create_type_numerical(ig_num_features), 
        create_type_categorical(ig_cat_features),
        is_this_classification )

    # I write here each example
    for i in range(int(n)):
        zap.write_one_example(
            take_ith_numerical(num_features, i),
            take_ith_cathegorical(cat_features, i), 
            take_ith_numerical(ig_num_features, i), 
            take_ith_cathegorical(ig_cat_features, i), 
            take_ith_label(labels, i), 
            lnt, lkt, ig_nt, ig_kt, writer, is_this_classification
        )
    #i have to close writer. This is very important. 
    writer.close()

