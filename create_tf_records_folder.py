import tensorflow as tf
import numpy as np
import user_defined_function as user 
import zapisywacz as zap

is_this_classification, num_features, cat_features, ig_num_features, \
            ig_cat_features, labels = user.get_data()

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


def create_type_numerical(data_num):
    res = {}
    for k in data_num.keys():
        res[k] = len(data_num[k][0])
    return res

def take_ith_numerical(data_num, i):
    res = {}
    for k in data_num.keys():
        res[k] = data_num[k][i]
    return res

def create_type_categorical(data_cat):
    res = {}
    for k in data_cat.keys():
        res[k] = \
            list[set[list[data_cat[k].reshape((-1))]]]
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
    
nazwa_tfr_folderu = "tfr_folder"

lnt, lkt, ig_nt, ig_kt, writer = \
  zap.tworz_folder_do_zapisu(nazwa_tfr_folderu, 
    create_type_numerical(num_features), 
    create_type_categorical(cat_features), 
    create_type_numerical(ig_num_features), 
    create_type_categorical(ig_cat_features),
     is_this_classification )

for i in range(int(n / 2)):
    zap.zapisz_jeden_przyklad(
        take_ith_numerical(num_features, i),
        take_ith_cathegorical(cat_features, i), 
        take_ith_numerical(ig_num_features, i), 
        take_ith_cathegorical(ig_cat_features, i), 
        take_ith_label(labels, i), 
        lnt, lkt, ig_nt, ig_kt, writer, is_this_classification
    )

