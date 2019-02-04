#This module lets you create datasets
#that are compatible with module model_dnn.py

#It writes data in TFRecords file format. You can read about
#this format here https://www.tensorflow.org/api_guides/python/reading_data 
#under standard tensorflow format. 
#You can also read about it here  https://www.tensorflow.org/tutorials/load_data/tf-records 
#and here https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564 

#This TFRecords format lets you store dictionaries which
#have strings as keys and lists of floats, ints or 
#raw bytes as values. In this module we will store 
#float and int values. 

#This module stores data in folder, which contains 
#file "dane" in TFRecords format and some other helper
#files, that contain metadata (names of features, 
# how many floats make that feature etc). 
 
#file "dane" stores dictionaries with the following keys:
# - 'l' stores label. If one is intrested in classification
#       values are 0 for background and 1 for signal. 
#       If on the other hand you are interesed in 
#       regression (for instance you are reconstructing
#       mass of some particle) this is float number. 
# - 'n' stores numerical features. This is one 
#       list of floats that has all numerical features 
#       concatenated. 
# - 'c' stores cathegorical features. This is an list
#       of ints that is an concatenation of all cathegorical
#       features 
# - 'ig_n' stores numerical values that are ignored during 
#          training, but are stored to later see correlations
#          between those ignored values and predictions. 
# - 'ig_c' the same as 'ig_n' but for  cathegorical features. 

#In dataset folder there are also other files with metadata. 
#Those folders are 
# - 'n' is an file that is used to reconstruct individual features
#       from concatenated list of features stored in file 'dane' under
#       key 'n'. So this is json list with elements 
#       [feature_name, index_beginning, index_end] such that
#       in the concatenated list feature feature_name is on positions
#       index_beginning, ..., index_end - 1. 
# - 'c' is an file that is used to reconstruct cathegorical features from 
#       concatenated list stored under key 'c' in dictionary in file "dane". 
#       This is an list where each element is 
#       [feature_name, list_of_possible_values], and feature_names
#       appear in the same order as in concatenated list (stored in file 
#       'dane' under key 'c')
# - 'ig_n' is analogus to 'n', but stores information for ignored numerical
#          features.
# - 'ig_c' is analogus to 'c', but stores information for ignored
#          cathegorical features. 
# - 'is_this_classification' is a file containing true or false. 
#          if it contains true then this is classification problem and 
#          values in file 'dane' under key 'l' are ints (0 and 1). In other 
#          case this is regression problem and values stored with key 'l' are 
#          floats.   

import tensorflow as tf
import numpy as np
import json
import os



#those are general functions to write and read 
#json files.
def write_json(what, where):
    f = open(where, 'w')
    f.write(json.dumps(what))

def read_json(from_where):
    f = open(from_where,'r')
    return json.loads(f.read())

# The following functions can be used to convert a value to a type compatible
# with tf.Example. This is needed to write them to TFRecords. values are 
# python lists.  

def _float_feature(values):
  return tf.train.Feature(float_list= tf.train.FloatList(value= values))

def _int64_feature(values):
  return tf.train.Feature(int64_list= tf.train.Int64List(value= values))

#def _bytes_feature(values):
#  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

#reads types of numerial features from file 'n' from frolder
# data_folder_name
def read_num_types(data_folder_name):
    return read_json(data_folder_name + '/n')

#reads types of cathegorical features from file 'c' from frolder
# data_folder_name
def read_cat_types(data_folder_name):
    return read_json(data_folder_name + '/c')

#reads types of ignored numerial features from file 'ig_n' from frolder
# data_folder_name
def read_ig_num_types(data_folder_name):
    return read_json(data_folder_name + '/ig_n')

#reads types of ignored cathegorical features from file 'ig_c' from frolder
# data_folder_name
def read_ig_cat_types(data_folder_name):
    return read_json(data_folder_name + '/ig_c')

#calculates how many floats are when you concatenate all float features
def num_types_length(num_types):
    if (len(num_types) == 0):
        return 0
    else:
        return num_types[len(num_types) - 1][2]

#Creates folder data_folder_name, writes metadata files with information
#about what will be stored there, and returns a bunch of objects, 
#that will be useful to the function write_one_example. 
#
#arguments are 
# - 'data_folder_name' is the name of created folder
# - 'n' is dictionary that binds names of numerical features with 
#       their lengths. (for instance four-vector will have length 4)
# - 'c' is dictionary that binds name of cathegorical feature with 
#       the list of its possible values. 
# - 'ig_n' is analogous to 'n' but for ignored numerical features
# - 'ig_c' is analogous to 'c' but for ignored cathegorical featuers. 
# - 'is_this_classification' is of type bool and is true if 
#         this dataset is used to classify (label is 0 and 1) and
#         false if this dataset is used to regression (label is floats)
# 
#returns 
# - results of functions read_num_types, read_cat_types, read_ig_num_types, read_ig_cat_types.
# - writer that will be used to actually write to dataset. 
#           it is importatn to close this writer after we wrote to file by executing 
#           writer_name.close()
def create_empty_data_folder(data_folder_name, n, c, ig_n, ig_c, is_this_classification):
    assert not os.path.isdir(data_folder_name)
    os.mkdir(data_folder_name)

    write_json(is_this_classification, data_folder_name + "/is_this_classification") 
    
    #see functions read_num_types and read_ig_num_types for explanation
    num_types = []
    ig_num_types = []
    size_of_numerical_features = 0
    for k in n.keys():
        num_types.append((k, size_of_numerical_features, size_of_numerical_features + n[k]))
        size_of_numerical_features += n[k]

    size_of_numerical_features = 0
    for k in ig_n.keys():
        ig_num_types.append((k, size_of_numerical_features, size_of_numerical_features + ig_n[k]))
        size_of_numerical_features += ig_n[k]
    write_json(num_types, data_folder_name + "/n")
    write_json(ig_num_types, data_folder_name + "/ig_n")
    
    #see read_cat_types and read_ig_cat_types for explanation
    cat_types = []
    ig_cat_types = []
    for k in c.keys():
        cat_types.append((k, c[k]))
    for k in ig_c.keys():
        ig_cat_types.append((k, ig_c[k]))
    write_json(cat_types, data_folder_name + "/c")
    write_json(ig_cat_types, data_folder_name + "/ig_c")
    writer = tf.python_io.TFRecordWriter(data_folder_name + '/dane')

    return num_types, cat_types, ig_num_types, \
        ig_cat_types, writer

#writes one example to dataset. 
#arguments are 
# - 'n' is an dictionary that takes name of numerical feature and returns 
#       its value that is float list or numpy float array (it can be also float)
# - 'c' is an dictionary that takes name of cathegorical feature and returns 
#       its int value
# - 'l' is label. If this is classificaton problem it is 0 or 1, in other case
#             this is float
# - 'ig_n' is analoguous to 'n' but for ignored numerical features
# - 'ig_c' is analoguosu to 'c' but for ignored cathegorical features
# - the next 5 arguments are the result of function create_empty_data_folder call
# - 'is_this_classification' is bool that is true if this is classificaiton problem
#      and 'l' is 0 or 1, and false in other case. 
# after writing close writer 'writer' with commend writer.close()
def write_one_example(n, c, ig_n, ig_c, l, lnt, lkt, ig_lnt, ig_lkt, writer, is_this_classification):
    #i am creating data_dictionary which will be serialized and written to TFRecords file. 
    data_dictionary = {}
    #add label
    if (is_this_classification):
        data_dictionary['l'] = _int64_feature([l])
    else:
        data_dictionary['l'] = _float_feature([l])
    #funkcja robiaca numpy tablice 1d z swojego argumentu
    def turn_to_list(numer_or_list_or_numpy_array):
        return list(np.array(numer_or_list_or_numpy_array).reshape((-1,)))
    
    #concatenate numerical feature and add to data_dictionary
    concatenated_num = []
    for i in range(len(lnt)):
        considered = lnt[i]
        concatenated_num += turn_to_list( n[ considered[0] ] )
    data_dictionary['n']= _float_feature(concatenated_num)
    ig_concatenated_num = []
    for i in range(len(ig_lnt)):
        considered = ig_lnt[i]
        ig_concatenated_num += turn_to_list( ig_n[ considered[0] ] )
    data_dictionary['ig_n']= _float_feature(ig_concatenated_num)

    #concatenate cathegorical features and add to data_dictionary
    concatenated_cat= []
    for i in range(len(lkt)):
        considered = lkt[i]
        concatenated_cat += [c[considered[0]]]
    data_dictionary['c']= _int64_feature(concatenated_cat)
    ig_concatenated_cat = []
    for i in range(len(ig_lkt)):
        considered = ig_lkt[i]
        ig_concatenated_cat += [ig_c[considered[0]]]
    data_dictionary['ig_c']= _int64_feature(ig_concatenated_cat)
    
    #serialize data_dictionary and write it 
    tr_ex = tf.train.Example(features= tf.train.Features(
        feature = data_dictionary))
    writer.write(tr_ex.SerializeToString())

#Creates dictionary of types of data written in 'dane' file. That is
# to read data written in TFRecords file you have to know what is the
# 'shape' of data written there.
def create_dictionary_of_types_needed_to_parse(data_folder_name):
    is_this_classification = read_json(data_folder_name + "/is_this_classification")
    concatenated_num = read_num_types(data_folder_name)

    ig_concatenated_num = read_ig_num_types(data_folder_name)

    #this is just for declaration of variables
    len_num = num_types_length(concatenated_num)
    len_ig_num = num_types_length(ig_concatenated_num)
    
    cat_types = read_cat_types(data_folder_name)
    ig_cat_types = read_ig_cat_types(data_folder_name)

    read_features= {}
    read_features['n'] = tf.FixedLenFeature([len_num], dtype=tf.float32)
    read_features['ig_n'] = tf.FixedLenFeature([len_ig_num], dtype=tf.float32)
    read_features['c'] = tf.FixedLenFeature([len(cat_types)], dtype=tf.int64)
    read_features['ig_c'] = tf.FixedLenFeature([len(ig_cat_types)], dtype=tf.int64)
    if is_this_classification:
        read_features['l'] = tf.FixedLenFeature([1], dtype=tf.int64)
    else:
        read_features['l'] = tf.FixedLenFeature([1], dtype=tf.float32)
    return read_features

#Read dataset, but keep it still in raw format. This is where shuffling
# of data occurs. 
#perform_shuffle is bool, buffer_size is the size of buffer used to shuffle
def read_raw_dataset(data_folder_name, perform_shuffle, buffer_size):
    dataset = tf.data.TFRecordDataset(data_folder_name + '/dane')
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset

#Reads dataset, both ignored and not ignored features. 
#it parses each example individually so it is slower. 
#perform_shuffle is bool, buffer_size is the size of buffer used to shuffle
def parse_indivdually_all(data_folder_name, perform_shuffle, buffer_size):
    read_features = create_dictionary_of_types_needed_to_parse(data_folder_name)
    def parsuj(serialized_example):
        return tf.parse_single_example(serialized=serialized_example,
                                        features=read_features)
    dataset = read_raw_dataset(data_folder_name, perform_shuffle, buffer_size)
    dataset = dataset.map(parsuj)
    return dataset

#Reads dataset, both ignored and not ignored features. 
#it parses batch of examples so it is supposedly faster. 
#num_of_threads is the numer of threads that are used to perform this persing. 
#perform_shuffle is bool, buffer_size is the size of buffer used to shuffle
def parse_batch_all(data_folder_name, batch_size, num_of_threads, perform_shuffle, buffer_size):
    read_features = create_dictionary_of_types_needed_to_parse(data_folder_name)
    def parsuj(serialized_examples):
         return tf.parse_example(serialized=serialized_examples,
                                        features=read_features)
    dataset = read_raw_dataset(data_folder_name, perform_shuffle, buffer_size)
    dataset= dataset.batch(batch_size)
    dataset = dataset.map(parsuj, num_parallel_calls = num_of_threads)
    return dataset

#function that drops ignored features
def drop_ignored(s):
    s.pop('ig_n', None)
    s.pop('ig_c', None)
    return s

#Reads dataset, drops ignored features.
#it parses each example individually so it is slower. 
#perform_shuffle is bool, buffer_size is the size of buffer used to shuffle
def parse_individualy_not_ignored(data_folder_name, perform_shuffle, buffer_size):
    read_features = create_dictionary_of_types_needed_to_parse(data_folder_name)
    def parsuj(serialized_example):
        return tf.parse_single_example(serialized=serialized_example,
                                        features=read_features)
    dataset = read_raw_dataset(data_folder_name, perform_shuffle, buffer_size)
    dataset = dataset.map(parsuj)
    dataset = dataset.map(drop_ignored)
    return dataset

#Reads dataset, drops ignored features.
#it parses batch of examples so it is supposedly faster. 
#num_of_threads is the numer of threads that are used to perform this persing. 
#perform_shuffle is bool, buffer_size is the size of buffer used to shuffle
def parse_batch_not_ignored(data_folder_name, batch_size, num_of_threads, perform_shuffle, buffer_size):
    read_features = create_dictionary_of_types_needed_to_parse(data_folder_name)
    def parsuj(serialized_examples):
         return tf.parse_example(serialized=serialized_examples,
                                        features=read_features)
    dataset = read_raw_dataset(data_folder_name, perform_shuffle, buffer_size)
    dataset= dataset.batch(batch_size)
    dataset = dataset.map(parsuj, num_parallel_calls = num_of_threads)
    dataset = dataset.map(drop_ignored)
    return dataset


    
    



