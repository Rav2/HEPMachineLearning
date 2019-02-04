#deep neural network model that can be trained on datasets created by 
#module io_TFRecords.py. It uses tensorflow build-in estimators
#namely tf.estimator.DNNClassifier for classification and 
# tf.estimator.DNNRegressor for regression. I am trying to use
# performance tips from https://www.tensorflow.org/guide/performance/datasets
# that is i am using prefetch, you can use multiple threads 
# to parse data, you can also cache dataset, if it is small enough. 

#It creates folder for our model. In this folder there are the following
#files:
# - model folder, that stores our tensorflow estimator.
# - source_of_data, that stores path to our dataset created by
#       io_TFRecords.py
# - dropout that stores how much dropout we used
# - hidden_units that stores how many hidden_units are in our model
# - is_this_classification that stores wheather this model is for classification. 
# - mean that stores means of numerical features from source_of_data
# - var that stores variance of data from source_of_data. 

import tensorflow as tf
import numpy as np
import json
import os
import io_TFRecords

#those are general functions to write and read 
#json files.
def write_json(what, where):
    f = open(where, 'w')
    f.write(json.dumps(what))

def read_json(from_where):
    f = open(from_where,'r')
    return json.loads(f.read())

#changes name of feature so that it won't get mixed
def make_name_longer(name):
    return 'kat_fet_' + name

#creates feature columns for our model. If you want to change
#something use this documentation
# https://www.tensorflow.org/guide/feature_columns
# source_name is the name of folder with dataset created by 
# modle io_TFRecords.
def create_feature_columns(source_name):
    cat_types = io_TFRecords.read_cat_types(source_name)

    num_types = io_TFRecords.read_num_types(source_name)
    length_of_numerical = io_TFRecords.num_types_length(num_types)
    result = []
    result.append(tf.feature_column.numeric_column(
        key = 'n', shape=(length_of_numerical,)  ))
    for cat in cat_types:
        new_feature = tf.feature_column.categorical_column_with_vocabulary_list(
            key = make_name_longer( cat[0]),
            vocabulary_list = cat[1],
            dtype = tf.int32
        )
        #Here instead of indicator column that does one hot encoding one can use embedding_column
        #https://www.tensorflow.org/api_docs/python/tf/feature_column/embedding_column
        #which learns good encoding of your cathegorical data into some R^n
        hot = tf.feature_column.indicator_column(new_feature)
        result.append(hot)
    return result

#this function creates input function for model. Arguments are 
# - source_of_data is path to folder created by module io_TFRecords.py
# - model_folder is path to our model direcotory created by function create_model_folder. 
# - batch_size is batch size
# - perform_shuffle is bool that tells weather or not we should shuffle data
# - num_of_threads tells how many parallel processes should parse data for 
#      our model
# - perform_repeat is bool that tells weather we should repeat data
#           when we run out of it
# - buffer_size is related to perform_shuffle. This is an size of buffer used to 
#       shuffle data
# - perform_cache is bool that tells weather we should keep dataset on ram. 
def input_fn(source_of_data, model_folder, batch_size,  
         perform_shuffle, num_of_threads, perform_repeat, buffer_size,
         perform_cache):
    def result_input_function():
        mean, var = read_stored_mean_var(model_folder)
        dataset = io_TFRecords.parse_batch_not_ignored(source_of_data, batch_size, num_of_threads, perform_shuffle, buffer_size)
        cathegorical_data = io_TFRecords.read_cat_types(source_of_data)
        if(perform_repeat):
            dataset = dataset.repeat()
        def normalization(dictionary):
            dictionary['n'] = (dictionary['n'] - mean) / (var**0.5)
            return dictionary
        def unpack_cathegorical(dictionary):
            cat_tab = dictionary.pop('c')
            for i in range(len(cathegorical_data)):
                dictionary[ make_name_longer(cathegorical_data[i][0])] = cat_tab[:, i]
            return dictionary
        def unpack_label(dictionary):
            l= dictionary.pop('l')
            return dictionary, l
        dataset = dataset.map(unpack_cathegorical, num_parallel_calls = num_of_threads)
        dataset = dataset.map(normalization,num_parallel_calls = num_of_threads)
        dataset = dataset.map(unpack_label,num_parallel_calls = num_of_threads)
        dataset = dataset.prefetch(1)
        if perform_cache:
            dataset = dataset.cache()
        return dataset
    return result_input_function

#creates folder with our model. arguments are 
# - model_folder is the name of created folder
# - source_of_data is an folder created with module io_TFRecords.py
# - hidden_units are hidden units, that is list of ints. 
#    from left those are units that touch input, and on the right
#    are units that touch output.
# - dropout is from 0. to 1. and tells how much dropout to apply
# - num_examples_for_statistics is the number of examples on which 
#      model will calculate mean and variance of data. 
# returns model that was created. You can ignore this returned model
def create_model_folder(model_folder, source_of_data, hidden_units, 
    dropout, num_examples_for_statistics = 10000):
    assert not os.path.isdir(model_folder)
    os.mkdir(model_folder)
    write_json(source_of_data, model_folder + '/source_of_data')
    is_this_classification = read_json(source_of_data + "/is_this_classification")
    write_json(is_this_classification, model_folder + "/is_this_classification")
    write_json(hidden_units, model_folder + '/hidden_units')
    write_json(dropout, model_folder + '/dropout')
    def calculate_mean_var(source_name, on_how_many_examples):
        dataset = io_TFRecords.parse_batch_not_ignored(source_name, on_how_many_examples, 1, False, -1)
        iterator = dataset.make_one_shot_iterator()
        tab = iterator.get_next()['n']
        mean, variance = tf.nn.moments(tab, axes=[0])
        return mean, variance
    
    mean, variance = calculate_mean_var(source_of_data, num_examples_for_statistics)
    with tf.Session() as sess:
        mean, variance = sess.run((mean, variance))
        def are_threre_zeros_in_variance(variance):
            for i in range(len(variance)):
                if (variance[i] == 0):
                    return True
            return False
        if (are_threre_zeros_in_variance(variance)):
            assert False
        np.savetxt(model_folder + '/mean', mean)
        np.savetxt(model_folder + '/var',variance)
    feature_columns = create_feature_columns(source_of_data)
    model = "" #declaration
    if (is_this_classification):
        model = tf.estimator.DNNClassifier(
        hidden_units = hidden_units,
        feature_columns = feature_columns,
        model_dir = model_folder + '/model',
        n_classes = 2,
        dropout = dropout
    )
    else:
        model = tf.estimator.DNNRegressor(
            hidden_units = hidden_units,
            feature_columns = feature_columns,
            model_dir = model_folder + '/model',
            dropout = dropout
    )

    #You have to train in order to have initial weights of model stored.
    model.train(
        input_fn = input_fn(source_of_data, model_folder,1,False, 1,False, 1, False),
        steps = 1
    )
    return model
    
#lets you read stored mean and variacce 
def read_stored_mean_var(model_folder):
    mean = np.loadtxt(model_folder + '/mean')
    var = np.loadtxt(model_folder + '/var')
    return mean, var

#Lets you read model that was already created. This function returns
# loaded model. 
def load_model(model_folder):
    assert os.path.isdir(model_folder + '/model')
    source_of_data = read_json(model_folder + '/source_of_data')
    feature_columns = create_feature_columns(source_of_data)
    hidden_units = read_json(model_folder + '/hidden_units')
    dropout = read_json(model_folder + '/dropout')
    is_this_classification = read_json(model_folder + "/is_this_classification")
    if is_this_classification:
        return tf.estimator.DNNClassifier(
        hidden_units = hidden_units,
        feature_columns = feature_columns,
        model_dir = model_folder + '/model',
        n_classes = 2,
        dropout = dropout
    )
    else:
        return tf.estimator.DNNRegressor(
        hidden_units = hidden_units,
        feature_columns = feature_columns,
        model_dir = model_folder + '/model',
        dropout = dropout
    )

#trains model. arguments are 
# - source_of_data is folder created with module io_TFRecords
# - model_folder is an folder created with functinon create_model_folder. 
# - steps is number of batches that we want to train on
# - perform_cache, batch_size, perform_shuffle, num_of_threads, perform_repeat, buffer_size like
#       in function input_fn
def train(source_of_data, model_folder, steps, perform_cache, batch_size = 128,
         perform_shuffle = True, num_of_threads = 1, perform_repeat = True, buffer_size = 1000
         ):
    model = load_model(model_folder)
    model.train(
        input_fn = input_fn(source_of_data, model_folder, batch_size,  
         perform_shuffle, num_of_threads, perform_repeat, buffer_size,
         perform_cache ),
        steps = steps
    )

#evaluates model. arguments are 
# - source_of_data is folder created with module io_TFRecords
# - model_folder is an folder created with functinon create_model_folder. 
# - steps is number of batches that we want to train on
# - perform_cache, batch_size, perform_shuffle, num_of_threads, perform_repeat, buffer_size like
#       in function input_fn
def evaluate(source_of_data, model_folder, steps, batch_size = 128,  
        num_of_threads = 1):
    model = load_model(model_folder)
    return model.evaluate(
        input_fn = input_fn(source_of_data, model_folder, batch_size,  
         perform_shuffle = False, num_of_threads = num_of_threads, perform_repeat = False, buffer_size = 100,
         perform_cache = False)
    )



    