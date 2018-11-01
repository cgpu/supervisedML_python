#!/usr/bin/env python3

#==============================================================================
#  KNN Classifier
#==============================================================================


import random
import numpy as np
import timeit
#==============================================================================
# DOWNLOAD DATASET AND SHUFFLE:
#==============================================================================
#%% DWNLOAD  
import os

# DOWNLOAD yeast data:
dwnload_yeast = "wget http://mlearn.ics.uci.edu/databases/yeast/yeast.data"
os.system(dwnload_yeast)


with open("yeast.data") as f:

    LIST_yeast = [line.split() for line in f]
#    from  random import shuffle
    
#==============================================================================
# # SHUFFLE THE OBSERVATIONS:
#==============================================================================
    random.shuffle(LIST_yeast)

    cols       = [list(x) for x in zip(*LIST_yeast)]
    
#%%

# LEAVING OUT FEATURES cols[5] ---0.50 and cols[6]----0.00 , 
#                              variance is pretty much non-existant, 
#                              so it probably won't add to the classification all that much


#==============================================================================
# Selecting features, leaving out cols[5], cols[6] with the slicing below:
#==============================================================================
matrix_strings = np.matrix(cols[1:5]+cols[7:-1])             # only values, but still as strings not as floats.
data_floats = matrix_strings.astype(float)                   # converting the LIST of lists to matrix, to use astype and conveniently convert strings to floats


#==============================================================================
# Storing the observations in a LIST of sublists, for indexing convenience later
# MATRIX ----> LIST
# vector ----> sublist
#==============================================================================

#==============================================================================
# data_LIST: A LIST of sublists; sublist == row of the initial data file
#==============================================================================
data_LIST = [np.ndarray.tolist(i)[0] for i in data_floats.T]    #converting each observation from array to a simple list
data_class_labels = cols[-1]                                    #keeping the labels of the dataset; index here corresponds to elements in data_LIST

#%%
#==============================================================================
# TRAINING/TEST SPLIT: 90% train + 10% test
# training_sample_size = N, a number, how many training observations
#==============================================================================
training_split_percentage = 0.9
training_sample_size = int(training_split_percentage * len(data_LIST))

#%%
#==============================================================================
# SLICING TO SPLIT TRAIN AND TEST, ALONG WITH RESPECTIVE LABELS:
#==============================================================================

training_labels = cols[-1][:training_sample_size]
training_set    = data_LIST[:training_sample_size]
#training_observation_names = cols[0][:training_sample_size]

test_labels     = cols[-1][training_sample_size:] 
test_set        = data_LIST[training_sample_size:]
#test_observation_names = cols[0][training_sample_size:]
#%%
#==============================================================================
# 
# #==============================================================================
# # KNN CLASSIFIER:
# #==============================================================================
#     
# #==============================================================================
# # TAKES AS ARGUMENTS:
# #==============================================================================
#     -k                   : number of nearest neighbours for the votes
#     -training_set        : a LIST of sublists; each observation as a sublist of len = number_of_attributes
#     -training_labels     : a list with the corresponding class label of each training_set's sublist, aka class label of each observation
#     -test_vector_as_list : one observation, as a list with len = number_of_attributes; the observation for which the class will be predicted
# #==============================================================================
# # RETURNS:
# #==============================================================================
#     The prediction for the class label of the test_vector_as_list
# 
#==============================================================================

def KNN_Classifier (k_neighbours, training_set, training_labels, test_vector_as_list):
    
    
    #distances[]: will be populated with the euclidean distances, between the test_vector_as_list and each observation of the training set
    distances  =  []                  
    for j in training_set:
        distances.append(np.linalg.norm(np.matrix(j) - np.matrix(test_vector_as_list)))
    
    #Zipping each calculated distance with the respective class label; we need these to count the neighbour votes
    #Also, zipping with distances in index zero, to have sorted straight up instead of using lambdas

    distances_label_pairs = list(zip(distances, training_labels)) 
    
    ## Sorting by x[index] with lambda if needed: (uncomment below in case of emergency)
    #  sorted_labels_lambda = list(sorted(distances_label_pairs, key=lambda x: x[0]))
    
    
    #sorted() sorts by element in index zero [0]
    sorted_labels_by_distance = list(sorted(distances_label_pairs))
    
    # Extracting the labels from the distances_label_pairs in the sorted_labels_by_distance list:
    # Labels are sorted in ascending order, from previous step.
    # Thus the k first elements correspond to the labels of the k nearest neighbours.
    votes_of_k_neighbours  = [sorted_labels_by_distance[i][1] for i in range(0,k_neighbours)]
    
    #Counting how many times a class labels appears in the k-neighbours:
    votes            = list(zip([votes_of_k_neighbours.count(i) for i in set(votes_of_k_neighbours)], [i for i in set(votes_of_k_neighbours)]))
    
    # Sort class labels by vote count:
    # The prediction of the class label for the test_vector_as_list will be the last element of the 'votes_sorted' list
    votes_sorted     = list(sorted(votes))
    
    # The last one since they are sorted in ascending order; 
    # thus the last one is the most-times-counted label.
    final_label = votes_sorted[-1][1]
    return(final_label)
    
#%%
import sys

saveout = sys.stdout
save = open("KNN_TEST_accuracy_k_from_10_to_50_step_3.txt", 'w')   #saving in a file the output of print
sys.stdout = save

accuracy_summary = []
for k_neighbours in range(10,50 , 3):
    successful_predictions = 0
    for i in range(0,len(test_set)):
        prediction = KNN_Classifier (k_neighbours, training_set, training_labels, test_set[i])
        if prediction == test_labels[i]:
            successful_predictions+=1
        accuracy = successful_predictions/len(test_labels)
        accuracy_summary.append(accuracy)
    print("KNN - TEST, k_neighbours = "+str(k_neighbours)+", accuracy:", round(accuracy, 3))


sys.stdout = saveout
save.close()



