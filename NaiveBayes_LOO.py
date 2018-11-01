#!/usr/bin/env python3



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

#%%
#==============================================================================
#     
# FUNCTION: NaiveBayes_Classifier ()
# TAKES AS ARGUMENTS
# -training_set   : a LIST of sublists; each sublists has as elements the attributes of each observations as floats
# -training_labels: a list with the labels of each observation in the training set. Order should stay true for the training set. 
# -test_vector    : a list which has as elements the attributes of the unknown-label observation to be guessed
#  
#==============================================================================
def NaiveBayes_Classifier (training_set, training_labels, test_vector_as_list):

    #==============================================================================
    #  Allocate in each of N = number_of_classes lists, the respective genes aka observations:
    #==============================================================================
    # class_categorised_LIST        , a LIST with as many sublists as the number of classes
    # class_categorised_LIST[i][0]  , the labels of each class
    # class_categorised_LIST[i][1:] , the observations that belong in each class
    
    class_categorised_LIST = []
    class_labels = tuple(set(training_labels))          # finding out how many classes there are
    
    #==============================================================================
    # CREATE LIST of sublists; sublist = a list with one element, the class label                   
    #==============================================================================
    for i in class_labels:                             # for each class, 
        class_categorised_LIST.append([i])             # add a sublist in the class_categorised_LIST with the class label
    
    #==============================================================================
    # APPEND RESPECTIVE OBSERVATION IN EACH SUBLIST WITH THE CORRESPONDING CLASS LABEL
    #==============================================================================
    for j in range(0,len(class_categorised_LIST)):                    # for each class,
        for i in range(0,len(training_set)):                          # for each training observation
            if data_class_labels[i] == class_categorised_LIST[j][0]:
                class_categorised_LIST[j].append(data_LIST[i])
    
    #==============================================================================
    # #%% Just for counting observations in each class
    # for i in range(len(class_categorised_LIST)):
    #     print(class_categorised_LIST[i][0],(len(class_categorised_LIST[i]) - 1))
    #==============================================================================
    
    
    #==============================================================================
    # For each attribute, for each class estimate the mean of attribute and the sd.
    #==============================================================================
    
    #==============================================================================
    # 
    # attributes_matrices: A LIST with as many matrices as the classes:
    # Each matrix, matrix.shape  =  (number_of_observations_in_class, number_of_attributes)
    #==============================================================================
    attributes_matrices = []
    
    #==============================================================================
    # means : a list // means[0] the label of the class // means[1:] the mean of each attribute in class // len(means[1:]) = number_of_attributes
    # stdevs: <look at means; pretty much the same > 
    #==============================================================================
    means  = []
    stdevs = []
    
    
    classes_attributes  = []
    for i in range(0,len(class_categorised_LIST)):
        
        classes_attributes.append([class_categorised_LIST[i][0]])
        classes_attributes[i].append(np.matrix(class_categorised_LIST[i][1:]))      # apo to [1:] giati to [0] einai to label
        
        attributes_matrices.append(classes_attributes[i][1])
        
        means.append([class_categorised_LIST[i][0]])
        means[i].append(np.ndarray.tolist(np.mean(attributes_matrices[i], axis=0))[0])
        
        stdevs.append([class_categorised_LIST[i][0]])
        stdevs[i].append((np.ndarray.tolist(np.std(attributes_matrices[i], axis=0))[0]))
        
    
    
    from scipy import stats
    pdfs = []
    
    
    #pdf from scipy found here:  https://oneau.wordpress.com/2011/02/28/simple-statistics-with-scipy/#probability-density-function-pdf-and-probability-mass-function-pmf
    
    # For every class:
    pdfs = []
    intersection_pdfs = []
    for i in range(0,len(class_categorised_LIST)):
        # Calculate each attribute pdf for given input test vector
        attribute_pdf = [stats.norm.pdf(test_vector_as_list, loc=means[i][1][j], scale=stdevs[i][1][j]) for j in range(0,len(means[0][1]))]
        pdfs.append(attribute_pdf)
        intersection_pdfs.append(np.prod(attribute_pdf))
        
    
    
    
    sum_lens = sum([(len(class_categorised_LIST[i])-1) for i in range(0,len(class_categorised_LIST))])
    marginal_probs_of_classes = []
    for i in range(0,len(class_categorised_LIST)):
        class_marginal_prob = (len(class_categorised_LIST[i])-1)/sum_lens
        marginal_probs_of_classes.append(class_marginal_prob) 
        #sum(marginal_probs_of_classes)  #check this, should be 1.0
                          
                              
    
    likelihood_of_belonging_to_class = [marginal_probs_of_classes[i]*intersection_pdfs[i] for i in range(0, len(intersection_pdfs))]
    
    
    
    
    label_based_on_max_ginomeno_prob = class_labels[likelihood_of_belonging_to_class.index(max(likelihood_of_belonging_to_class))]                  # the index of the max gamma is the label of the gaussian that the x_vector belongs to
    
    #print(test_labels[0])
    #print(label_based_on_max_ginomeno_prob)
#    end = timeit.default_timer()
#    runtime = end - start
    #print(runtime)
    
    
    
    return(label_based_on_max_ginomeno_prob)




#%%
def Naive_Bayes_Leave_One_Out_Accuracy_Validator(training_set, training_labels):
    successful_predictions = 0
    for i in range(0,len(training_set)):
        prediction = NaiveBayes_Classifier(training_set[:i]+training_set[i+1:], training_labels, training_set[i])
        if prediction == training_labels[i]:
            successful_predictions+=1
    accuracy = successful_predictions/len(training_labels)
    #print("Estimated accuracy - Leave one out method:", round(accuracy, 3))

    return("Estimated accuracy - Leave one out method:", round(accuracy, 3))
    

#%%
import sys

saveout = sys.stdout
save = open("NaiveBayes_LOO.txt", 'w')   #saving in a file the output of print
sys.stdout = save

print(Naive_Bayes_Leave_One_Out_Accuracy_Validator(training_set, training_labels))

sys.stdout = saveout
save.close()  





