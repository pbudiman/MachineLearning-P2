# MachineLearning-P2
Gaussian Naive Bayes
COMP30027 Machine Learning - Assignment 1, Semester 1 2021
By: Janice Theresia Sutrisno (1013239) & Patricia Angelica Budiman (1012861)

Objective: 
1. Implement a supervised Naive Bayes learner to classify pose from key- points provided by a deep convolutional neural network
2. The classifier is trained, tested, & evaluated on the provided dataset. 

Assumption:
Assuming dataset is located in : 
    - train_file_name="COMP30027_2021_assignment1_data/train.csv"
    - test_file_name="COMP30027_2021_assignment1_data/test.csv"
Assuming :
    - first column of data set is class
    - the rest of the columns are attributes value 
  
To Re-Create Results 
A. To re-create **Gaussian Naive Bayes** result:
1. Run the block under ***PREPROCESSING*** heading
    - Run the Preprocess functions and all the supporting functions
    - Read the file and store it in required structure
    - Global variables to locate test and train dataset
        - train_file_name="COMP30027_2021_assignment1_data/train.csv"
        - test_file_name="COMP30027_2021_assignment1_data/test.csv"
2. Run the block under ***TRAIN*** heading
    - Run the Train functions and all the supporting functions to find mean, standard deviation and class probabilities
3. Run the block under ***PREDICT*** heading
    - Run the Predict functions and all the supporting functions to count probability and make predictions
4. Run the ***EVALUATE*** heading
    - Run the Evaluate functions to find the accuracy of the Gaussian Naive Bayes Classifier
5. Run the block under ***RUN HERE*** heading
    - To run the Gaussian Naive Bayes Classifier with the provided train and test datasets
    - preprocess,  train, predict, & evaluate is used here

B. To re-create results for **Q 1,3,4,5**:
1. Run each block under each questions
2. If error occurs, please run Section A.

Questions Description
Q1. 
* Block 1: functions to calculate micro average & macro average and create confusion matrix
* Block 2: Run the functions and print the confusion matrix and results

Q3. 
* Block 1: functions to train, calculate probability, & predict classes using KDE and to calculate similarity between when using the Gaussian Naive Bayes
* Block 2: Run the functions and print the confusion matrix, accuracy %, similarity % with the Gaussian Naive Bayes 

Q4.
* Block 1: functions to 
  * preprocess train dataset for Q4 
  * do cross validation to choose best bandwidth
  * display graph of kde vs x1 attribute for class 'bridge' to show the difference when using best bandwidth and using arbitrary bandwidth
* Block 2: 
  * Run the functions to find the best bandwidth 
  * Run KDE Naive Bayes using the bandwidth
  * Display the graph
  * Show the accuracy using KDE with the best bandwidth & accuracy when using arbitrary bandwidth
        
Q5.
  * Block 1: functions to find mean of attribute per class & impute missing value with respective mean
  * Block 2: 
    * Run the functions and the Naive Bayes Classifier on the mean-imputed training dataset
    * Compare accuracy % when using 0-imputation and mean-imputation for Gaussian and KDE Naive Bayes Classifier

Functions Description:
1. PREPROCESSING
preprocess(train_attr_vals, train_classes, class_dict, class_count_dict):
input : 
    - train_attr_vals : array to store all instance of training data except their class
    - train_classes : array to store all instances' actual class
    - class_dict : dictionary to store instances based on class, format: {classname: [[instance]]}
    - class_count_dict : dictionary to store number of instance in each class format: {classname: num of instance}
output:
    - no return value but each parameter is filled

open_train_file(train_attr_vals, train_classes,filename):
input: 
    - train_attr_vals, train_classes : same as preprocess()'s
    - filename: take the file name from the global variable using train_file_name & test_file_name
output:
    - no return value but train_attr_vals, train_classes is filled

remove_missing_values(train_attr_vals):
input: same as preprocess()'s
output: missing values in every instance is imputed to 0

str_to_float(attributes):
input: attributes - array containing an instance's attributes' values i.e [54.8598,33.0166,....] = [x1,x2,...]
output: no return value but string changed to float

put_into_dict(train_attr_vals,train_classes,class_dict,class_count_dict)
input: train_attr_vals,train_classes,class_dict,class_count_dict same as preprocess()'s
output: no return value but dictionary filled 

2. Train
mean(array):
input: array of an attribute's values from all instances of a class i.e class is 'bridge' & attribute is x1, array is all values of x1 whose class is bridge
output: mean of input array

standard_deviation(array,mean_val):
input: 
    - array is same as mean()'s
    - mean of that array
output: standard deviation of that array

train(class_dict, class_count_dict,training_details,attr_length,total_inst):
input:
    - class_dict : dictionary to store instances based on class, format: {classname: [[instance]]}
    - class_count_dict : dictionary to store number of instance in each class format: {classname: num of instance}
    - training_details : dictionary to store details after being trained, format: {classname:[[mean of every feature/attribute],[std deviation of every attributes], class_probability]}
    - attr_length : number of attributes/feature
    - total_inst : total number of training instance
output: no return value but training_details dictionary is filled

3. Predict:
gaussian_distribution(val,mean,std_dev):
input: 
    - val : x in 𝜙σ(𝑥−𝜇)
    - mean : 𝜇 in 𝜙σ(𝜇)
    - std dev : std deviation of the attribute
output: gaussian distribution of the value

take_log(val):
    - val to take log of
    - log of the val

probability(instance,training_details,attr_num,class_num):
input:
    - instance whose probability being classified as each class that we want to find
    - training_details : dictionary to store details after being trained, format: {classname:[[mean of every feature/attribute],[std deviation of every attributes], class_probability]}
    - attr_num : number of attributes/features
    - class num : number of the classes
output: dictionary of probabilities of the instance being classified to each classes 

predict(training_details,attr_num,class_num,test_attr_vals,test_actual_class):
input:
    - training_details : dictionary to store details after being trained, format: {classname:[[mean of every feature/attribute],[std deviation of every attributes], class_probability]}
    - attr_num : number of attributes
    - class_num : number of classes
    - test_attr_vals: array to hold all test instances' attributes values ,i.e [[instance]]
    - test_actual_class: array to hold all test instances' actual classes
output: array to hold all predicted classes of each test instance

4. Evaluate
evaluate(test_actual_class,predicted_classes): 
input:   
    -  test_actual_class : array of test instances' actual class
    -  predicted_classes : array of test instances' predicted class


Q1:
confusion_matrix(predicted_classes, test_actual_class, class_dict):
input: 
    - predicted_classes : array of predicted classes of test instances
    - test_actual_class : : array of actual classes of test instances
    - class_dict : dictionary to store instances based on class, format: {classname: [[instance]]}
output: return dataframe of confusion matrix

macro_averaging(predicted_classes, test_actual_class, class_dict):
input: same as confusion_matrix()'s
output: 
    - macro_precision: precision of the test set using macro averaging
    - macro_recall: recall of the test set using macro averaging
    - fscore_macro: fscore of the test set using macro averaging

micro_averaging(predicted_classes, test_actual_class, class_dict):
input: same as confusion_matrix()'s
output: 
    - micro_precision: precision of the test set using micro averaging
    - micro_recall: recall of the test set using micro averaging
    - fscore_micro: fscore of the test set using micro averaging

Q3:
KDE(kernel_bandwidth, test_instance, class_dict):
input: 
    - kernel_bandwidth
    - test_instance : current instance to find the probabilities classified to each class
    - class_dict: dictionary to store instances based on class, format: {classname: [[instance]]} from the Gaussian Naive Bayes
output: dictionary of probabilities of the instance being classified to each classes 

predict_kde(bandwidth,test_attr_vals_q3,class_dict):
input: 
    - bandwidth: kernel bandwidth
    - test_attr_vals_q3: array of test instances
    - class_dict: dictionary to store instances based on class, format: {classname: [[instance]]} from the Gaussian Naive Bayes
output: predicted_classes - array of test instances' predicted classes

kde_vs_gaussian(predicted_kde, predicted_gaussian):
input:
    - predicted_kde: - array of test instances' predicted classes using kde naive bayer classifier
    - predicted_gaussian: - array of test instances' predicted classes using gaussian naive bayer classifier

Q4:
preprocess_q4(train_attr_vals_q4, train_classes_q4):
input:
    - train_attr_vals_q4: array storing all training instances' attributes' values
    - train_classes_q4: array of storing all training instances' classes
output: train_attr_vals_q4, train_classes_q4 is filled

cross_val_bandwidth(class_dict_q4,class_count_dict_q4,train_attr_vals_q4,train_classes_q4)
input:
    - train_attr_vals_q4,train_classes_q4 same as preprocess_q4()'
    - class_dict_q4: dictionary storing train instances based on class 
    - class_count_dict_q4: dictionary storing number train instances based on class 
output: best kernel bandwidth

display_kde_graph(best_bandwidth, arbitrary_bandwidth):
input:
    - best_bandwidth: best bandwidth found
    - arbitrary_bandwidth: bandwidth used in Q3
output: no return value but plot is displayed

Q5:
mean_of_class(class_dict, mean_dict):
input:
    - class_dict: dictionary storing train instances based on class 
    - mean_dict: dictionary storing of attributes based on class
output: mean_dict

impute_mean(mean_dict, new_class_dict):
input:
    - new_class_dict: dictionary storing train instances based on class 
    - mean_dict: dictionary storing of attributes based on class
output: new_class_dict filled with imputed training instances

