File Location Assumption:
 - train_file = "./COMP30027_2021_Project2_datasets/recipe_train.csv"
 - test_file = "./COMP30027_2021_Project2_datasets/recipe_test.csv"
 - count_vec_folder = './COMP30027_2021_Project2_datasets/recipe_text_features_countvec/' - consists:
    - train_name_countvectorizer.pkl
    - train_steps_countvectorizer.pkl
    - train_ingr_countvectorizer.pkl
    - train_name_vec.npz
    - test_name_vec.npz
    - train_steps_vec.npz
    - test_steps_vec.npz
    - train_ingr_vec.npz
    - test_ingr_vec.npz

File Generated: All files is in the same folder as all the notebooks
    - Count_Vectoriser_Real_DataSet.ipynb:
        - All Features (all matrices & n_steps & n_ingredients):
            - df_CV_res_MNB_full.csv (using Multinomial Naive Bayes for prediction)
            - df_CV_res_Decision Tree_full.csv (using Decision Tree for prediction)
            - df_CV_res_Logistic Regression_full.csv (using Logistic Regression for prediction)
            - df_CV_stack_sklearn_Log_Reg.csv (Stacking, Meta: Logistic Regression, Base: Multinomial Naive Bayes + Decision Tree)
        - Chi-Square, k=1000:
            - df_CV_res_chi2_MNB_full.csv (using Multinomial Naive Bayes for prediction)
            - df_CV_res_chi2_Decision Tree_full.csv (using Decision Tree for prediction)
            - df_CV_res_chi2_Logistic Regression_full.csv (using Logistic Regression for prediction)
            - df_CV_chi2_stack_sklearn_Log_Reg.csv (Stacking, Meta: Logistic Regression, Base: Multinomial Naive Bayes + Decision Tree)
    - TFIDF_Real_DataSet.ipynb :
        - All Features (all matrices & n_steps & n_ingredients):
            - df_TFIDF_res_MNB_full.csv (using Multinomial Naive Bayes for prediction)
            - df_TFIDF_res_Decision Tree_full.csv (using Decision Tree for prediction)
            - df_TFIDF_res_Logistic Regression_full.csv (using Logistic Regression for prediction)
            - df_TFIDF_stack_sklearn_Log_Reg.csv (Stacking, Meta: Logistic Regression, Base: Multinomial Naive Bayes + Decision Tree)
        - Chi-Square, k=1000:
            - df_TFIDF_res_chi2_MNB_full.csv (using Multinomial Naive Bayes for prediction)
            - df_TFIDF_res_chi2_Decision Tree_full.csv (using Decision Tree for prediction)
            - df_TFIDF_res_chi2_Logistic Regression_full.csv (using Logistic Regression for prediction)
            - df_TFIDF_chi2_stack_sklearn_Log_Reg.csv (Stacking, Meta: Logistic Regression, Base: Multinomial Naive Bayes + Decision Tree)


(1) Count_Vectoriser_Real_DataSet.ipynb
        Uses Count Vectoriser for 'name', 'steps' & 'ingredients' text features in recipe_train.csv & recipe_test.csv

To Run (1):
1. Run cell under 'Raw Train & Test DataSet'
2. Run cell under 'Count Vectoriser for text features'
3. Run cell under 'Individual Classifiers - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from  'name', 'steps' & 'ingredients' - Count Vectorizer
    - Multinomial Naive Bayes, Decision Tree, Logistic Regression Used
    - Resulting prediction csv file :
        - df_CV_res_MNB_full.csv (using Multinomial Naive Bayes for prediction)
        - df_CV_res_Decision Tree_full.csv (using Decision Tree for prediction)
        - df_CV_res_Logistic Regression_full.csv (using Logistic Regression for prediction)
4. Run cell under 'Stacking - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from 'name', 'steps' & 'ingredients'
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Resulting prediction csv file : 
        - df_CV_stack_sklearn_Log_Reg.csv 
5. Run cell under 'CHI SQUARE , K=1000' : for feature selection 
6. Run cell under 'Individual Classifiers - CHI SQUARE, K=1000'
    - Using 1000 best features selected using chi square
    - Resulting prediction csv file :
        - df_CV_res_chi2_MNB_full.csv (using Multinomial Naive Bayes for prediction)
        - df_CV_res_chi2_Decision Tree_full.csv (using Decision Tree for prediction)
        - df_CV_res_chi2_Logistic Regression_full.csv (using Logistic Regression for prediction)
7. Run cell under 'Stacking - CHI SQUARE , K =1000'
    - Using 1000 best features selected using chi square
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Resulting prediction csv file :
        - df_CV_chi2_stack_sklearn_Log_Reg.csv

(2) TFIDF_Real_DataSet.ipynb
        Uses TF-IDF Vectoriser for 'name', 'steps' & 'ingredients' text features in recipe_train.csv & recipe_test.csv

To Run (2):
1. Run cell under 'TF-IDF on Text Features'
2. Run cell under 'Individual Classifiers - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from 'name', 'steps' & 'ingredients' - TFIDF vectorizer
    - Multinomial Naive Bayes, Decision Tree, Logistic Regression Used
    - Resulting prediction csv file :
        - df_TFIDF_res_MNB_full.csv (using Multinomial Naive Bayes for prediction)
        - df_TFIDF_res_Decision Tree_full.csv (using Decision Tree for prediction)
        - df_TFIDF_res_Logistic Regression_full.csv (using Logistic Regression for prediction)
3. Run cell under 'Stacking - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from 'name', 'steps' & 'ingredients'
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Resulting prediction csv file :
        - df_TFIDF_stack_sklearn_Log_Reg.csv 
4. Run cell under 'CHI SQUARE , K=1000' : for feature selection
5. Run cell under 'Individual Classifiers - CHI SQUARE, K=1000'
    - Using 1000 best features selected using chi square
    - Resulting prediction csv file :
        - df_TFIDF_res_chi2_MNB_full.csv (using Multinomial Naive Bayes for prediction)
        - df_TFIDF_res_chi2_Decision Tree_full.csv (using Decision Tree for prediction)
        - df_TFIDF_res_chi2_Logistic Regression_full.csv (using Logistic Regression for prediction)
6. Run cell under 'Stacking - CHI SQUARE , K =1000'
    - Using 1000 best features selected using chi square
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Resulting prediction csv file :
        - df_TFIDF_chi2_stack_sklearn_Log_Reg.csv

(3) Count_Vectoriser_TrainTestSplit.ipynb:
        Split recipe_train.csv to train & test , where test = 0.33 - for testing & initial accuracy purpose
        Uses Count Vectoriser for 'name', 'steps' & 'ingredients' text features

To Run (3):
1. Run cell under 'Train Test Split on recipe_train.csv'
2. Run cell under 'Individual Classifiers - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from 'name', 'steps' & 'ingredients'
    - Multinomial Naive Bayes, Decision Tree, Logistic Regression Used
    - Result: Accuracy & Time run using each classifier
3. Run cell under 'Stacking - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from 'name', 'steps' & 'ingredients'
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Result: Accuracy for Stacking
4. Run cell under 'CHI SQUARE , K=1000' : for feature selection 
5. Run cell under 'Individual Classifiers - CHI SQUARE, K=1000'
    - Using 1000 best features selected using chi square
    - Multinomial Naive Bayes, Decision Tree, Logistic Regression Used
    - Result: Accuracy & Time run using each classifier on 1000 features from feature selection
6. Run cell under 'Stacking - CHI SQUARE , K =1000'
    - Using 1000 best features selected using chi square
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Result: Accuracy for Stacking

(4) TFIDF_TrainTestSplit.ipynb:
        Split recipe_train.csv to train & test , where test = 0.33 - for testing & initial accuracy purpose
        Uses TF IDF Vectoriser for 'name', 'steps' & 'ingredients' text features

To Run (4):
1. Run cell under 'Train Test Split on recipe_train.csv'
2. Run cell under 'Individual Classifiers - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from 'name', 'steps' & 'ingredients'
    - Multinomial Naive Bayes, Decision Tree, Logistic Regression Used
    - Result: Accuracy & Time run using each classifier
3. Run cell under 'Stacking - All features'
    - Using all features including n_steps, n_ingredients, & all matrices generated from 'name', 'steps' & 'ingredients'
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Result: Accuracy for Stacking
4. Run cell under 'CHI SQUARE , K=1000' : for feature selection 
5. Run cell under 'Individual Classifiers - CHI SQUARE, K=1000'
    - Using 1000 best features selected using chi square
    - Multinomial Naive Bayes, Decision Tree, Logistic Regression Used
    - Result: Accuracy & Time run using each classifier on 1000 features from feature selection
6. Run cell under 'Confusion Matrix for TFIDF - CHI SQUARE K=1000'
    - Display Confusion Matrix for Classification using Decision Tree, on 1000 features from chi square feature selection
7. Run cell under 'Stacking - CHI SQUARE , K = 1000'
    - Using 1000 best features selected using chi square
    - Base Learners: Decision Tree & Multinomial Naive Bayes
    - Meta Learner : Logistic Regresssion
    - Result: Accuracy for Stacking
        