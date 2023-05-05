Understanding the Notebooks
===========================

1. First of all the codes have been merged with data preprocessing team so first copy and run the `notebook <https://github.com/DSCI-Admissions-project/data-processing/blob/main/Data%20Preprocessing%20Updated.ipynb notebook to get the merged data.(It includes the functionalities from Data_Preprocessing_Analytics.ipynb and Feature Engineering.ipynb so you don’t need to run them separately>`_
2. Run the Feature Selection.ipynb for the feature Selection process.

**Power BI dashboard:** `Dashboard <https://app.powerbi.com/groups/me/reports/80cd2e24-4199-4c97-965b-87dedfadbf97/ReportSection?ctid=3c71cbab-b5ed-4f3b-ac0d-95509d6c0e93>`_

Analysis.ipynb
**************

Preliminary Data **Analysis.ipynb** file has the initial Data analysis where we have used Statistical Techniques to:

• Measure central tendency (mean, median, mode) and dispersion (range, standard deviation, variance) for different columns
• Frequency count for some categorical column

Feature Engineering.ipynb
*************************

In the Feature **Engineering.ipynb** we have engineered the following features:

* Max Job Number: The total number of jobs done by the applicant
* Number of Degrees – the total number of degrees a person has
* Has Masters or Doctoral Degree – if the person has masters or doctoral degree (1= True, 0= False)
* latest_school_duration_in_year – The duration of their latest school
* days_between_application_submitted_and_decision- days between application submitted and decision made (We can visualize this to see the timeframe of application since several years)
* Is_Fresher : If the student has recently graduated or not
* Total_experience_years : For some 58 rows, we are getting negative experiences, as looks like the start date is greater than the end date for them. Data processing team has been instructed to see if it is possible and reasonable to swap the start and end date while calculating months_of_experience(in months) . For now, carry on with what we have.
* School_Gap_in_Years : The academic gap of student in years
* Applying_while_in_school : See if the student is applying while in school ( 1=True, 0=False)
* Gap_from_last_job : The number of years that represents the gap from last year. If Gap_from_last_job=0, then the applicants are employed during application.
* Initial_Enrollment_from_Student : Since we don’t have enrolled column, this column is made from Applicantion I-20 sent date, such that it has a value of 1 if Application I-20 Sent Date is not null, -1 if the Admission Decision is "Denied" I.e. Not applicable, and 0 otherwise.
* Standard test given: Whether GRE is given (1) or not(0)

All the codes from feature Engineering have been added to the notebook from Data Preprocessing `LINK <https://github.com/DSCI-Admissions-project/data-processing/blob/main/Data%20Preprocessing%20Updated.ipynb>`_ for the one step processing of data to be ingested my Machine Learning Team. The documentation for the generation of these features is `here <https://github.com/DSCI-Admissions-project/data-processing/blob/main/Documentation.pdf>`_

Feature Selection.ipynb
***********************

Let’s see the codes for the Feature Selection :

3. pd.read_csv('/Volumes/Admis-Shared/Visualization/Data_analytics_merged_data.csv'):

    This line of code reads the CSV file located at the path /Volumes/Admis-Shared/Visualization/Data_analytics_merged_data.csv using pandas and assigns it to a pandas DataFrame named merged_data_with_new_features.

4. merged_data_with_new_features.drop(merged_data_with_new_features.columns[merged_data_with_new_features.columns.str.contains('Unnamed',case=False)], axis=1, inplace=True):

    This line of code drops any columns from merged_data_with_new_features that contain the string 'Unnamed' in their column
    name. The axis=1 argument specifies that columns should be dropped, and the inplace=True argument specifies that the DataFrame should be modified in place.

5. Next, we drop multiple columns from df_final DataFrame, which will not be used in the model.

6. We divided the data by Application Area of Study to see the importance feature for different program which are named as df_DS fro Data Science and df_EM for Engineering Management.

7. corr_matrix_DS = df_DS.corr():

    This line of code calculates the correlation matrix for the df_DS DataFrame using the pandas corr() method and assigns it to the corr_matrix_DS variable and then we display it.

8. We do the similar for Engineering Management.

9. Next, we have applied many different techniques such as :

**Recursive Feature Elimination.**
----------------------------------

.. code-block:: python

    import pandas as pd
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    # replace null values with 0
    df_final.fillna(0, inplace=True)

    X = df_DS.drop(columns=['Decision Name'])
    y = df_DS['Decision Name']
    # create the RFE object with a logistic regression estimator and select the top 10 features
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=15)
    rfe.fit(X, y)

    # print the top 10 selected features
    print("For Data Science: ")
    print('Top 10 Features:', X.columns[rfe.support_])

The code provided is a Python script that performs feature selection on a dataset using the scikit-learn library. The script first imports the pandas library as pd and two classes from scikit-learn, RFE and LogisticRegression. It then replaces any null values in a dataframe, df_final, with 0 using the fillna method. The script creates two variables, X and y, by assigning the 'Decision Name' column of the dataframe to y and dropping the 'Decision Name' column from the dataframe and assigning the remaining columns to X.

Next, the script creates an instance of the RFE class, specifying a LogisticRegression estimator and selecting the top 15 features to retain. The fit method is then called on the RFE object, passing in X and y as arguments, to perform the feature selection. Finally, the script prints the top 10 selected features, as determined by the support attribute of the RFE object, to the
console. The output is labeled "For Data Science" and provides useful information for further analysis of the dataset.

**We did the same for the Engineering management data (df_EM) :**

.. code-block:: python

    import pandas as pd
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    # replace null values with 0
    df_final.fillna(0, inplace=True)
    X = df_EM.drop(columns=['Decision Name'])
    y = df_EM['Decision Name']

    # create the RFE object with a logistic regression estimator and select the top 10 features
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=15)
    rfe.fit(X, y)

    # print the top 10 selected features
    print("For Engineering Management: ")
    print('Top 10 Features:', X.columns[rfe.support_])

**Next, to test these selected features , we make a model with Logistic Regression to see the accuracy**

.. code-block:: python

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    # print the top 10 selected features print('-----------------')
    print('For Data Science')
    print("Selected Features Using RFE:")

    for i in range(len(X.columns)):
        if rfe.support_[i]:
            print(X.columns[i])
    X = df_DS.drop(columns=['Decision Name'])
    y = df_DS['Decision Name']

    # create a new feature matrix with the selected features
    X_new = X[X.columns[rfe.support_]]
    # split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

    # train a logistic regression model on the selected features
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = clf.predict(X_test)
    # evaluate the performance of the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

This code performs a classification task on a dataset called 'df_DS'. First, the top 15 features are selected using recursive feature elimination (RFE) with a logistic regression estimator. Then, the dataset is split into training and test sets using the train_test_split function from sklearn. The logistic regression model is trained on the selected features using the training set, and then used to predict the class labels of the test set. The performance of the model is evaluated using the classification_report function from sklearn.metrics, which outputs precision, recall, F1-score, and support for each class label. The results of the evaluation are printed to the console. Overall, this code demonstrates how to perform a classification task using RFE for feature selection and logistic regression for modeling, and how to evaluate the performance of the model using the classification report.

**Chi_Square Test :**
---------------------

The code snippet performs a chi-square test to select the top 15 features for a logistic regression model. The code imports the chi2 function from the feature_selection module of the sklearn library. The dataset used for analysis is for data science. The dataset is loaded into a variable named 'df_DS'. The dataset is preprocessed to remove the 'Decision Name' column from the feature matrix X, which is the predictor variables matrix. The target variable 'Decision Name' is assigned to the variable y.

The chi-square scores and p-values are calculated using the chi2 function with the feature matrix X and target variable y. The chi-square scores are then sorted in descending order, and the top 15 features are selected for the logistic regression model.

A new feature matrix is created using the selected top 15 features, and the data is split into training and test sets. A logistic regression model is then trained on the selected features, and predictions are made on the test set. Finally, the performance of the model is evaluated using the classification_report function from the metrics module of the sklearn library, which prints a report containing the precision, recall, f1-score, and support for each class.

.. code-block:: python

    from sklearn.feature_selection import chi2
    # Calculate chi-square statistics and p-values print ('-----------')
    print('Data Science')

    # Fit the Random Forest model
    X = df_DS.drop(columns=['Decision Name'])
    y = df_DS['Decision Name']

    X = np.abs(X)
    chi_scores, p_values = chi2(X, y)

    # Create a dataframe to store feature names and chi-square scores
    scores_df_chi_test = pd.DataFrame({'Feature': X.columns, 'Chi-Square Score': chi_scores})

    # Sort the dataframe by chi-square score in descending order and print the top 10 features
    top_10_features = scores_df_chi_test.sort_values('Chi-Square Score', ascending=False).head(15)
    print(top_10_features)

    # Select the top features
    top_features = scores_df_chi_test['Feature'][:15].tolist()

    # print the top 15 selected features
    print("Selected Top 15 Features using Chi_square:")
    for i in range(len(top_features)):
        print((i+1), top_features[i])

    # Create a new feature matrix with the selected features
    X_new = X[top_features]
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

    # Train a logistic regression model on the selected features
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the performance of the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


**Random Forest Feature Importance:**
-------------------------------------

.. code-block:: python

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    # Fit the Random Forest model
    X = df_EM.drop(columns=['Decision Name'])
    y = df_EM['Decision Name']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importance scores
    importance_scores = rf.feature_importances_

    # Create a DataFrame to store the feature importance scores
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})
    # Sort the features by their importance scores in descending order
    importance_df = importance_df.sort_values('Importance', ascending=False)
    # Select the top features
    top_features = importance_df['Feature'][:15].tolist() print('-----------------')
    print('For Engineering Management ')
    # print the top 15 selected features
    print("Selected Top 15 Features using Random Forest Feature Importance:")
    print(importance_df[:15])
    # Create a new feature matrix with the selected features
    X_new = X[top_features]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    # Train a logistic regression model on the selected features
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    # Evaluate the performance of the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

This code is performing feature selection for the Engineering Management dataset (we also did for Data Science dataset) using the Random Forest feature importance method.
First, the code defines the predictor variables (X) and the response variable (y).
Then, a Random Forest Regressor model is fit on the data with 100 trees and a random seed of 42.
The feature importance scores are then calculated using the **feature_importances_** attribute of the model.
These scores are stored in a DataFrame with the corresponding feature names.
The DataFrame is sorted by importance scores in descending order, and the top 15 features are selected.
The selected features are printed, and a new feature matrix is created with the top 15 features.
The data is split into training and testing sets using a 80/20 ratio, and a logistic regression model is trained on the selected features.
The model is then used to make predictions on the test set, and the performance of the model is evaluated using the **classification_report** function from the **sklearn.metrics** module.

Here, among all, Random Forest Feature Importance has the best accuracy so we will be using this method for the feature selection .

Therefore, we again dropped some more columns from df_DS and df_EM which are not that important and have redundancy in information to generate our final feature list.

.. code-block:: python

    df_DS= df_DS.drop(columns=['TOEFL Reading',
    'TOEFL Writing' , 'TOEFL Listening',
    'TOEFL Speaking' , 'IELTS Reading' ,
    'IELTS Writing' , 'IELTS Listening',
    'IELTS Speaking', 'TOEFL Reading',
    'TOEFL Writing', 'TOEFL Listening',
    'TOEFL Speaking', 'School GPA_1',
    'School GPA_2', 'School GPA_3',
    'School GPA_4', 'Job Organization_2',
    'Job Organization_3', 'Job Organization_4',
    'Job Organization_5', 'Job Organization_6',
    'Job Organization_7', 'Job Organization_8',
    'Job Organization_9', 'Duolingo Literacy',
    'Duolingo Comprehension', 'Duolingo Conversation',
    'Written_TOEFL Listening Comprehension', 'Written_TOEFL Reading Comprehension',
    'Written_TOEFL Structure/Written Expression', 'Written_TOEFL Test of Written English'], axis=1)

And applied Random Forest Feature Importance to generate the top 10 features and the accuracy metrics.

Conclusion
**********

So, based on the analysis, the top 10 Features using Random Forest Feature Importance were:

**Feature Importance**
1. School GPA (Recalculated)_1 --> 0.182289

2. Application Area of Study --> 0.136539

3. School Backlogs_1 --> 0.102131

4. Job Organization_1 --> 0.061276

5. Total_experience_years --> 0.059822

6. Application Term of Entry --> 0.059276

7. School Major_1 --> 0.058107

8. latest_school_duration_in_year --> 0.056608

9. Converted English Prof Score --> 0.051250

10. School_Gap_in_Years --> 0.050931

11. GRE Quantitative --> 0.025911

12. GRE Verbal --> 0.021301

13. Gap_from_last_job --> 0.020369

**Understanding the selected features:**

**School GPA (Recalculated)_1 :** Contains the recalculated GPA for the latest degree ( can be masters or undergrad)

**Application Area of Study :** The area of study

**School Backlogs_1 :** Contains the backlog for the latest school of the students

**Application Term of Entry:** The term of entry for admission ( Eg: Fall 2020, Spring 2021)

**latest_school_duration_in_year :** It contains the total years of school attended

**Job Organization_1 :** The latest employer of the students

**Total_experience_years :** The total years of experience

**School_Gap_in_Years :** The total gap between their school finished date and the application submitted date

**Converted English Prof Score :** It is the converted english proficiency test(every test converted to IELTS)

**GRE Quantitative :** Quantitative Score of GRE

**GRE Verbal :** Verbal Score of GRE

**Gap_from_last_job:** The total gap between the last job and application submitted date.

We obtained these features by employing the Random Forest Feature Importance method, which observes the correlation from the data. While developing the model, we suggest considering other features as well, as we may be excluding relevant information.

We also suggest considering the following additional features: Instead of using GRE Quantitative and GRE Verbal, we can try using Standardized test given.

#. Number of Degrees : The total number of degrees a student has.

#. Has Masters or Doctoral Degree : Whether the student has masters or Doctoral degree

#. Standardized test given : 1 for Standard Test Given and 0 for Not Given