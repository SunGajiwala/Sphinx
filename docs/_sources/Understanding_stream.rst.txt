Understanding the Streamlit dashboard
=====================================

**The Welcome Page**

.. image:: Images_Used/WelcomePage.png
  :width: 1000
  :alt: Sample image of the Welcome page

**To Get Started, please select an option from the sidebar.**

#. Predict Student Admissions: *To predict whether a student should be admitted or not and a probability of admit based on features.*

#. Analytical Dashboard: *Explore various analytical metrics about students that were admitted and rejected over the past years.*

#. Model Explanation: *Access model documentation and details about the features used, their importance in making the decision and accuracy of the model.*

#. About the team: *Learn about the team members who contributed to this project.*

Predict Students Admissions
****************************

.. image:: Images_Used/PredictionUpload.png
  :width: 1000
  :alt: Sample Image of the Prediction Using Upload Option

**Choose graduate option** allows user to choose from two different models based on the type of course:

#. Data Science
#. Engineering and Operations Management

**Please choose a csv or xls file** allows user to upload either a csv or an xls file for prediction.

.. warning::
    The size of the file needs to be less than or equal to 200mb

.. image:: Images_Used/PredictionManual.png
  :width: 1000
  :alt: Sample Image of the Prediction Using Manual Input Option

The **Manual Input** option of the left pane allows user to manually input values of an individual sample for predictions.

Click the Predict button to predict the results

.. note::
    The results are displayed on the dashboard with Three additional features in the same uploaded file.

    *Prediction* is a binary feature which displays Admit or Reject

    *Admit_Rank* is rank of all the samples sorted according to probability. Highest probability as  Rank 1 and lowest as last.

    *Acceptance_Probability* displays probability of each sample. *Admit_Rank is based on this feature*

.. note::
    **What happens when you click the predict button?**

    When you click the *predict* button, the uploaded csv or the xls file is passed down to an already pretrained model which predicts the
    results of all the samples and return a csv file with results. The steps to retrain the model are included here
    :ref:`Retraining the model`

Analytical Dashboard
*********************

.. image:: Images_Used/AnalyticalDashboard.png
  :width: 1000
  :alt: Sample image of the Analytical Dashboard Page

**The View Analytical Dashboard** button takes user to a power BI dashboard for data analysis and exploration

Model Explanation
******************

.. image:: Images_Used/ModelExplainer.png
  :width: 1000
  :alt: Sample image of the The Model Explainer Page

* This is where the explainability part of the model comes into play.
* The dashboard has 5 different tabs:
    #. Feature Importance: To understand more about the features that played important role in prediction.
    #. Classification Stats: To understand statics related to model performance which includes confusion matrix, ROC plot, PR plot, Lift curve etc.
    #. Individual Predictions: To understand performance related to an individual sample.
    #. What If: Includes feature twitching to see *what if* the feature used had a different value.
    #. Feature Dependence: Displays SHAP plots for the features used.