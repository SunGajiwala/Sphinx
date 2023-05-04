Understanding the Classifier used
=================================

HistGradientBoostingClassifier()
---------------------------------

* Importing the classifier

    **sklearn.ensemble.HistGradientBoostingClassifier¶**

* Histogram-based Gradient Boosting Classification Tree.

    #. This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).

    #. This estimator has native support for missing values (NaNs). During training, the tree grower learns at each split point whether samples with missing values should go to the left or right child, based on the potential gain. When predicting, samples with missing values are assigned to the left or right child consequently. If no missing values were encountered for a given feature during training, then samples with missing values are mapped to whichever child has the most samples.

    #. This implementation is inspired by LightGBM.

**Official documentation can be found here** `HistGradientBoostingClassifier() <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`_


