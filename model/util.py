import yaml, pickle, os
import numpy as np
import pandas as pd
import warnings
import plotly.express as px

warnings.filterwarnings("ignore")
import json
from datetime import datetime

# Libraries for model creation....
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


class ADMSMClassifier:
    def __init__(
        self,
        config=None,
        training_file_path="data/unh/merged_data_wnf.csv",
        study_area="ms_data_science",
        model_save_filename="finalized_model",
        datalore_repo = None
    ):
        """Initialize class. This is the constructor for the class. If config is None the config will be loaded from yaml
         
         :param config: Configuration object to be used

         :param training_file_path:  Path to the training file

         :param study_area: Area of the study ( 'ms_data_science' or 'ms_engineering_om')

         :param model_save_filename: Saved model file path
        """

        # Get the current working directory
        current_dir = '' #os.getcwd()
        main_dir = '' #os.getcwd()
        if datalore_repo is not None:
            main_dir = datalore_repo
        print("MAIN:", main_dir)
    
        # Append the file path to the current working directory
        self.training_file_path = training_file_path

        # Load the config file and load it.
        if config is not None:
            self.config = config

        else:
            config_filepath = os.path.join(main_dir, "model/", "config.yml")
            with open(config_filepath) as f:
                self.config = yaml.safe_load(f)

        self.study_area = study_area
        self.feature_cols = (
            self.config[self.study_area]["features"]["num_cols"]
            + self.config[self.study_area]["features"]["cat_cols"]
        )
        self.target_col = self.config[self.study_area]["features"]["target_col"]

        # make saved directory if not exist
        os.makedirs(current_dir + "saved/", exist_ok=True)
        print(f"Making new directory to save files at {current_dir}saved")

        # save model,model_metadata as json and feature importance as csv
        self.model_save_path = os.path.join(current_dir, f"saved/{study_area}_{model_save_filename}.sav")
        self.model_metadata_filepath = os.path.join(current_dir, f"saved/{self.study_area}_metadata.json")
        self.feat_imp_filepath = os.path.join(current_dir, f"saved/{self.study_area}_model_feature_imp.csv")

    def load_and_preprocess_data(self):
        """Load and preprocess data.

         :return: A dataframe with the data and preprocessed
        """
        df = pd.read_csv(self.training_file_path)
        df = self.preprocess_train(df)
        return df

    def preprocess_train(self, df):
        """Preprocess data for training. This is called after the training has been completed and all data has been loaded into the data frame
         
         :param df: Data frame that is being used to train the model
         
         :return: The data frame with preprocessed data ready for training and training_set_data ( df ) =
        """
        # preprocess name
        df["Application Area of Study"] = np.where(
            df["Application Area of Study"] == "Data Science, MS",
            "ms_data_science",
            "ms_engineering_om",
        )

        # select subset
        df = df[df["Application Area of Study"] == self.study_area]
        df = df.drop("Application Area of Study", axis=1)

        # select columns
        df = df[self.feature_cols + self.target_col]

        # Map values for decision
        df["Decision Name"] = np.where(df["Decision Name"].str.contains("Accept"), 1, 0)

        # Label encode each of the columns with object type
        print("Label Encoding..")
        # Fit label encoder for each column in the dataframe.
        for column in df.columns:
            # Fit label transform to the column.
            if df[column].dtype == type(object):
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column])

        # remove the data with no English prof score (temporary)
        df = df[df["Converted English Prof Score"].notna()]

        # df['GRE Analytical Writing'] = df['GRE Analytical Writing'].fillna(0)
        # df['GRE Quantitative'] = df['GRE Quantitative'].fillna(0)
        # df['GRE Verbal'] = df['GRE Verbal'].fillna(0)
        return df

    ## Modeling
    def train_model(self):
        """Train and return the model. This is the main function for the model selection process. It loads the data and pre - processes
        """
        self.df = self.load_and_preprocess_data()
        print(
            f"Training model for {len(self.feature_cols)} features for {self.study_area}"
        )
        # Assigning the features as X and target as y
        X = self.df.drop(self.target_col, axis=1)
        y = self.df[self.target_col]

        #         scaler = StandardScaler()
        #         X_scaled = scaler.fit_transform(X)

        # Splitting the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=7
        )

        # Building piplines for model selection
        # clf = DecisionTreeClassifier()
        clf = HistGradientBoostingClassifier()
        clf.fit(X_train, y_train)

        # Calculating accuracy of predicted labels against true labels
        accuracy = round(accuracy_score(y_test, clf.predict(X_test)) * 100, 3)

        print(f"Accuracy score: {accuracy} %")

        # Extracting important features
        importance = self.get_feat_imp(clf, X_train, y_train).reset_index(drop=True)
        print(f"Feature importance: {importance}")

        # save as json

        # Define a dictionary to save as JSON
        savedata = {}

        savedata["model"] = self.study_area
        savedata["model_name"] = clf.__class__.__name__
        savedata["accuracy"] = accuracy
        savedata["feature_importance"] = importance.to_dict()
        savedata["trained_dateTime"] = str(datetime.now())

        # Save the dictionary as a JSON file
        with open(self.model_metadata_filepath, "w") as f:
            json.dump(savedata, f)

        return clf

    def load_or_train_model(self, retrain=False):
        """Load or train the model. This is a wrapper around : meth : ` ~gensim. models. SchwarzStudy. load_model `
         
         :param retrain: If True the model will be retrained.
         
         :return: A trained model or None if the model has been trained and saved to disk. Note that it is possible to get a model from the model_save_path
        """
        # check if model has been trained and saved
        # Load and train the model.
        if os.path.exists(self.model_save_path) and not retrain:
            model = self.load_model()
            print(
                f"Loading trained model for {self.study_area} at {self.model_save_path}."
            )
        else:
            model = self.train_model()
            # save the model to disk
            pickle.dump(model, open(self.model_save_path, "wb"))
            print(
                f"Saved trained model for {self.study_area} at {self.model_save_path}."
            )

        return model

    def load_model(self):
        """Load and return the model from the file. This is useful for testing purposes. If you want to do something other than save it yourself use : meth : ` save_model `
         
         :return: The model that was
        """
        model = pickle.load(open(self.model_save_path, "rb"))
        return model

    def predict_single(self, input_data, model):
        """Predicts a single class. Prediction and probability of being accepted are returned. In case of multiple classes the probability of being accepted is returned
         
         :param input_data: Data to be predicted.

         :param model: Model to be used. This is a subclass of BaseModel.
         
         :return: Tuple of predicted class and probability of being accepted ( 0. 0 - 1. 0 ). Note that we do not use model. predict_proba () because it is called from another thread
        """
        prediction = model.predict(input_data)[0]
        # prob_prediction = model.predict_proba(input_data)[0][prediction]
        prob_accept = model.predict_proba(input_data)[0][1]  # Get probability of being accepted
        return prediction, prob_accept

    def predict(self, input_data, model):
        """
         Predict and probabilistically predict. This is a wrapper around predict and predict_proba to allow customization of the model
         
         :param input_data: input data to be used for prediction
         :param model: model to be used for prediction. It must have a predict method
         
         :return: Returns the tuple of prediction and prob_prediction for the input_data. Prediction is a list of strings
        """
        labels = ["reject", "admit"]
        prediction = model.predict(input_data)
        prob_prediction = model.predict_proba(input_data)
        return prediction, prob_prediction

    def get_feat_imp(self, model, X_train, y_train):
        """Get feature importance for each feature. It is used to determine the importance of features based on how many times they were used in the training set
        
        :param model: The model to train on

        :param X_train: The training set as a pandas dataframe

        :param y_train: The target as a pandas dataframe ( labels )
        
        :return: A pandas dataframe with feature importance as the column " Importance " and the mean importance as the
        """

        importance = pd.DataFrame()
        importance["Features"] = self.feature_cols
        result = permutation_importance(
            model, X_train, y_train, n_repeats=10, random_state=0, n_jobs=-1
        )
        importance["Importance"] = result.importances_mean
        importance = importance.sort_values(by=["Importance"], ascending=False)
        importance.to_csv(self.feat_imp_filepath)
        print(f"Feature importances saved to {self.feat_imp_filepath}")
        return importance

    def load_feat_imp(self):
        """Load feat_imp. csv and return dataframe. This is used for testing. If you don't want to load it yourself you can use load_feat_imp_filepath

         :return: A pandas dataframe with feature
        """
        return pd.read_csv(self.feat_imp_filepath)

    def plot_feat_imp(self, importance):
        """Create a bar chart showing the importance of the features. It is used to visualize the feature impedance
         
         :param importance: The importance of the features
         
         :return: The plotly bar chart as a streamlit Figure object Example : >>> importance = pySpark. Spark. PsychoPy () >>> streamlit_plot. plot_feat_imp
        """
        # Create the Plotly bar chart
        fig = px.bar(importance, x="Importance", y="Features", width = 600)
        # Display the chart using Streamlit
        return fig
