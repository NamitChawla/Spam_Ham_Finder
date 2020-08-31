import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from com_in_ineuron_ai_utils.utils import preprocess_training_data
from imblearn.over_sampling import SMOTE
from app_logging import logfile
import os

class TrainApi:

    def __init__(self, stopWordsFilePath):
        self.stop_words =stopWordsFilePath
        self.log_writer = logfile.App_logger()

    def training_model(self, trainingFilePath, modelPath, retraining_file_object):
        self.log_writer.log(retraining_file_object, 'Start of model training')

        data_df, target = preprocess_training_data(trainingFilePath, self.stop_words, retraining_file_object)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data_df).toarray()

        """
        CLASS BALANCING to deal with the imbalanced data so that 
        equal intances are created and the proposed model could avoid the 
        problem of overfitting due to less exposure towards minority data. 
        To do so, Synthetic Minorty Oversampling TEchnique is used, 
        also known as SMOTE. SMOTE generates synthetic samples from the minority class.
        """

        sm = SMOTE(random_state=50)
        X_res, y_res = sm.fit_sample(X, target.ravel())

        rfc = RandomForestClassifier(n_estimators=37, random_state=50)

        """ n_estimators is used for the number of trees you want to build 
        before taking the maximum voting or average of predictions"""

        rfc.fit(X_res, y_res)

        # deleting previous files present in trainingData folder
        model_files = modelPath
        list_of_files = os.listdir(model_files)
        for mfile in list_of_files:
            try:
                os.remove(modelPath+ "/" + mfile)
                self.log_writer.log(retraining_file_object, 'deleted existing models')
            except Exception as e:
                self.log_writer.log(retraining_file_object, 'error deleting existing models')
                raise e

        # save the model to disk
        # save vector instance
        try:
            with open(modelPath + '/vectorizer.pickle', 'wb') as f:
                pickle.dump(vectorizer, f)
                pickle.dump(rfc, open(modelPath+'/final_spam_vs_ham_model.pickle', "wb"))
                self.log_writer.log(retraining_file_object, 'End of model training')


        except Exception as e:
            self.log_writer.log(retraining_file_object, 'End of model training')
            raise e

        return "You have successfully retrained the model"


