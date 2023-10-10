from com_in_ineuron_ai_utils.utils import data_preprocessing_predict, df_preprocessing_predict
import pickle
from app_logging import logfile

# predictApi class for prediction
class predictApi:
    def __init__(self, stopwordsFilepath):
        self.stop_words_path =stopwordsFilepath
        self.log_writer = logfile.App_logger()
        self.prediction_file_object = open("Logs_Prediction/ModelPredictionLog.txt", 'a+')
        self.uploading_file_object = open("Logs_Uploading/ModelUploadingLogs.txt", 'a+')

    def executeProcessing(self, text, trainingFilePath, modelPath, vectorPath, prediction_file_object):
        self.log_writer.log(prediction_file_object, 'Prediction begins now')
        try:
            with open(vectorPath,'rb') as f:
                vectorizer = pickle.load(f)
            with open(modelPath, 'rb') as f:
                loaded_model = pickle.load(f)
            self.log_writer.log(prediction_file_object,'Prediction model and vectorizer instance loaded successfully')

        except Exception as e:
            self.log_writer.log(prediction_file_object,
                                'Prediction unsuccessful due to model loading')

            raise e

        text = data_preprocessing_predict(text, self.stop_words_path, prediction_file_object)
        x = vectorizer.transform(text).toarray()
        prediction = loaded_model.predict(x)

        self.log_writer.log(prediction_file_object, 'Predictions generated')
        return prediction[0]


    def executeFileProcessing(self, newDF, modelPath, vectorPath, uploading_file_object):
        try:
            with open(vectorPath,'rb') as f:
                vectorizer = pickle.load(f)
            with open(modelPath, 'rb') as f:
                loaded_model = pickle.load(f)
            self.log_writer.log(uploading_file_object, 'Preprocessing Uploaded data from file')
        except Exception as e:
            self.log_writer.log(uploading_file_object, 'Preprocessing failed due to failed model loading')
            raise e
        clean_df = df_preprocessing_predict(newDF, self.stop_words_path, uploading_file_object)
        self.log_writer.log(uploading_file_object, 'Preprocessing of Uploaded data finished')
        x = vectorizer.transform(clean_df).toarray()

        self.log_writer.log(uploading_file_object, 'Prediction of Uploaded data started')
        prediction = loaded_model.predict(x)
        self.log_writer.log(uploading_file_object, 'Prediction of Uploaded data finished')
        return prediction

