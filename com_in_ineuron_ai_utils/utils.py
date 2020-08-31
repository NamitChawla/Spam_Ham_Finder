import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from app_logging import logfile
from flask import request
import os

log_writer = logfile.App_logger()



def dataFromTextFile(filepath):
    stop_wordslist= []
    try:
        with open(filepath) as f:
            lines = f.read().splitlines()
            for line in lines:
                stop_wordslist.append(line)
        return stop_wordslist
    except Exception as e:
        return "Excetpion with stopwords: "+e

def data_preprocessing_predict(message, filepath, prediction_file_object):
    #prediction_file_object = open("Logs_Prediction/ModelPredictionLog.txt", 'a+')
    try:
        stop_words = dataFromTextFile(filepath)
        log_writer.log(prediction_file_object, "Preprocessing user's message")
    except Exception as e:
        log_writer.log(prediction_file_object, "Preprocessing failed due to error in stopwords")
        raise e

    ps = PorterStemmer()
    rev = re.sub("[^a-zA-Z]", " ", message)
    rev = rev.lower()
    rev = rev.split()
    rev = [ps.stem(word) for word in rev if not word in stop_words]
    rev = " ".join(rev)
    corp = [rev]
    log_writer.log(prediction_file_object, "Preprocessing of user's message completed")
    return corp

def df_preprocessing_predict(message, filepath, uploading_file_object):
    try:
        stop_words = dataFromTextFile(filepath)
        log_writer.log(uploading_file_object, 'Good to go with stopwords')
    except Exception as e:
        log_writer.log(uploading_file_object, 'Error with stopwords')
        raise e

    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(message)):
        review = re.sub('[^a-zA-Z]', ' ', message['Message'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

def data_preprocessing_train(messages, filepath):
    try:
        stop_words =dataFromTextFile(filepath)
    except Exception as e:
        raise e

    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['Message'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus.append(review)

    target = pd.get_dummies(messages['Datalabel'])
    # 1 for SPAM and 0 for HAM
    target = target.iloc[:, 1].values

    return corpus, target

#Data preprocessing
def preprocess_training_data(trainingFilePath, stop_words, retraining_file_object):
    #prediction_file_object = open("Logs_Prediction/ModelPredictionLog.txt", 'a+')
    try:
        messages = pd.read_csv(trainingFilePath, sep="\t", names=["Datalabel", "Message"])
        #Data cleaning
        clean_df, target =data_preprocessing_train(messages, stop_words)
        log_writer.log(retraining_file_object, "Preprocessing completed")
    except Exception as e:
        log_writer.log(retraining_file_object, "Error with CSV")
        raise e
    return clean_df, target

def conditions_check(uploading_file_object):
    #uploading_file_object = open("Logs_Uploading/ModelUploadingLogs.txt", 'a+')
    try:
        uploaded_file = request.files['upload_file']
        filename = uploaded_file.filename
        log_writer.log(uploading_file_object, 'Upload successful')
    except Exception as e:
        log_writer.log(uploading_file_object, 'Error with uploaded file')
        raise e
    return uploaded_file, filename

def delete_existing_files(uploading_file_object):
    csv_files = './csv_files'
    list_of_files = os.listdir(csv_files)

    #Delete existing csv files if present
    for csfile in list_of_files:
        try:
            os.remove("./csv_files/" + csfile)
            log_writer.log(uploading_file_object, 'Successfully deleted existing files')
        except Exception as e:
            log_writer.log(uploading_file_object, 'Error in deleting existing files')
            raise e
    return "Success"




