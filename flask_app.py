from flask import Flask, render_template, send_file, request
from flask_cors import CORS, cross_origin
import pandas as pd
import flask_monitoringdashboard as dashboard
from com_in_ineuron_ai_utils.utils import conditions_check, delete_existing_files

from com_in_ineuron_ai_prediction.predictApp import predictApi
from com_in_ineuron_ai_training.trainApp import TrainApi
from app_logging import logfile
import os

app = Flask(__name__)
dashboard.bind(app)
CORS(app)

trainingDataFolderPath = "trainingData"
trainingFilePath = "./SMSSpamCollection.txt"
modelPath = trainingDataFolderPath +"/final_spam_vs_ham_model.pickle"
vectorPath = trainingDataFolderPath +"/vectorizer.pickle"
result_file = './output_data/result_output_data.csv'

class Appapi:
    def __init__(self):
        stopwordsFilePath = "data/stopwords.txt"
        self.predictObj = predictApi(stopwordsFilePath)
        self.trainObj = TrainApi(stopwordsFilePath)
        self.log_writer = logfile.App_logger()

#route for homepage
@app.route("/", methods=["GET"])
@cross_origin()
def homepage():
    return render_template("index.html")

#prediction route
@app.route("/predict", methods=["POST", "GET"])
@cross_origin()
def index():
    prediction_file_object = open("Logs_Prediction/ModelPredictionLog.txt", 'a+')
    if request.method == "POST":

        try:
            message = request.form["msg"]
            flaskapp.log_writer.log(prediction_file_object, 'Prediction begins')
            result = flaskapp.predictObj.executeProcessing(message, trainingFilePath, modelPath, vectorPath, prediction_file_object)
            print(result)

            if result == 0:
                #For HAM
                flaskapp.log_writer.log(prediction_file_object, 'Prediction finished')
                return render_template("results.html")
            else:
                #For SPAM
                flaskapp.log_writer.log(prediction_file_object, 'Prediction finished')
                return render_template("results_2.html")
        except Exception as e:
            flaskapp.log_writer.log(prediction_file_object, 'Prediction failed due to '+e)
            raise e

    else:
        flaskapp.log_writer.log(prediction_file_object, 'failed, POST method expected')
        prediction_file_object.close()
        return render_template("index.html")

#model retraining route
@app.route('/retrainmodel',methods=['POST','GET'])
@cross_origin()
def retrainModel():
    retraining_file_object = open("Logs_Retraining/ModelRetrainingLog.txt", 'a+')
    if request.method=="POST":
        try:
            flaskapp.log_writer.log(retraining_file_object, 'Retraining started')
            result = flaskapp.trainObj.training_model(trainingFilePath, trainingDataFolderPath, retraining_file_object)
            flaskapp.log_writer.log(retraining_file_object, 'Successfully retrained')
            retraining_file_object.close()
            return result

        except Exception as e:
            flaskapp.log_writer.log(retraining_file_object, 'Retraining failed '+e)
            retraining_file_object.close()
            raise e
    else:
        flaskapp.log_writer.log(retraining_file_object, 'Failed, POST method expected')
        retraining_file_object.close()
        return render_template("retrain.html")

#for bulk upload option
@app.route('/uploadfile',methods=['POST','GET'])
@cross_origin()
def uploadfile():
    return render_template('upload.html')

@app.route('/csv',methods=['POST','GET'])
@cross_origin()
def csv():
    if request.method == 'POST':
        uploading_file_object = open("Logs_Uploading/ModelUploadingLogs.txt", 'a+')
        try:
            #reading csv file
            uploaded_file, filename = conditions_check(uploading_file_object)
            #procede only if file is available
            if uploaded_file.filename != '':
                uploaded_file.save(filename)
                data = pd.read_csv(filename, names=["Message"])

                # procede only if file is in correct format
                if len(data.columns) == 1:
                    flaskapp.log_writer.log(uploading_file_object, 'Process initiated')
                    result = flaskapp.predictObj.executeFileProcessing(data, modelPath, vectorPath, uploading_file_object)

                    #deleting previous files present in csv_file folder
                    flaskapp.log_writer.log(uploading_file_object, 'Deleting existing files')
                    delete_existing_files(uploading_file_object)

                    #output mapping
                    data['Result'] = result
                    data['Result'] = data["Result"].replace(to_replace=[0, 1], value=["HAM", "SPAM"])

                    #saving pandas dataframe as a csv file in csv_file folder
                    data.to_csv(result_file)
                    flaskapp.log_writer.log(uploading_file_object, 'Saved prediction results')
                    uploading_file_object.close()
                    return render_template('csv.html')
                else:
                    return 'Error: Please Make Sure that csv file is in standard acceptable format,Please go through given Sample csv file format'
            else:
                return 'File Not Found'
        except Exception as e:
            flaskapp.log_writer.log(uploading_file_object, 'Unsuccessful with upload task')
            uploading_file_object.close()
            raise e
    else:
        return render_template('index.html')

#predictions download for bulk upload option
@app.route('/download_file', methods=['POST', 'GET'])
@cross_origin()
def download_file():
    uploading_file_object = open("Logs_Uploading/ModelUploadingLogs.txt", 'a+')
    try:
        p = './output_data/result_output_data.csv'
        flaskapp.log_writer.log(uploading_file_object, 'File available for download')
        uploading_file_object.close()
        return send_file(p, as_attachment=True)

    except Exception as e:
        flaskapp.log_writer.log(uploading_file_object, 'File unavailable for download')
        uploading_file_object.close()
        raise e

port = int(os.getenv("PORT"))
if __name__=="__main__":
    #app.run(debug=True)
    flaskapp = Appapi()
    #app.run(host="0.0.0.0", port=8000) #For testing in local system
    app.run(host="0.0.0.0", port=port)
    #app.run(host="127.0.0.1", debug=True)