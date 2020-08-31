FROM continuumio/anaconda3:4.4.0
COPY . /Users/Namit/PycharmProjects/SPAMORHAM
EXPOSE 5000
WORKDIR /Users/Namit/PycharmProjects/SPAMORHAM
RUN pip install -r requirements.txt
CMD python flask_app.py