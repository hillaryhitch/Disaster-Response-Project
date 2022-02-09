# Disaster-Response-Project

During and immediately after natural disaster there are millions of communication to disaster response organizations either direct or through social media. Disaster response organizations have to to filter and pull out the most important messages from this huge amount of communications a and redirect specific requests or indications to the proper organization that takes care of medical aid, water, logistics ecc. Every second is vital in this kind of situations, so handling the message correctly is the key


This is an NLP project pipeline that builds, trains, evaluates and serves a model to predict category of messages received for disaster response.

There are 3 steps used:

- Text Processing: Take raw input text, clean it, normalize it, and convert it into a form that is suitable for feature extraction.
- Feature Extraction: Extract and produce feature representations that are appropriate for the type of NLP task you are trying to accomplish and the type of model you are planning to use.
- Modeling: Design a statistical or machine learning model, fit its parameters to training data, use an optimization procedure, and then use it to make predictions about unseen data.

### Files in the repo:

app

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- InsertDatabaseName.db # database to save clean data to

models

|- train_classifier.py

|- classifier.pkl # saved model

README.md



### How to run from training model to serving as app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

#### You can just run this part only for the app since the model is already trained and saved
3. Run your web app: `python run.py`



