# Disaster-Response-Pipeline

Figure Eight has provided a database with pre-labeled tweets and text messages from real life disasters. The purpose of this project is to build a machine learning pipeline that can automatically classify disasters messages to allow agencies to determine more efficiently the nature of the disaster.

The process is divided in three parts:
1. ETL pipeline to clean the dataset and then store it in a SQLite database
2. ML pipeline that uses the disaster message column to predict classifications for 35 categories (multi-output classification)
3. Deployment in a web app to visualise the data 

# Instructions

1. Run the following commands in the current working directory  to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Files description 

**Data Folder** 
- disaster_categories.csv: Raw categories data
- disaster_messages.csv: Raw message data 
- DisasterResponse.db: Database to store the clean data 
- process_data.py - ETL pipeline to clean the data to clean the dataset and then store it in a SQLite database

**Model Folder** 
- train_classifier.py: ML pipeline that uses the disaster message column to predict multi-output classifications
- classifier.pkl: Saved model after running the ML pipeline

**App Folder** 
- run.py: flask web app to visualise the data 
- templates: html templates

**Main Folder**
- ETL Pipeline Preparation Notebook
- ML Pipeline Notebook







