# Disaster-Response-Pipeline

Figure Eight has provided a database with pre-labeled tweets and text messages from real life disasters. The purpose of this project is to build a machine learning pipeline that can automatically classify disasters messages to allow agencies to determine more efficiently the nature of the disaster.

The process is divided in three parts:
1. ETL pipeline to clean the dataset and then store it in a SQLite database
2. ML pipeline that uses the disaster message column to predict classifications for 35 categories (multi-output classification)

