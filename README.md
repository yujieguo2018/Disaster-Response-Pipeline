# Disaster Response Pipeline Project

### Project Overview

The goal is to build ETL + NLP pipeline and machine learning model that classifies disaster message and categorize these events so that it helps the organization to send the messages to an appropriate disaster relief agency.

### Folder Structure

- app
    -  template
    - master.html  # main page of web app
    - go.html  # classification result page of web app
    - run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data to process 
    - disaster_messages.csv  # data to process
    - process_data.py
    - DisasterResponse.db   # database to save clean data to

- models
    - train_classifier.py
    - classifier.pkl  # saved model/too big to save on github but should be in local

- README.md

### Technical Details 

1. ETL Pipeline
In a Python script, process_data.py, there is a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py, there is a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App

A web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions on running the code:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
