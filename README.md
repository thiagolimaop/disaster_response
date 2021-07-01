# Disaster Response Pipeline Project

During a disaster event millions of communications are exchanged. People need water, medical supplies, food and other basic needs. This application is responsible for helping organizations to prioritise the important messages, taking the disaster messages in english and then classifying these messages according to 36 classes indicating what is relevant to help people.

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Repository

    - app
    | - template
    | |- master.html # main page of web app
    | |- go.html # classification result page of web app
    |- run.py # Flask file that runs app
    - data
    |- disaster_categories.csv # data to process
    |- disaster_messages.csv # data to process
    |- process_data.py # cleaning data and structure it to save in database
    |- DisasterResponse.db # database to save clean data
    - models
    |- train_classifier.py # read the data from a database and train the classifier
    |- classifier.pkl # saved model
    - README.md



### Data structure
 
The dataset provided is divided in two parts:
- First part the data has the original message, the message in english and the id, besides the genre that is equal to all messages.
- The second has the categories that a certain message belongs to and the id message.

### Cleaning data

The process consists of separating the categories in 36 different columns, indicating if some message belongs to a class or not, then the duplicated rows are dropped according to the “original” column, containing the original messages. Finally the data is saved in a database called DisasterResponse.db.

### Modeling process

The first step for data modeling is to tokenize the message, splitting it into tokens, which passes for a lemmatization, i.e, each token is reduced to your root form. 
 
The second step is to use these lemmatized tokens to build a vocabulary vector with unique words, giving the frequency time for each word in a message. This step is occur in the TFIDF estimator.
 
The final step is to apply the random forest tree estimator.
 
The data imbalance, which easily causes high accuracy, was treated using hyperparameter tuning with GridSearchCV, changing the performance metric, using precision, recall and F1 score, which provides a better insight for each class.
