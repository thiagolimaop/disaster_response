import sys
import re
import sqlite3
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn.externals import joblib

def load_data(database_filepath):
    """
    INPUT
        database_filepath - the file path to database
    OUTPUT
        X - a panda series with the messages in english
        y - a numpy array with all 36 binary categories
        labels - a list of strings with the categories name 
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM DisasterMessages', con = engine)
    
    labels = [item for item in list(df.columns) if item not in ['id', 'message', 'original', 'genre']]
    X = df['message']
    y = df[labels].values
    return X, y, labels


def tokenize(text):
    """
    INPUT
        text - a string with an english message
    OUTPUT
        return - a list with the clean tokens based on the english message
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    OUTPUT
        return - a GridSearchCV instance with the model pipeline
    """
    pipeline = Pipeline([
           ('vect', CountVectorizer(tokenizer=tokenize)),
           ('tfidf', TfidfTransformer()),
           ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        # 'tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [5, 10, 50, 100, 200],
        # 'clf__estimator__min_samples_split': [2, 3, 4],
    }

    return GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT
        model - a trained model
        X_test - the portion data to evaluate the model
        Y_test - the categories represented to the X_test
        category_names - a list of strings with the category names
    """
    y_pred = model.predict(X_test)
    print('Best params')
    print(model.best_params_)
    
    for index in range(len(category_names)):
        class_report = classification_report(Y_test[:, index], y_pred[:, index], target_names=[category_names[index]])
        print(class_report)


def save_model(model, model_filepath):
    """
    INPUT
        model - a trained model
        model_filepath - a file path to store the trained model
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
