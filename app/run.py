import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)
import numpy as np
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

def top_ten_correlations(df, labels, item):
    """
    INPUT
        df - a dataframe already structured
        labels - a list with all possible labels a message could be classified
        item - a string representing the item to trace correlation
    OUTPUT
        return - a dataframe containing the ten classifications with most correlation
                compared to the main item
    """
    correlation = df[df[item] == 1][labels].sum().sort_values(ascending=False)
    total = correlation[item]
    correlation.drop(labels=['related', item], inplace=True)
    top_ten = correlation[:10]
    top_ten = top_ten.append(pd.Series([correlation[10:].sum()], index=['others']))
    labels = [label.capitalize().replace('_', ' ') for label in list(top_ten.index)]
    return pd.DataFrame(np.array(top_ten), index=labels, columns=['Correlation'])/total

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for finding the ten most other correlated classes to
    # earthquake, fire and missing_people
    labels = list(df.columns)[4:]
    earthquake = top_ten_correlations(df, labels, 'earthquake')
    fire = top_ten_correlations(df, labels, 'fire')
    missing_people = top_ten_correlations(df, labels, 'missing_people')
  
    
    graphs = [
        {
            'data': [
                Bar(
                    x=earthquake.index,
                    y=earthquake['Correlation']
                )
            ],

            'layout': {
                'title': 'The ten most correlated classes with "Earthquake"',
                'yaxis': {
                    'title': "Correlation"
                },
                'xaxis': {
                    'title': "Classes"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=fire.index,
                    y=fire['Correlation']
                )
            ],
            
            'layout': {
                'title': 'The ten most correlated classes with "Fire"',
                'yaxis': {
                    'title': "Correlation"
                },
                'xaxis': {
                    'title': "Classes"
                }
            }
        },        
        {
            'data': [
                Bar(
                    x=missing_people.index,
                    y=missing_people['Correlation']
                ),
            ],
            
            'layout': {
                'title': 'The ten most correlated classes with "Missing people"',
                'yaxis': {
                    'title': "Correlation"
                },
                'xaxis': {
                    'title': "Classes"
                }
            }
        },        
        {
            'data': [
                Heatmap(
                    z=df[labels].corr().values,
                    x=labels,
                    y=labels
                )
            ],
            
            'layout': {
                'title': 'Other Correlations',
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
