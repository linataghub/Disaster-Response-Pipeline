import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
df = pd.read_sql_table('disasterresponses', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Distribution by genre  
    genre_distribution = df['genre'].value_counts(normalize=True)
    genre_names = list(genre_distribution.index)
    
    # Distribution by related  
    related_distribution = df['related'].value_counts(normalize=True)
    related_names = list(related_distribution.index)
    
    # Calculate percentage of other categories (binary)
    other_categories = df.drop(['id', 'message', 'original', 'genre', 'related'], axis = 1)
    other_distribution = other_categories.sum()/len(other_categories)
    other_distribution = other_distribution.sort_values(ascending = False)
    other_names = list(other_distribution.index)
    
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_distribution
                )
            ],

            'layout': {
                'title': 'Distribution of Genres',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
           'data': [
                Bar(
                    x=related_names,
                    y=related_distribution
                )
            ],

            'layout': {
                'title': 'Distribution of related',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Related"
                }
            }
        }, 
        {
            'data': [
                Bar(
                    x=other_names,
                    y=other_distribution
                )
            ],

            'layout': {
                'title': 'Distribution for other categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Category",
                }
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