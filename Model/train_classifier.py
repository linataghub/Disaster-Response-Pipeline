# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disasterresponses",engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns.values
    
    return X, Y, category_names 


def tokenize(text):
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression(multi_class='multinomial')))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'tfidf__use_idf': (True, False),
        'clf__estimator__solver':['newton-cg', 'lbfgs']
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return cv

def display_results(y_test, y_pred, category_names):
    
    metrics = []
    
    sum_accuracy = sum_f1 = sum_precision =sum_recall = 0
    
    # Report the f1 score, precision and recall for each output category of the dataset
    for i in range(len(category_names)):
        accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
        f1 = f1_score(y_test[:, i], y_pred[:, i], average='weighted')
        precision = precision_score(y_test[:, i], y_pred[:, i], average='weighted')
        recall = recall_score(y_test[:, i], y_pred[:, i],average='weighted')
        metrics.append([accuracy, f1, precision, recall])
        
    # Format the results in a dataframe
    results = pd.DataFrame(metrics, index = category_names, columns = ['Accuracy','F1 score', 'Precision', 'Recall'])
    
    return results 

def evaluate_model(model, X_test, Y_test, category_names):
    
    # Predict on test data
    Y_pred = model.predict(X_test)
    
    # Display evaluation metrics 
    evaluation_metrics = display_results(np.array(Y_test), Y_pred, category_names)
    
    return evaluation_metrics 
  


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))
    
    pass


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