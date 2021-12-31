import sys
import pandas as pd
import nltk
import re
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
import pickle

def load_data(database_filepath):
    # load data from database
    '''Loads data from the database given. 
    INPUT: dbname
    OUTPUT: input variables, output variables and the category names'''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from table1', engine)
    X = df['message']
    y = df.iloc[:, 4:39]
    category_names=list(y)
    return X, y,category_names


def tokenize(text):
    '''INPUT: message 
    OUTPUT: remove spaces, convert words to their base,
    converts to lowercase and return an array of each word
    '''
    text = re.sub(r'[^\w\s]','',text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_toks = []
    for i in tokens:
        clean = lemmatizer.lemmatize(i).lower().strip()
        clean_toks.append(clean)

    return clean_toks


def build_model():
    '''use grid search to train adaboost.'''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])
    parameters = {
            'vect__max_df': (0.25, 1.0)
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Calculates precision, recall & f1-score for each of the category names'''
    pred = model.predict(X_test)
    pred_df = pd.DataFrame(pred, columns=category_names)
    evals = {}
    for cols in Y_test.columns:
        evals[cols] = []
        evals[cols].append(precision_score(Y_test[cols], pred_df[cols],average='micro'))
        evals[cols].append(recall_score(Y_test[cols], pred_df[cols],average='micro'))
        evals[cols].append(f1_score(Y_test[cols], pred_df[cols],average='micro'))
    print(pd.DataFrame(evals))


def save_model(model, model_filepath):
    '''dunmps model to a pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))


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