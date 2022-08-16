import sys

def load_data(database_filepath):
    
    '''
    load data from the sql database, extract key information for model training
    '''
    import re
    import numpy as np
    import pandas as pd
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM Disaster_data", engine)

    category_columns = ['related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather', 'direct_report']
    X = df['message']
    Y = df[category_columns]

    return X,Y,category_columns

def tokenize(text):
    
    '''
    NLP data processing 
    '''
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    # Remove punctuation characters
    import re
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    # tokenize
    raw_tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [w for w in raw_tokens if w not in stopwords.words("english")]
    
    # lemmatize and normalize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''
    build machine learning pipeline with RandomizedSearch
    '''

    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__n_estimators': [20, 30, 40],
    'clf__estimator__min_samples_split': [2,5,8]}

    model = RandomizedSearchCV(pipeline, param_distributions=parameters, n_jobs=4, verbose=2)
    
    return model
    

def evaluate_model(model, X_test, y_test, category_names):
    
    '''
    evaluate our machine learning model by printing out the precision, recall and f-1 score
    '''
    import pandas as pd
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred,columns = category_names)
    from sklearn.metrics import classification_report
    
    metrics = pd.DataFrame(columns = ['category','accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1', 
                                      'weighted_avg_precision','weighted_avg_recall', 'weighted_avg_f1'])

    for col in category_names:

        report = classification_report(y_test[col], y_pred[col],output_dict=True)

        accuracy = report['accuracy']
        macro_avg_precision = report['macro avg']['precision']
        macro_avg_recall = report['macro avg']['recall']
        macro_avg_f1_score = report['macro avg']['f1-score']
        weighted_avg_precision = report['weighted avg']['precision']
        weighted_avg_recall = report['weighted avg']['recall']
        weighted_avg_f1_score = report['weighted avg']['f1-score']

        list_col = [col,accuracy,
                    macro_avg_precision,macro_avg_recall,macro_avg_f1_score,
                    weighted_avg_precision,weighted_avg_recall,weighted_avg_f1_score]
        metrics.loc[len(metrics)] = list_col
        
    return metrics


def save_model(model, model_filepath):

    '''
    save model as a pickle object 
    '''

    import pickle
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():

    '''
    Load data -> build model -> train model -> evaluate model -> save model
    
    To run this file in the terminal:
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    '''

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
        
        print('Evaluating model...')
        pd.options.display.precision = 2
        evaluate_model(best_model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(best_model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()