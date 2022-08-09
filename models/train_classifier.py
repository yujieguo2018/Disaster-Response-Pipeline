import sys

def load_data(database_filepath):
    
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
    
    y_pred = model.predict(X_test)

    category_columns = category_names
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred,target_names = category_columns))


def save_model(model, model_filepath):
    import pickle
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
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
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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