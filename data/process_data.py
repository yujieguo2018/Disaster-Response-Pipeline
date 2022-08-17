import sys
import numpy as np
import pandas as pd

def load_data(messages_filepath, categories_filepath):

    '''
    load data from two csv files and merge them together
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df 

def clean_data(df):

    '''
    data cleansing -- focus on the categories columns
    get the category for the messages

    '''
    # create a dataframe of the 36 individual category columns
    # extract the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    categories = df['categories'].str.split(';',expand = True)
    first_row = categories.iloc[0]
    category_colnames = list(first_row.map(lambda x:x[0:-2]))

    # add columns of `categories` to our dataset and drop the old one
    df[category_colnames] = df.categories.apply(lambda x: pd.Series(str(x).split(";")))
    df = df.drop('categories',axis=1)

    # convert category values to 0/1 in the categories dataframe
    for column in category_colnames:
        # set each value to be the last character of the string
        df[column] = df[column].map(lambda x:x[-1]).astype(str)
        # convert column from string to numeric
        df[column] = df[column].astype(int)
        # convert values that great than 1 to be 1 to ensure the value is binary
        df.loc[df[column]>1,column] = 1
    # drop duplicates
    df = df[-df.duplicated()]

    return df

def save_data(df, database_filename):

    '''
    save data into a SQL database
    '''
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_data', engine, index=False,if_exists='replace')


def main():

    '''
    Call the functions we defined above: load data - clean data - save data

    To run this file in the terminal
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]



        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
