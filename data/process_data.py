import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np


def load_data(messages_filepath, categories_filepath):
    '''INNPUT
    messages_filepath; path with messages
    categories_filepath: path with the categories
    
    OUTPUT
    df: a dataframe with messages and labels (categories)'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,how='left',on='id')
    return df


def clean_data(df):
        '''INNPUT
    df: a dataframe with messages and labels (categories)
    
    OUTPUT
    df: cleaned dataframe
    '''
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    row2=row.apply(lambda x: x[:-2])
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(row2)
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: str(x))
        df=df.drop('categories',1)
        df = pd.concat([df,categories],axis=1)
        df=df.drop_duplicates()
        return df


def save_data(df, database_filename):
    '''writes dataframe to db
    INPUT: a dataframe and a db filename'''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('table1', engine, index=False) 


def main():
    '''main function to call the others'''
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
