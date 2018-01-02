"""
Titanic Data - exploring  our titanic data set 
"""
import pandas as pd 
import numpy as np 

def main():
    # import data 
    df = pd.read_csv('../data/titanic-train.csv')

    print('------------COLUMNS------------')
    print(df.columns)

    btwn_70_and_75 = df[(df['Age'] > 70) & (df['Age'] < 75)]
    print('------------AGES BETWEEN 70 AND 75------------')
    print(btwn_70_and_75.head())

    embarked_vals = df['Embarked'].unique()
    print('------------EMBARKED VALUES------------')
    print(embarked_vals)

    sorted_by_age = df.sort_values('Age', ascending=False)
    print('------------AGES SORTED------------')
    print(sorted_by_age.head())
    print(sorted_by_age.tail())

    # inspect the correlations 
    corr_survived = df.corr()['Survived'].sort_values
    print('------------CORR SURVIVED-----------------')
    print(corr_survived)
    
if __name__ == '__main__':
    main()