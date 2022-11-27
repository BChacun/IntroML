# Import Libraries : 

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#Libraries for MLR:
from sklearn import linear_model

#Libraries for SVM
from sklearn.model_selection import KFold
from sklearn import svm

#Libraries for Random Forest:
from sklearn.ensemble import RandomForestRegressor

# Libraries for PCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import scale 

# Visualization Libraries :
from matplotlib import pyplot as plt 
import seaborn as sns 

# RMSE :
from sklearn.metrics import mean_squared_error

""" Data Treatment : Filling missing values and tranforming Data"""


#Function filling missing values of a dataframe df

def missing_values_table(df):
    """mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe had " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    print(missing_values_table(df))
    df = df.fillna(df.mean())"""
    return df

#Function that normalize our data so our algorithms work better:

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

#Split the data Frame into the label and the features    

def split_data_XY(df,label):
    y=df[label]
    X = df.drop(label,axis= 1)
    return(X,y)

#funtion that binarize categorical data:

def binarisation_categorical_data(df):
    return pd.get_dummies(df)


#Function that returns a cleaned DataFrame
def Data_treatment(df,label_name):
    df=missing_values_table(df)
    print("missing values filled")
    df=binarisation_categorical_data(df)
    print("categorical data binarised")
    df=normalize(df)
    X,y = split_data_XY(df,label_name)
    print("values normalized")
    return(X,y)


"""Visualization of the Original Data"""

#Correlation matrix

def corr_matrix(df):
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(),annot=True)

def scatter_matrix(df,label):
    sns.pairplot(df, hue=label)



