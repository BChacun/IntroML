# Import Libraries : 

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split

#Libraries for MLR:
from sklearn import linear_model

#Libraries for the MLP 
from sklearn.neural_network import MLPRegressor

#Libraries for SVM
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import preprocessing
from sklearn import utils

#Libraries for Random Forest:
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  ShuffleSplit
from sklearn.model_selection import cross_val_score

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
    print(missing_values_table(df))"""
    df = df.fillna(df.mean())
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
    X,y = split_data_XY(df,label_name)
    X=normalize(X)
    print("values normalized")
    return(X,y)


"""Visualization of the Original Data"""

#Correlation matrix

def corr_matrix(df):
    plt.figure(figsize=(20,10))
    sns.heatmap(df.corr(),annot=True)

def scatter_matrix(df,label):
    sns.pairplot(df, hue=label)


"""Different methods to evaluate the problem"""

# Trees and forest

#We find the best depth for our trees

def depht_Tuning(X,y):
    # Define the cvp (cross-validation procedure) with random 1000 samples, 2/3 training size, and 1/3 test size
    cvp = ShuffleSplit(n_splits=1000,test_size=1/3)

    # Define the max depths between 1 and 10
    n_depths = 10
    depths = np.linspace(1, 10, n_depths)
    
    # Loop on the max_depth parameter and compute median RMSE
    tab_RMSE_tree = np.zeros(n_depths)
    for i in range(n_depths):
        reg_tree = DecisionTreeRegressor(max_depth=depths[i])
        tab_RMSE_tree[i] = np.median(np.sqrt(-cross_val_score(reg_tree, X, y, scoring='neg_mean_squared_error', cv=cvp)))

     # Plot
    plt.plot(depths, tab_RMSE_tree)
    plt.xlabel('Max depth of the tree', size=20)
    plt.ylabel('RMSE', size=20)

    result = np.where(tab_RMSE_tree == min(tab_RMSE_tree))
    optimal_dephts = depths[result]
    return optimal_dephts

#We generate a forest model with the optimum tree depth

def Forest_model_generator(X,y):
    t_init = time.time()

    Optimal_depth=depht_Tuning(X,y)

    print("The optimal depth of the trees are :" , Optimal_depth)
    #Split the data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

     # create regressor object
    regressor = RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth=Optimal_depth)
  
    # fit the regressor with x and y data
    regressor.fit(X, y)  

    t_fin = time.time()

    # Visualising the Random Forest Regression results
    y_pred = regressor.predict(X_test)
    print("the mean squared error is : ",mean_squared_error(y_test, y_pred))
    print("it took this time for the training : ", t_fin-t_init)

    return regressor,mean_squared_error(y_test, y_pred),t_fin-t_init

# A single layer neuron regression

def MLR_neuron_regression(X,y):

    t_init = time.time()

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)


    # Multi Linear Model :
    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)  

    t_fin = time.time()

    # Visualising the Random Forest Regression results
    y_pred = regr.predict(X_test)
    print("the mean squared error is : ",mean_squared_error(y_test, y_pred))
    print("it took this time for the training : ", t_fin-t_init)

    return regr,mean_squared_error(y_test, y_pred),t_fin-t_init

#We do a multilinear polynomial regression with a neural network

#definition of the model
def MLP_neurons_model(X,Y,hidden_layer_sizes):
    
    #on utilise cette activation et ce solver par défaut
    regr = MLPRegressor(hidden_layer_sizes=(hidden_layer_sizes),activation='tanh', solver='lbfgs').fit(X, Y)
    return regr

def MLP_neurons_tuning(X,Y):
    MSE=[]
    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3, random_state=42)

    for q in range(300):
        model=MLP_neurons_model(X_train,y_train,q+1) #création du modèle
        y_pred = model.predict(X_test)
        MSE.append(mean_squared_error(y_test, y_pred))

    plt.plot(MSE)

def MLP_neurons_regression(X,Y,hidden_layer_sizes):
    t_init = time.time()

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3, random_state=42)


    # Multi Linear polynomial Model :
    # with sklearn
    regr = MLP_neurons_model(X_train,y_train,hidden_layer_sizes)

    t_fin = time.time()

    # Visualising the Random Forest Regression results
    y_pred = regr.predict(X_test)
    print("the mean squared error is : ",mean_squared_error(y_test, y_pred))
    print("it took this time for the training : ", t_fin-t_init)

    return regr,mean_squared_error(y_test, y_pred),t_fin-t_init

#We do a linear regression after a PCA but for both data the dimension reduction is not worth it

def PCA_Observations(X,y):

    t_init = time.time()

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)

    #scale predictor variables
    pca = PCA()
    X_reduced = pca.fit_transform(scale(X))

    #define cross validation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

    regr = LinearRegression()
    mse = []

    # Calculate MSE with only the intercept
    score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
    mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in np.arange(1, X.shape[1]):
        score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
        mse.append(score)

    t_fin = time.time()
    #We plot the explained variance ratios    

    print("The explained varianc ratios are: " , np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100))
    
    # Plot cross-validation results    
    plt.subplots()
    plt.plot(mse)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('MSE')
    plt.title('hp')

    #Plot the data above the first three components
    plt.subplots()
    plt.scatter(X_reduced[:,1],X_reduced[:,2],c=X_reduced[:,3],cmap='Blues',label="Component 3")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()

    print("it took this time for the PCA : ", t_fin-t_init)


def PCA_linear_regression(X,y):

    t_init = time.time()
    
    pca = PCA()

    #split the dataset into training (70%) and testing (30%) sets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

    #scale the training and testing data
    X_reduced_train = pca.fit_transform(scale(X_train))
    X_reduced_test = pca.transform(scale(X_test))[:,:1]

    #train PCR model on training data 
    regr = LinearRegression()
    regr.fit(X_reduced_train[:,:1], y_train)

    t_fin = time.time()

    #calculate RMSE
    pred = regr.predict(X_reduced_test)
    print("the mean squared error is : ",mean_squared_error(y_test, pred))
    print("it took this time for the training with PCA : ", t_fin-t_init)

    return regr,mean_squared_error(y_test, pred),t_fin-t_init

def SVM_regression(X,y):

    t_init = time.time()

    #We encode the continous data into variable
    lab = preprocessing.LabelEncoder()

    y_transformed = lab.fit_transform(y)


    #split the dataset into training (70%) and testing (30%) sets
    X_train,X_test,y_train,y_test = train_test_split(X,y_transformed,test_size=0.3, random_state=42) 

    #model
    #We arbitrary choose this kernel that gave us the best results during the lab
    clf = svm.SVC(kernel="poly",gamma=0.25,degree=2,coef0=0.8, C=1.0)
    clf.fit(X_train, y_train)

    t_fin = time.time()

    y_pred = clf.predict(X_test)
    y_pred = lab.inverse_transform(y_pred)
    y_test = lab.inverse_transform(y_test)

    #calculate RMSE
    print("the mean squared error is : ",mean_squared_error(y_test, y_pred))
    print("it took this time for the training  : ", t_fin-t_init)

    return clf,mean_squared_error(y_test, y_pred),t_fin-t_init


