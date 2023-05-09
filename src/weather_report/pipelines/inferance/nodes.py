import logging
from typing import Dict,Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
# import os 
# import pickle
# pickled_model = pickle.load(open("logreg.pkl"))

def inferance_data(df:pd.DataFrame):
  """Drop date column from dataframe.
  Arg=df
  output=Dataframe with no date column."""
  df=df.drop(["Date"],axis=1)
  df_inferance=df[df['RainTomorrow'].isna()] 
  return df_inferance

def treat_missing_inferance(df_inferance:pd.DataFrame):
  """treat missing values with ffill and bfill method

  Arg=df_training_data

  output=nan values from training data get fill with the ffill and bfill method."""
  df1_treat_inferance_data=df_inferance.fillna(method="ffill",axis=0).fillna(method="bfill",axis=0)
  return df1_treat_inferance_data

def inferance_data_split(df1_treat_inferance_data:pd.DataFrame):
  """Training data spliting ie.separating predictors and response variables.

  Arg=df_treat_training_data

  Output=Target variable get separated from main traing data."""
  X_inferance=df1_treat_inferance_data.drop(["RainTomorrow"],axis=1)
  y_inferance=df1_treat_inferance_data["RainTomorrow"]
  return X_inferance, y_inferance

def lebel_encoding_inferance(X_inferance:pd.DataFrame):
  """Lebal encoding on the discrite varibales.

  Arg=X_training

  Output=converting categorical data into numeric variable."""
  
  label_encoder = preprocessing.LabelEncoder()
  
  X_inferance["Location"]=label_encoder.fit_transform(X_inferance["Location"])
  X_inferance["WindGustDir"]=label_encoder.fit_transform(X_inferance["WindGustDir"])
  X_inferance["WindDir9am"]=label_encoder.fit_transform(X_inferance["WindDir9am"])
  X_inferance["WindDir3pm"]=label_encoder.fit_transform(X_inferance["WindDir3pm"])
  X_inferance["RainToday"]=label_encoder.fit_transform(X_inferance["RainToday"])
  df2_inferance = X_inferance.copy()
  return df2_inferance


def inferance_prediction(logreg, df2_inferance:pd.DataFrame):
  """calculate and logs the coefficient of determination.
  Args:
      logreg:Trined model
      X_test:Testing data of independent features
      y_test:Testing data for price
  Returns:
      y_pred_test:prediction on x test
      acc        :accuracy_score
  """
  y_pred_inferance=logreg.predict(df2_inferance)
  #Check accuracy score
  y_pred_inferance=pd.DataFrame(y_pred_inferance)
  return y_pred_inferance


def final_report(df_inferance, y_pred_inferance):
  """creating a final project dataset
  Args:
      Inferance_data
      y_pred_inferance
  output: Dataset  
  """
  result_Dataset = pd.concat([df_inferance,y_pred_inferance],axis=1)
  result_Dataset=pd.DataFrame(result_Dataset)
  return result_Dataset
  