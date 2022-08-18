#!/usr/bin/env python
# coding: utf-8

# prep exercise
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import acquire
import warnings
import pandas as pd

# iris data
def prep_iris(df):
    df_iris=acquire.get_iris_data()
    df_iris.drop(columns=['species_id','measurement_id','Unnamed: 0'],inplace=True)
    df_iris.rename(columns={'species_name':'species'},inplace=True)
    dummy_df_iris= pd.get_dummies(df_iris.species, dummy_na=False, drop_first=[True])
    df_iris=pd.concat([df_iris,dummy_df_iris],axis=1)
    return df_iris

def split_iris(df):
    df_iris=prep_iris(df)
    train, iris_test = train_test_split(df_iris, test_size=.2, 
                               random_state=123, stratify=df_iris['versicolor'])
    iris_train, iris_validate = train_test_split(train, test_size=.25, 
                 random_state=123, stratify=train.versicolor)
    
    return iris_train, iris_validate, iris_test


# titanic data

def prep_titanic(df):
    df_titanic=acquire.get_titanic_data()
    df_titanic.drop(columns=['Unnamed: 0','embarked','age','deck','class'],inplace=True)
    dummy_df_titanic = pd.get_dummies(df_titanic[['sex','embark_town']], dummy_na=False, drop_first=[True])
    df_titanic=pd.concat([df_titanic ,dummy_df_titanic],axis=1)
    return df_titanic


def split_titanic(df):
    df_titanic=prep_titanic(df)
    train, titanic_test = train_test_split(df_titanic, test_size=.2, 
                               random_state=123, stratify=df_titanic['survived'])
    titanic_train, titanic_validate = train_test_split(train, test_size=.25, 
                 random_state=123, stratify=train.survived)
    
    return titanic_train, titanic_validate, titanic_test

# telco data

def prep_telco(df):
    df_telco=acquire.get_telco_data()
    df_telco.drop(columns=['internet_service_type_id','contract_type_id','payment_type_id',
                       ],inplace=True)
    dummy_df_telco = pd.get_dummies(df_telco[['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type']], dummy_na=False, drop_first=[True])
    df_telco=pd.concat([df_telco ,dummy_df_telco],axis=1)
    df_telco.drop(columns=['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type'],inplace=True)
    return df_telco


def split_telco(df):
    df_telco=prep_telco(df)
    train, telco_test = train_test_split(df_telco, test_size=.2, 
                               random_state=123)
    telco_train, telco_validate = train_test_split(train, test_size=.25, 
                 random_state=123)
    
    return telco_train, telco_validate, telco_test





# %%
