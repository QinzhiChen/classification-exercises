#!/usr/bin/env python
# coding: utf-8

# In[1]:


# prep exercise
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import acquire
import warnings
import pandas as pd
df_iris=acquire.get_iris_data()

# In[2]:


df_iris.drop(columns=['species_id','measurement_id','Unnamed: 0'],inplace=True)



df_iris.rename(columns={'species_name':'species'},inplace=True)


dummy_df_iris = pd.get_dummies(df_iris.species, dummy_na=False, drop_first=[True])
df_iris=pd.concat([df_iris,dummy_df_iris],axis=1)


# In[ ]:


def prep_iris(df):
    df_iris=acquire.get_iris_data()
    df_iris.drop(columns=['species_id','measurement_id','Unnamed: 0'],inplace=True)
    df_iris.rename(columns={'species_name':'species'},inplace=True)
    dummy_df_iris= pd.get_dummies(df_iris.species, dummy_na=False, drop_first=[True])
    df_iris=pd.concat([df_iris,dummy_df_iris],axis=1)
    return df_iris


def clean_iris(df):
    df_iris=prep_iris(df)
    train, test = train_test_split(df_iris, test_size=.2, 
                               random_state=123, stratify=df_iris['versicolor'])
    n_train, validate = train_test_split(train, test_size=.25, 
                 random_state=123, stratify=train.versicolor)
    
    return n_train, validate, test



# In[ ]:


df_titanic=acquire.get_titanic_data()


# In[ ]:


df_titanic.drop(columns=['Unnamed: 0','embarked','age','deck','class'],inplace=True)


# In[ ]:


df_titanic


# In[ ]:


dummy_df_titanic = pd.get_dummies(df_titanic[['sex','embark_town']], dummy_na=False, drop_first=[True])
df_titanic=pd.concat([df_titanic ,dummy_df_titanic],axis=1)


# In[ ]:


def prep_titanic(df):
    df_titanic=acquire.get_titanic_data()
    df_titanic.drop(columns=['Unnamed: 0','embarked','age','deck','class'],inplace=True)
    dummy_df_titanic = pd.get_dummies(df_titanic[['sex','embark_town']], dummy_na=False, drop_first=[True])
    df_titanic=pd.concat([df_titanic ,dummy_df_titanic],axis=1)
    return df_titanic


# In[ ]:


def split_titanic(df):
    df_titanic=prep_titanic(df)
    train, test = train_test_split(df_titanic, test_size=.2, 
                               random_state=123, stratify=df_titanic['survived'])
    n_train, validate = train_test_split(train, test_size=.25, 
                 random_state=123, stratify=train.survived)
    
    return n_train, validate, test


# In[ ]:


prep_titanic(df_titanic)


# In[ ]:


df_telco=acquire.get_telco_data()


# In[ ]:


df_telco.drop(columns=['Unnamed: 0','internet_service_type_id','phone_service.1','multiple_lines.1','internet_service_type_id.1','online_security.1','online_backup.1','device_protection.1','contract_type_id','payment_type_id','tech_support.1','streaming_tv.1','streaming_movies.1','contract_type_id.1' ,'paperless_billing.1','payment_type_id.1','monthly_charges.1','total_charges.1','internet_service_type_id.2'],inplace=True)


# In[ ]:


df_telco


# In[ ]:


dummy_df_telco = pd.get_dummies(df_telco[['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type']], dummy_na=False, drop_first=[True])
df_telco=pd.concat([df_telco ,dummy_df_telco],axis=1)


# In[ ]:


df_telco.drop(columns=['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type'],inplace=True)


# In[ ]:


def prep_telco(df):
    df_telco=acquire.get_telco_data()
    df_telco.drop(columns=['Unnamed: 0','internet_service_type_id','phone_service.1','multiple_lines.1','internet_service_type_id.1','online_security.1','online_backup.1','device_protection.1','contract_type_id','payment_type_id','tech_support.1','streaming_tv.1','streaming_movies.1','contract_type_id.1' ,'paperless_billing.1','payment_type_id.1','monthly_charges.1','total_charges.1','internet_service_type_id.2'],inplace=True)
    dummy_df_telco = pd.get_dummies(df_telco[['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type']], dummy_na=False, drop_first=[True])
    df_telco=pd.concat([df_telco ,dummy_df_telco],axis=1)
    df_telco.drop(columns=['partner','dependents','phone_service','multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies','paperless_billing','churn','internet_service_type'],inplace=True)
    return df_telco


# In[ ]:





# In[ ]:


def split_telco(pre_telco):
    prep_telco=prep_telco(df)
    n_train, test = train_test_split(prep_telco, test_size=.2, 
                               random_state=123)
    train, validate = train_test_split(n_train, test_size=.25, 
                 random_state=123)
    
    return train, validate, test


# In[ ]:





# In[ ]:




