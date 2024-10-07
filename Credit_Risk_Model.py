#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os



# In[2]:


a1 = pd.read_excel(r"C:\Users\deepd\Credit Risk Model\case_study1.xlsx")
a2 = pd.read_excel(r"C:\Users\deepd\Credit Risk Model\case_study2.xlsx")


# In[3]:


df1 = a1.copy()
df2= a2.copy()


# In[ ]:


df2.info()


# In[4]:


# Remove nulls
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]




columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed .append(i)
        
        
df2 = df2.drop(columns_to_be_removed, axis =1)

for i in df2.columns:
    df2 = df2.loc[ df2[i] != -99999 ]


# In[ ]:


df2.info()


# In[5]:


df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )


# In[18]:


df.info()


# In[6]:


for i in df.columns:
    if df[i].dtype == 'object':
        print(i)


# In[7]:


for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)


# In[8]:


numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numeric_columns.append(i)
print (numeric_columns)


# In[9]:


vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0



for i in range (0,total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,'---',vif_value)
    
    if vif_value <= 6:
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)

    


# In[10]:


vif_data.info()


# In[11]:


from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)
        


# In[ ]:


print (columns_to_be_kept_numerical)


# In[12]:


features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

df = df[features + ['Approved_Flag']]


# In[13]:


df.loc[df['EDUCATION'] == 'SSC', ['EDUCATION']]             =1 
df.loc[df['EDUCATION'] == '12TH', ['EDUCATION']]            =2
df.loc[df['EDUCATION'] == 'GRADUATE', ['EDUCATION']]        =3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', ['EDUCATION']]  =3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', ['EDUCATION']]   =4
df.loc[df['EDUCATION'] == 'OTHERS', ['EDUCATION']]          =1 
df.loc[df['EDUCATION'] == 'PROFESSIONAL', ['EDUCATION']]    =3


# In[14]:


df['EDUCATION'] = df['EDUCATION'].astype(int)


# In[19]:


df.info()


# In[15]:


df_encoded = pd.get_dummies(df,columns = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])


# In[16]:


df_encoded.info()


# In[32]:


#Random Forest 244 
y=df_encoded ['Approved_Flag'] 
x = df_encoded. drop (['Approved_Flag'], axis = 1) 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators= 200, random_state=42) 
rf_classifier.fit(x_train, y_train) 

y_pred = rf_classifier.predict(x_test) 
accuracy = accuracy_score(y_test, y_pred) 
print() 
print(f'Accuracy: {accuracy}') 
print(f'Accuracy: {accuracy:.2f}') 
print () 

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)  


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']): 
        print(f"Class (v):") 
        print(f"Recall: {recall[i]}") 
        print(f"Precision: {precision[i]}") 
        print(f"F1 Score: {f1_score[i]}") 
        print() 


# In[27]:


# xgboost
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder 

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4)


y=df_encoded['Approved_Flag'] 
x= df_encoded. drop (['Approved_Flag'], axis=1) 

label_encoder= LabelEncoder() 
y_encoded = label_encoder.fit_transform(y) 
print () 

print(f'Accuracy: {accuracy}') 
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42) 

print() 

accuracy = accuracy_score(y_test, y_pred) 



xgb_classifier.fit(x_train, y_train) 
y_pred = xgb_classifier.predict(x_test)  
accuracy = accuracy_score(y_test, y_pred) 
print ()  
print(f'Accuracy: {accuracy:.2f}') 
print ()
                                 
# y = df_encoded ['Approved_Flag'] 
# x = df_encoded. drop (['Approved_Flag'], axis = 1) 
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# rf_classifier = RandomForestClassifier(n_estimators= 200, random_state=42) 
# rf_classifier.fit(x_train, y_train) 

# y_pred = rf_classifier.predict(x_test) 
# accuracy = accuracy_score(y_test, y_pred) 
# print() 
# print(f'Accuracy: {accuracy}') 
# print () 

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)  


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']): 
        print(f"Class (v):") 
        print(f"Precision: {precision[i]}")
        print(f"Recall: {recall[i]}") 
        print(f"F1 Score: {f1_score[i]}") 
        print() 


# In[31]:


#Decision Tree 339 
from sklearn.tree import DecisionTreeClassifier 
y = df_encoded ['Approved_Flag'] 
x = df_encoded. drop (['Approved_Flag'], axis = 1) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) 

dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10) 
dt_model.fit(x_train, y_train) 
                                                    
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) 
print ()  
print(f'Accuracy: {accuracy:.2f}') 
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)  


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']): 
        print(f"Class (v):") 
        print(f"Precision: {precision[i]}")
        print(f"Recall: {recall[i]}") 
        print(f"F1 Score: {f1_score[i]}") 
        print() 


# In[ ]:


#hypertuning

Define the hyperparameter grid 
param grid={ 
    colsample bytree: [0.1, 0.3, 0.5, 0.7, 0.9], 
    learning rate :   [0.001, 0.01, 0.1, 1], 
    max depth:        [3, 5, 8, 10], 
    alpha :            [1, 10, 100],
    n estimators : (10,50,100)   
    index= 0
    
    answers grid={ 
        'combination'    :[], 
        'train Accuracy':[],
        'test Accuracy' :[],
        'colsample bytree':[], 
        'learning rate' :[],
        'max depth':[],
        'alpha':[], 
        'n estimators':[]
    }
    
    
    for colsample bytree in param_grid['learning_rate']:
        for learning rate in param_grid['learning_rate']: 
            for max depth in param_grid['max_depth"]: 
                for alpha in param_grid['alpha']: 
                    for n_estimators in param_grid['n_estimators"]: 
                        index index + 1 

                        Define and train the XGBoost model 
    
                        model xgb.XGBClassifier(objective='multi:softmax', 
                                                num_class=4, 
                                                colsample_bytree=colsample_bytree, 
                                                learning_rate=learning_rate, 
                                                max_depth=max_depth, 
                                                alpha-alpha, 
                                                n_estimators=n_estimators)  
    
                                                   
y = df_encoded [Approved_Flag"] 
x = df_encoded. drop (['Approved Flag'], axis 1) 
label encoder LabelEncoder() 
                y_encoded = label_encoder.fit_transform(y) 
                x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random=42) 
                
                
                model.fit(x_train, y_train) 
                
                
                Predict on training and testing setS 
                            
                y_pred_train = model.predict(x_train)
                y_pred_test = model.predict(x_test)

