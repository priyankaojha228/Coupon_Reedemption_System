#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic(u'matplotlib inline')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ## Reading the .csv files from the dataset

# In[2]:


train=pd.read_csv('train.csv')
campaign_data=pd.read_csv('campaign_data.csv')
coupon_item_mapping=pd.read_csv('coupon_item_mapping.csv')
customer_demographics=pd.read_csv('customer_demographics.csv')
customer_transaction_data=pd.read_csv('customer_transaction_data.csv')
ctd=customer_transaction_data[0:50000]#taken only 50000 rows because I was getting memory error otherwise as it contained 1.324566e+06 rows which was out of the memory of my laptop.
item_data=pd.read_csv('item_data.csv')


# ## Merging all the dataframes together into one single dataframe 

# In[3]:


r1=pd.merge(train,campaign_data,on='campaign_id')
r2=pd.merge(r1,coupon_item_mapping,on='coupon_id')
r3=pd.merge(r2,customer_demographics,on='customer_id')
r5=pd.merge(r3,item_data,on='item_id')
r5.dropna(inplace=True)
result=pd.merge(r5,ctd,on='item_id')
result.info()


# ## Visualizing the data

# In[4]:


result.hist(bins=50,figsize=(20,15))


# In[5]:


sns.countplot(x="redemption_status",data=result)


# In[6]:


sns.countplot(x="redemption_status",hue="campaign_id",data=result)


# In[7]:


sns.countplot(x="redemption_status",hue="category",data=result)


# In[8]:


sns.countplot(x="redemption_status",hue="campaign_type",data=result)


# In[9]:


sns.countplot(x="redemption_status",hue="brand_type",data=result)


# In[10]:


sns.countplot(x="redemption_status",hue="age_range",data=result)


# In[11]:


sns.countplot(x="redemption_status",hue="marital_status",data=result)


# In[12]:


sns.countplot(x="redemption_status",hue="income_bracket",data=result)


# ## Data preprocessing and cleaning

# In[13]:


result.head()


# In[14]:


#checking if any of the features has null value 
result.isnull().sum()


# ## Data aggregation and Feature Engineering

# In[15]:


#I checked the features which had values in character and string format and converted them to numerical categorical 
#format using pd.getdummies() function. Concatenated it with the complete dataframe and removed the earlier features 
#which were now converted in categorical format.
camp_type_x=pd.get_dummies(result['campaign_type'],drop_first=True)
marital_status_s=pd.get_dummies(result['marital_status'],drop_first=True)
brandtype_e=pd.get_dummies(result['brand_type'],drop_first=True)
category_b=pd.get_dummies(result['category'],drop_first=True)
ar=pd.get_dummies(result['age_range'],drop_first=True)


# In[16]:


pd.concat([result,camp_type_x,marital_status_s,brandtype_e,category_b,ar],axis=1)


# ## Final complete dataframe after cleaning and aggregation

# In[17]:


result.drop("date",axis=1,inplace=True)
result.drop("customer_id_y",axis=1,inplace=True)
result.drop("quantity",axis=1,inplace=True)
result.drop("selling_price",axis=1,inplace=True)
result.drop("other_discount",axis=1,inplace=True)
result.drop("coupon_discount",axis=1,inplace=True)
result.drop(['campaign_type','marital_status','start_date','end_date','brand_type','category','age_range','family_size','no_of_children'],axis=1,inplace=True)


# ## Validation or train_test_split

# In[18]:


x=result.drop("redemption_status",axis=1)
y=result["redemption_status"]


# In[19]:


#Splitted the trainign and the testing set in 70:30 ratio 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=2)


# ## Training using Logistic Regression 

# In[33]:


logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)


# ## Accuracy, ROC AUC score in %  and Roc curve but the model is underfitted

# In[34]:


predictions = logmodel.predict(X_test)
print ("Accuracy score :",accuracy_score(y_test,predictions)*100)
lr_auc = roc_auc_score(y_test, predictions)
print('Logistic: ROC AUC=%.3f' % (lr_auc*100))
lr_fpr, lr_tpr, _ = roc_curve(y_test, predictions)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ## After removing underfitting final accuracy score and ROC curve

# In[24]:


#Removed underfitting by changing some of the parameters as shown below
logmodel=LogisticRegression(class_weight='balanced', random_state=42, multi_class='auto')
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print ("Accuracy score :",accuracy_score(y_test,predictions)*100)
lr_fpr, lr_tpr, _ = roc_curve(y_test, predictions)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ## Final ROC AUC score after removing underfitting in logistic regression

# In[27]:


lr_auc = roc_auc_score(y_test, predictions)
print('Logistic: ROC AUC=%.3f' % (lr_auc*100))


# ## Training using Decision Tree Classifier

# In[29]:


clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)


# ## Accuracy, ROC AUC score in %  and Roc curve but the model is overfitted

# In[30]:


predictions = clf.predict(X_test)
print ("Accuracy score :",accuracy_score(y_test,predictions)*100)
lr_auc = roc_auc_score(y_test, predictions)
print('Decision Trees : ROC AUC=%.3f' % (lr_auc*100))
lr_fpr, lr_tpr, _ = roc_curve(y_test, predictions)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ## After removing overfitting final accuracy score and ROC curve

# In[31]:


#Removed overfitting by changing some of the parameters as shown below
clf=DecisionTreeClassifier(min_samples_split=7, random_state=42, max_depth=None,max_features=None, max_leaf_nodes=29)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print ("Accuracy score :",accuracy_score(y_test,predictions)*100)
lr_fpr, lr_tpr, _ = roc_curve(y_test, predictions)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Decision Tree')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# ## Final ROC AUC score after removing overfitting in Decision trees

# In[32]:


lr_auc = roc_auc_score(y_test, predictions)
print('Decision Tree: ROC AUC=%.3f' % (lr_auc*100))


# In[ ]:




