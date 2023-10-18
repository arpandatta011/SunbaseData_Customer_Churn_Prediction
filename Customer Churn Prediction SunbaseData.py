#!/usr/bin/env python
# coding: utf-8

# ## Assignment Title : Customer Churn Prediction

# In[1]:


#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# In[2]:


#Loading the path
customar_data = pd.read_excel("customer_churn_large_dataset.xlsx")


# In[3]:


# display the table
customar_data


# In[4]:


customar_data.shape


# ##  Data Preprocessing

# In[5]:


# Checking null values
customar_data.isnull().sum()


# In[6]:


# Check the description of customar data table
customar_data.describe()


# In[7]:


customar_data.info()


# In[8]:


# drop two column customar id and name in this table
final_result = customar_data.drop(columns = ['CustomerID', 'Name'])


# In[9]:


#check duplicated value
customar_data.duplicated().sum()


# ## Age

# In[10]:


plt.figure(figsize=[10,8])
plt.hist(final_result['Age'], bins=50, edgecolor='pink', linewidth=2) 
plt.xlabel('Age', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Age', fontsize=20)
plt.show()


# ## Total Gender counting

# In[11]:


sns.set(style="darkgrid") 
plt.figure(figsize=(10, 8))
sns.countplot(data=final_result, x='Gender', color="salmon", facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("BrBG", 2))
plt.xlabel('Gender', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Count of Gender', fontsize=20)
plt.xticks(fontsize=18)
plt.show()


# ## Location

# In[12]:


sns.set(style="whitegrid") 
plt.figure(figsize=(16, 7))  
sns.countplot(data=customar_data, x='Location', order=final_result['Location'].value_counts().index, palette = "Set2")
plt.xlabel('Location', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Location',fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=18)  
plt.tight_layout()  
plt.show()


# ## Subscription Length  

# In[13]:


sorted_lengths = sorted(customar_data['Subscription_Length_Months'].unique())

# Create a countplot for the 'Subscription_Lengths_Months' using Seaborn
sns.set(style="whitegrid")  
plt.figure(figsize=(16, 9)) 

sns.countplot(data=customar_data, x='Subscription_Length_Months', order=sorted_lengths)
plt.xlabel('Subscription Length (Months)', fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title('Subscription Length', fontsize=20)
plt.xticks(rotation=45, ha='right',fontsize=16) 
plt.tight_layout() 
plt.show()


# ## Monthly Bills

# In[14]:


fig = px.histogram(customar_data, x='Monthly_Bill', title='Distribution of Monthly Bills',
                   histfunc='count', nbins=20)


fig.update_traces(marker=dict(line=dict(color='black', width=1)))
fig.update_xaxes(title='Monthly Bill')
fig.update_yaxes(title='Count')
fig.show()


# ## Total Usage GB

# In[15]:


plt.figure(figsize=(16, 9))
plt.hist(final_result['Total_Usage_GB'], bins=10, color = "lightblue", ec="red")
plt.xlabel('Total_Usage_GB', fontsize=20)
plt.ylabel('Frequency', fontsize=20)
plt.title('Total_Usage_GB', fontsize=20)
plt.show()


# ## Churn Count

# In[16]:


fig = px.bar(customar_data['Churn'].value_counts(), x=customar_data['Churn'].value_counts().index, y='Churn',
             color=['Black', 'Red'], labels={'x': 'Churn', 'y': 'Count'})

fig.update_layout(xaxis={'categoryorder': 'total descending'}) 
fig.show()


# ## Location Vs Monthly Bill

# In[17]:


plt.figure(figsize=(16, 9))
plt.boxplot([customar_data[customar_data['Location'] == loc]['Monthly_Bill'] for loc in customar_data['Location'].unique()],
            labels=customar_data['Location'].unique(), vert=False)


plt.xlabel('Monthly Bill', fontsize=20)
plt.ylabel('Location', fontsize=20)
plt.title('Location vs Monthly Bill', fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Using BI-Variate Analysis to find 
# 
# ## Relation Between Location And Total Usage GB

# In[18]:


plt.figure(figsize=(16, 9))  

plt.scatter(customar_data['Location'], customar_data['Total_Usage_GB'],color = '#88c999', alpha=0.5)
plt.xlabel('Location', fontsize=20)
plt.ylabel('Total Usage (GB)', fontsize=20)
plt.title('Location vs Total Usage', fontsize=20)
plt.xticks(rotation=40, fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Age Vs Monthly Bill

# In[19]:


fig = px.scatter(customar_data, x='Age', y='Monthly_Bill', title='Age vs Monthly Bill')
fig.update_layout(xaxis_title='Age', yaxis_title='Monthly Bill')
fig.show()


# ## Age Vs Total Usage GB

# In[20]:


plt.figure(figsize=(16, 19))
sns.jointplot(data=customar_data, x='Age', y='Total_Usage_GB', kind='hex', cmap='viridis')
plt.xlabel('Age',fontsize=20)
plt.ylabel('Total Usage (GB)',fontsize=20)
plt.title('Hexbin Plot of Age vs Total Usage',fontsize=20)
plt.colorbar(label='Density')
plt.tight_layout()
plt.show()


# Here in this picture , we see about that there is not relation between two different picture column (Total Usage GB) column.

# ## Correlation

# In[21]:


correlation_matrix = final_result.corr()

fig = px.imshow(correlation_matrix, color_continuous_scale='Mint',
                title='Correlation Heatmap')
fig.update_layout(coloraxis_colorbar=dict(title='Correlation'))
fig.show()     


# ## Outliers

# In[22]:


Q1 = final_result.quantile(0.25)
Q3 = final_result.quantile(0.75)

# Calculate the IQR(Interquartile Range) for each column
IQR = Q3 - Q1

# Identify potential outliers 
outliers = ((final_result < (Q1 - 1.5 * IQR)) | (final_result > (Q3 + 1.5 * IQR)))
print(outliers.any())


# In[23]:


numeric_columns = final_result.select_dtypes(include=['int', 'float']).columns


# In[24]:


plt.figure(figsize=(15, 8))
for column in numeric_columns:
    plt.subplot(2, 3, numeric_columns.get_loc(column) + 1)
    sns.boxplot(data=final_result[column])
    plt.title(f'Violin Plot of {column}')
plt.xticks(fontsize=18)
plt.tight_layout()
plt.show()


# ## Using Feature Engineering

# In[25]:



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

second_final_result = {
    'Age_Bin': ['0-20', '21-40', '21-40', '41-60', '61+'],
    'Churn': [0, 0, 1, 0, 1] }

second_final_result = pd.DataFrame(second_final_result)


# ## Relationship Age Bin and Churn

# In[26]:


plt.figure(figsize=(18, 9))
sns.countplot(data=second_final_result, x='Age_Bin', hue='Churn',palette = "Set2")
plt.xlabel('Age Bin',fontsize=20)
plt.ylabel('Count',fontsize=20)
plt.title('Relationship between Age Bin and Churn',fontsize=20)
plt.xticks(rotation=40, fontsize = 18)
plt.tight_layout()
plt.legend(title='Churn', labels=['Not Churned', 'Churned'])
plt.show()


# ## Normalization

# In[27]:


scaler = MinMaxScaler()
normalized_age = scaler.fit_transform(final_result[['Age']])
normalized_total_usage = scaler.fit_transform(final_result[['Total_Usage_GB']])

print("Normalized Age:")
print(normalized_age)

print("Normalized Total Usage GB:")
print(normalized_total_usage)


# ## Model Building

# In[28]:


# Divide the data in `features` and `target`
X = final_result.iloc[:, :-1]
y = final_result.iloc[:, -1]


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[31]:


# Dividing the features in categorical and numerical features
Numerical_Features = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']
Categorical_Features = ['Gender', 'Location']


# In[32]:


# Create StandardScalar and `OneHotEncoder` Object
ohe = OneHotEncoder()
ss = StandardScaler()


# In[33]:


# Create ColumnTransformer Object for Preprocessing
preprocesser = ColumnTransformer(transformers=(('encode_gender', ohe, Categorical_Features),
                                               ('standardization', ss, Numerical_Features)))


# In[34]:


clf = Pipeline(steps=(('preprocessing', preprocesser),('classifier', LogisticRegression())
                     ))


# In[36]:


clf.fit(X_train, y_train)
print("Accuracy of Logistic Regression: ", clf.score(X_test, y_test))


# In[37]:


# Check score using metrics like Precision Score, Recall Score, F1 Score
y_pred = clf.predict(X_test)

print("The precision score of Logistic Regression is: ", precision_score(y_test, y_pred))
print("The recall score of Logistic Regression is: ", recall_score(y_test, y_pred))
print("The F1 score of Logistic Regression is: ", f1_score(y_test, y_pred))


# In[38]:


# using Model Pipeline for `RandomForestClassifier`
clf1 = Pipeline(steps=[ ('preprocessing', preprocesser), ('classifier', RandomForestClassifier(random_state = 42))
                          ])


# In[40]:


clf1.fit(X_train, y_train)
print("Accuracy score of Random Forest Classifier:", clf1.score(X_test, y_test))


# In[41]:


# Check score using other metrics like Precision Score, Recall Score, F1 Score..
y_pred = clf1.predict(X_test)

print("The precision score of Logistic Regression is: ", precision_score(y_test, y_pred))
print("The recall score of Logistic Regression is: ", recall_score(y_test, y_pred))
print("The F1 score of Logistic Regression is: ", f1_score(y_test, y_pred))
     


# ## Using Neural Network

# In[42]:


features = preprocesser.fit_transform(X_train)
targets = y_train


# In[43]:


# Build model
model = keras.Sequential(layers=[
    keras.layers.Dense(units=64, activation="relu", input_shape=(features.shape[1], )),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=128, activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=64, activation="relu"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=1, activation="sigmoid")
])


# In[44]:


# Add `Optimizer` and `Loss` function
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# In[45]:


model.summary()


# In[46]:


# Train the model
model.fit(x=features, y=targets, batch_size=1000, epochs=100, validation_split=0.2)


# In[47]:


#Model Testing
test_features = preprocesser.transform(X_test)
test_targets = y_test

model.evaluate(test_features, test_targets)


# ## Model deployment and integration

# In[48]:


# Deploy the model using `pickle` module
import pickle

pickle.dump(clf, open("model.pkl", 'wb'))


# In[49]:


# Loading the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))


# In[ ]:





# In[ ]:




