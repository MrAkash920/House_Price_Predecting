#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df = pd.read_csv('Housing.csv')
df.head(20)


# In[37]:


print(df.shape) #print shape of the dataset


# In[38]:


df.dropna()  #to drop null values


# In[39]:


#Count Null values
total_na = df.isna().sum().sum()
print(total_na)


# In[40]:


import seaborn as sns
sns.boxplot(df['price'])


# In[41]:


#histogram of house price
df['price'].plot(kind = 'hist')
plt.show()


# In[42]:


bedroom_groups = df.groupby('bedrooms')['price']
price_stats = bedroom_groups.agg(['mean', 'median', 'count','std'])
print(price_stats)


# In[43]:


# Bar chart of mean prices
plt.figure(figsize=(6, 4))
plt.bar(price_stats.index, price_stats['mean'])
plt.xlabel('Number of Bedrooms')
plt.ylabel('Mean Price')
plt.title('Mean Price by Number of Bedrooms')
plt.show()


# In[44]:


plt.figure(figsize=(6,6))
plt.bar(price_stats.index,price_stats['count'])
plt.xlabel("Number of Beadrooms")
plt.ylabel("Count Price")
plt.title('Count Price by Number of Beadrooms')
# Annotate each bar with its count
for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords='offset points',
                 ha='center', va='bottom')

plt.show()
plt.show()


# In[50]:


#barplot using saeborn
sns.barplot(data=df, x="bedrooms", y="bathrooms", hue="stories")
plt.xlabel("Beadrooms")
plt.ylabel("Bathrooms")
plt.show()


# In[52]:


#pair plot for multiple variables
plt.figure(figsize=(20,12))
sns.pairplot(df[['price', 'area', 'bedrooms', 'bathrooms']])
plt.title('Pair Plot of Numerical Variables')
plt.show()


# In[55]:


#Price Distribution by Furnishing Status using boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='furnishingstatus', y='price', data=df)
plt.xlabel('Furnishing Status')
plt.ylabel('Price')
plt.title('Price Distribution by Furnishing Status')
plt.xticks(rotation=0)
plt.show()


# In[58]:


# Create a DataFrame subset with the selected columns
subset = df[['price', 'area', 'bedrooms', 'bathrooms', 'stories']]

# Calculate the correlation matrix for the selected columns
correlation_matrix = subset.corr()

# Create a heatmap to visualize the correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[77]:


# Numerical features
numerical_features = ['bedrooms', 'bathrooms', 'area', 'stories','parking']

# Categorical features
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Select the subsets of data
X_numerical = df[numerical_features]
X_categorical = df[categorical_features]
y = df['price']


# In[78]:


# Apply one-hot encoding to categorical features
X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)


# In[79]:


# Concatenate numerical and encoded categorical features
X_combined = pd.concat([X_numerical, X_categorical_encoded], axis=1)


# In[80]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)


# In[81]:


#training the model using linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# In[82]:


#Making prediction
y_pred = model.predict(X_test)


# In[85]:


# Create a DataFrame with the same column names as your original dataset
new_house = pd.DataFrame({
    'bedrooms': [2],
    'bathrooms': [2.5],
    'area': [6000],
    'stories': [2],
    'mainroad': ['no'],
    'guestroom': ['yes'],
    'basement': ['no'],
    'hotwaterheating': ['yes'],
    'airconditioning': ['yes'],
    'parking': [1],
    'prefarea': ['yes'],
    'furnishingstatus': ['furnished']
})

# Apply one-hot encoding to categorical features
new_house_encoded = pd.get_dummies(new_house, columns=categorical_features, drop_first=True)

# Ensure the order and names of columns match the training data
new_house_encoded = new_house_encoded.reindex(columns=X_combined.columns, fill_value=0)

# Make predictions using the trained model
predicted_price = model.predict(new_house_encoded)

print("Predicted Price:", predicted_price[0])


# In[ ]:




