#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[22]:


pd.set_option('display.max_columns', None)


# In[27]:


df1 = pd.read_csv(r"C:\Users\HP\Downloads\zomato (1).csv",encoding='latin-1')


# In[28]:


df1


# In[29]:


df1.head()


# In[30]:


df1.shape


# In[33]:


df_country = pd.read_excel(r"C:\Users\HP\Downloads\Country-Code.xlsx")
df_country.head()


# In[34]:


data=pd.merge(df1,df_country,on='Country Code',how='left')


# In[35]:


data.head()


# In[36]:


data.shape


# In[37]:


data.drop_duplicates()
data.shape


# In[38]:


data.nunique()


# In[39]:


data.isnull().sum()


# In[40]:


# visualising null values in dataset
sns.heatmap(data.isnull())


# In[41]:


sns.countplot(x='Country Code',data=data)
plt.show()


# In[42]:


sns.countplot(x='Price range',data=data)
plt.show()


# In[43]:


sns.countplot(y='Currency',data=data)
plt.show()


# In[44]:


sns.countplot(x='Has Table booking',data=data)
plt.show()


# In[45]:


sns.countplot(x='Has Online delivery',data=data)
plt.show()


# In[46]:


## Which countries do have online deliveries option
a=data[data['Has Online delivery']=='Yes'].Country.value_counts()
a.plot.pie(autopct = '%1.1f%%')
plt.title('Country has online delivery')
plt.show()


# In[47]:


sns.countplot(x='Is delivering now',data=data)
plt.show()


# In[48]:


plt.figure(figsize=(15,7))
sns.countplot(x='Aggregate rating',data=data)
plt.show()


# In[49]:


sns.countplot(x='Rating color',data=data,)
plt.show()


# In[50]:


sns.countplot(x='Rating text',data=data,)
plt.show()


# In[51]:


plt.figure(figsize=(20,5))
sns.countplot(x='Country',data=data)
plt.show()


# In[52]:


city_values = data.City.value_counts().values
city_labels = data.City.value_counts().index
plt.pie(city_values[:5],labels=city_labels[:5],autopct='%1.2f%%')
plt.show()


# In[53]:


# Find top 10 Cuisine
cuisin_val=data.Cuisines.value_counts().values
cuisin_label = data.Cuisines.value_counts().index
plt.pie(cuisin_val[:10],labels=cuisin_label[:10],autopct='%1.2f%%')
plt.show()


# In[54]:


# Find top 10 location
val=data.Locality.value_counts().values
label = data.Locality.value_counts().index
plt.pie(val[:10],labels=label[:10],autopct='%1.2f%%')
plt.show()


# In[55]:


sns.distplot(data['Average Cost for two'])
plt.show()


# In[56]:


plt.figure(figsize=(15,6))
df_good = data.sort_values(by="Average Cost for two",ascending=False)
sns.barplot(x="Country",y="Average Cost for two",data=df_good)
plt.show()


# In[57]:


plt.figure(figsize=(20,6))
df_good = data.sort_values(by="Average Cost for two",ascending=False).iloc[0:10]
sns.barplot(x="Restaurant Name",y="Average Cost for two",data=df_good)
plt.show()


# In[58]:


df_good = data.sort_values(by="Average Cost for two",ascending=False).iloc[0:10]
sns.barplot(y="City",x="Average Cost for two",data=df_good)
plt.show()


# In[59]:


sns.boxplot(x='Has Online delivery',y='Votes',data=data)
plt.title('Has Online delivery VS Votes')
plt.show()


# In[60]:


data.groupby('Restaurant Name')['Aggregate rating'].mean().nlargest(10).plot.bar()
plt.show()


# In[61]:


sns.boxplot(x='Has Online delivery',y='Average Cost for two',data=data)
plt.title('Has Online delivery VS Average Cost for two')
plt.show()


# In[62]:


data.groupby('Restaurant Name')['Votes'].mean().nlargest(10).plot.bar()
plt.show()


# In[63]:


sns.scatterplot(x='Average Cost for two',y='Longitude',data=data)
plt.show()


# In[64]:


sns.scatterplot(x='Average Cost for two',y='Latitude',data=data)
plt.show()


# In[65]:


rating=data.groupby(['Aggregate rating','Rating color','Rating text']).size().reset_index().rename(columns={0:'Rating count'})
plt.figure(figsize=(12,6))
sns.barplot(x='Aggregate rating',y='Rating count',data=rating)
plt.show()


# In[66]:


plt.figure(figsize=(10,6))
df_good = data.sort_values(by="Latitude",ascending=False).iloc[0:10,:]
sns.barplot(y="Restaurant Name",x="Latitude",data=df_good)
plt.show()


# In[67]:


sns.barplot(x='Price range',y='Average Cost for two',data=data)
plt.show()


# In[68]:


sns.barplot(x='Rating text',y='Average Cost for two',data=data)
plt.show()


# In[69]:


plt.figure(figsize=(12,6))
sns.barplot(x='Aggregate rating',y='Rating text',data=rating)
plt.show()


# In[70]:


# Find the countrries name that has given 0 rating
data[data['Rating color']=='White'].groupby(['Aggregate rating','Country']).size().reset_index()


# In[71]:


Delhi = data[(data.City == 'New Delhi')]
top_locality = Delhi.Locality.value_counts().head(10)
plt.figure(figsize=(12,6))
sns.countplot(y= "Locality", hue="Has Online delivery", data=Delhi[Delhi.Locality.isin(top_locality.index)])
plt.title('Resturants Online Delivery')
plt.show()


# In[72]:


sns.countplot(x='Has Online delivery',hue='Has Table booking',data=data)
plt.show()


# In[73]:


sns.countplot(x='Rating color',hue='Price range',data=data)
plt.show()


# In[74]:


sns.scatterplot(x='Votes',y='Average Cost for two',data=data)
plt.show()


# In[75]:


sns.scatterplot(x='Votes',y='Aggregate rating',data=data)
plt.show()


# In[76]:


plt.figure(figsize=(10,7))
sns.scatterplot(x='Aggregate rating',y='Average Cost for two',hue='Has Online delivery',data=data)
plt.show()


# In[77]:


plt.figure(figsize=(12,6))
sns.barplot(x='Aggregate rating',y='Rating count',hue='Rating color',data=rating, palette=['blue','red','orange','yellow','green','darkgreen'])
plt.show()


# In[78]:


plt.figure(figsize=(15,5))
sns.boxplot(x="Country", y="Average Cost for two", hue="Has Online delivery",data=data)
plt.show()


# In[79]:


plt.figure(figsize=(12,6))
sns.scatterplot(x="Average Cost for two", y="Aggregate rating", hue='Price range', data=Delhi)

plt.xlabel("Average Cost for two")
plt.ylabel("Aggregate rating")
plt.title('Rating vs Cost of Two')
plt.show()


# In[80]:


sns.scatterplot(x='Votes',y='Aggregate rating',data=data,hue='Has Online delivery')
plt.show()


# In[81]:


data.drop(columns=['Restaurant ID','Locality Verbose'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for col in data[data.columns[data.dtypes == 'object']]:
    data[col] = le.fit_transform(data[col])
data.head()


# In[82]:


data.describe()


# In[83]:


data.corr()['Average Cost for two'].sort_values()


# In[84]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True, fmt ='.2f')
plt.show()


# In[85]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), annot=True, fmt ='.2f')
plt.show()


# In[86]:


data.shape


# In[87]:


plt.figure(figsize=(25,30))
plotnumber = 1

for column in data:
    if plotnumber <=20:
        ax = plt.subplot(5,4,plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=20)
    plotnumber +=1
plt.show()


# In[88]:


data.skew()


# In[89]:


plt.figure(figsize=(25,30))
plotnumber = 1

for column in data:
    if plotnumber <=20:
        plt.subplot(5,4,plotnumber)
        ax = sns.boxplot(data=data[column])
        plt.xlabel(column, fontsize=20)
    plotnumber +=1
plt.show()


# In[90]:


data.head(1)


# In[91]:


from scipy.stats import zscore

z_score = zscore(data[['Average Cost for two','Votes']]) # Only removing outliers from continuous data
abs_z_score = np.abs(z_score)    # Apply the formula and get the scaled data

filtering_entry = (abs_z_score  < 3).all(axis=1)

df = data[filtering_entry]


# In[92]:


df.shape


# In[93]:


data.shape


# In[94]:


data_loss = ((9551 - 9362)/9551*100)
print(data_loss,'%')


# In[95]:


x = df.drop(columns=['Average Cost for two'],axis=1)
y = df['Average Cost for two']


# In[96]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaled_X = scaler.fit_transform(x)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["Features"] = x.columns
vif['vif'] = [variance_inflation_factor(scaled_X,i) for i in range(scaled_X.shape[1])]
vif


# In[97]:


df.drop(columns=['Country Code','Switch to order menu','Country'],axis=1,inplace=True)
df.shape


# In[98]:


x = df.drop(columns=['Average Cost for two'],axis=1)
y = df['Average Cost for two']


# In[99]:


from sklearn.feature_selection import SelectKBest, f_classif
bestfeat = SelectKBest(score_func = f_classif, k = 'all')
fit = bestfeat.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
fit = bestfeat.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
dfcolumns.head()
featureScores = pd.concat([dfcolumns,dfscores],axis = 1)
featureScores.columns = ['Feature', 'Score']
print(featureScores.nlargest(35,'Score'))


# In[100]:


x_best = x.drop(columns=['Restaurant Name','Is delivering now']).copy()


# In[101]:


data.skew()


# In[102]:


from sklearn.preprocessing import power_transform
x = power_transform(x_best,method='yeo-johnson')
trans = pd.DataFrame(x)
trans.skew()


# In[103]:


x = scaler.fit_transform(x)
x


# In[ ]:




