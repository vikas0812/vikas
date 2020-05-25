#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('emails1(1).csv')
df


# In[3]:


pd.array(df["Class"])


# In[4]:


df["Class"].value_counts()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df["Class"].hist(figsize=(30,10))


# In[9]:


sns.pairplot(df,hue="Class")


# In[10]:


sns.countplot('Class',data=df)


# In[11]:


df.drop_duplicates(keep='first')


# In[12]:


from textblob import TextBlob

polarity = lambda x: TextBlob(x).sentiment.polarity
subjectivity = lambda x: TextBlob(x).sentiment.subjectivity


# In[13]:


df['polarity'] = df['content'].apply(polarity)
df['subjectivity'] = df['content'].apply(subjectivity)
df


# In[14]:


data=df.drop(df.columns[[0,1,2]],axis=1)
data


# In[15]:


#Data cleaning and preprocessing
#re=Regular Expression
import re


# In[16]:


# when ever ur using stopwords,lematization,bag of words
import nltk


# In[17]:


nltk.download('stopwords')


# In[18]:


#StopWords are used to remove words that are of no use [to,of,for,etc..]
from nltk.corpus import stopwords


# In[19]:


#PorterStemmer is used for Stemming and stemming is used to find the base root of the word
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[20]:


corpus=[]


# In[21]:


for i in range(0 ,len(df['content'])):
    review=re.sub('[^a-zA-Z]',' ',df['content'][i])#remove all other than a-zA-Z
    review=review.lower()
    review=review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[22]:


corpus


# In[23]:


#train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(corpus,df.Class,test_size=0.3)


# In[24]:


#TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()


# # Naive Baye's Classifier

# In[25]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[26]:


X_train_transformed=vectorizer.fit_transform(X_train)
X_test_transformed=vectorizer.transform(X_test)


# In[27]:


X_train_transformed


# In[96]:


clf=MultinomialNB()
clf


# In[97]:


clf.fit(X_train_transformed,Y_train)


# In[98]:


Y_pred=clf.predict(X_test_transformed)
Y_pred


# In[99]:


accuracy_score(Y_test,Y_pred)


# In[100]:


confusion_matrix(Y_test,Y_pred)


# In[101]:


print(classification_report(Y_test,Y_pred))


# # KNN Classifier

# In[34]:


from sklearn.neighbors import KNeighborsClassifier


# In[35]:


knn=KNeighborsClassifier()
knn


# In[36]:


knn.fit(X_train_transformed,Y_train)


# In[37]:


y_pred1=knn.predict(X_test_transformed)
y_pred1


# In[38]:


accuracy_score(Y_test,y_pred1)


# In[39]:


print(classification_report(Y_test, y_pred1))


# # Logistic Regression

# In[40]:


from sklearn.linear_model import LogisticRegression


# In[41]:


LR=LogisticRegression()


# In[42]:


LR.fit(X_train_transformed,Y_train)


# In[43]:


y_pred2=LR.predict(X_test_transformed)
y_pred2


# In[44]:


accuracy_score(Y_test,y_pred2)


# In[45]:


print(classification_report(Y_test, y_pred2))


# # SupportVectorMachine Classifier

# In[46]:


from sklearn.svm import SVC


# In[47]:


svc=SVC()
svc


# In[48]:


svc.fit(X_train_transformed,Y_train)


# In[49]:


y_pred3=svc.predict(X_test_transformed)
y_pred3


# In[50]:


accuracy_score(Y_test,y_pred3)


# In[51]:


print(classification_report(Y_test, y_pred3))


# # RandomForest Classifier

# In[52]:


from sklearn.ensemble import RandomForestClassifier


# In[53]:


RFC=RandomForestClassifier()
RFC


# In[54]:


RFC.fit(X_train_transformed,Y_train)


# In[55]:


y_pred4=RFC.predict(X_test_transformed)
y_pred4


# In[56]:


accuracy_score(Y_test,y_pred4)


# In[57]:


print(classification_report(Y_test, y_pred4))


# # NeuralNetworks Classifier

# In[58]:


from sklearn.neural_network import MLPClassifier


# In[59]:


MLP=MLPClassifier()
MLP


# In[60]:


MLP.fit(X_train_transformed,Y_train)


# In[61]:


y_pred5=MLP.predict(X_test_transformed)
y_pred5


# In[62]:


accuracy_score(Y_test,y_pred5)


# In[63]:


print(classification_report(Y_test, y_pred5))


# In[104]:


pickle.dump(vectorizer, open('transform.pkl','wb'))


# In[105]:


filename='Nlp.pkl'


# In[106]:


pickle.dump(clf, open(filename,'wb'))


# In[ ]:





# In[ ]:





# In[ ]:




