#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing Libraries: 
import string
import re
import codecs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[ ]:


#Loading English Raw Data:
eng_df = pd.read_csv("train\\english.txt","utf-8",header=None,names=["English"], engine = 'python')
eng_df.head()


# In[ ]:


#Loading German Raw Data:
ger_df = pd.read_csv("train\\german.txt","utf-8",header=None,names=["German"], engine = 'python')
ger_df.head()


# In[ ]:


#Loading French Raw Data:
fre_df = pd.read_csv("train\\french.txt","utf-8",header=None,names=["French"], engine = 'python')
fre_df.head()


# In[ ]:


#Loading Spanish Raw Data:
spa_df = pd.read_csv("train\\spanish.txt","utf-8",header=None,names=["Spanish"], engine = 'python')
spa_df.head()


# In[ ]:


#Loading Chinese Raw Data:
chi_df = pd.read_csv("train\\chinese.txt","utf-8",header=None,names=["Chinese"], engine = 'python')
chi_df.head()


# In[ ]:


#Data Pre-Processing:
for char in string.punctuation:
    print(char, end=" ")
translate_table = dict((ord(char), None) for char in string.punctuation)


# In[ ]:


#Data Pre-Processing for English:
data_eng = []
lang_eng = []
for i,line in eng_df.iterrows():
    line = line['English']
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+","", line)
        line = line.translate(translate_table)
        data_eng.append(line)
        lang_eng.append("English")


# In[ ]:


#Data Pre-Processing for German:
data_ger = []
lang_ger = []
for i,line in ger_df.iterrows():
    line = line['German']
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+","", line)
        line = line.translate(translate_table)
        data_ger.append(line)
        lang_ger.append("German")


# In[ ]:


#Data Pre-Processing for French:
data_fre = []
lang_fre = []
for i,line in fre_df.iterrows():
    line = line['French']
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+","", line)
        line = line.translate(translate_table)
        data_fre.append(line)
        lang_fre.append("French")


# In[ ]:


#Data Pre-Processing for Spanish:
data_spa = []
lang_spa = []
for i,line in spa_df.iterrows():
    line = line['Spanish']
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+","", line)
        line = line.translate(translate_table)
        data_spa.append(line)
        lang_spa.append("Spanish")


# In[ ]:


#Data Pre-Processing for Chinese:
data_chi = []
lang_chi = []
for i,line in chi_df.iterrows():
    line = line['Chinese']
    if len(line) != 0:
        line = line.lower()
        line = re.sub(r"\d+","", line)
        line = re.sub(r"[a-zA-Z]+","", line)
        line = line.translate(translate_table)
        data_chi.append(line)
        lang_chi.append("Chinese (Traditional)")


# In[ ]:


#Data After Pre-Processing:
df = pd.DataFrame({"Text":data_eng+data_ger+data_fre+data_spa+data_chi,
                   "language":lang_eng+lang_ger+lang_fre+lang_spa+lang_chi})
print(df.shape)


# In[ ]:


#Splitting Data into Train and Test sets (80:20):
X,y = df.iloc[:,0],df.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


#Vectorizer and Model fitting Pipeline:
vectorizer = feature_extraction.text.TfidfVectorizer(ngram_range=(1,3), analyzer='char')

pipe_lr_r13 = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', linear_model.LogisticRegression())
])


# In[ ]:


#Model Fitting:
pipe_lr_r13.fit(X_train, y_train)


# In[ ]:


#Model Prediction:
y_predicted = pipe_lr_r13.predict(X_test)


# In[ ]:


#Model Evaluation:
acc = (metrics.accuracy_score(y_test, y_predicted))*100
print(acc,'%')


# In[ ]:


matrix = metrics.confusion_matrix(y_test,y_predicted)
print('Confusion matrix : \n',matrix)


# In[ ]:


#Model Saving:
import pickle
lrFile = open('LRModel.pckl', 'wb')
pickle.dump(pipe_lr_r13, lrFile)
lrFile.close()


# In[ ]:


#Model Loading
global lrLangDetectModel
lrLangDetectFile = open('LRModel.pckl', 'rb')
lrLangDetectModel = pickle.load(lrLangDetectFile)
lrLangDetectFile.close()


# In[ ]:


#Method Definition to call Trained Model and Make Predictions:
def lang_detect(text):
    import numpy as np
    import string
    import re
    import pickle
    translate_table = dict((ord(char), None) for char in string.punctuation)
    
    global lrLangDetectModel
    lrLangDetectFile = open('LRModel.pckl', 'rb')
    lrLangDetectModel = pickle.load(lrLangDetectFile)
    lrLangDetectFile.close()
    
    text = " ".join(text.split())
    text = text.lower()
    text = re.sub(r"\d+","", text)
    text = text.translate(translate_table)
    pred = lrLangDetectModel.predict([text])
    prob = lrLangDetectModel.predict_proba([text])
    return pred[0]


# In[ ]:


#Predictions
lang_detect("他的妻子是宝莱坞女影星安努舒卡·莎瑪")

