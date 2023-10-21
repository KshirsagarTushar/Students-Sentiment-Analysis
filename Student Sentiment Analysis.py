#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Necessary Imports


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import defaultdict,Counter


stop=set(stopwords.words('english'))
plt.style.use('seaborn')


# In[3]:


#read and load data


# In[4]:


df = pd.read_csv('/content/sample_data/Student Sentiment Data.csv')


# In[5]:


df


# In[6]:


df = df[['Polarity','Comment']]


# In[7]:


df.Polarity.unique()


# In[8]:


aspect_mapping = {'Positive': 1 ,'Negative': 2 , 'Neutral': 0}

df['Polarity_label'] = df['Polarity'].map(aspect_mapping)


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


#Exploratory Data Analysis


# In[13]:


#Funnel Chart


# In[14]:


from plotly import graph_objs as go
temp = df.groupby('Polarity').count()['Comment'].reset_index().sort_values(by='Comment',ascending=False)

fig = go.Figure(go.Funnelarea(
    text =temp.Polarity,
    values = temp.Comment,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()


# In[15]:


df['Polarity'].value_counts()


# In[16]:


#Pie Chart


# In[17]:


class_df = df.groupby('Polarity').count()['Comment'].reset_index().sort_values(by='Comment',ascending=False)
percent_class=class_df.Comment

labels= ['Positive','Negative','Neutral']

colors = ['#17C37B','#F92969','#FACA0C']

my_pie,_,_ = plt.pie(percent_class,radius = 1.2,labels=labels,colors=colors,autopct="%.1f%%")

plt.setp(my_pie, width=0.6, edgecolor='white')

plt.show()


# In[18]:


#There is an uneven distribution in the data with the largest portion belongs to Positive followed by Negative and Neutral.


# In[19]:


#Number of Characters in Review


# In[20]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))

tweet_len=df[df['Polarity']=="Positive"]['Comment'].str.len()
ax1.hist(tweet_len,color='#17C37B')
ax1.set_title('Positive Sentiments')

tweet_len=df[df['Polarity']=="Negative"]['Comment'].str.len()
ax2.hist(tweet_len,color='#F92969')
ax2.set_title('Negative Sentiments')

tweet_len=df[df['Polarity']=="Neutral"]['Comment'].str.len()
ax3.hist(tweet_len,color='#FACA0C')
ax3.set_title('Neutral Sentiments')

fig.suptitle('Characters in a Review')
plt.show()


# In[21]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
text_len=df[df['Polarity_label']==1]['Comment'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='green')
ax1.set_title('Positive Review')
text_len=df[df['Polarity_label']==-1]['Comment'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='red')
ax2.set_title('Negative Review')
text_len=df[df['Polarity_label']==0]['Comment'].str.split().map(lambda x: len(x))
ax3.hist(text_len,color='gray')
ax3.set_title('Neutral Review')
fig.suptitle('Distribution of Words in Review')
plt.show()


# In[22]:


fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,5))
word=df[df['Polarity_label']==1]['Comment'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='green')
ax1.set_title('Positive feedback')
word=df[df['Polarity_label']==-1]['Comment'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='red')
ax2.set_title('Negative feedback')
word=df[df['Polarity_label']==0]['Comment'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax3,color='grey')
ax3.set_title('Neutral feedback')
fig.suptitle('Average word length in each feedback')


# In[23]:


from sklearn.feature_extraction.text import CountVectorizer
def get_top_text_ngrams(corpus, n, g):
    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[24]:


import sys
sys.version


# In[25]:


plt.figure(figsize = (16,9))
most_common_uni = get_top_text_ngrams(df.Comment,10,1)
most_common_uni = dict(most_common_uni)
sns.barplot(x=list(most_common_uni.values()),y=list(most_common_uni.keys()))


# In[26]:


stop


# In[27]:


#Common Stopwards in Text


# In[28]:


import numpy as np

def create_corpus(target):
    corpus=[]

    for x in df[df['Polarity']==target ]['Comment'].str.split():
        for i in x:
            if i  in stop:
                corpus.append(i)
    return corpus

np.array(stop)


# In[29]:


comment_words = ''
stopwords = set(STOPWORDS)


for val in stop:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "


wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize = (8, 5), facecolor = "white")
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()


# In[30]:


corpus=create_corpus("Positive")

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
x,y=zip(*top)
plt.bar(x,y, color='#17C37B')


# In[31]:


corpus=create_corpus("Negative")

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
x,y=zip(*top)
plt.bar(x,y, color='#F92969')


# In[32]:


corpus=create_corpus("Neutral")

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
x,y=zip(*top)
plt.bar(x,y, color='#FACA0C')


# In[33]:


#World Cloud of Every Polarity


# In[34]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[30, 15])

df_pos = df[df["Polarity"]=="Positive"]
df_neg = df[df["Polarity"]=="Negative"]
df_neu = df[df["Polarity"]=="Neutral"]

comment_words = ''
stopwords = set(STOPWORDS)

for val in df_pos.Comment:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "


wordcloud1 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Greens",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Positive Sentiment',fontsize=35);

comment_words = ''

for val in df_neg.Comment:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "


wordcloud2 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Reds",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Negative Sentiment',fontsize=35);



comment_words = ''
for val in df_neu.Comment:

    # typecaste each val to string
    val = str(val)

    # split the value
    tokens = val.split()

    # Converts each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "


wordcloud3 = WordCloud(width = 800, height = 800,
                background_color ='white',
                colormap="Greys",
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Neutral Sentiment',fontsize=35);


# In[35]:


import tensorflow as tf


# In[36]:


get_ipython().system('pip install transformers')


# In[37]:


from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

#for displaying 500 results in pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


from sklearn.model_selection import train_test_split

import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict,Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import string
nltk.download('stopwords')


stop=set(stopwords.words('english'))


# In[38]:


import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[39]:


get_ipython().system('pip install openpyxl')


# In[40]:


df.head(10)


# In[41]:


df = df[['Polarity_label','Comment']]


# In[42]:


index = df.index
number_of_rows = len(index)
print(number_of_rows)


# In[43]:


df.head(10)


# In[44]:


#Preprocessing


# In[45]:


#Remove Urls and HTML links
def remove_urls(text):
    url_remove = re.compile(r'https?://\S+|www\.\S+')
    return url_remove.sub(r'', text)
df['comment_new']=df['Comment'].apply(lambda x:remove_urls(x))

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
df['comment_new']=df['comment_new'].apply(lambda x:remove_html(x))


# In[46]:


# Lower casing
def lower(text):
    low_text= text.lower()
    return low_text
df['comment_new']=df['comment_new'].apply(lambda x:lower(x))


# In[47]:


# Number removal
def remove_num(text):
    remove= re.sub(r'\d+', '', text)
    return remove
df['comment_new']=df['comment_new'].apply(lambda x:remove_num(x))


# In[48]:


#Remove stopwords & Punctuations
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPWORDS = set(stopwords.words('english'))

def punct_remove(text):
    punct = re.sub(r"[^\w\s\d]","", text)
    return punct
df['comment_new']=df['comment_new'].apply(lambda x:punct_remove(x))



def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df['comment_new']=df['comment_new'].apply(lambda x:remove_stopwords(x))


# In[49]:


#Remove extra white space left while removing stuff
def remove_space(text):
    space_remove = re.sub(r"\s+"," ",text).strip()
    return space_remove
df['comment_new']=df['comment_new'].apply(lambda x:remove_space(x))


# In[50]:


from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

def lemmatize_words(text):
    wnl = WordNetLemmatizer()
    lem = ' '.join([wnl.lemmatize(word) for word in text.split()])
    return lem

df['comment_new'] = df['comment_new'].apply(lemmatize_words)


# In[51]:


reviews = df['comment_new'].values.tolist()
labels = df['Polarity_label'].tolist()


# In[52]:


print(reviews[:2])
print(labels[:2])


# In[53]:


from sklearn.model_selection import train_test_split, GridSearchCV
training_sentences, validation_sentences, training_labels, validation_labels = train_test_split(reviews, labels, test_size=.2,random_state = 23)
validation_sentences, test_sentences, validation_labels, test_labels = train_test_split(validation_sentences, validation_labels, test_size=.5,random_state = 23)


# In[54]:


len(training_sentences)


# In[55]:


len(validation_sentences)


# In[56]:


len(test_sentences)


# In[57]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df = 2, max_df = 0.5, ngram_range = (1, 2))

#training_sentences_Tf = tfidf.fit_transform(training_sentences)

#test_sentences_Tf = tfidf.transform(test_sentences)


# In[58]:


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# In[59]:


tokenizer([training_sentences[0]], truncation=True, padding=True, max_length=512)


# In[60]:


train_encodings = tokenizer(training_sentences,
                            truncation=True,
                            padding=True)
val_encodings = tokenizer(validation_sentences,
                            truncation=True,
                            padding=True)


# In[61]:


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    training_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    validation_labels
))


# In[62]:


model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=3)


# In[63]:


model.save_pretrained("./sentiment")


# In[64]:


loaded_model = TFDistilBertForSequenceClassification.from_pretrained("./sentiment")


# In[65]:


test_sentence = "I absolutely loved the lecturer, George Siedel. He presents in a way where I wanted to listen and I feel like I learned a lot from him just with the way that he spoke. I definitely think that this course is worth doing and I am really happy that did it. The fact that the course was online, didn't feel like it was a barrier to me at all. In fact, I preferred it because I was able to go through the work in my own pace and was thus able to complete the entire course in one week. I really enjoyed this course, and definitely think that the lecturer, George Siedel, played a big part in it. I haven't done many other online courses before, but I'm definitely motivated to try more of Coursera's courses due to the great experience that I had taking this course. Due to the introduction of the University of Michigan made, I am now also strongly considering to apply for the MBA course when I am able."


predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

tf_output = loaded_model.predict(predict_input)[0]


tf_prediction = tf.nn.softmax(tf_output, axis=1)
label = tf.argmax(tf_prediction, axis=1)
label = label.numpy()
print(label)


# In[66]:


test_labels[0]


# In[67]:


predict_input = []
y_hat = []
predictions = []

for x in test_sentences:
   predict_input.append(tokenizer.encode(x,truncation=True,padding=True,return_tensors="tf"))

for x in predict_input:
  y_hat.append(loaded_model.predict(x)[0])


for j in y_hat:
  tf_prediction = tf.nn.softmax(j, axis=1)
  label = tf.argmax(tf_prediction, axis=1)
  label = label.numpy()
  predictions.append(label)


# In[68]:


tokenizer.save_pretrained("./sentiment")


# In[69]:


label


# In[70]:


from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score


# In[71]:


#NAIVE BAYES


# In[72]:


# Model 1 - default parameter
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

nb_classifier1 = MultinomialNB()

# define the stages of the pipeline
pipeline = Pipeline(steps= [('tfidf', tfidf),
                            ('model',nb_classifier1) ])

pipeline.fit(training_sentences,training_labels)


pred1 = pipeline.predict(test_sentences)

print(classification_report(test_labels,pred1, target_names = ['Positive','Negative','Neutral']))


# In[73]:


precision = precision_score(test_labels, pred1,average='weighted')
print('Precision: %f' % precision)
recall = recall_score(test_labels, pred1,average='weighted')
print('Recall: %f' % recall)
f1 = f1_score(test_labels, pred1,average='weighted')
print('F1 Score: %f' % f1)


# In[74]:


# Model 2
from sklearn.svm import SVC

svc_model1 = SVC(C=1, kernel='linear', gamma= 1)

# define the stages of the pipeline
Model = Pipeline(steps= [('tfidf', tfidf),
                            ('model',svc_model1) ])

Model.fit(training_sentences,training_labels)

pred1 = Model.predict(test_sentences)

print(classification_report(test_labels, pred1, target_names = ['Positive','Negative','Neutral']))


# In[75]:


accuracy = accuracy_score(test_labels, pred1)
print('Accuracy: %f' % accuracy)
precision = precision_score(test_labels, pred1,average='weighted')
print('Precision: %f' % precision)
recall = recall_score(test_labels, pred1,average='weighted')
print('Recall: %f' % recall)
f1 = f1_score(test_labels, pred1,average='weighted')
print('F1 Score: %f' % f1)


# In[76]:


acccuracy = accuracy_score(test_labels, pred1)
precision = precision_score(test_labels, pred1,average='weighted')
recall = recall_score(test_labels, pred1,average='weighted')
f1_score = f1_score(test_labels, pred1,average='weighted')

print("********* Support Vector Classifier *********")
print("\tAccuracy    : ", acccuracy)
print("\tPrecision   : ", precision)
print("\tRecall      : ", recall)
print("\tF1 Score    : ", f1_score)


# In[77]:


# Model 3
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
# define the stages of the pipeline
pipeline = Pipeline(steps= [('tfidf', tfidf),
                            ('model',rf) ])

pipeline.fit(training_sentences,training_labels)

prediction1 = pipeline.predict(test_sentences)

print(classification_report(test_labels, prediction1, target_names = ['Positive','Negative','Neutral']))


# In[78]:


precision = precision_score(test_labels, prediction1,average='weighted')
print('Precision: %f' % precision)
recall = recall_score(test_labels, prediction1,average='weighted')
print('Recall: %f' % recall)


# In[79]:


get_ipython().system('pip install catboost')
from catboost import CatBoostClassifier

clfs = CatBoostClassifier(
    iterations=5,
    learning_rate=0.1,
    #loss_function='CrossEntropy'
)

# define the stages of the pipeline
pipeline = Pipeline(steps= [('tfidf', tfidf),
                            ('model',clfs) ])

pipeline.fit(training_sentences,training_labels)

pred1 = pipeline.predict(test_sentences)

print(classification_report(test_labels, pred1, target_names = ['Positive','Negative','Neutral']))


# In[80]:


precision = precision_score(test_labels, pred1,average='weighted')
print('Precision: %f' % precision)
recall = recall_score(test_labels, pred1,average='weighted')
print('Recall: %f' % recall)


# In[81]:


get_ipython().system('pip install AdaBoost')
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier()

# define the stages of the pipeline
pipeline = Pipeline(steps= [('tfidf', tfidf),
                            ('model',abc) ])

pipeline.fit(training_sentences,training_labels)

pred1 = pipeline.predict(test_sentences)

print(classification_report(test_labels, pred1, target_names = ['Positive','Negative','Neutral']))


# In[82]:


precision = precision_score(test_labels, pred1,average='weighted')
print('Precision: %f' % precision)
recall = recall_score(test_labels, pred1,average='weighted')
print('Recall: %f' % recall)


# In[83]:


get_ipython().system('pip install GradientBoostingClassifier')
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

# define the stages of the pipeline
pipeline = Pipeline(steps= [('tfidf', tfidf),
                            ('model',gbc) ])

pipeline.fit(training_sentences,training_labels)

pred1 = pipeline.predict(test_sentences)

print(classification_report(test_labels, pred1, target_names = ['Positive','Negative','Neutral']))


# In[84]:


precision = precision_score(test_labels, pred1,average='weighted')
print('Precision: %f' % precision)
recall = recall_score(test_labels, pred1,average='weighted')
print('Recall: %f' % recall)


# In[85]:


# table of all models

Pipelines = [Pipeline(steps= [('tfidf', tfidf),('model',nb_classifier1) ]), Pipeline(steps= [('tfidf', tfidf),('model',svc_model1) ]),
            Pipeline(steps= [('tfidf', tfidf),('model',rf) ]), Pipeline(steps= [('tfidf', tfidf),('model',abc) ]),
            Pipeline(steps= [('tfidf', tfidf),('model',gbc) ])]

model_names = ['Naive Bayes', 'SVC', 'Random Forest', 'Ada Boost', 'Gradient Boosting']

accuracy = []
precision = []

for i in Pipelines:
    i.fit(training_sentences,training_labels)
    y_pred = i.predict(test_sentences)
    accuracy.append(accuracy_score(test_labels, y_pred))
    precision.append(precision_score(test_labels, y_pred,average='weighted'))

model_comparison = pd.DataFrame({'Model': model_names, 'Accuracy': accuracy, 'Precision': precision})
model_comparison.sort_values(by='Accuracy', ascending=False)


# In[86]:


#Pickled Model


# In[87]:


import pickle


# In[88]:


pickle.dump(Model,open('Model.pkl','wb'))


# In[89]:


pickled_model = pickle.load(open('Model.pkl','rb'))


# In[90]:


print(pickled_model)

