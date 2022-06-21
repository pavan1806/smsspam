


import streamlit as st
import numpy as np
import nltk
import pandas as pd

import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv("spam.csv", encoding='latin-1')
#df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns = {'v1':'class_label','v2':'message'},inplace=True)


#Preprocess the data
df['class_label'] = df['class_label'].map( {'spam': 1, 'ham': 0})

# Replace email address with 'emailaddress'
df['message'] = df['message'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')

# Replace urls with 'webaddress'
df['message'] = df['message'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

# Replace money symbol with 'money-symbol'
df['message'] = df['message'].str.replace(r'£|\$', 'money-symbol')

# Replace 10 digit phone number with 'phone-number'
df['message'] = df['message'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3?[\d]{4}$', 'phone-number')

# Replace normal number with 'number'
df['message'] = df['message'].str.replace(r'\d+(\.\d+)?', 'number')

# remove punctuation
df['message'] = df['message'].str.replace(r'[^\w\d\s]', ' ')

# remove whitespace between terms with single space
df['message'] = df['message'].str.replace(r'\s+', ' ')

# remove leading and trailing whitespace
df['message'] = df['message'].str.replace(r'^\s+|\s*?$', ' ')

# change words to lower case
df['message'] = df['message'].str.lower()


nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
df['message'] = df['message'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

#Snowball Stemmer
ss = nltk.SnowballStemmer("english")
df['message'] = df['message'].apply(lambda x: ' '.join(ss.stem(term) for term in x.split()))


#Splitting
X,y = df['message'],df['class_label']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


text_clf = Pipeline([('tfidf',TfidfVectorizer()),('svm',LinearSVC())])
text_clf.fit(X_train,y_train)









st.title("SMS Spam Classifier")
st.write("Build with Streamlit & Python")
ms = st.text_area("Enter the message")
msg = [ms]

if st.button('Predict'):
    if len(msg[0]) <= 1:
        st.info("Please type your messgae or paste your message")

    # 4. Display
    elif text_clf.predict(msg) == 1:
        st.error("This is A Spam SMS")


    else:
        st.success("This is Not A Spam SMS")

