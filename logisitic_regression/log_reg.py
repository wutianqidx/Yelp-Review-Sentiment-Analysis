import numpy as np
import re
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import ssl
import nltk
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')


file_path = '../yelp_review_polarity_csv/'
x_train = []
y_train = []
with open(file_path+'train.csv') as f:
    for line in f:
        l = line.strip().split(',')
        senti, text = l[0], re.sub('[^a-zA-Z \']', '', re.sub('\\\\n', ' ', ','.join(l[1:]))).lower()
        #tokens = nltk.word_tokenize(text)
        x_train.append(text)
        y_train.append(int(senti[1])-1)
    f.close()

x_test = []
y_test = []
with open(file_path+'test.csv') as f:
    for line in f:
        l = line.strip().split(',')
        senti, text = l[0], re.sub('[^a-zA-Z \']', '', re.sub('\\\\n', ' ', ','.join(l[1:]))).lower()
        #tokens = nltk.word_tokenize(text)
        x_test.append(text)
        y_test.append(int(senti[1])-1)
    f.close()

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
start_time = time.time()
vectorizer = TfidfVectorizer(max_features=4000, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))

X_train = vectorizer.fit_transform(x_train).toarray()
X_test = vectorizer.transform(x_test).toarray()

lr_classfier = LogisticRegression()
lr_classfier.fit(X_train,y_train)
train_acc = lr_classfier.score(X_train,y_train)
test_acc = lr_classfier.score(X_test,y_test)

print('%.2f' % float(train_acc*100), '%.2f' % float(test_acc*100),'%.4f' % float(time.time()-start_time))
