from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

# __file__ =''
cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


vect = HashingVectorizer(decode_error='ignore', n_features=2 ** 21, preprocessor=None, tokenizer=tokenizer)
print('finish')

clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))

print(clf)

import numpy as np

label = {0: 'negative', 1: 'positive'}
example = ["I love this movie. It's amazing."]
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X)) * 100))
