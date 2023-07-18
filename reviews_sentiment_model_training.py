# Init
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pickle
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')

# Prep
def prep():
  amazon = pd.read_csv(os.path.join(data_dir, 'amazon_cells_labelled.txt'), sep='\t', encoding='ISO-8859-1', on_bad_lines='skip', names=['Review', 'Label'])
  imdb = pd.read_csv(os.path.join(data_dir, 'imdb_labelled.txt'), sep='\t', encoding='ISO-8859-1', on_bad_lines='skip', names=['Review', 'Label'])
  yelp = pd.read_csv(os.path.join(data_dir, 'yelp_labelled.txt'), sep='\t', encoding='ISO-8859-1', on_bad_lines='skip', names=['Review', 'Label'])
  reviews_df = pd.concat([amazon, imdb, yelp], axis=0, ignore_index=True)

  stop_words = stopwords.words('english')
  stop_words.extend(['.', ',', "'", '"', '?', '!', '-', '/', ':', '(', ')', '\n', '@'])
  to_remove = ["but", "or", "against", "on", "off", "both", "no", "nor", "not", "only", "same", "don'", "don't", "ain'", "aren'", "aren't", "could'", "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'wouldn', "wouldn't", 'ain', 'aren', 'couldn']
  stop_words = [word for word in stop_words if word not in to_remove]
  sent_list = reviews_df['Review'].apply(sent_tokenize).tolist()
  reviews_list = [' '.join(inner_list) for inner_list in sent_list]

  return reviews_df, stop_words, reviews_list

# Preprocess
def preprocess_reviews(stop_words, reviews_list):
    filtered_reviews = []
    for sentence in reviews_list:
        words = WordPunctTokenizer().tokenize(sentence)
        filtered_words = []

        for word in words:
            word_lower = word.lower()
            if word_lower not in stop_words:
                pos = pos_tag([word])[0][1]

                if word_lower.endswith('ness'):  # Nouns
                    stemmer = LancasterStemmer()
                    filtered_word = stemmer.stem(word_lower)
                elif pos.startswith('VB'):  # Verbs
                    lemmatizer = WordNetLemmatizer()
                    filtered_word = lemmatizer.lemmatize(word_lower, pos='v')
                else:
                    filtered_word = word_lower

                filtered_words.append(filtered_word)

        filtered_sentence = " ".join(filtered_words)
        filtered_reviews.append(filtered_sentence)
    return filtered_reviews

# Vectorize
def vectorize(reviews):
  vectorizer = TfidfVectorizer()
  f_vectors = vectorizer.fit(reviews)
  vectorized_reviews = f_vectors.transform(reviews)
  return f_vectors, pd.DataFrame(vectorized_reviews.toarray(), columns=f_vectors.get_feature_names_out())

# Define | Split X and Y
def define_split_x_y(vectorized_reviews, labels):
    vectorized_reviews['labels'] = labels

    x = vectorized_reviews.drop(['labels'], axis=1)
    y = vectorized_reviews['labels']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    return x_train, x_test, y_train, y_test

# Train | Predict
def train_predict(x_train, x_test, y_train):
  xgb_c = xgb.XGBClassifier()
  xgb_model = xgb_c.fit(X=x_train, y=y_train)
  x_test_pred = xgb_model.predict(x_test)
  return xgb_model, x_test_pred

# Pickle
def pickles(stop_words, f_vectors, xgb_model):
  pickle.dump(stop_words, open(os.path.join(data_dir, 'stop_words.pickle'), 'wb'))
  pickle.dump(f_vectors, open(os.path.join(data_dir, 'f_vectors.pickle'), 'wb'))
  pickle.dump(xgb_model, open(os.path.join(data_dir, 'xgb_model.pickle'), 'wb'))

# Evaluate
def evaluate(y_test, x_test_pred):
  return roc_auc_score(y_test, x_test_pred)

# FN Call
if __name__ == '__main__':
    reviews_df, stop_words, reviews_list = prep()
    filtered_reviews = preprocess_reviews(stop_words, reviews_list)
    f_vectors, vectorized_reviews = vectorize(filtered_reviews)
    x_train, x_test, y_train, y_test = define_split_x_y(vectorized_reviews, reviews_df['Label'])
    xgb_model, x_test_pred = train_predict(x_train, x_test, y_train)
    pickles(stop_words, f_vectors, xgb_model)
    auc_acore = evaluate(y_test, x_test_pred)
    print('auc_acore', auc_acore)
else:
    pass
