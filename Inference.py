# Init
import os
from reviews_sentiment_model_training import preprocess_reviews, vectorize
import pandas as pd
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')

# List
reviews_list = ['I really loved the product.', 'Althogh it took a long time for the item to deliver, I was happy with the item', 'Bad customer service, dont recommend it!']

# Execute
def execute(list):

  # Pickles
  stop_words = pickle.load(open(os.path.join(data_dir, 'stop_words.pickle'), 'rb'))
  f_vectors = pickle.load(open(os.path.join(data_dir, 'f_vectors.pickle'), 'rb'))
  xgb_model = pickle.load(open(os.path.join(data_dir, 'xgb_model.pickle'), 'rb'))

  # Preprocess
  filtered_reviews = preprocess_reviews(stop_words, list)

  # Vectorize
  def vectorize(filtered_reviews):
    vectorized_reviews = f_vectors.transform(filtered_reviews)
    return pd.DataFrame(vectorized_reviews.toarray(), columns=f_vectors.get_feature_names_out())

  # Predict
  def predict(vectorized_reviews):
    prediction = xgb_model.predict(vectorized_reviews)
    return prediction

  # Pred DF
  def pred_df(reviews_list, prediction):
    data = {
    'Reviews': reviews_list,
    'Labels': prediction
    }
    return pd.DataFrame(data)

  # FN Call
  vectorized_reviews = vectorize(filtered_reviews)
  prediction = predict(vectorized_reviews)
  df = pred_df(list, prediction)
  return df

# Execution
if __name__ == '__main__':
  print(execute(reviews_list))
else:
  pass
