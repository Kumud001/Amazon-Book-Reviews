import streamlit as st
import numpy as np
import base64
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

books = pd.read_csv('D:/DM/DMProject-main/DMProject-main/book_list.csv', header=None)
# books = books.transpose().values
book_t = books.transpose().values.tolist()
book_list = [item for sublist in book_t for item in sublist]

def process_tokens(tokens):
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            filtered_tokens.append(lemma)
    return filtered_tokens

def preprocess_data(text_input):
    # Remove HTML tags
    text = re.sub('<[^<]+?>', '', text_input)
    
    # Remove punctuation and special characters
    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    # Tokenize
    tokens = nltk.tokenize.word_tokenize(text)

    # Lemmatize
    # processed_tokens = tokens.apply(process_tokens)
    processed_tokens = process_tokens(tokens)
    processed_string = ' '.join(processed_tokens)
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=3000)
    print(processed_string)
    vectorized_text = vectorizer.fit_transform([processed_string])
    return vectorized_text

col1, col2 = st.columns(2)
with col1:
    option = st.selectbox(
        'Select book!', book_list)

st.write('You selected:', option)

with col2:
    text_input = st.text_input(
        "Enter some text 👇"
    )

    if text_input:
        st.write("You entered: ", text_input)
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_pipeline = pipeline("sentiment-analysis")
        # "D:\DM\DMProject-main\DMProject-main\Logistic_Regression_model.joblib"
        # loaded_model = joblib.load("D:/DM/DMProject-main/DMProject-main/Logistic_Regression_model.joblib")
        # vectorized_text = preprocess_data(text_input)
        # loaded_model.predict(vectorized_text)
        sentiment_dict = sid_obj.polarity_scores(text_input)
        transformer_output = sentiment_pipeline(text_input)
        if sentiment_dict['compound'] >= 0.05 :
            overall_sentiment = "POSITIVE"

        elif sentiment_dict['compound'] <= - 0.05 :
            overall_sentiment = "NEGATIVE"

        else :
            overall_sentiment = "NEUTRAL"
        st.write("VADER says {} with confidence level: {}".format(overall_sentiment, sentiment_dict['compound']))
        #st.write(sentiment_dict)
        st.write("Transformer says {} with confidence level: {}".format(transformer_output[0]['label'], transformer_output[0]['score']))
        #st.write(transformer_output)

# with col3:
#     text_input = st.text_input(
#         "Enter some text"
#     )

#     if text_input:
#         st.write("You entered: ", text_input)


# with st.container():
   

#    # You can call any Streamlit command, including custom components:
#    st.text_input(
#         "Enter some text!"
#     )
#https://storyset.com/illustration/book-lover/amico
@st.cache_data
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://github.com/HirenRupchandani/March-Kaggle-TPS/raw/main/img.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

#sample plotly code
# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
values = [20, 30, 50]
labels = ['NEU', 'NEG', 'POS']
df = pd.DataFrame({'values': values, 'labels': labels})
# Group data together
hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']

# Create distplot with custom bin_size
# fig = ff.create_distplot(
#         hist_data, group_labels, bin_size=[.1, .25, .5])
import plotly.graph_objs as go

fig = go.Figure(data=[go.Pie(labels=df['labels'], values=df['values'])])
# fig = ff.create_pie(df, values='values', names='labels')

fig.update_layout({
'plot_bgcolor': 'rgba(0, 0, 0, 0)',
'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
# Plot!
st.plotly_chart(fig, use_container_width=True, theme= 'streamlit',sharingMode = "streamlit")


