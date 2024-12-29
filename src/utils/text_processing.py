import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def download_nltk_data():
    for resource in ["punkt", "stopwords", "wordnet", "punkt_tab"]:
        nltk.download(resource)


def preprocess_text(text, stop_words=set(stopwords.words("english"))):
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalnum() and token not in stop_words
    ]
    return " ".join(tokens)


def preprocess_dataset(df):
    df["title_clean"] = df["title"].apply(preprocess_text)
    df["text_clean"] = df["text"].apply(preprocess_text)
    df["title_length"] = df["title"].str.len()
    df["text_length"] = df["text"].str.len()
    return df
