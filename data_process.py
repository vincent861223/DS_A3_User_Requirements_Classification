import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def nltk_prepare():
    print("Downloading NLTK files...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    wordnet_lemmatizer = WordNetLemmatizer()

def replace_symbols(df):
    df['tmp'] = df['tmp'].str.replace("\r", " ")
    df['tmp'] = df['tmp'].str.replace("\n", " ")
    df['tmp'] = df['tmp'].str.replace("  ", " ")
    df['tmp'] = df['tmp'].str.replace('"', '')
    df['tmp'] = df['tmp'].str.lower()

    punctuation_signs = list("?:!.,;")

    for punct_sign in punctuation_signs:
        df['tmp'] = df['tmp'].str.replace(punct_sign, '')

    df['tmp'] = df['tmp'].str.replace("'s", "")

    return df

def lemmatize(df):
    nrows = len(df)
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_text_list = []

    for row in range(0, nrows):
        # Create an empty list containing lemmatized words
        lemmatized_list = []

        # Save the text and its words into an object
        text = df.loc[row]['tmp']
        text_words = text.split(" ")

        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        # Join the list
        lemmatized_text = " ".join(lemmatized_list)

        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)

    df['tmp'] = lemmatized_text_list

    return df

def replace_stopwords(df):
    stop_words = list(stopwords.words('english'))

    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['tmp'] = df['tmp'].str.replace(regex_stopword, '')

    return df
