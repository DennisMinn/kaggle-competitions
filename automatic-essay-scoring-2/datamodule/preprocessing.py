import re
import string
import spacy
import polars as pl

with open('data/vocabulary.txt') as infile:
    vocabulary = set(word.strip().lower() for word in infile)

nlp = spacy.load('en_core_web_sm')

def count_spelling_errors(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_.lower() for token in doc]
    spelling_errors = sum(1 for token in lemmatized_tokens if token not in vocabulary)
    return spelling_errors

def remove_HTML(x):
    html=re.compile(r'<.*?>')
    return html.sub(r'',x)

def sanitize_text(text):
    # Convert words to lowercase
    text = text.lower()

    # Remove HTML
    text = remove_HTML(text)

    # Delete strings starting with @
    text = re.sub(r'@\w+', '', text)

    # Delete Numbers
    text = re.sub(r"'\d+", '', text)
    text = re.sub(r'\d+', '', text)

    # Delete URL
    text = re.sub(r'http\w+', '', text)

    # Replace consecutive empty spaces with a single space character
    text = re.sub(r'\s+', ' ', text)

    # Replace consecutive commas and periods with one comma and period character
    text = re.sub(r'\.+', '.', text)
    text = re.sub(r'\,+', ',', text)

    # Remove empty characters at the beginning and end
    text = text.strip()

    return text

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def Paragraph_Preprocess(df):
    df = df.with_columns(
        pl.col('full_text')
        .str.split(by="\n\n")
        .alias("paragraph")
    )

    # Expand the paragraph list into several lines of data
    df = df.explode('paragraph')

    df = df.with_columns(
        pl.col('paragraph')
        .map_elements(sanitize_text)
    )

    df = df.with_columns(
        pl.col('paragraph')
        .map_elements(remove_punctuation)
        .map_elements(count_spelling_errors)
        .alias("paragraph_error_num")
    )

    # Calculate the number of sentences and words in each paragraph
    df = df.with_columns(
        pl.col('paragraph')
        .map_elements(lambda x: len(x))
        .alias("paragraph_len"),
        pl.col('paragraph')
        .map_elements(lambda x: len(x.split('.')))
        .alias("paragraph_sentence_cnt"),
        pl.col('paragraph')
        .map_elements(lambda x: len(x.split(' ')))
        .alias("paragraph_word_cnt"),
    )
    return df

def Paragraph_Eng(train_df):
    paragraph_feature = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt', 'paragraph_error_num']
    feature_aggregates = []

    feature_aggregates += [
        pl.col('paragraph')
        .filter(pl.col('paragraph_len') >= i)
        .count()
        .alias(f"paragraph_{i}_cnt") for i in [0, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700]
    ]

    feature_aggregates += [
        pl.col('paragraph')
        .filter(pl.col('paragraph_len') <= i)
        .count()
        .alias(f"paragraph_{i}_cnt") for i in [25,49]
    ]

    for feature in paragraph_feature:
        feature_aggregates += [
            pl.col(feature).max().alias(f'{feature}_max'),
            pl.col(feature).min().alias(f"{feature}_min"),
            pl.col(feature).mean().alias(f"{feature}_mean"),
            pl.col(feature).sum().alias(f"{feature}_sum"),
            pl.col(feature).first().alias(f"{feature}_first"),
            pl.col(feature).last().alias(f"{feature}_last"),
            pl.col(feature).kurtosis().alias(f"{feature}_kurtosis"),
            pl.col(feature).quantile(0.25).alias(f"{feature}_q1"),
            pl.col(feature).quantile(0.75).alias(f"{feature}_q3")
        ]

    df = train_df.group_by(['essay_id'], maintain_order=True).agg(feature_aggregates).sort("essay_id")
    df = df.to_pandas()
    return df