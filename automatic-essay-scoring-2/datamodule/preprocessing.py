import re
import string
import spacy
import polars as pl
import pandas as pd

with open('data/vocabulary.txt') as infile:
    vocabulary = set(word.strip().lower() for word in infile)

nlp = spacy.load('en_core_web_sm')

def count_spelling_errors(text):
    doc = nlp(text)
    lemmatized_tokens = set([token.lemma_.lower() for token in doc])
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

def create_paragraph_features(df):
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
        .map_elements(lambda paragraph: len(paragraph))
        .alias("paragraph_len"),
        pl.col('paragraph')
        .map_elements(lambda paragraph: len(paragraph.split('.')))
        .alias("paragraph_sentence_cnt"),
        pl.col('paragraph')
        .map_elements(lambda paragraph: len(paragraph.split(' ')))
        .alias("paragraph_word_cnt"),
    )

    # Aggreate features
    feature_aggregates = []

    feature_aggregates += [
        pl.col('paragraph')
        .filter(pl.col('paragraph_len') >= i)
        .count()
        .alias(f"paragraph_{i}_cnt")
        for i in [0, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700]
    ]

    feature_aggregates += [
        pl.col('paragraph')
        .filter(pl.col('paragraph_len') <= i)
        .count()
        .alias(f"paragraph_{i}_cnt")
        for i in [25,49]
    ]

    paragraph_feature = ['paragraph_len','paragraph_sentence_cnt','paragraph_word_cnt', 'paragraph_error_num']
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

    df = df.group_by(['essay_id'], maintain_order=True).agg(feature_aggregates).sort("essay_id")
    df = df.to_pandas()
    return df

def create_word_features(df):
    df = df.with_columns(
        pl.col('full_text')
        .map_elements(sanitize_text)
        .str.split(by=" ").
        alias("word")
    )

    df = df.explode('word')

    df = df.with_columns(
        pl.col('word')
        .map_elements(lambda x: len(x))
        .alias("word_len")
    )

    df = df.filter(pl.col('word_len') != 0)

    # Aggregate features
    feature_aggregates = []
    feature_aggregates += [
        pl.col('word')
        .filter(pl.col('word_len') >= i + 1)
        .count()
        .alias(f"word_{i + 1}_cnt")
        for i in range(15)
    ]

    feature_aggregates += [
        pl.col('word_len').max().alias(f"word_len_max"),
        pl.col('word_len').mean().alias(f"word_len_mean"),
        pl.col('word_len').std().alias(f"word_len_std"),
        pl.col('word_len').quantile(0.25).alias(f"word_len_q1"),
        pl.col('word_len').quantile(0.50).alias(f"word_len_q2"),
        pl.col('word_len').quantile(0.75).alias(f"word_len_q3"),
    ]

    df = df.group_by(['essay_id'], maintain_order=True).agg(feature_aggregates).sort("essay_id")
    df = df.to_pandas()
    return df


def create_tfidf_features(df, vectorizer, stage):
    essays = df['full_text'].to_list()
    if stage == 'train':
        tfidf_sparse_matrix = vectorizer.fit_transform(essays)
    else:
        tfidf_sparse_matrix = vectorizer.transform(essays)

    tfidf_dense_matrix = tfidf_sparse_matrix.toarray()
    tfidf_df = pd.DataFrame(tfidf_dense_matrix)

    tfidf_df.columns = [f'tfidf_{i}' for i in range(tfidf_df.shape[1])]
    tfidf_df['essay_id'] = df['essay_id']
    return tfidf_df

def create_bag_of_words_features(df, vectorizer, stage):
    essays = df['full_text'].to_list()
    if stage == 'train':
        bow_sparse_matrix = vectorizer.fit_transform(essays)
    else:
        bow_sparse_matrix = vectorizer.transform(essays)

    bow_dense_matrix = bow_sparse_matrix.toarray()
    bow_df = pd.DataFrame(bow_dense_matrix)

    bow_df.columns = [f'tfidf_{i}' for i in range(bow_df.shape[1])]
    bow_df['essay_id'] = df['essay_id']
    return bow_df

def create_features(df, tfidf_vectorizer=None, count_vectorizer=None):
    paragraph_feature_df = create_paragraph_features(df)
    word_feature_df = create_word_features(df)

    if tfidf_vectorizer:
        tfidf_df = create_tfidf_features(df, tfidf_vectorizer)
    if count_vectorizer:
        bow_df = create_bag_of_words_features(df, count_vectorizer)

    feature_df = df[('essay_id')].to_pandas().to_frame()
    feature_df = feature_df.merge(word_feature_df, on='essay_id', how='left')
    feature_df = feature_df.merge(paragraph_feature_df, on='essay_id', how='left')
    if tfidf_vectorizer:
        feature_df = feature_df.merge(tfidf_df, on='essay_id', how='left')
    if count_vectorizer:
        feature_df = feature_df.merge(bow_df, on='essay_id', how='left')
    return feature_df