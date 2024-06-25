import re
import string
import spacy
import polars as pl
import pandas as pd
import pkg_resources
from datasets import Dataset
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

vocabulary = {
    word.strip().lower()
    for word in pkg_resources.resource_string(__package__, 'vocabulary.txt')
    .decode('utf-8')
    .splitlines()
}

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
        .str.split(by='\n\n')
        .alias('paragraph')
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
        .alias('paragraph_num_errors')
    )

    # Calculate the number of sentences and words in each paragraph
    df = df.with_columns(
        pl.col('paragraph')
        .map_elements(lambda paragraph: len(paragraph))
        .alias('paragraph_num_characters'),
        pl.col('paragraph')
        .map_elements(lambda paragraph: len(paragraph.split(' ')))
        .alias('paragraph_num_words'),
        pl.col('paragraph')
        .map_elements(lambda paragraph: len(paragraph.split('.')))
        .alias('paragraph_num_sentences'),
    )

    # Aggreate features
    feature_aggregates = []

    feature_aggregates += [
        pl.col('paragraph')
        .filter(pl.col('paragraph_num_characters') >= i)
        .count()
        .alias(f'paragraph_num_character_greater_equal_{i}')
        for i in [0, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700]
    ]

    paragraph_feature = ['paragraph_num_characters','paragraph_num_words','paragraph_num_sentences', 'paragraph_num_errors']
    for feature in paragraph_feature:
        feature_aggregates += [
            pl.col(feature).max().alias(f'{feature}_max'),
            pl.col(feature).min().alias(f'{feature}_min'),
            pl.col(feature).mean().alias(f'{feature}_mean'),
            pl.col(feature).sum().alias(f'{feature}_sum'),
            pl.col(feature).first().alias(f'{feature}_first'),
            pl.col(feature).last().alias(f'{feature}_last'),
            pl.col(feature).kurtosis().alias(f'{feature}_kurtosis'),
            pl.col(feature).quantile(0.25).alias(f'{feature}_q1'),
            pl.col(feature).quantile(0.75).alias(f'{feature}_q3')
        ]

    df = df.group_by(['essay_id'], maintain_order=True).agg(feature_aggregates).sort('essay_id')
    df = df.to_pandas()
    return df

def create_word_features(df):
    df = df.with_columns(
        pl.col('full_text')
        .map_elements(sanitize_text)
        .str.split(by=' ').
        alias('word')
    )

    df = df.explode('word')

    df = df.with_columns(
        pl.col('word')
        .map_elements(lambda x: len(x))
        .alias('word_length')
    )

    df = df.filter(pl.col('word_length') != 0)

    # Aggregate features
    feature_aggregates = []
    feature_aggregates += [
        pl.col('word')
        .filter(pl.col('word_length') >= i + 1)
        .count()
        .alias(f'word_length_greater_equal_{i + 1}')
        for i in range(15)
    ]

    feature_aggregates += [
        pl.col('word_length').max().alias(f'word_length_max'),
        pl.col('word_length').mean().alias(f'word_length_mean'),
        pl.col('word_length').std().alias(f'word_length_std'),
        pl.col('word_length').quantile(0.25).alias(f'word_length_q1'),
        pl.col('word_length').quantile(0.50).alias(f'word_length_q2'),
        pl.col('word_length').quantile(0.75).alias(f'word_length_q3'),
    ]

    df = df.group_by(['essay_id'], maintain_order=True).agg(feature_aggregates).sort('essay_id')
    df = df.to_pandas()
    return df


def create_tfidf_features(df, vectorizer):
    essays = df['full_text'].map_elements(sanitize_text)
    tfidf_sparse_matrix = vectorizer.transform(essays)

    tfidf_dense_matrix = tfidf_sparse_matrix.toarray()
    tfidf_df = pd.DataFrame(tfidf_dense_matrix)

    tfidf_df.columns = [f'tfidf_{i}' for i in range(tfidf_df.shape[1])]
    tfidf_df['essay_id'] = df['essay_id']
    return tfidf_df


def create_bag_of_words_features(df, vectorizer):
    essays = df['full_text'].map_elements(sanitize_text)
    bow_sparse_matrix = vectorizer.transform(essays)

    bow_dense_matrix = bow_sparse_matrix.toarray()
    bow_df = pd.DataFrame(bow_dense_matrix)

    bow_df.columns = [f'bow_{i}' for i in range(bow_df.shape[1])]
    bow_df['essay_id'] = df['essay_id']
    return bow_df


def create_llm_features(df, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    df = df.to_pandas()
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        tokenizer,
        fn_kwargs={'truncation': True, 'max_length': 1024},
        input_columns='full_text'
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer
    )

    predictions = trainer.predict(dataset).predictions

    llm_df = pd.DataFrame({
        f'{model_id}_label_{column_index}': predictions[:, column_index]
        for column_index in range(predictions.shape[1])
    })

    return llm_df