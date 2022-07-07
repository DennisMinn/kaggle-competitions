from datetime import datetime


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def token_len(text, tokenizer):
    return len(tokenizer.encode_plus(text).tokens())
