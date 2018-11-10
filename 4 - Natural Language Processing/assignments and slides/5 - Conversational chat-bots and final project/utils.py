import nltk
import pickle
import re
import numpy as np
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype
	
    embeddings = pd.read_csv(embeddings_path, sep='\t', header=None)
    idx = embeddings.iloc[:,1:].values
    embeddings = embeddings.iloc[:,0].values
    
    res = {}
    for i, j in zip(idx, embeddings):
        res.update({i: j})
    
    dim = embeddings.shape[1]
    
    return res, dim

	
def question_to_vec(question, embeddings, dim):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    tokens = question.split()
    word_embeddings = []
    
    # In case of words with no embeddings, then they are skipped
    for word in tokens:
        if word in embeddings:
            word_embeddings.append(embeddings[word])
    
    # If the question doesn't contain any known word with embedding
    if not word_embeddings:
        return np.zeros(dim)
    
    question_embeddings = np.mean(word_embeddings, axis=0)
    
    return question_embeddings


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
