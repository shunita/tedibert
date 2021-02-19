from scipy.sparse import lil_matrix
from tqdm import tqdm


def get_vocab(list_of_word_lists):
    vocab_set = set()
    for lst in list_of_word_lists:
        for w in lst:
            vocab_set.add(w)
    vocab = sorted(list(vocab_set))
    word_to_index = {w: i for i, w in enumerate(vocab)}
    print("vocab size: {}".format(len(vocab)))
    return word_to_index, vocab


def texts_to_BOW(texts_list, vocab):
    """embed_with_bert abstracts using BOW. (set representation).
    :param texts_list: list of abstracts, each is a list of words.
    :param vocab: dictionary of word to index
    :return:
    """
    X = lil_matrix((len(texts_list), len(vocab)))
    for i, abstract in tqdm(enumerate(texts_list), total=len(texts_list)):
        # if the word is unknown we ignore it (could possibly add UNK token)
        word_indices = [vocab[w] for w in sorted(set(abstract)) if w in vocab]
        X[i, word_indices] = 1
    return X.tocsr()