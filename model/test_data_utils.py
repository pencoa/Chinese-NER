import pickle
import numpy as np
from data_utils import getDataset, get_vocabs, get_processing_word, get_polyglot_vocab, UNK, write_vocab, load_vocab, \
    export_trimmed_polyglot_vectors, get_trimmed_polyglot_vectors, pad_sequences, minibatches, get_chunks


# test without words idx dict
def test_processing_words_without_words_idx_dict():
    processing_word = get_processing_word()
    word1 = processing_word("娃哈哈")
    word2 = processing_word("12345")
    print(word1, word2)

# test using words idx dict and allow unknow words
def test_processing_words_with_words_idx_dict_and_allow_unknow():
    d = dict()
    d['娃哈哈'] = 1
    d['#####'] = 3
    d['<UNK>'] = 0
    processing_word = get_processing_word(d, True)
    word1 = processing_word("娃哈哈")
    word2 = processing_word("12345")
    word3 = processing_word("xixihehe")
    print(word1, word2, word3)


def test_dataset():
    # test getDataset and get_vocabs
    processing_word = get_processing_word()
    dev = getDataset("../data/test_ner.txt", processing_word)
    vocab_words, vocab_tags = get_vocabs([dev])

    # get common vocab from dev file and polyglot
    vocab_poly = get_polyglot_vocab("../data/polyglot-zh.pkl")
    vocab = vocab_words & vocab_poly
    vocab.add(UNK)

    write_vocab(vocab, "../data/words.txt")
    write_vocab(vocab_tags, "../data/tags.txt")

    vocab = load_vocab("../data/words.txt")
    export_trimmed_polyglot_vectors(vocab, "../polyglot-zh.pkl", "../data/polyglot.trimmed.npz", 64)
    data = get_trimmed_polyglot_vectors("../data/polyglot.trimmed.npz")



def check_npz():
    vocab = load_vocab("../data/words.txt")
    idx = vocab['硕士']

    with open('../data/polyglot-zh.pkl', 'rb') as f:
        words, embeddings = pickle.load(f, encoding="latin1")
        words = list(words)
        embeddings = list(embeddings)
    word_idx = words.index('硕士')

    return (data[idx] == embeddings[word_idx])


def test_seq_padding():
    a = np.array([[1, 2, 3, 5], [2, 3, 2], [3, 1, 4, 1, 5, 9]])
    seq, length = pad_sequences(a, 0)


def test_minibatch():
    processing_word = get_processing_word()
    dev = getDataset("../data/test_ner.txt", processing_word)
    for i, (w, t) in enumerate(minibatches(dev, 5)):
        print(w, t)


def test_chunk():
    tags_dict = load_vocab("../data/tags.txt")
    seq = [10, 3, 6, 12, 12, 6]
    chunks = get_chunks(seq, tags_dict)
    return chunks


if __name__ == '__main__':
    b = test_chunk()
