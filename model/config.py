import os


from .general_utils import get_logger
from .data_utils import get_trimmed_polyglot_vectors, load_vocab, \
        get_processing_word

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)

        self.nwords     = len(self.vocab_words)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words)
        self.processing_tag  = get_processing_word(self.vocab_tags, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_polyglot_vectors(self.filename_trimmed)
                if self.use_pretrained else None)


    # general config
    dir_output = "./results/test/"
    dir_model  = dir_output + "model.weights/" # directory to save models
    path_log   = dir_output + "log.txt"
    restore_model = "./results/test/model.weights/early_best.ckpt"

    # embeddings
    dim = 64

    filename_dev = filename_test = filename_train = "./data/test_ner.txt" # test

    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "./data/words.txt"
    filename_tags = "./data/tags.txt"

    # polyglot file
    filename_polyglot = "./data/polyglot-zh.pkl"
    # trimmed embeddings
    filename_trimmed = "./data/polyglot.trimmed.npz"

    use_pretrained = True

    max_iter = None # if not None, max number of examples in Dataset

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3
    early_stop       = True
    max_train_step   = 100000

    # model hyperparameters
    hidden_size_lstm = 64 # lstm on word embeddings
    use_crf = True
