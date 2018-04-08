from model.config import Config

config = Config(load=True)

print(config.vocab_words)
print(config.ntags)
print(config.embeddings)
print(config.processing_word('副'))
print(config.processing_tag('B-TIME'))
