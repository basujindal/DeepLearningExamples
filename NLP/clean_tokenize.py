from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from collections import defaultdict
import pickle
import numpy as np
# !wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en -P ../data
# !wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de -P ../data

## Create argparse for the file
import argparse
parser = argparse.ArgumentParser(description='Clean and tokenize the data')
parser.add_argument('--en', type=str, default='../data/train.en', help='Path to english data')
parser.add_argument('--de', type=str, default='../data/train.de', help='Path to german data')
parser.add_argument('--max_len', type=int, default=50, help='Maximum length of the sentence')
parser.add_argument('--min_count', type=int, default=1000, help='Minimum count of the character')
parser.add_argument('--vocab_size', type=int, default=25000, help='Size of the vocabulary')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training BPE tokenizer')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

## Set random seed
np.random.seed(args.seed)
min_count = args.min_count
max_len = args.max_len
vocab_size = args.vocab_size
batch_size = args.batch_size



class TextLoader:
    def __init__(self, PATH):
        
        self.corpus = open(PATH, 'r').read()
        self.sentences = self.corpus.split('\n')

    def remove_rare_words(self,vocab_dict, min_repeat):

        wordList = [k for k,v in vocab_dict.items() if v >= min_repeat]
        vocab = set(wordList)
        print(len(vocab), len(wordList))

    def get_all_chars(self):
        chars = defaultdict(lambda: 0)
        for sen in self.sentences:
            for char in sen:
                chars[char] += 1

        return chars

    def remove_sentence_with_rare_chars(self, min_count, dict):
        sens_idx = []
        for idx, sen in enumerate(self.sentences):
            flag = 1
            for char in sen:
                if(dict[char] < min_count):
                    flag = 0
                    break
            if flag:
                sens_idx.append(idx)
        return sens_idx

    def create_cleaned_data(self, idxs, ext):
        cleaned_sentences = [self.sentences[i] for i in idxs]
        self.sentences = cleaned_sentences
        print(len(self.sentences))

        clean_file = open('../data/cleaned_train.' + ext, 'w+')
        for i in range(len(self.sentences)):  
            clean_file.write(self.sentences[i] + '\n')
            
    def batch_iterator(self, batch_size):
        for i in range(0, len(self.sentences), batch_size):
            yield self.sentences[i : i + batch_size]


## Loading data
print("Loading data")
loader_en = TextLoader(args.en)
loader_de = TextLoader(args.de)

print("Cleaning data")

## Get all unique chars
chars_en = loader_en.get_all_chars()
chars_de = loader_de.get_all_chars()

chars = defaultdict(lambda:0)
for k,v in chars_en.items():
    chars[k] += v
for k,v in chars_de.items():
    chars[k] += v

chars = dict(sorted(chars.items(), key=lambda item: item[1], reverse=True))
# dict(sorted(chars_de.items(), key=lambda item: item[1], reverse=True))

## Remove sentences with rare chars
cleaned_en = loader_en.remove_sentence_with_rare_chars(1000, chars)
cleaned_de = loader_de.remove_sentence_with_rare_chars(1000, chars)
cleaned_idx = list(set(cleaned_en) & set(cleaned_de))
len(cleaned_idx), len(cleaned_en), len(cleaned_de)

loader_de.create_cleaned_data(cleaned_idx, 'de')
loader_en.create_cleaned_data(cleaned_idx, 'en')

## Loading cleaned data
loader_en = TextLoader('../data/cleaned_train.en')
loader_de = TextLoader('../data/cleaned_train.de')

assert(len(loader_de.sentences) == len(loader_en.sentences))

# ## Byte Pair Encoding
#  
# - Starts with a dictionary of individual characters and merges them to create new words upto the max size of vocabulary. 
# - No need for UNK token sice if the word will be tokenized into the largest available words in dictionary and into individual characters in the worst case.
# - White space information between the words is not preserved.


print("Training BPE tokenizer")
tokenizer_en = Tokenizer(models.BPE())
tokenizer_en.normalizer = normalizers.Lowercase()
# tokenizer_en.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer_en.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

trainer_en = trainers.BpeTrainer(vocab_size=25000,special_tokens=["[PAD]"])
tokenizer_en.train_from_iterator(loader_en.batch_iterator(1000), trainer=trainer_en)

# tokenizer_en.decoder =  decoders.ByteLevel()
tokenizer_en.save("../data/tokenizer_en_25000.json")


li = []
for batch in loader_en.batch_iterator(1024):
    encoding = tokenizer_en.encode_batch(batch)
    for i in range(len(encoding)):
        li.append(len(encoding[i].ids))

li = sorted(li, reverse=True)


tokenizer_de = Tokenizer(models.BPE())
tokenizer_de.normalizer = normalizers.Lowercase()
# tokenizer_de.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer_de.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# print(tokenizer.pre_tokenizer.pre_tokenize_str("Ich liebe das wirklich ."))

trainer_de = trainers.BpeTrainer(vocab_size=25000, special_tokens=["[SOS]","[EOS]","[PAD]"])
tokenizer_de.train_from_iterator(loader_de.batch_iterator(1000), trainer=trainer_de)

tokenizer_de.post_processor = processors.TemplateProcessing(
    single=f"[SOS]:0 $A:0 [EOS]:0",
    special_tokens=[("[SOS]", 0), ("[EOS]", 1)])
# tokenizer_de.decoder =  decoders.ByteLevel()

tokenizer_de.save("../data/tokenizer_de_25000.json")

## Tokenize the data
en_tokens= []
encodes_len = []
for idx, batch in enumerate(loader_en.batch_iterator(1024)):
    encoding = tokenizer_en.encode_batch(batch)
    for i in range(len(encoding)):
        en_tokens.append(encoding[i].ids)
        encodes_len.append(len(encoding[i].ids))

de_tokens = []
for idx, batch in enumerate(loader_de.batch_iterator(1024)):
    encoding = tokenizer_de.encode_batch(batch)
    for i in range(len(encoding)):
        de_tokens.append(encoding[i].ids)

encodes_len , en_tokens, de_tokens = (list(t) for t in zip(*sorted(zip(encodes_len , en_tokens, de_tokens))))


# Dump the tokenized lists as pickle files for faster retrieval
with open('../data/en_tokenized.pkl', 'wb') as f:
    en_tokens = [np.array(i, dtype=np.uint16) for i in en_tokens]
    pickle.dump(en_tokens, f)

with open('../data/de_tokenized.pkl', 'wb') as f:
    de_tokens = [np.array(i, dtype=np.uint16) for i in de_tokens]
    pickle.dump(de_tokens, f)

# Load the tokenized lists from pickle files
with open('../data/en_tokenized.pkl', 'rb') as f:
    en_tokens = pickle.load(f)

with open('../data/de_tokenized.pkl', 'rb') as f:
    de_tokens = pickle.load(f)

print(len(en_tokens))
assert(len(de_tokens) == len(en_tokens))


tokenizer_de = Tokenizer.from_file("../data/tokenizer_de_25000.json")
tokenizer_en = Tokenizer.from_file("../data/tokenizer_en_25000.json")