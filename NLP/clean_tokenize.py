from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from collections import defaultdict
import pickle
from tqdm import tqdm
import numpy as np
# !wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en -P ../data
# !wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de -P ../data

## Create argparse for the file
import argparse
parser = argparse.ArgumentParser(description='Clean and tokenize the data')
parser.add_argument('--data_root', type=str, default='../data/', help='Root directory of the data')
parser.add_argument('--d1', type=str, default='train.en', help='Name of dataset to be translated')
parser.add_argument('--d2', type=str, default='train.de', help='Name of dataset to be translated into')
parser.add_argument('--max_len', type=int, default=50, help='Maximum length of the sentence')
parser.add_argument('--min_count', type=int, default=1000, help='Minimum count of the character')
parser.add_argument('--vocab_size', type=int, default=25000, help='Size of the vocabulary')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training BPE tokenizer')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()

##  python clean_tokenize.py --data_root ../data/en_hin/ --d1 IITB.en-hi.en --d2 IITB.en-hi.hi --max_len 50 --min_count 1000 --vocab_size 25000 --batch_size 1000 --seed 42

## Set random seed
np.random.seed(args.seed)
min_count = args.min_count
max_len = args.max_len
vocab_size = args.vocab_size
batch_size = args.batch_size
data_root = args.data_root
d1 = args.d1
d2 = args.d2


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

    def remove_sentence_with_zero_length(self):
        sens_idx = []
        for idx, sen in enumerate(self.sentences):
            if len(sen) > 0:
                sens_idx.append(idx)
        return sens_idx

    def create_cleaned_data(self, idxs, ext):
        cleaned_sentences = [self.sentences[i] for i in idxs]
        self.sentences = cleaned_sentences

        clean_file = open(data_root + 'cleaned_train.' + ext, 'w+')
        for i in (range(len(self.sentences))):  
            clean_file.write(self.sentences[i] + '\n')
            
    def batch_iterator(self, batch_size):
        for i in range(0, len(self.sentences), batch_size):
            yield self.sentences[i : i + batch_size]


## Loading data
print("Loading data")
loader_d1 = TextLoader(data_root + d1)
loader_d2 = TextLoader(data_root + d2)

print("Number of sentence pairs = ", len(loader_d1.sentences))

print("Cleaning data")

## Get all unique chars
chars_d1 = loader_d1.get_all_chars()
chars_d2 = loader_d2.get_all_chars()

chars = defaultdict(lambda:0)
for k,v in chars_d1.items():
    chars[k] += v
for k,v in chars_d2.items():
    chars[k] += v

chars = dict(sorted(chars.items(), key=lambda item: item[1], reverse=True))

## Remove sentences with rare chars
valid_idxs = []
cleaned_en = loader_d1.remove_sentence_with_rare_chars(1000, chars)
cleaned_de = loader_d2.remove_sentence_with_rare_chars(1000, chars)
non_zero_len_en = loader_d1.remove_sentence_with_zero_length()
non_zero_len_de = loader_d2.remove_sentence_with_zero_length()

print("len of cleaned_en = ", len(cleaned_en), "len of cleaned_de = ", len(cleaned_de), "len of non_zero_len_en = ", len(non_zero_len_en), "len of non_zero_len_de = ", len(non_zero_len_de))

cleaned_idx = list(set(cleaned_en) & set(cleaned_de) & set(non_zero_len_en) & set(non_zero_len_de))
print("len of cleaned idxs = ", len(cleaned_idx))

loader_d1.create_cleaned_data(cleaned_idx, 'd1')
loader_d2.create_cleaned_data(cleaned_idx, 'd2')


## Loading cleaned data
loader_d1 = TextLoader(data_root + 'cleaned_train.d1')
loader_d2 = TextLoader(data_root + 'cleaned_train.d2')

assert(len(loader_d1.sentences) == len(loader_d2.sentences))

# ## Byte Pair Encoding
#  
# - Starts with a dictionary of individual characters and merges them to create new words upto the max size of vocabulary. 
# - No need for UNK token sice if the word will be tokenized into the largest available words in dictionary and into individual characters in the worst case.
# - White space information between the words is not preserved.


print("Training BPE tokenizer")
tokenizer_d1 = Tokenizer(models.BPE())
tokenizer_d1.normalizer = normalizers.Lowercase()
# tokenizer_en.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer_d1.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

trainer_d1 = trainers.BpeTrainer(vocab_size=25000,special_tokens=["[PAD]"])
tokenizer_d1.train_from_iterator(loader_d1.batch_iterator(1000), trainer=trainer_d1)

# tokenizer_en.decoder =  decoders.ByteLevel()
tokenizer_d1.save(data_root + "tokenizer_d1_25000.json")


tokenizer_d2 = Tokenizer(models.BPE())
tokenizer_d2.normalizer = normalizers.Lowercase()
# tokenizer_de.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer_d2.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

# print(tokenizer.pre_tokenizer.pre_tokenize_str("Ich liebe das wirklich ."))

trainer_d2 = trainers.BpeTrainer(vocab_size=25000, special_tokens=["[SOS]","[EOS]","[PAD]"])
tokenizer_d2.train_from_iterator(loader_d2.batch_iterator(1000), trainer=trainer_d2)

tokenizer_d2.post_processor = processors.TemplateProcessing(
    single=f"[SOS]:0 $A:0 [EOS]:0",
    special_tokens=[("[SOS]", 0), ("[EOS]", 1)])
# tokenizer_de.decoder =  decoders.ByteLevel()

tokenizer_d2.save(data_root +  "tokenizer_d2_25000.json")

## Tokenize the data

print("Tokenizing data")
d1_tokens= []
encodes_len = []
for batch in (loader_d1.batch_iterator(1024)):
    encoding = tokenizer_d1.encode_batch(batch)
    for i in range(len(encoding)):
        d1_tokens.append(encoding[i].ids)
        encodes_len.append(len(encoding[i].ids))

d2_tokens = []
for batch in (loader_d2.batch_iterator(1024)):
    encoding = tokenizer_d2.encode_batch(batch)
    for i in range(len(encoding)):
        d2_tokens.append(encoding[i].ids)

encodes_len , d1_tokens, d2_tokens = (list(t) for t in zip(*sorted(zip(encodes_len , d1_tokens, d2_tokens))))


## Get index of sentences with length > 0

idxs_d1 = [i for i in range(len(d1_tokens)) if len(d1_tokens[i]) > 0]
idxs_d2 = [i for i in range(len(d2_tokens)) if len(d2_tokens[i]) > 0]

idxs = list(set(idxs_d1) & set(idxs_d2))

d1_tokens = [d1_tokens[i] for i in idxs]
d2_tokens = [d2_tokens[i] for i in idxs]


## assert len of all d1  > 0
for i in range(len(d1_tokens)):
    assert(len(d1_tokens[i]) > 0)
    assert(len(d2_tokens[i]) > 0)

print("Number of Sentences = ", len(d1_tokens), len(d2_tokens))

# Dump the tokenized lists as pickle files for faster retrieval
with open(data_root + 'd1_tokenized.pkl', 'wb') as f:
    d1_tokens = [np.array(i, dtype=np.uint16) for i in d1_tokens]
    pickle.dump(d1_tokens, f)

with open(data_root + 'd2_tokenized.pkl', 'wb') as f:
    d2_tokens = [np.array(i, dtype=np.uint16) for i in d2_tokens]
    pickle.dump(d2_tokens, f)

# Load the tokenized lists from pickle files
# with open(data_root + 'd1_tokenized.pkl', 'rb') as f:
    # d1_tokens = pickle.load(f)

# with open(data_root + 'd2_tokenized.pkl', 'rb') as f:
    # d2_tokens = pickle.load(f)

# print("Number of Sentences = ", len(d1_tokens))
# assert(len(d2_tokens) == len(d1_tokens))

# tokenizer_d1 = Tokenizer.from_file(data_root + "tokenizer_d1_25000.json")
# tokenizer_d2 = Tokenizer.from_file(data_root + "tokenizer_d2_25000.json")