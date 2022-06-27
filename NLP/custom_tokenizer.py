from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from collections import defaultdict

class TextLoader:
    def __init__(self) -> None:
        pass

    def create_vocab(self, PATH):

        # vocab_dict = defaultdict(lambda :0)

        corpus = open(PATH, 'r').read()

        self.sentences = corpus.split('\n')
        # print(1)
        # wordList = []
        # for line in self.sentences:
        #     wordList += line.lower().split(' ') 
        # print(2)

        # for word in wordList:
        #     vocab_dict[word] += 1

        # return vocab_dict, self.sentences
        print(3)

        return 1,1

    def remove_rare_words(self,vocab_dict, min_repeat):
        # vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))
        wordList = [k for k,v in vocab_dict.items() if v >= min_repeat]
        print(4)
        vocab = set(wordList)
        print(len(vocab), len(wordList))

    def batch_iterator(self, batch_size):
        for i in range(0, len(self.sentences), batch_size):
            yield self.sentences[i : i + batch_size]


loader = TextLoader()
vde, sentences = loader.create_vocab('train.en')

# loader.batch_iterator(5)
# loader.remove_rare_words(vde, 30)
# loader.remove_rare_words(vde, 35)
# vd, _ = loader.create_vocab('train.de')


tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# print(tokenizer.pre_tokenizer.pre_tokenize_str("Ich liebe das wirklich ."))

trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(loader.batch_iterator(100), trainer=trainer)