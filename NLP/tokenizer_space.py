from collections import defaultdict

class TextLoader:
    def __init__(self) -> None:
        pass


    def create_vocab(self, PATH):

        vocab_dict = defaultdict(lambda :0)

        corpus = open(PATH, 'r').read()

        sentences = corpus.split('\n')
        print(1)
        wordList = []
        for line in sentences:
            wordList += line.lower().split(' ') 
        print(2)

        for word in wordList:
            vocab_dict[word] += 1

        return vocab_dict

        print(3)
    def remove_rare_words(self,vocab_dict, min_repeat):
        # vocab_dict = dict(sorted(vocab_dict.items(), key=lambda item: item[1]))
        wordList = [k for k,v in vocab_dict.items() if v >= min_repeat]
        print(4)
        vocab = set(wordList)
        print(len(vocab), len(wordList))

loader = TextLoader()
vde = loader.create_vocab('train.en')
loader.remove_rare_words(vde, 30)
loader.remove_rare_words(vde, 35)
vd = loader.create_vocab('train.de')
# loader.remove_rare_words(vd, 70)

        