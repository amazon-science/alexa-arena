# Modified from https://github.com/pytorch/tutorials/blob/master/intermediate_source/seq2seq_translation_tutorial.py - Licensed under the
# BSD-3-Clause License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2count = {"<<stop>>": 0, "<<pad>>": 0}
        self.index2word = {0: "<<stop>>", 1: "<<pad>>"}
        self.word2index = {"<<stop>>": 0, "<<pad>>": 1}
        self.n_words = 2  # Count SOS and EOS
    
    def __len__(self):
        return self.n_words

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def contains_same_content(self, lang):
        if lang.index2word == self.index2word:
            return True
        return False


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index]

