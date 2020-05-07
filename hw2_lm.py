import itertools
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def sum_str(str_array):
    sum_ = ''
    for str_ in str_array:
        sum_ += str_
    return sum_

class LanguageModel:
    def __init__(self, n_gram, is_laplace_smoothing, backoff=None):
        self.n = n_gram
        self.is_laplace = is_laplace_smoothing
        self.b = backoff
        self.add_term = 0
        self.n_grams = {}
        self.filenames = sys.argv
        
    def train(self, training_file_path): #train statistical language model on the training data. 
        with open(training_file_path, 'r') as f: #open the file. 
            vocabulary = {}
            vocabulary["<UNK>"] = 0 
            tokens = []
            for line in f.readlines():
                words = line.split()
                for word in words:
                   # print(word)
                    tokens.append(word)
                    if word in vocabulary: #populate vocab dict
                        vocabulary[word] += 1
                    else:
                        vocabulary[word] = 1 
            
            for key in vocabulary:
                if vocabulary[key] == 1:
                   # del vocabulary[key]
                    vocabulary["<UNK>"] += 1
                    for i, t in enumerate(tokens):
                        if t == key:
                            tokens[i] = "<UNK>" #replace unknown words
            self.tokens = tokens
            for i, token in enumerate(tokens):
                if i >= self.n - 1:
                    key = sum_str([tokens[i - j] for j in range(0, self.n)]) #put all unique n_grams in the dictionary. 
                #    print("key = ", key)
                    if key in self.n_grams:
                        self.n_grams[key] += 1
                    else:
                        self.n_grams[key] = 0      
            
            self.add_term = len(set(tokens))
            self.sum_add = 0   
            all_keys = itertools.permutations(vocabulary, r=self.n)
            all_keys = list(all_keys)
            self.allkeys = all_keys
            
            if self.is_laplace == True:
                for key in all_keys:
                    if not sum_str(key) in self.n_grams:
                        self.n_grams[sum_str(key)] = 1
                        self.sum_add += self.add_term
            else:
                for key in all_keys:
                    if not sum_str(key) in self.n_grams:
                        self.n_grams[sum_str(key)] = 0
            
            self.lenTokens = len(tokens)
            
    def generate(self, num_sentences): #generate list of strings of length num_sentences
        self.n_probs = {key:(value / (self.lenTokens / self.n)) for (key,value) in self.n_grams.items()}
        start_vals = []
        num_sentences *= 2
        start_token = ''
        end_token = ''
        if self.n > 1:
            for _ in range(self.n - 1):
                start_token += '<s>'
                end_token += '</s>'
        else:
            start_token = '<s>'
            end_token = '</s>'
            
        for (key, value) in self.n_probs.items():
           # if start_token in key:
           #     start_vals.append((key, value))
           # flag = False
            if key[0:2] == start_token[0:2]:
                start_vals.append((key, value))
            
        randvalues = np.random.rand(num_sentences - 1)
       # print(randvalues)
        sentence = []
        iter_ = 0
        rsum = 0
        val = randvalues[0]
        for (key, prob) in start_vals:
            rsum += (prob * 100) / (len(start_vals))
      #      print(rsum)
            if rsum > val - 0.1 and rsum <= val + 0.1:
                sentence.append(key)
              # print("append: ", key)
                iter_ += 1
                break
        if len(sentence) == 0 and self.n > 1:
            sentence.append("<s>")
        if self.n > 1:
            sword = sentence[0]
            s_word_fields = sword.split('<s>')
            prev_word = s_word_fields[self.n-1] #split off prev word
          # print("previous word: ", prev_word)
            ii = 1
            while ii < len(randvalues):
               # print("in gen while")
                grams = []
                for (key, value) in self.n_probs.items():
                    flag = True
                    if len(prev_word) < len(key):
                        for i, ch in enumerate(prev_word):
                            if ch != key[i]:
                                flag = False
                        if flag:
                        #   print("grams appended: ", key)
                            grams.append((key, value))
                for g in grams:
                        val = randvalues[ii]
                        rsum = 0
                        for (key, prob) in grams:
                            rsum += prob
                            if rsum > val -0.05 and rsum <= val + 0.05:
                                sentence.append(key[len(prev_word):])
                                prev_word = key[len(prev_word):] #split off prev word
                                if prev_word == "</s>":
                                    sentence.append("</s>")
                                    return sentence
                               #print("new previous word: ", prev_word)
                                break
                        break
                ii += 1
        else:
            sentence.append("<s>")
            ii = 1
            while ii < len(randvalues):
                val = randvalues[ii]
                rsum = 0
                for (key, prob) in self.n_grams.items():
                    rsum += prob
                    if rsum > val - 0.05 and rsum <= val + 0.05:
                        sentence.append(key)
                ii += 1
        sentence.append("</s>")
        return sentence
                        
    
    def score(self, sentence): #return probability of a sentence given your language model.
        p = 1
        number_ngrams = len(self.n_grams)
        for i, word in enumerate(sentence):
            if i >= self.n -1:
                word = sum_str([self.tokens[i - j] for j in range(0, self.n)])
                if word in self.n_grams:
                 #  print("word was in dictionary: ", word)
                    count_of_word = self.n_grams[word]
                    p_word = count_of_word / number_ngrams
                    if not p_word == 0:
                   #    print("p_word = ", p_word)
                        p = p * p_word
                   #print("p = ", p)
                else:
                    pass
                 #  print("key not in dictionary: ", word)
        return p
    
    
def main():
    bigramModel = LanguageModel(n_gram=2, is_laplace_smoothing=True)
  # path = sys.argv[0]
  # path2 = sys.argv[1]
    path = 'berp-training.txt'
    path2 = 'hw2-test.txt'
    print(path, path2)
    bigramModel.train(path)
    print("first bigram model trained")
    random_corpus = [bigramModel.generate(10) for _ in range(3)]
    ps = []
    f1 = open('hw2-bigram-generated.txt', 'w+')
    for i in random_corpus:
        score = bigramModel.score(i)
        ps.append(score)
        f1.write(str(i))
        f1.write(str(score))
        f1.write('\n')
    f2 = open('hw2-bigram-out.txt', 'w+')
    f3 = open('hw2-unigram-out.txt', 'w+')
    
    test_file = open(path)
    for lin in test_file.readlines():
        f2.write(lin)
        f2.write(str(bigramModel.score(lin.split())))
        
    print("score written from first bigram model")
 
    with PdfPages(r'hw2-bigram-histogram.pdf') as pdf_out:
        plt.hist(ps)
        plt.grid(True)
        pdf_out.savefig()
        plt.close()

    bigram_model_two = LanguageModel(n_gram=2, is_laplace_smoothing=True)
    bigram_model_two.train(path2)
    print("second bigram model trained")
    corp2 = [bigram_model_two.generate(10) for _ in range(5)]

    ps2 = []

    for i in corp2:
        ps2.append(bigram_model_two.score(i))
    plt.hist(ps2)

    unigramModel = LanguageModel(n_gram=1, is_laplace_smoothing=True)
    unigramModel.train(path2)
    print("first unigram model trained")
    for lin in test_file.readlines():
        f3.write(lin)
        f3.write(unigramModel.score(lin.split()))

    randomCorp = [unigramModel.generate(10) for _ in range(5)]
    f2 = open('hw2-unigram-generated.txt', 'w+')
    ps_uni = []
    for i in randomCorp:
        s = unigramModel.score(i)
        ps_uni.append(s)
        f2.write(str(i))
        f2.write(str(s))
        f2.write('\n')
    
    with PdfPages(r'hw2-unigram-histogram.pdf') as pdf_out:
        plt.hist(ps_uni)
        plt.grid(True)
        pdf_out.savefig()
        plt.close()

    uniModel2 = LanguageModel(n_gram=1, is_laplace_smoothing=True)

    uniModel2.train(path)
    print("second unigram model trained")
    random_sent = [uniModel2.generate(10) for _ in range(5)]

    ps_uni_two = []

    for i in random_sent:
        ps_uni_two.append(uniModel2.score(i))

    plt.hist(ps_uni_two)
    
    file_unigram_gen = open('hw2-unigram-generated.txt', 'w+')
    for r in random_sent:
        file_unigram_gen.write(str(r))
        file_unigram_gen.write('\n')
    
    file_bigram_gen = open('hw2-bigram-generated.txt', 'w+')
    for r in random_corpus:
        file_bigram_gen.write(str(r))
        file_bigram_gen.write('\n')
    
    f1.close()
    f2.close()
    f3.close()
    #f4.close()
if __name__=="__main__":
    main()