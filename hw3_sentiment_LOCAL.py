# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math
np.random.seed(1024)
"""
Your name and file comment here: Theodore Margoles
"""


"""
Cite your sources here: 
    1. Speech and Language Processing (3rd edition)
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""
def generate_tuples_from_file(training_file_path):
  tfile = open(training_file_path, 'r', encoding='utf-8')
  examples = []
  for line in tfile.readlines():
      line_fields = line.split('\t')
      if len(line_fields) == 3:
          X = line_fields[1]
          y = line_fields[2][:-1]
          _id = line_fields[0]
          examples.append((_id, X, y))
      else:
          continue
  return examples      
  
def precision(gold_labels, classified_labels):
  tp = 0
  fp = 0
  for (true_label, pred_label) in zip(gold_labels, classified_labels):
      if true_label == "1": #true label is positive
          if pred_label == "1": #true positive
              tp += 1
          else: #false negative
              pass
      else: #true_label is 0
          if pred_label == "1": #false positive
              fp += 1
          else: #true negative
              pass
  return tp / (tp + fp)

def recall(gold_labels, classified_labels):
  tp = 0
  fn = 0
  for (true_label, pred_label) in zip(gold_labels, classified_labels):
      if true_label == "1":
          if pred_label == "1": #true positive
              tp += 1
          elif pred_label == "0": #true label was 1 but predicted label was 0
              fn += 1
  return tp / (tp + fn)

def f1(gold_labels, classified_labels):
  P = precision(gold_labels, classified_labels)
  R = recall(gold_labels, classified_labels)
  return (2*P*R) / (P + R)

"""
Implement any other non-required functions here
"""
def regularize(Theta, Lambda):
    return (1/Lambda) * np.linalg.norm(Theta)

def predict_labels_sa(Sent_Analyzer, examples):
    y_hats = []
    for (_, X, _) in examples:
        X = X.lower()
        X = Sent_Analyzer.featurize(X)
        y_hats.append(Sent_Analyzer.classify(X))
    return y_hats

def get_y_golds(examples):
    y_golds = []
    for (_, _, y) in examples:
        y_golds.append(y)
    return(y_golds)
    
def accuracy_score(gold_labels, classified_labels):
    wrong = 0
    total = len(gold_labels)
    for (i, j) in zip(gold_labels, classified_labels):
        if i != j:
            wrong += 1
    return (total - wrong) / total

def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

def propogate(x, w, bias):
    return np.dot(x, w) + bias

def update_weights(weights, x, b, y, lr):
    z = propogate(x, weights, b)
    los = sigmoid(z) - float(y) #+ reg_term
    for i, w in enumerate(weights):
        term = los * x[i]
        weights[i] = weights[i] - (lr * term) #
    return weights

"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:
  def __init__(self):
    self.y_labels = []
    self.vocabulary = {'<UNK>' : 1}
    self.tokens = []
    self.x_features = []
    self.pos_text_total = ''
    self.neg_text_total = ''
    self.is_trained = False
    
  def train(self, examples):
    for (_, X, y) in examples:
        X = X.lower()
        self.x_features.append(self.featurize(X))
        self.y_labels.append(y)
        if y == "1":
            self.pos_text_total += X
        elif y == "0":
            self.neg_text_total += X
        else:
            print('error { return code 1 }:  label not in {''1'', ''0''}\nquitting now\n')
            sys.exit(1)
        words = X.split(' ')
        for w in words:
            if w in self.vocabulary:
                self.vocabulary[w] += 1
            else:
                self.vocabulary[w] = 1
            self.tokens.append(w)    
    self.vocab_list = list(self.vocabulary.keys())
    self.bigdocs = {"0" : self.neg_text_total, "1" : self.pos_text_total}
    nv = len(self.vocabulary)
    prior = np.log(0.5)
    self.log_prior = {"0" : prior, "1" : prior}
    pos_words = self.pos_text_total.split(' ')
    neg_words = self.neg_text_total.split(' ')
    self.length_pos = len(pos_words)
    self.length_neg = len(neg_words)
    self.loglhood = {}
    for word in self.vocabulary:
        count_pos = len([w for w in pos_words if w == word]) #count of word in positive class.
        count_neg = len([w for w in neg_words if w == word]) #count of word in the negative class.
        llpos = np.log((count_pos + 1) / (self.length_pos + nv))
        llneg = np.log((count_neg + 1) / (self.length_neg + nv))
        self.loglhood[word] = {"0": llneg, "1": llpos}

  def score(self, data):
    #returns dictoinaty with p(data | c)
    sum_dict = {"0" : self.log_prior["0"], "1" : self.log_prior["1"],}
    for key in sum_dict:
        for word in data:
            if word in self.vocabulary:
              #  print('known word: ', word)
                sum_dict[key] += self.loglhood[word][key]
           # else:
              #  print('unknown word: ', word)
    sum_dict["1"] = np.exp(sum_dict["1"])
    sum_dict["0"] = np.exp(sum_dict["0"])
    return sum_dict

  def classify(self, data):
    words = [f[0] for f in data]
    sum_dict = self.score(words)
    ppos = sum_dict["1"]
    pneg = sum_dict["0"]
    if ppos > pneg:
        return "1"
    else:
        return "0"

  def featurize(self, data):
    x_feature = []
    for xword in data.split(' '):
        x_feature.append((xword, True))
    return x_feature

  def __str__(self):
    return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:
  def __init__(self):
    self.x_strings = []
    self.y_labels = []
    self.vocabulary = {}
    self.tokens = []
    self.bias = 0
    self.lr = 0.01
    self.epochs = 10

  def train(self, examples):
    for (_, X, y) in examples:
        X = X.lower()
        self.x_strings.append(X)
        self.y_labels.append(y)
        
        words = X.split(' ')
        for w in words:
            if w in self.vocabulary:
                self.vocabulary[w] += 1
            else:
                self.vocabulary[w] = 1
            self.tokens.append(w) 
    self.x_features = []
    self.vocab_list = list(self.vocabulary)
    vlen = len(self.vocabulary)
    for(_, X, _) in examples:
        self.x_features.append(self.featurize(X))
    self.weights = np.random.random(vlen) - np.random.random(vlen)
    xy_pairs = zip(self.x_features, self.y_labels)
    for _ in range(1, self.epochs+1):
    #   print("epoch: %d / %d" % (_, self.epochs))
        for (x, y) in xy_pairs:
            self.weights = update_weights(self.weights, x, self.bias, y, self.lr)
            
  def score(self, data):
    sigma = sigmoid(propogate(data, self.weights, self.bias))
    
    scores = {"1" : sigma, 
             "0" : 1 - sigma}
    return scores

  def classify(self, data):
    scores = self.score(data)
    if scores["1"] > scores["0"]:
        return "1"
    else:
        return "0"

  def featurize(self, data):
    words = data.split()
    feature = np.zeros(len(self.vocabulary))
    for w in words:
        if w in self.vocabulary:
            indx = self.vocab_list.index(w)
            feature[indx] += 1
    return feature
            

  def __str__(self):
    return "Logistic Regression - Bag of Words"

if __name__ == "__main__":
#  if len(sys.argv) != 3:
#    print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
#    sys.exit(1)

#  training = sys.argv[1]
#  testing = sys.argv[2]
  training = 'train_file.txt'
  testing = 'dev_file.txt'
  train_ex = generate_tuples_from_file(training)
  test_ex = generate_tuples_from_file(testing)

  sa = SentimentAnalysis()
 
  sa.train(train_ex)
  y_hats = predict_labels_sa(sa, test_ex)
  y_gold = get_y_golds(test_ex)
  print(sa)
  print("accuracy: ", accuracy_score(y_gold, y_hats))
  print("recall: ", recall(y_gold, y_hats))
  print("precision: ", precision(y_gold, y_hats))
  print("f1-score: ", f1(y_gold, y_hats))
  # do the things that you need to with your base class
  
  improved = SentimentAnalysisImproved()
  improved.train(train_ex)
  yh = predict_labels_sa(improved, test_ex)
  yg = get_y_golds(test_ex)
  print(improved)
  print("accuracy: ", accuracy_score(yg, yh))
  print("recall: ", recall(yg, yh))
  print("precision: ", precision(yg, yh))
  print("f1-score: ", f1(yg, yh))
  # do the things that you need to with your improved class










