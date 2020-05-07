import sys
import numpy as np
from collections import Counter


"""
Theodore Margoles
"""


"""
Cite your sources here:
"""

def generate_tuples_from_file(file_path):
  """
  Implemented for you. 

  counts on file being formatted like:
  1 Comparison  O
  2 with  O
  3 alkaline  B
  4 phosphatases  I
  5 and O
  6 5 B
  7 - I
  8 nucleotidase  I
  9 . O

  1 Pharmacologic O
  2 aspects O
  3 of  O
  4 neonatal  O
  5 hyperbilirubinemia  O
  6 . O

  params:
    file_path - string location of the data
  return:
    a list of tuples in the format [(token, label), (token, label)...]
  """
  current = []
  f = open(file_path, "r", encoding="utf8")
  examples = []
  for line in f:
   if len(line.strip()) == 0 and len(current) > 0:
          examples.append(current)
          current = []
   else:
           pieces = line.strip().split()
           current.append(tuple(pieces[1:]))
  if len(current) > 0:
             examples.append(current)
             f.close()
  return examples

def get_words_from_tuples(examples):
  """
  You may find this useful for testing on your development data.

  params:
    examples - a list of tuples in the format [[(token, label), (token, label)...], ....]
  return:
    a list of lists of tokens
  """
  return [[t[0] for t in example] for example in examples]


def decode(data, probability_table, pointer_table):
  """
  TODO: implement
  params: 
    data - a list of tokens
    probability_table - a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
    pointer_table - a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
  return:
    a list of tuples in the format [(token, label), (token, label)...]
  """
  st_to_index = {'I':0, 'O':1, 'B':2}
  in_to_st = {value:key for key,value in st_to_index.items()}
  n = len(data)
  d_ = probability_table[n-1] #dictionary for last word. 
  dl = [d_['I'], d_['O'], d_['B']] 
  last_state = in_to_st[np.argmax(np.array(dl))] #max valued state of last word in sentence. 
  tuples = []
  tuples.append((data[n-1], last_state)) #append the last predicted token and word
  prev_state = last_state #keep track of max state to backtrack on
  for i in range(1, n):
      j = n - i
      k = n - i - 1
      back = pointer_table[j]
      cur_state = back[prev_state]
      tuples.append((data[k], cur_state))
      prev_state = cur_state
      
  ordered_tuples = []
  for j in range(n): #the resulting list is in wrong order, this code reverses its order. 
      z = n - j - 1
      ordered_tuples.append(tuples[z])
      
  return ordered_tuples
      
      
      
     
def precision(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of precision at the entity level
  """
  tp = 0
  tn = 0
  fn = 0
  fp = 0
  #for ((token, yg), (_, yh)) in zip(gold_labels, classified_labels):
  for (yg, yh) in zip(gold_labels, classified_labels):
    if yg == 'B':
        if yh == 'B' or yh == 'I':
            tp += 1
        elif yh == 'O':
            fn += 1
    elif yg == 'O':
        if yh == 'O':
            tn += 1
        elif yh == 'I' or yh == 'B':
            fp += 1
    elif yg == 'I':
        if yh == 'I' or yh == 'B':
            tp += 1
        elif yh == 'O':
            fn += 1
#  print("in precision: \n---->true positives = ", tp)
#  print("---->false positives = ", fp)
#  print("---->true negatives = ", tn)
#  print("---->false negatives = ", fn)      
  p = tp / (tp + fp)
  return p


def recall(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of recall at the entity level
  """
  tp = 0
  tn = 0
  fn = 0
  fp = 0
  for (yg, yh) in zip(gold_labels, classified_labels):
    if yg == 'B':
        if yh == 'B' or yh == 'I':
            tp += 1
        elif yh == 'O':
            fn += 1
    elif yg == 'O':
        if yh == 'O':
            tn += 1
        elif yh == 'I' or yh == 'B':
            fp += 1
    elif yg == 'I':
        if yh == 'I' or yh == 'B':
            tp += 1
        elif yh == 'O':
            fn += 1
  r = tp / (tp + fn)
  return r

def f1(gold_labels, classified_labels):
  """
  TODO: implement
  params:
    gold_labels - a list of tuples in the format [(token, label), (token, label)...]
    classified_labels - a list of tuples in the format [(token, label), (token, label)...]
  return:
    float value of f1 at the entity level
  """
  p = precision(gold_labels, classified_labels)
  r = recall(gold_labels, classified_labels)
  
  f = (2*p*r) / (p + r)
  return f

def pretty_print_table(data, list_of_dicts):
  """
  Pretty-prints probability and backpointer lists of dicts as nice tables.
  Truncates column header words after 10 characters.
  params:
    data - list of words to serve as column headers
    list_of_dicts - list of dicts with len(data) dicts and the same set of
      keys inside each dict
  return: None
  """
  # ensure that each dict has the same set of keys 155, 156 163 164 165 166 173-175
  keys = None
  for d in list_of_dicts:
            if keys is None:
                  keys = d.keys()
            else:
                  if d.keys() != keys:
                    print("Error! not all dicts have the same keys!")
                    return
  header = "\t" + "\t".join(['{:11.10s}']*len(data))
  header = header.format(*data)
  rows = []
  for k in keys:
            r = k + "\t"
            for d in list_of_dicts:
                if type(d[k]) is float:
                    r += '{:.9f}'.format(d[k]) + "\t"
                else:
                    r += '{:10.9s}'.format(str(d[k])) + "\t"
  rows.append(r)
  print(header)
  for row in rows:
            print(row)

"""
Implement any other non-required functions here
"""

"""
Implement the following class
"""
class NamedEntityRecognitionHMM:
  
  def __init__(self):
    self.vocab = {}
    self.transitions = {}
    self.emissions = {}
    self.count_states = {'I':0, 'O':0, 'B':0}
    self.pi = {'I':0, 'O':0, 'B':0}
    self.sentences = []
    self.states = {'I', 'O', 'B'}

  def train(self, examples):
        """
        Trains this model based on the given input data
        params: examples - a list of lists of (token, label) tuples
        return: None
        """
        for state in self.states:
            for state1 in self.states:
                t = state + state1
                self.transitions[t] = 0 #initialize so that every transition is in the dicionary with count 0
        current_sentence = []
        
        for i in range(len(examples)):
            sentence = examples[i]
           
            for (token, label) in sentence:
                if token not in self.vocab: #put word in vocab
                    self.vocab[token] = 1
                else:
                    self.vocab[token] += 1 #increment count in dict
                
                current_sentence.append((token, label))
                if token == '.': #if its a period we know another sentence comes next. 
                    self.sentences.append(current_sentence)
                    current_sentence = []
        for sentence in self.sentences:
            (t0, s0) = sentence[0]
            self.pi[s0] += 1 #track which tags start sentences for pi distribution
            
            prev = (t0, s0)
            d = {'I':0, 'O':0, 'B':0}
            if t0 not in self.emissions:
                d[s0] += 1
                self.emissions[t0] = d
            else:
                self.emissions[t0][s0] += 1
            for i, (ti, si) in enumerate(sentence):
                self.count_states[si] += 1 #count occurence of each tag type
                if i == 0:
                    continue
                else:
                    trans = str(prev[1] + si)
                    if trans not in self.transitions: #get p(si | si-1)
                        self.transitions[trans] = 1
                    else:
                        self.transitions[trans] += 1
                    if ti not in self.emissions: #get emissions prob = P(ti | si) (here ti means tokeni and si means state i)
                        d = {'I':0, 'O':0, 'B':0}
                        d[si] += 1
                        self.emissions[ti] = d
                    else:
                        self.emissions[ti][si] += 1
                    prev = (ti, si)
                    
        for wi in self.emissions: #get emission probabilities
            d_wi = self.emissions[wi] 
            d_wi['B'] = (d_wi['B'] + 1) / (self.count_states['B'] + 3) #laplace 
            d_wi['I'] = (d_wi['I'] + 1) / (self.count_states['I'] +  3) #
            d_wi['O'] = (d_wi['O'] + 1) / (self.count_states['O'] + 3)#
            self.emissions[wi] = d_wi
        denom = len(self.sentences) #get prior probabilities
        self.pi['O'] = self.pi['O'] / denom #prob we start a sentence with state O
        self.pi['B'] = self.pi['B'] / denom# ''     ''   state B
        self.pi['I'] = self.pi['I'] / denom
        for ti in self.transitions: #get transition probabilities. 
            tagi_1 = ti[0] #previous tag
            #tagi = ti[1] #next tag
            self.transitions[ti] = self.transitions[ti] / self.count_states[tagi_1]
        pass

  def generate_probabilities(self, data):
    """
    params: data - a list of tokens
    return: two lists of dictionaries --
      - first a list of dictionaries of states to probabilities, 
      one dictionary per word in the test data that represents the
      probability of being at that state for that word
      - second a list of dictionaries of states to states, 
      one dictionary per word in the test data that represents the 
      backpointers for which previous state led to the best probability
      for the current state
    """
    st_to_index = {'I':0, 'O':1, 'B':2}
    in_to_st = {value:key for key,value in st_to_index.items()}
    N = len(self.states)
       
    
   # data = [i for i in data if i in self.vocab] #dealing with words in vocabulary. 
    
    T = len(data)

    viterbi_table = np.zeros((N, T)) #the viterbi table
    back_table = np.zeros((N, T)) #the table of backpointers
    #will convert these to dictionaries after. 
    #w0 is the first word. 
    w0 = data[0]
    for state in self.states: #set up first column. 
        j = st_to_index[state] #j will be 0 for I, 1 for O and 2 for B
        if w0 in self.vocab:
            viterbi_table[j][0] = self.pi[state] * self.emissions[w0][state]
        else:
            viterbi_table[j][0] = 0
        back_table[j][0] = 0
    for t in range(1, T):
        for state in self.states:
            j = st_to_index[state]
            wi = data[t]
            #next line puts each of the three options that any cell chooses the max of for its value 
            #this is because the beam width is 3 for |{I, O, B}| = 3. 
            #each option is the value of the table for the previous word state (I, O and B) times the transition 
            #probability p(s | s_i-1) of the previous state transitioning to the current state
            #times the emission probability of the current word actually being that state. 
            #the best of these options (max probability) is chosen. 
            #the argmax is the index of the previous state that led to the maximum probability
            if wi in self.vocab:
                options = np.array([viterbi_table[st_to_index[sprime]][t-1] * 
                                    self.transitions[str(sprime) + str(state)] * 
                                    self.emissions[wi][state] for sprime in self.states])
            else:
                options = np.array([0.0, 0.0, 0.0])
    
            viterbi_table[j][t] = np.max(options) #fill in probability table
            back_table[j][t] = np.argmax(options) #index 0 for I, index 1 for O index 2 for B
    v_list = [] #make lists to return
    b_list = [] #
    for col in viterbi_table.T:
        d = {}
        for i in range(len(col)):
            st = in_to_st[i]
            d[st] = col[i] #one dictionary represents each column. 
        v_list.append(d)
    
    for k in range(len(back_table.T)):
        b = {}
        col = back_table.T[k]
        if k == 0:
            b['I'] = 'O' 
            b['O'] = 'O'
            b['B'] = 'O'
            b_list.append(b)
        if k > 0:
            for i in range(len(col)):
                st1 = in_to_st[i]
                st2 = in_to_st[col[i]]
                b[st1] = st2
            b_list.append(b)
        
    return v_list, b_list
        
  def __str__(self):
    return "HMM"


def softmax(z):
    return 1 / (1 + np.exp(-1 * z))

def get_one_hot(tag):
    if tag == 'I':
        return np.array([1, 0, 0])
    elif tag == 'O':
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])

"""
Implement the following class
"""
class NamedEntityRecognitionMEMM:
  def __init__(self):
        # implement as needed
        self.n_features = 5
        self.weights = np.random.randn(3, self.n_features)
        self.biases = np.random.randn(3)
        self.epochs = 10
        self.lr = 0.01525
        self.states = {'I', 'O', 'B'}
        
  def train(self, examples):
    for _ in range(self.epochs):
        for e in examples:
            for (token, label) in e:
                xi = self.featurize(token)
                z = np.dot(self.weights, xi) + self.biases
                parr = softmax(z)
                ytrue = get_one_hot(token)
                los = parr - ytrue
                
                for i in range(3):
            
                    loss_i = los[i]
                    
                    term = loss_i * xi
                    self.weights[i] = self.weights[i] - (self.lr * term)
                    #self.biases[i] = self.biases[i] - (self.lr * term)
        

  def featurize(self, example): #list of words to featurize. 
    """
    CHOOSE YOUR OWN PARAMS FOR THIS FUNCTION
    CHOOSE YOUR OWN RETURN VALUE FOR THIS FUNCTION
    """
    xi = [0 for i in range(5)]
    if '-' in example:
        xi[0] = 1
    if example[0].isupper():
        xi[1] = 1
    if len(example) > 10:
            xi[2] = 1
    if len(example) < 5:
            xi[3] = 1
    caps = True
    for c in example:
        if c.isupper() != True:
                caps = False
    if caps:
            xi[4] = 1
    return np.array(xi)
    


  def generate_probabilities(self, data):
    st_to_index = {'I':0, 'O':1, 'B':2}
    in_to_st = {value:key for key,value in st_to_index.items()}
    N = len(self.states)
       
    
   # data = [i for i in data if i in self.vocab] #dealing with words in vocabulary. 
    
    T = len(data)

    viterbi_table = np.zeros((N, T)) #the viterbi table
    back_table = np.zeros((N, T)) #the table of backpointers
    #will convert these to dictionaries after. 
    #w0 is the first word. 
    w0 = data[0][0]    
    
    
    
    xi = self.featurize(w0)
    parr = softmax(np.dot(self.weights, xi) + self.biases)
    
    for i in range(len(parr)):
        viterbi_table[i][0] = parr[i]
        back_table[i][0] = 0
    
    for w in range(1, T):
        xi = self.featurize(data[w][0])
        parr = softmax(np.dot(self.weights, xi) + self.biases)
        for state in self.states:
            j = st_to_index[state]
            
            choices_j = [viterbi_table[k][w-1] * parr[j] for k in range(3)]
          
           
            p = np.max(choices_j)
            y = np.argmax(choices_j)
            viterbi_table[j][w] = p
            back_table[j][w] = y
            
    v_list = [] #make lists to return
    b_list = [] #
    for col in viterbi_table.T:
        d = {}
        for i in range(len(col)):
            st = in_to_st[i]
            d[st] = col[i] #one dictionary represents each column. 
        v_list.append(d)
    
    for k in range(T):
        b = {}
        col = back_table.T[k]
        if k == 0:
            b['I'] = 'O' 
            b['O'] = 'O'
            b['B'] = 'O'
            b_list.append(b)
        if k > 0:
            for i in range(len(col)):
                st1 = in_to_st[i]
                st2 = in_to_st[col[i]]
                b[st1] = st2
            b_list.append(b)
        
    return v_list, b_list
        

  def __str__(self):
    return "MEMM"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw5_ner.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]
    training_examples = generate_tuples_from_file(training)
    testing_examples = generate_tuples_from_file(testing)
    

    nerHmm = NamedEntityRecognitionHMM()
    nerHmm.train(training_examples)
     
    example_answers = []
    
    for i in range(len(testing_examples)):
        e1 = testing_examples[i]
        tokens = [e[0] for e in e1]
        v, b = nerHmm.generate_probabilities(tokens)
        y = decode(tokens, v, b)
        example_answers.append(y)
        
        
    predicted_labels = []
    for e in example_answers:
        for t in e:
            predicted_labels.append(t[1])
    
    real_labels = []
    for e in testing_examples:
        for t in e:
            real_labels.append(t[1])
            
            
    r = recall(real_labels, predicted_labels)
    p = precision(real_labels, predicted_labels)
    f = f1(real_labels, predicted_labels)
    
    print("Model: HMM")
    print("precision: ", p)
    print("recall: ", r)
    print("f1: ", f)
    
    training_examples = training_examples[0:10]
    nerMemm = NamedEntityRecognitionMEMM()
    
    nerMemm.train(training_examples)
    memm_labels = []
    memm_answers = []
    for i in range(len(testing_examples)):
        e1 = testing_examples[i]
        tokens = [e[0] for e in e1]
        v, b = nerMemm.generate_probabilities(tokens)
        y = decode(tokens, v, b)
        memm_answers.append(y)
    

    predicted_labels = []
    for e in memm_answers:
        for t in e:
            memm_labels.append(t[1])
    
    rmemm = recall(real_labels, memm_labels)
    pmemm = precision(real_labels, memm_labels)
    fmemm = f1(real_labels, memm_labels)
    
    print("Model: MEMM")
    print("precision: ", pmemm)
    print("recall: ", rmemm)
    print("f1: ", fmemm)
  

