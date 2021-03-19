# Author: Joyce Tan
# NLP Fall 2020 - HW1
import math

# this method pads each sentence and lowercase all words
# returns a processed sentence
def preprocess(text):
    cleanText = '<s> '
    text = text.lower()
    cleanText += text
    cleanText += ' </s>'
    return cleanText

# this method pads and lowercase each sentence in a file
# returns a list of processed sentences
def create_processed_list(filename):
    processed_list_of_lines = []
    with open(filename) as file:
        for line in file:
            line = preprocess(line)
            split_line = line.split()
            processed_list_of_lines.append(split_line)
    return processed_list_of_lines

# this method calculates the the log probability under the unigram maximum likelihood model
def log_probability_unigram(unigram, total_tokens, sentence, print_param):
    log_probability = 0
    undefined = False
    for word in sentence:
        if word not in unigram:
            undefined = True
            if print_param:
                print('p(', word, ') = 0')
                print('log(p(', word, ')) = NaN')
        else:
            if print_param:
                print('p(', word, ') =', (unigram[word] / total_tokens))
                print('log(p(', word, ')) =', math.log((unigram[word] / total_tokens), 2))
            log_probability += math.log((unigram[word] / total_tokens), 2)

    if undefined:
        return 'NaN'
    return log_probability

# this method calculates the the log probability under the bigram maximum likelihood model
def log_probability_bigram(bigram, unigram, sentence,print_param):
    log_probability = 0
    undefined = False
    for i in range(len(sentence)):
        word = sentence[i]
        if word != '</s>':
            nextWord = sentence[i+1]
            if (word,nextWord) not in bigram:
                undefined = True
                if print_param:
                    print('p(', nextWord, '|', word, ') = 0')
                    print('log(p(', nextWord, '|', word, ')) = NaN')
            else:
                if print_param:
                    print('p(', nextWord, '|', word, ') =', bigram[(word,nextWord)]/unigram[word])
                    print('log(p(', nextWord, '|', word, ')) =', math.log(bigram[(word,nextWord)]/unigram[word], 2))
                log_probability += math.log(bigram[(word,nextWord)]/unigram[word], 2)

    if undefined:
        return 'NaN'
    return log_probability

# this method is used to calculate the the log probability under the bigram model add-one smoothing
def log_probability_bigram_smoothing(bigram, unigram, sentence, print_param):
    log_probability = 0
    V = len(unigram)
    for i in range(len(sentence)):
        numerator = 1
        denominator = V
        word = sentence[i]
        if word != '</s>':
            nextWord = sentence[i+1]
            if (word,nextWord) in bigram:
                numerator = bigram[(word, nextWord)] + 1
            if word in unigram:
                denominator = unigram[word] + V
            if print_param:
                print('p(', nextWord, '|', word, ') =', numerator/denominator)
                print('log(p(', nextWord, '|', word, ')) =', math.log(numerator/denominator, 2))
            log_probability += math.log(numerator/denominator, 2)

    return log_probability

# this method is used to calculate perplexity
def perplexity(total_tokens,log_prob_sentence):
    if log_prob_sentence == 'NaN':
        return 'NaN'
    l = log_prob_sentence/total_tokens
    p = 2**(l*-1)
    return p

# create processed list of sentences
processed_sentences_train = create_processed_list('train.txt')
processed_sentences_test = create_processed_list('test.txt')

# create training corpus unigram before mapping <unk>
train_unigram = {}
train_unigram_tokens = 0
for line in processed_sentences_train:
    for i in range(len(line)):
        train_unigram_tokens += 1
        word = line[i]
        if word in train_unigram:
            train_unigram[word] += 1
        else:
            train_unigram[word] = 1

# create training corpus unigram after mapping <unk>
train_unigram_with_unk = {}
train_with_unk_tokens = 0
for line in processed_sentences_train:
    for i in range(len(line)):
        train_with_unk_tokens += 1
        word = line[i]
        if train_unigram[word] == 1:
            word = '<unk>'
        if word in train_unigram_with_unk:
            train_unigram_with_unk[word] += 1
        else:
            train_unigram_with_unk[word] = 1

print('# of word types in training corpus after mapping <unk>:', len(train_unigram_with_unk))
print('# of word tokens in training corpus:', train_with_unk_tokens)

# find words in test corpus that do not occur in training corpus
test_unigram = {}
test_total_tokens = 0
test_tokens_not_in_train = 0
word_not_in_train = [] # words not in training corpus before mapping <unk>
word_not_in_train_with_unk = [] # words not in training corpus after mapping <unk>
for line in processed_sentences_test:
    for i in range(len(line)):
        test_total_tokens += 1
        word = line[i]
        if word not in train_unigram:
            test_tokens_not_in_train += 1
            if word not in test_unigram:
                word_not_in_train.append(word)
        if word not in train_unigram_with_unk:
            if word not in test_unigram:
                word_not_in_train_with_unk.append(word)
        if word in test_unigram:
            test_unigram[word] += 1
        else:
            test_unigram[word] = 1

# calculate percentage of word tokens in test corpus that did not occur in training corpus
test_token_not_in_train_percentage = test_tokens_not_in_train/test_total_tokens
test_token_not_in_train_percentage = "{:.1%}".format(test_token_not_in_train_percentage)

# calculate percentage of unique words in test corpus that did not occur in training corpus
test_word_not_in_train_percentage = len(word_not_in_train)/len(test_unigram)
test_word_not_in_train_percentage = "{:.1%}".format(test_word_not_in_train_percentage)

print('percentage of word tokens in test corpus not in training corpus before mapping <unk> =', test_token_not_in_train_percentage)
print('percentage of word types in test corpus not in training corpus before mapping <unk> =', test_word_not_in_train_percentage)

# create test corpus unigram after mapping <unk>
test_unigram_with_unk = {}
test_with_unk_tokens = 0
for line in processed_sentences_test:
    for i in range(len(line)):
        test_with_unk_tokens += 1
        if line[i] in word_not_in_train_with_unk:
            line[i] = '<unk>'
        if line[i] in test_unigram_with_unk:
            test_unigram_with_unk[line[i]] += 1
        else:
            test_unigram_with_unk[line[i]] = 1

# create training corpus bigram after mapping <unk>
train_bigram_with_unk = {}
train_bigram_token_with_unk = 0
for line in processed_sentences_train:
    for i in range(len(line)):
        if line[i] != '</s>':
            word = line[i]
            nextWord = line[i + 1]
            train_bigram_token_with_unk += 1
            if train_unigram[word] == 1:
                word = '<unk>'
            if train_unigram[nextWord] == 1:
                nextWord = '<unk>'
            if (word, nextWord) in train_bigram_with_unk:
                train_bigram_with_unk[word, nextWord] += 1
            else:
                train_bigram_with_unk[word, nextWord] = 1

# create test corpus bigram after mapping <unk> words not observed in the training corpus (with mapped <unk>)
test_bigram_with_unk = {}
test_bigram_tokens_with_unk = 0
for line in processed_sentences_test:
    for i in range(len(line)):
        if line[i] != '</s>':
            word = line[i]
            nextWord = line[i + 1]
            test_bigram_tokens_with_unk += 1
            if word in word_not_in_train_with_unk:
                word = '<unk>'
            if nextWord in word_not_in_train_with_unk:
                nextWord = '<unk>'
            if (word, nextWord) in test_bigram_with_unk:
                test_bigram_with_unk[word, nextWord] += 1
            else:
                test_bigram_with_unk[word, nextWord] = 1

# find bigrams in test corpus (with mapped <unk>) that do not occur in training corpus (with mapped <unk>)
bigrams_not_in_train = []
test_bigram_types_not_in_train = 0
test_bigram_tokens_not_in_train = 0
for bigram in test_bigram_with_unk:
    if bigram not in train_bigram_with_unk:
        bigrams_not_in_train.append(bigram)
        test_bigram_types_not_in_train += 1
        test_bigram_tokens_not_in_train += test_bigram_with_unk[bigram]

# calculate the percentage of word tokens in test corpus did not occur in training corpus
test_bigram_tokens_not_in_train_percentage = test_bigram_tokens_not_in_train/test_bigram_tokens_with_unk
test_bigram_tokens_not_in_train_percentage = "{:.1%}".format(test_bigram_tokens_not_in_train_percentage)

# calculate the percentage of word types in test corpus did not occur in training corpus
test_bigram_types_not_in_train_percentage = test_bigram_types_not_in_train/len(test_bigram_with_unk)
test_bigram_types_not_in_train_percentage = "{:.1%}".format(test_bigram_types_not_in_train_percentage)

print('percentage of bigram tokens in test corpus not in training corpus =', test_bigram_tokens_not_in_train_percentage)
print('percentage of bigram types in test corpus not in training corpus =', test_bigram_types_not_in_train_percentage)
print('')

# process sentence
sentence = "• I look forward to hearing your reply ."
sentence = preprocess(sentence).split()

# calculate the log probability of the sentence under the unigram maximum likelihood model
print('Sentence log probability unigram parameters:')
log_prob_sentence_unigram = log_probability_unigram(train_unigram_with_unk, train_with_unk_tokens, sentence, True)
print('Sentence log probability: unigram =', log_prob_sentence_unigram)
print('')

# calculate the log probability of the sentence under the bigram maximum likelihood model
print('Sentence log probability bigram parameters:')
log_prob_sentence_bigram = log_probability_bigram(train_bigram_with_unk, train_unigram_with_unk, sentence, True)
print('Sentence log probability: bigram =', log_prob_sentence_bigram)
print('')

# calculate the log probability of the sentence under the bigram model with add-one smoothing
print('Sentence log probability bigram add one smoothing parameters:')
log_prob_sentence_bigram_smoothing = log_probability_bigram_smoothing(train_bigram_with_unk, train_unigram_with_unk, sentence, True)
print('Sentence log probability: bigram add one smoothing =', log_prob_sentence_bigram_smoothing)
print('')

# calculate the perplexity of the sentence under the unigram maximum likelihood model
print('Perplexity of sentence under unigram model =', perplexity(len(sentence),log_prob_sentence_unigram))
# calculate the perplexity of the sentence under the bigram maximum likelihood model
print('Perplexity of sentence under bigram model =', perplexity(len(sentence),log_prob_sentence_bigram))
# calculate the perplexity of the sentence under the bigram model with add-one smoothing
print('Perplexity of sentence under bigram add one smoothing model =', perplexity(len(sentence),log_prob_sentence_bigram_smoothing))
print('')

# calculate perplexity of the entire test corpus under the unigram maximum likelihood model
log_prob_test_unigram = 0
for sentence in processed_sentences_test:
    if log_probability_unigram(train_unigram_with_unk, train_with_unk_tokens, sentence, False) == 'NaN':
        log_prob_test_unigram = 'NaN'
        break
    else:
        log_prob_test_unigram += log_probability_unigram(train_unigram_with_unk, train_with_unk_tokens, sentence, False)
perplexity_test_unigram = perplexity(test_with_unk_tokens, log_prob_test_unigram)
print('Perplexity of test corpus under unigram model =', perplexity_test_unigram)

# calculate perplexity of the entire test corpus under the bigram maximum likelihood model
log_prob_test_bigram = 0
for sentence in processed_sentences_test:
    if log_probability_bigram(train_bigram_with_unk, train_unigram_with_unk, sentence, False) == 'NaN':
        log_prob_test_bigram = 'NaN'
        break
    else:
        log_prob_test_bigram += log_probability_bigram(train_bigram_with_unk, train_unigram_with_unk, sentence, False)
perplexity_test_bigram = perplexity(test_with_unk_tokens, log_prob_test_bigram)
print('Perplexity of test corpus under bigram model =', perplexity_test_bigram)

# calculate perplexity of the entire test corpus under the bigram model with add-one smoothing
log_prob_test_bigram_smoothing = 0
for sentence in processed_sentences_test:
    if log_probability_bigram_smoothing(train_bigram_with_unk, train_unigram_with_unk, sentence, False) == 'NaN':
        log_prob_test_bigram_smoothing = 'NaN'
        break
    else:
        log_prob_test_bigram_smoothing += log_probability_bigram_smoothing(train_bigram_with_unk, train_unigram_with_unk, sentence, False)
perplexity_test_bigram_smoothing = perplexity(test_with_unk_tokens, log_prob_test_bigram_smoothing)
print('Perplexity of test corpus under bigram add one smoothing model =', perplexity_test_bigram_smoothing)
