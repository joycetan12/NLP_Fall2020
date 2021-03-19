# Author: Joyce Tan
import sys
import math

# this function returns string label1 and string label2 of the vector file
def get_labels(vector_file):
    label1 = ''
    label2 = ''
    for line in vector_file:
        line = line.split()
        if label1 == '':
            label1 = line[0]
        if line[0] != label1:
            label2 = line[0]
            break

    return label1, label2

# this function prints all model parameters to an output file
def print_params(prob_label1, prob_label2, label1_word_counts, label2_word_counts, total_label1_words, total_label2_words, vocab):
    output = open(sys.argv[4], "w")
    output.write("Prior Probabilities:")
    string = ["\n", "p( ", label1, " ): ", str(prob_label1), "\n"]
    output.writelines(string)
    string = ["p( ", label2, " ): ", str(prob_label2), "\n"]
    output.writelines(string)
    output.write("Feature Probabilities:")
    output.write("\n")

    V = len(vocab)

    for word in vocab:
        if word in label1_word_counts:
            prob_label1_word = (label1_word_counts[word] + 1)/(total_label1_words + V)
        else:
            prob_label1_word = 1/(total_label1_words + V)

        if word in label2_word_counts:
            prob_label2_word = (label2_word_counts[word] + 1)/(total_label2_words + V)
        else:
            prob_label2_word = 1/(total_label2_words + V)

        string_label1 = ["p( ", word, " | ", label1, " ) = ", str(prob_label1_word), "\n"]
        output.writelines(string_label1)
        string_label2 = ["p( ", word, " | ", label2, " ) = ", str(prob_label2_word), "\n"]
        output.writelines(string_label2)

# this function calculates the log probability given the label
def calc_log_prob(word, given_label, label1_word_counts, label2_word_counts, total_label1_words, total_label2_words, vocab):
    V = len(vocab)

    log_prob = 0

    if given_label == label1:
        if word in label1_word_counts:
            log_prob = math.log((label1_word_counts[word] + 1)/(total_label1_words + V), 2)
        else:
            log_prob = math.log(1/(total_label1_words + V), 2)

    if given_label == label2:
        if word in label2_word_counts:
            log_prob = math.log((label2_word_counts[word] + 1) / (total_label2_words + V), 2)
        else:
            log_prob = math.log(1 / (total_label2_words + V), 2)

    return log_prob

# adds all words inside the vocab file into a dict
vocab_path = sys.argv[1]  # path of the vocab file
vocab = {}
with open(vocab_path) as file:
    for line in file:
        for word in line.split():
            vocab[word] = None

train_file = open(sys.argv[2], "r")  # train vector form file
test_file = open(sys.argv[3], "r")  # test vector form file

# gets the two different types of label
label1, label2 = get_labels(train_file)

train_label1_counts = {}  # key: words in 'label1' docs, value: count of words
train_label2_counts = {}  # key: words in 'label2' docs, value: count of words
label1_files = 0  # total # of 'label1' docs
label2_files = 0  # total # of 'label2' docs

# go back to the beginning of the train vector file
train_file.seek(0)

# read the train vector file and save the counts of each word in 'label1' and 'label2' in their respective dict
for line in train_file:
    line = line.split()
    label = line[0]
    if label == label1:
        label1_files += 1
    else:
        label2_files += 1

    for vector in line[1:]:
        w = vector.split(':')
        word = w[0]
        wordCount = w[1]
        if label == label1:
            if word not in train_label1_counts:
                train_label1_counts[word] = int(wordCount)
            else:
                train_label1_counts[word] += int(wordCount)
        if label == label2:
            if word not in train_label2_counts:
                train_label2_counts[word] = int(wordCount)
            else:
                train_label2_counts[word] += int(wordCount)

# calculate p(label1) and p(label2)
prob_label1 = label1_files/(label1_files + label2_files)
prob_label2 = label2_files/(label1_files + label2_files)

# calculate total # of words in 'label1' class and total # of words in 'label2' class
label1_words_counts = train_label1_counts.values()
total_label1_words = sum(label1_words_counts)
label2_words_counts = train_label2_counts.values()
total_label2_words = sum(label2_words_counts)

# creates an output file of all parameters
print_params(prob_label1, prob_label2, train_label1_counts, train_label2_counts, total_label1_words, total_label2_words, vocab)

# creates the output file that will save all the predictions
output = open(sys.argv[5], "w")

num_of_docs = 0 # total # of docs
num_accurate = 0 # total # of accurate predictions

# read and calculate the log probability of each doc given label in the test vector file to determine the most likely label
for line in test_file:
    line = line.split()
    label = line[0]

    i = 1
    predict_label1_prob = math.log(prob_label1,2)
    predict_label2_prob = math.log(prob_label2,2)

    for vector in line[1:]:
        w = vector.split(':')
        word = w[0]
        wordCount = int(w[1])
        predict_label1_prob += calc_log_prob(word, label1, train_label1_counts, train_label2_counts, total_label1_words, total_label2_words, vocab)*wordCount
        predict_label2_prob += calc_log_prob(word, label2, train_label1_counts, train_label2_counts, total_label1_words, total_label2_words, vocab)*wordCount


    # if p(label1|doc) >= p(label2|doc) predicted label is label1
    if predict_label1_prob >= predict_label2_prob:
        if label == label1:
            num_accurate += 1
            string = ["correct label: ", label, "   predicted label: ", label1, "   1", "\n"]
            output.writelines(string)
        else:
            string = ["correct label: ", label, "   predicted label: ", label1, "   0", "\n"]
            output.writelines(string)
    # else predicted label is label2
    else:
        if label == label2:
            num_accurate += 1
            string = ["correct label: ", label, "   predicted label: ", label1, "   1", "\n"]
            output.writelines(string)
        else:
            string = ["correct label: ", label, "   predicted label: ", label2, "   0", "\n"]
            output.writelines(string)

    num_of_docs += 1

# calculate accuracy percentage
accuracy = num_accurate/num_of_docs
accuracy_percentage = "{:.1%}".format(accuracy)

# write total # of files, # of accurate predictions and accuracy to the output predictions file
string = ["\n", "total # of files: ", str(num_of_docs), "\n"]
output.writelines(string)
string = ["# of accurate predictions: ", str(num_accurate), "\n"]
output.writelines(string)
string = ["accuracy: ", accuracy_percentage, "\n"]
output.writelines(string)
