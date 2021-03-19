# Author: Joyce Tan
import sys
import os
import string

path_vocab = sys.argv[1]  # path of vocab file
path_pos = sys.argv[2]  # path of 'pos' directory
path_neg = sys.argv[3]  # path of 'neg' directory
output = open(sys.argv[4], "w")  # output vector file

# adds all words inside the vocab file into a dict
vocab = {}
with open(path_vocab) as file:
   for line in file:
      for word in line.split():
         vocab[word] = None

# reads all docs in the 'pos' directory
# lowercase all words and separates punctuations
for filename in sorted(os.listdir(path_pos)):
   printed = set()
   word_dict = {}
   with open(os.path.join(path_pos, filename)) as file:
      for line in file:
         line = line.lower()
         for word in line.split():
            table = str.maketrans('', '', string.punctuation)
            word = word.translate(table)
            if word not in word_dict:
               word_dict[word] = 1
            else:
               word_dict[word] += 1

      # writes the doc in vector form onto the output vector file
      output.write("pos ")
      for word in line.split():
         table = str.maketrans('', '', string.punctuation)
         word = word.translate(table)
         if word in vocab and word not in printed:
            s = [word, ":", str(word_dict[word]), " "]
            output.writelines(s)
            printed.add(word)
      output.write("\n")

# reads all docs in the 'neg' directory
# lowercase all words and separates punctuations
for filename in sorted(os.listdir(path_neg)):
   printed = set()
   word_dict = {}
   with open(os.path.join(path_neg, filename)) as file:
      for line in file:
         line = line.lower()
         for word in line.split():
            table = str.maketrans('', '', string.punctuation)
            word = word.translate(table)
            if word not in word_dict:
               word_dict[word] = 1
            else:
               word_dict[word] += 1

      # writes the doc in vector form onto the output vector file
      output.write("neg ")
      for word in line.split():
         table = str.maketrans('', '', string.punctuation)
         word = word.translate(table)
         if word in vocab and word not in printed:
            s = [word, ":", str(word_dict[word]), " "]
            output.writelines(s)
            printed.add(word)
      output.write("\n")


