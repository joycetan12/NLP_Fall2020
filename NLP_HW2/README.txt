Joyce Tan 
README File - HW2 

Language - Python 
IDE - PyCharm 

How to run 'pre-process.py' in Terminal:
1) cd to "Joyce_Tan_NLP_HW2" folder 

2) To run 'pre-process.py':  
   Run this: python ./pre-process.py arg1 arg2 arg3 arg4 
   arg1: path of the vocab file 
   arg2: path of the 'pos' directory 
   arg3: path of the 'neg' directory 
   arg4: name of vector form output file 

   'pre-process.py' will return the vector form of docs in the 'pos' and 'neg' directories in a txt file (name of file: arg passed into arg4)

3) To run 'NB.py': 
   Run this: python ./NB.py arg1 arg2 arg3 arg4 arg5 
   arg1: path of the vocab file 
   arg2: train vector form file 
   arg3: test vector form file 
   arg4: name of parameter output file 
   arg5: name of prediction output file 
   
   'NB.py' will return the a parameter output file (name of file: arg passed into arg4) and a prediction output file (name of file: arg passed into arg5)
