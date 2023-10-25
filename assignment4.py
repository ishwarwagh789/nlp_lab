'''
Name:-Wagh Ishwar Santosh
Roll no:-65
Assignment No:-04
Batch:-B-4
'''

from nltk import ngrams

sentence = 'This is a sample text generated by Pulkit Ahuja'


arr = ngrams(sentence.split(" "), 1)
print("1-grams for given sentence are : ")
for j in arr:
    print(j, end=" ")
print()

'''
-----------------OUTPUT--------------------
1-grams for given sentence are : 
('This',) ('is',) ('a',) ('sample',) ('text',) ('generated',) ('by',) ('Pulkit',) ('Ahuja',) 
'''
arr = ngrams(sentence.split(" "), 2)
print("2-grams for given sentence are : ")
for j in arr:
    print(j, end=" ")
print()
'''
--------------OUTPUT----------------------
2-grams for given sentence are : 
('This', 'is') ('is', 'a') ('a', 'sample') ('sample', 'text') ('text', 'generated') ('generated', 'by') ('by', 'Pulkit') ('Pulkit', 'Ahuja') 
'''
arr = ngrams(sentence.split(" "), 3)
print("3-grams for given sentence are : ")
for j in arr:
    print(j, end=" ")
print()

'''
--------------OUTPUT-----------------------
2-grams for given sentence are : 
('This', 'is', 'a') ('is', 'a', 'sample') ('a', 'sample', 'text') ('sample', 'text', 'generated') ('text', 'generated', 'by') ('generated', 'by', 'Pulkit') ('by', 'Pulkit', 'Ahuja') 
'''
from nltk import ngrams
file = open("/home/exam/pr4.txt")
for i in file.readlines():
    cumulative = i
    sentences = i.split(".")
    counter = 0
    for sentence in sentences:
        print("For sentence", counter + 1, ", trigrams are: ")
        trigrams = ngrams(sentence.split(" "), 3)
        for grams in trigrams:
            print(grams)
        counter += 1
        print()
''' 
------------OUTPUT---------------------------
For sentence 1 , trigrams are: 
('Ishwar', 'Wagh', 'is')
('Wagh', 'is', 'currently')
('is', 'currently', 'pursuing')
('currently', 'pursuing', 'a')
('pursuing', 'a', 'BTech')
('a', 'BTech', 'in')
('BTech', 'in', 'IT')

For sentence 2 , trigrams are: 
('', 'They', 'are')
('They', 'are', 'actively')
('are', 'actively', 'involved')
('actively', 'involved', 'in')
('involved', 'in', 'NLP')
('in', 'NLP', 'tasks')
('NLP', 'tasks', 'as')
('tasks', 'as', 'part')
('as', 'part', 'of')
('part', 'of', 'their')
('of', 'their', 'studies')
('their', 'studies', 'and')
('studies', 'and', 'projects')

For sentence 3 , trigrams are: 
('', 'NLP', 'involves')
('NLP', 'involves', 'natural')
('involves', 'natural', 'language')
('natural', 'language', 'processing')
('language', 'processing', 'to')
('processing', 'to', 'analyze')
('to', 'analyze', 'and')
('analyze', 'and', 'work')
('and', 'work', 'with')
('work', 'with', 'text')
('with', 'text', 'data')

For sentence 4 , trigrams are: 
('', "Ishwar's", 'combination')
("Ishwar's", 'combination', 'of')
('combination', 'of', 'IT')
('of', 'IT', 'skills')
('IT', 'skills', 'and')
('skills', 'and', 'NLP')
('and', 'NLP', 'interest')
('NLP', 'interest', 'can')
('interest', 'can', 'lead')
('can', 'lead', 'to')
('lead', 'to', 'exciting')
('to', 'exciting', 'opportunities')
('exciting', 'opportunities', 'in')
('opportunities', 'in', 'the')
('in', 'the', 'field')

For sentence 5 , trigrams are: 
('', 'NLP', 'tasks')
('NLP', 'tasks', 'can')
('tasks', 'can', 'include')
('can', 'include', 'sentiment')
('include', 'sentiment', 'analysis,')
('sentiment', 'analysis,', 'text')
('analysis,', 'text', 'classification,')
('text', 'classification,', 'and')
('classification,', 'and', 'language')
('and', 'language', 'generation,')
('language', 'generation,', 'among')
('generation,', 'among', 'others')

For sentence 6 , trigrams are: 
('', 'With', 'dedication')
('With', 'dedication', 'and')
('dedication', 'and', 'continued')
('and', 'continued', 'learning,')
('continued', 'learning,', 'Ishwar')
('learning,', 'Ishwar', 'can')
('Ishwar', 'can', 'excel')
('can', 'excel', 'in')
('excel', 'in', 'the')
('in', 'the', 'realm')
('the', 'realm', 'of')
('realm', 'of', 'NLP')

For sentence 7 , trigrams are:
'''