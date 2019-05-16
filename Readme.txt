[How to Run]

My code file has 3 options. 
Python fasttext_subword_embed.py -NegativeSampling -Ngram -full/part

According to the paper and the presentation(on Youtube), the researchers suggest that a short n-grams(n=4) is good to capture syntactic information and longer n-grams(n=6) are good to capture semantic information. 
Following them, I used middle size of ngrams(n=5) for the test.

*** You should download 'text8' corpus! ***
* http://mattmahoney.net/dc/textdata.html *

(Ex) python fasttext_subword_embed.py 20 5 full

Thank you.
