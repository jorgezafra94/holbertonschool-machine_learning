# BLEU SCORE

Calculate the different methods to handle the bleu score, using:
* unigrams, group by word
* multigrams, grouping by n-groups of words
* cumulative Bleu score

# task0 - Unigram-bleu score
Write the function def uni_bleu(references, sentence): that calculates the unigram BLEU score for a sentence:<br>
<br>
* references is a list of reference translations
* each reference translation is a list of the words in the translation
* sentence is a list containing the model proposed sentence

Returns: the unigram BLEU score

```
$ cat 0-main.py
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))


$ ./0-main.py
0.6549846024623855
$
```

# task1 - n-grams bleu score
Write the function def ngram_bleu(references, sentence, n): that calculates the n-gram BLEU score for a sentence:<br>
<br>
* references is a list of reference translations
* each reference translation is a list of the words in the translation
* sentence is a list containing the model proposed sentence
* n is the size of the n-gram to use for evaluation

Returns: the n-gram BLEU score

```
$ cat 1-main.py
#!/usr/bin/env python3

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
$ ./1-main.py
0.6140480648084865
$
```

# task2 - cumulative n-grams bleu score
Write the function def cumulative_bleu(references, sentence, n): that calculates the cumulative n-gram BLEU score for a sentence:<br>
<br>
* references is a list of reference translations
* each reference translation is a list of the words in the translation
* sentence is a list containing the model proposed sentence
* n is the size of the largest n-gram to use for evaluation
* All n-gram scores should be weighted evenly

Returns: the cumulative n-gram BLEU score

```
$ cat 2-main.py
#!/usr/bin/env python3

cumulative_bleu = __import__('1-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
$ ./2-main.py
0.5475182535069453
$
```
