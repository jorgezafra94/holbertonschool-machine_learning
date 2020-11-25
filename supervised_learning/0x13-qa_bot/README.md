# QA_Bot

## Task0 - Question Answering
Write a function `def question_answer(question, reference):` that finds a snippet of text within a reference document to answer a question:<br>
<br>
* question is a string containing the question to answer
* reference is a string containing the reference document from which to find the answer

Returns: a string containing the answer
* If no answer is found, return None
* Your function should use the bert-uncased-tf2-qa model from the tensorflow-hub library
* Your function should use the pre-trained BertTokenizer, bert-large-uncased-whole-word-masking-finetuned-squad, from the transformers library

```
$ cat 0-main.py
#!/usr/bin/env python3

question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
$ ./0-main.py
on - site from 9 : 00 am to 3 : 00 pm
$
```

## Task1 - Create the loop
Create a script that takes in input from the user with the prompt Q: and prints A: as a response. If the user inputs exit, quit, goodbye, or bye, case insensitive, print A:<br> Goodbye and exit.<br>

```
$ ./1-loop.py
Q: Hello
A:
Q: How are you?
A:
Q: BYE
A: Goodbye
$
```

## Task2 -  Answer Questions
Based on the previous tasks, write a function `def answer_loop(reference):` that answers questions from a reference text:<br>
<br>
* reference is the reference text
* If the answer cannot be found in the reference text, respond with Sorry, I do not understand your question.

```
$ cat 2-main.py
#!/usr/bin/env python3

answer_loop = __import__('2-qa').answer_loop

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

answer_loop(reference)
$ ./2-main.py
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: Sorry, I do not understand your question.
Q: What does PLD stand for?
A: peer learning days
Q: EXIT
A: Goodbye
$
```

## Task3 - Semantic Search
Write a function def semantic_search(corpus_path, sentence): that performs semantic search on a corpus of documents:<br>
<br>
* corpus_path is the path to the corpus of reference documents on which to perform semantic search
* sentence is the sentence from which to perform semantic search

Returns: the reference text of the document most similar to sentence

```
$ cat 3-main.py
#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

print(semantic_search('ZendeskArticles', 'When are PLDs?'))
$ ./ 3-main.py
PLD Overview
Peer Learning Days (PLDs) are a time for you and your peers to ensure that each of you understands the concepts you've encountered in your projects, as well as a time for everyone to collectively grow in technical, professional, and soft skills. During PLD, you will collaboratively review prior projects with a group of cohort peers.
PLD Basics
PLDs are mandatory on-site days from 9:00 AM to 3:00 PM. If you cannot be present or on time, you must use a PTO. 
No laptops, tablets, or screens are allowed until all tasks have been whiteboarded and understood by the entirety of your group. This time is for whiteboarding, dialogue, and active peer collaboration. After this, you may return to computers with each other to pair or group program. 
Peer Learning Days are not about sharing solutions. This doesn't empower peers with the ability to solve problems themselves! Peer learning is when you share your thought process, whether through conversation, whiteboarding, debugging, or live coding. 
When a peer has a question, rather than offering the solution, ask the following:
"How did you come to that conclusion?"
"What have you tried?"
"Did the man page give you a lead?"
"Did you think about this concept?"
Modeling this form of thinking for one another is invaluable and will strengthen your entire cohort.
Your ability to articulate your knowledge is a crucial skill and will be required to succeed during technical interviews and through your career. 
$
```

## Task4 - QA_bot
Based on the previous tasks, write a function `def qa_bot(coprus_path):` that answers questions from multiple reference texts:<br>
<br>
corpus_path is the path to the corpus of reference documents

```
$ cat 4-main.py
#!/usr/bin/env python3

qa_bot = __import__('4-qa').qa_bot

qa_bot('ZendeskArticles')
$ ./4-main.py
Q: When are PLDs?
A: from 9 : 00 am to 3 : 00 pm
Q: What are Mock Interviews?
A: help you train for technical interviews
Q: What does PLD stand for?
A: peer learning days
Q: goodbye
A: Goodbye
$
```
