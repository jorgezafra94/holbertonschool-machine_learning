#!/usr/bin/env python3

semantic_search = __import__('3-semantic_search').semantic_search

print(semantic_search('ZendeskArticles', 'When are PLDs?'))

print(semantic_search('ZendeskArticles', 'What are Mock Interviews?'))

print(semantic_search('ZendeskArticles', 'What does PLD stand for?'))