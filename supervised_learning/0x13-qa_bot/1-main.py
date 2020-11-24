#!/usr/bin/env python3
"""
loop user, bot
"""

EXIT = ['exit', 'quit', 'goodbye', 'bye']

while(1):
    user = input("Q: ")

    if user.lower() in EXIT:
        print("A: Goodbye")
        break

    else:
        print("A:")

exit(0)
