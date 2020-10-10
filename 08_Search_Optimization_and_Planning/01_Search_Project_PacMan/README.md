# Search project - Pac-Man

This project has been built by UC Berkeley AI division:

http://ai.berkeley.edu/search.html

My only task was to implement search agents to solve the puzzles. All credits go to:

http://ai.berkeley.edu

## Licensing information

Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

## Compliance with License term (1)

To be compliant with license term (1), the module I have written with agents has been encrypted before being committed 
to GitHub. The purpose of this repository is to keep a track of my work, and I publish all my solutions 
as open code. I made an exception here to comply with license terms and as a courtesy to UC Berkeley AI division who 
gives me a chance to work on this really fun challenge ;)

## Credits for encryption functions

I have used code provided by Eli Bendersky for encrypting/decrypting the solutions, all credits to him for the 
`encrypt.py` module.

## Most important commands

Before running the auto-grader, it is mandatory to generate the `search.py` module by decrypting the 
`search.py.enc` file:
```
python encrypt.py -d
```

To generate an encrypted version of the file:
```
python encrypt.py
```

To run the auto-grader to validate the project:
```
python autograder.py
```

To play a game of Pac-Man:
```
python pacman.py
```

A more exhaustive list of commands can be found [here](./commands.txt).