# Wordlyzer

A simply Wordle clone and solver in Python.

## Usage

Play repeated games of Wordle:
```
❯ ./wordlyzer.py
```

Use Wordlyzer to cheat at another Wordle game:
```
❯ ./wordlyzer.py -x
```

Various other modes of operation:
```
❯ ./wordlyzer.py --help
usage: wordlyzer.py [-h] [-f WORDSFILE] [-l LENGTH] [-r ROUNDS] [-a] [-x] [-w WORD] [-o] [-v] [-q]

options:
  -h, --help            show this help message and exit
  -f WORDSFILE, --wordsfile WORDSFILE
                        File with words (default: ./words)
  -l LENGTH, --length LENGTH
                        What length words to analyze (default: 5)
  -r ROUNDS, --rounds ROUNDS
                        How many rounds per game (default: 6)
  -a, --assist          Provide analysis while playing
  -x, --external        Provide analysis (i.e. cheat) for external game
  -w WORD, --word WORD  Provide initial solution word (default: random choice)
  -o, --one             Play a one-off game, do not auto-repeat
  -v, --verbose         Increase log level
  -q, --quiet           Decrease log level

```
