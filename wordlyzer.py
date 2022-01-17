#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property
import logging
from pathlib import Path
import random
import readline  # noqa: F401 # gives edit history to rich's Prompt.ask
from typing import Callable, Iterable, Iterator, cast

import chardet
from rich import print
from rich.columns import Columns
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt

logger = logging.getLogger(__file__ if __name__ == "__main__" else __name__)


def detect_charset(path: Path) -> str:
    with path.open("rb") as f:
        # Look at first 64K of data
        return str(chardet.detect(f.read(64 * 1024))["encoding"])


def parse_words(
    path: Path, filters: Iterable[Callable[[str], bool]] = ()
) -> Iterator[str]:
    with path.open(encoding=detect_charset(path)) as f:
        for line in f:
            word = line.rstrip()
            if all(f(word) for f in filters):
                yield word


class CharResult(IntEnum):
    unknown = 0  # this character is not yet tested
    missing = 1  # this character does not exist in the word
    elsewhere = 2  # this character exists elsewhere in the word
    correct = 3  # this character exists at this position in the word

    def format(self, c: str) -> str:
        styles = [
            "black on white",
            "bold white on black",
            "bold white on yellow",
            "bold white on green",
        ]
        return f"[{styles[self.value]}]{c}[/]"


class WordlePrompt(Prompt):
    illegal_choice_message = "[prompt.invalid.choice]Not a valid word!"


@dataclass(frozen=True)
class Guess:
    word: str
    guess: str

    @classmethod
    def ask(cls, solution: str, choices: list[str]) -> Guess:
        assert solution in choices
        guess = WordlePrompt.ask(
            "[bold]What is your guess?[/]",
            choices=choices,
            show_choices=False,
        )
        return cls(solution, guess)

    def __str__(self) -> str:
        return self.format()

    def correct(self) -> bool:
        return self.guess == self.word

    @cached_property
    def tally(self) -> tuple[tuple[str, CharResult], ...]:
        assert len(self.word) == len(self.guess)
        incorrect = [w for g, w in zip(self.guess, self.word) if g != w]

        def inner() -> Iterator[tuple[str, CharResult]]:
            for g, w in zip(self.guess, self.word):
                if g == w:
                    yield g, CharResult.correct
                elif g in incorrect:
                    yield g, CharResult.elsewhere
                    incorrect.remove(g)
                else:
                    yield g, CharResult.missing

        return tuple(inner())

    def format(self, invisible: bool = False) -> str:
        return "".join(
            res.format(" " if invisible else c) for c, res in self.tally
        )

    def filter(self, words: list[str]) -> list[str]:
        """Return the words that match this guess"""
        return [w for w in words if Guess(w, self.guess).tally == self.tally]


def analyze(guess: str, words: list[str]) -> Counter[str]:
    assert guess in words
    count = Counter(Guess(guess, word).format(invisible=True) for word in words)
    return count


def find_best_guess_fast(words: list[str]) -> tuple[str, int]:
    min_count, best_guess = len(words), ""
    for guess in words:
        if len(set(guess)) < len(guess):
            continue  # ignore guesses with repeated chars
        count = 0
        for word in words:
            if len(set(guess + word)) == len(set(guess)) + len(set(word)):
                count += 1
            if count >= min_count:
                break
        else:
            if count < min_count:
                min_count = count
                best_guess = guess
    return best_guess, min_count


def find_best_guess_correct(words: list[str]) -> tuple[str, int]:
    count, word = min(
        (analyze(guess, words).most_common(1)[0][1], guess) for guess in words
    )
    return word, count


def find_best_guess(words: list[str]) -> tuple[str, int]:
    if len(words) > 1000:
        return find_best_guess_fast(words)
    else:
        return find_best_guess_correct(words)


def keyboard(guesses: list[Guess]) -> str:
    chars: dict[str, CharResult] = defaultdict(lambda: CharResult.unknown)
    for guess in guesses:
        for c, result in guess.tally:
            chars[c] = max(result, chars[c])

    return "\n".join(
        " " * i + " ".join(chars[c].format(c) for c in row)
        for i, row in enumerate(["qwertyuiop", "asdfghjkl", "zxcvbnm"])
    )


class Wordle:
    def __init__(self, words: Iterable[str], max_rounds: int = 6):
        self.words = sorted(words)
        self.max_rounds = max_rounds

    def valid_guess(self, guess: str) -> bool:
        return guess in self.words

    def play(self, assist: bool = True) -> list[Guess]:
        word = random.choice(self.words)
        logger.debug(f"Chose word: {word!r}")
        guesses: list[Guess] = []
        candidates = self.words

        while len(guesses) < self.max_rounds:
            if assist:
                logger.info(f"Best guess is {find_best_guess(candidates)!r}")
            guess = Guess.ask(word, self.words)
            guesses.append(guess)
            count = analyze(guess.guess, candidates)
            if assist:
                print(
                    " ".join(
                        f"{category}|{num}"
                        for category, num in count.most_common()
                    )
                )
                candidates = guess.filter(candidates)
            print(
                Columns(
                    [
                        Panel.fit(str(guess), padding=1),
                        Panel.fit(keyboard(guesses)),
                    ],
                    padding=5,
                )
            )
            if guess.correct():
                print("[bold]Correct![/]")
                break
        else:
            print(f"Sorry, you ran out of turns! The word was: [bold]{word}[/]")
        return guesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--wordsfile",
        type=Path,
        default="words",
        help="File with words (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=5,
        help="What length words to analyze (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log level",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=0,
        help="Decrease log level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO + 10 * (args.quiet - args.verbose),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    words = set(
        parse_words(
            args.wordsfile,
            filters=[
                lambda word: len(word) == cast(int, args.length),
                str.islower,
                str.isalpha,
                str.isascii,
            ],
        )
    )
    logger.info(f"Read {len(words)} words from {args.wordsfile}")
    guesses = Wordle(words).play()
    print(Panel.fit("\n".join(guess.format() for guess in guesses)))


if __name__ == "__main__":
    main()
