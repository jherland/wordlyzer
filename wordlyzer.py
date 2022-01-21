#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from itertools import product
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
from rich.rule import Rule

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


@cache
def tally(guess: str, solution: str) -> tuple[CharResult, ...]:
    assert len(guess) == len(solution)
    incorrect = [s for g, s in zip(guess, solution) if g != s]

    def inner() -> Iterator[tuple[str, CharResult]]:
        for g, s in zip(guess, solution):
            if g == s:
                yield CharResult.correct
            elif g in incorrect:
                yield CharResult.elsewhere
                incorrect.remove(g)
            else:
                yield CharResult.missing

    return tuple(inner())


@dataclass(frozen=True)
class Guess:
    guess: str
    tally: tuple[CharResult, ...]

    def __str__(self) -> str:
        return self.format()

    def __iter__(self) -> Iterator[tuple[str, CharResult]]:
        return zip(self.guess, self.tally)

    def correct(self) -> bool:
        return all(t == CharResult.correct for t in self.tally)

    def format(self, invisible: bool = False) -> str:
        if invisible:
            return "".join(res.format(" ") for res in self.tally)
        else:
            return "".join(res.format(c) for c, res in self)

    def filter(self, words: list[str]) -> list[str]:
        """Return the words that match this guess"""
        return [w for w in words if tally(self.guess, w) == self.tally]


def analyze(guess: str, words: list[str]) -> Counter[str]:
    assert guess in words
    count = Counter(
        Guess(guess, tally(guess, w)).format(invisible=True) for w in words
    )
    return count


quick_and_dirty_cache = {}


def find_best_guess_fast(words: list[str]) -> tuple[str, int]:
    if len(words) not in quick_and_dirty_cache:
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
        quick_and_dirty_cache[len(words)] = (best_guess, min_count)
    return quick_and_dirty_cache[len(words)]


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


class WordlePrompt(Prompt):
    illegal_choice_message = "[prompt.invalid.choice]Not a valid word!"


class Wordle:
    def __init__(self, words: Iterable[str], max_rounds: int = 6):
        self.words = sorted(words)
        self.max_rounds = max_rounds

    def ask_for_guess(self, choices: list[str]) -> str:
        return WordlePrompt().ask(
            "[bold]What is your guess?[/]",
            choices=choices,
            show_choices=False,
        )

    @staticmethod
    def keyboard(guesses: list[Guess]) -> str:
        chars: dict[str, CharResult] = defaultdict(lambda: CharResult.unknown)
        for guess in guesses:
            for c, result in guess:
                chars[c] = max(result, chars[c])

        return "\n".join(
            " " * i + " ".join(chars[c].format(c) for c in row)
            for i, row in enumerate(["qwertyuiop", "asdfghjkl", "zxcvbnm"])
        )

    def eval_guess(self, word: str) -> Guess:
        raise NotImplementedError

    def play(self, assist: bool = True) -> list[Guess]:
        guesses: list[Guess] = []
        candidates = self.words

        while len(guesses) < self.max_rounds:
            if assist:
                logger.info(f"Best guess is {find_best_guess(candidates)!r}")
            word = self.ask_for_guess(self.words)
            guess = self.eval_guess(word)
            guesses.append(guess)
            count = analyze(word, candidates)
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
                        Panel.fit(self.keyboard(guesses)),
                    ],
                    padding=5,
                )
            )
            if guess.correct():
                print("[bold]Correct![/]")
                break
        else:
            print(f"Sorry, out of turns!")
        return guesses


class InternalWordle(Wordle):
    def eval_guess(self, word: str) -> Guess:
        return Guess(word, tally(word, self.solution))

    def play(self, *args, **kwargs) -> list[Guess]:
        self.solution = random.choice(self.words)
        logger.debug(f"Chose word: {self.solution!r}")
        try:
            return super().play(*args, **kwargs)
        finally:
            print(f"The solution was: [bold]{self.solution}[/]")


class ExternalWordle(Wordle):
    def ask_for_tally(self, length: int) -> tuple[CharResult, ...]:
        answers = {
            "b": CharResult.missing,  # black
            "y": CharResult.elsewhere,  # yellow
            "g": CharResult.correct,  # green
        }
        choices = {"".join(answer) for answer in product(*(["byg"] * length))}
        colors = WordlePrompt.ask(
            "[bold]What colors did you get?[/] "
            "([bold]b[/]lack, [bold]y[/]ellow, [bold]g[/]reen)",
            choices=choices,
            show_choices=False,
        )
        return tuple(answers[c] for c in colors)

    def eval_guess(self, word: str) -> Guess:
        return Guess(word, self.ask_for_tally(len(word)))


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
        "-r",
        "--rounds",
        type=int,
        default=6,
        help="How many rounds per game (default: %(default)s)",
    )
    parser.add_argument(
        "-x",
        "--external",
        action="store_true",
        help="Provide analysis (i.e. cheat) for external game",
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
    # TODO: Hard vs. normal mode
    # TODO: Absurdle

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
    find_best_guess_fast(words)  # pre-compute initial guess
    logger.info(f"Read {len(words)} words from {args.wordsfile}")

    if args.external:
        game = ExternalWordle(words, args.rounds)
    else:
        game = InternalWordle(words, args.rounds)

    with suppress(KeyboardInterrupt):
        while True:
            print(Rule("New game! (Ctrl+C to quit)"))
            guesses = game.play()
            print(Panel.fit("\n".join(guess.format() for guess in guesses)))

    print()


if __name__ == "__main__":
    main()
