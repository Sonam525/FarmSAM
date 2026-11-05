#!/usr/bin/env python
"""Check that every number in the given Markdown files occurs in the results ledger or a registered source.

Usage: python tools/check_numbers.py LEDGER.md [EXTRA_SOURCE ...] -- FILE.md [FILE.md ...]

Prints numbers that are not found. A miss is not always an error: versions, years and file names also show up here,
so read the list by eye.
"""
import re
import sys

NUM = re.compile(r"(?<![\w.])[+−-]?\d[\d,]*(?:\.\d+)?(?![\w])")


def norm(s):
    return s.replace("−", "-").replace(",", "").lstrip("+-")


def tokens(text):
    return {norm(m.group(0)).rstrip(".") for m in NUM.finditer(text)}


def main(argv):
    i = argv.index("--")
    src = set()
    for p in argv[:i]:
        src |= tokens(open(p, encoding="utf-8").read())
    bad = 0
    for p in argv[i + 1:]:
        t = open(p, encoding="utf-8").read()
        t = re.sub(r"https?://\S+|`[^`]*`|\]\([^)]*\)", " ", t)  # links and code spans
        miss = sorted(x for x in tokens(t) if x not in src)
        print(p, len(miss), " ".join(miss))
        bad += len(miss)
    return bad


if __name__ == "__main__":
    print("not found:", main(sys.argv[1:]))
