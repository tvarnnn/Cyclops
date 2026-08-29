#!/usr/bin/env python3
"""A structural sanity check for Swift sources on a host with no toolchain.

This is NOT a compiler and must never be reported as one. It exists because
half of this repository is developed on a Windows host with no Swift
toolchain -- swift, swiftc, xcodebuild, SwiftLint and swift-format are all
absent -- so a scripted edit to a .swift file there is otherwise completely
unchecked until it reaches a Mac.

What it can prove: brackets balance, and every string and comment is
terminated. What it CANNOT prove: anything about types, actors,
availability, protocol conformance, or whether the file compiles. A green
run here is not a build and must never be reported as one.

With `--no-prose` it additionally enforces the rule that a named view
writes no user-facing sentences of its own -- see the section on it below. It walks the file
as a character stream, tracking line comments, block comments (nested, as
Swift allows), plain strings, multi-line strings and escapes, and then checks
that brackets balance outside all of those. It catches exactly the class of
damage a scripted edit does: an unterminated string, a triple-quote that
lost its closing pair, a stray brace.
"""

import re
import sys
from pathlib import Path

OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = {v: k for k, v in OPEN.items()}


def check(path: Path):
    src = path.read_text(encoding="utf-8")
    i, n = 0, len(src)
    line = 1
    stack = []
    problems = []
    block_depth = 0

    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1
            i += 1
            continue

        if block_depth:
            if src.startswith("/*", i):
                block_depth += 1
                i += 2
                continue
            if src.startswith("*/", i):
                block_depth -= 1
                i += 2
                continue
            i += 1
            continue

        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue

        if src.startswith("/*", i):
            block_depth = 1
            i += 2
            continue

        if src.startswith('"""', i):
            j = src.find('"""', i + 3)
            while j != -1 and src[j - 1] == "\\":
                j = src.find('"""', j + 3)
            if j == -1:
                problems.append(f"{path.name}:{line}: unterminated multi-line string")
                break
            line += src.count("\n", i, j)
            i = j + 3
            continue

        if ch == '"':
            j = i + 1
            closed = False
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "\n":
                    break
                if src[j] == '"':
                    closed = True
                    break
                j += 1
            if not closed:
                problems.append(f"{path.name}:{line}: unterminated string literal")
                break
            i = j + 1
            continue

        if ch in OPEN:
            stack.append((ch, line))
            i += 1
            continue
        if ch in CLOSE:
            if not stack:
                problems.append(f"{path.name}:{line}: unmatched closing {ch!r}")
                break
            opened, opened_line = stack.pop()
            if OPEN[opened] != ch:
                problems.append(
                    f"{path.name}:{line}: closing {ch!r} does not match "
                    f"{opened!r} opened on line {opened_line}"
                )
                break
            i += 1
            continue
        i += 1

    if block_depth:
        problems.append(f"{path.name}: unterminated block comment")
    for opened, opened_line in stack:
        problems.append(
            f"{path.name}:{opened_line}: {opened!r} was never closed"
        )
    return problems


# ---------------------------------------------------------------------------
# The "this view writes no prose" rule
# ---------------------------------------------------------------------------
#
# `ObjectMemoryWorkspaceView.swift` says, three times and in capitals, that no
# user-facing string literal may appear in it: every sentence comes from
# `ObjectMemoryCopy`, and `ObjectMemoryCopyTests` runs that type's whole output
# through the claims this cartridge is forbidden to make. A `Text("...")` added
# to the view escapes that test entirely.
#
# An independent reviewer checked and found the rule held -- and that NOTHING
# ENFORCED IT. It was asserted only in doc comments. This is the enforcement,
# and it lives here rather than in the XCTest bundle because the rule is about
# what is in a file, a test bundle cannot read the source tree it was compiled
# from, and this is the one check the Windows half of this project can actually
# run.
#
# WHAT COUNTS AS PROSE, AND WHY THE HEURISTIC IS THIS ONE.
#
# A SwiftUI view legitimately holds non-prose strings: SF Symbol names
# ("record.circle.fill"), accessibility identifiers, and format specifiers.
# Every one of those is a lowercase dotted token or a single word. Prose is not.
# So a literal is flagged when it contains a SPACE -- which no symbol name ever
# does and no sentence ever lacks.
#
# It will not catch a one-word label. Nothing that reads source text can catch
# everything, and a checker that tried would be turned off. This catches the
# thing that actually happened elsewhere in this repo: somebody writing a
# sentence where a sentence was cheap to write.

SYMBOLIC = re.compile(r"^[A-Za-z0-9_.\-]*$")


def string_literals(src: str):
    """Every string literal in a Swift source, with its line number.

    Shares the lexer above rather than a regex, which is the point: a
    regex over Swift source finds "strings" inside comments and comments
    inside strings, and this rule is not worth a checker that cries wolf.
    """
    i, n, line = 0, len(src), 1
    block_depth = 0
    found = []
    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if block_depth:
            if src.startswith("/*", i):
                block_depth += 1
                i += 2
                continue
            if src.startswith("*/", i):
                block_depth -= 1
                i += 2
                continue
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if src.startswith("/*", i):
            block_depth = 1
            i += 2
            continue
        if src.startswith('"""', i):
            j = src.find('"""', i + 3)
            if j == -1:
                break
            found.append((line, src[i + 3 : j]))
            line += src.count("\n", i, j)
            i = j + 3
            continue
        if ch == '"':
            j, buf = i + 1, []
            while j < n and src[j] != '"' and src[j] != "\n":
                if src[j] == "\\":
                    buf.append(src[j : j + 2])
                    j += 2
                    continue
                buf.append(src[j])
                j += 1
            if j >= n or src[j] != '"':
                break
            found.append((line, "".join(buf)))
            i = j + 1
            continue
        i += 1
    return found


def check_no_prose(path: Path):
    """Flag user-facing prose in a file that has forsworn it."""
    problems = []
    for line, text in string_literals(path.read_text(encoding="utf-8")):
        stripped = text.strip()
        if not stripped or SYMBOLIC.match(stripped):
            continue
        if " " not in stripped:
            continue
        problems.append(
            f"{path.name}:{line}: prose in a file that writes none: {stripped[:60]!r}"
        )
    return problems

def main(argv):
    prose_mode = "--no-prose" in argv
    paths = [Path(a) for a in argv if not a.startswith("--")]
    failures = []
    for path in paths:
        found = check(path)
        if prose_mode:
            found = found + check_no_prose(path)
        status = "ok" if not found else "PROBLEMS"
        print(f"{status:9s} {path}")
        failures.extend(found)
    for problem in failures:
        print("  " + problem)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
