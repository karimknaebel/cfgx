"""CLI assignments as ordered contributions to the resolver."""

import ast
import re

from .expressions import delete, namespace, replace


def parse_path(text):
    """Parse dotted keys and Python literal subscripts without executing code."""
    keys = []
    pos = 0
    while pos < len(text):
        if text[pos] == "[":
            start = pos
            depth = 1
            quote = None
            pos += 1
            while pos < len(text) and depth:
                char = text[pos]
                if quote:
                    if char == "\\":
                        pos += 2
                        continue
                    if char == quote:
                        quote = None
                elif char in "\"'":
                    quote = char
                elif char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                pos += 1
            if depth:
                raise ValueError(f"Unclosed subscript in override path: {text}")
            key = ast.literal_eval(text[start + 1 : pos - 1])
            hash(key)
            keys.append(key)
        else:
            match = re.match(r"[^.\[\]\s]+", text[pos:])
            if not match:
                raise ValueError(f"Invalid override path: {text}")
            keys.append(match[0])
            pos += len(match[0])
        if pos < len(text) and text[pos] == ".":
            pos += 1
            if pos == len(text) or text[pos] in ".[":
                raise ValueError(f"Invalid override path: {text}")
        elif pos < len(text) and text[pos] != "[":
            raise ValueError(f"Invalid override path: {text}")
    if not keys:
        raise ValueError("Override path must not be empty")
    return tuple(keys)


def parse_override(text):
    quote = None
    depth = 0
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif quote:
            if char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in "\"'" and depth:
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "=" and not depth:
            path, raw = text[:index], text[index + 1 :]
            if path.endswith("!"):
                if raw:
                    raise ValueError("Delete overrides must not include a value")
                return parse_path(path[:-1]), delete
            if path.endswith(("+", "-")):
                raise ValueError(
                    "Use path=expr:value.map(...) for sequence transformations"
                )
            return parse_path(path), replace(_parse_value(raw))
    raise ValueError(f"Override must use path=value or path!=: {text}")


def _parse_value(text):
    if text.startswith("expr:"):
        return eval(compile(text[5:], "<cfgx override>", "eval"), namespace())
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
