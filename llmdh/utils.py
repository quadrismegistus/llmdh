from .imports import *


def better_split(s):
    import re

    tokens = re.findall(r"\S+|\s+", s)
    return tokens


def truncate(s, n=100000, end=True):
    s = s.replace("\n\n", "\n")
    toks = better_split(s)
    toks = toks[: n * 2] if not end else toks[-n * 2 :]
    return "".join(toks)


def dirty_json_loads(s, as_list=False):
    if not "{" in s and "}" in s:
        return None
    s = "{" + s.split("{", 1)[-1]
    s = "}".join(s.split("}")[:-1]) + "}"
    if as_list:
        s = "[" + s + "]"
    return json.loads(s)


def dirty_jsonl_loads(s):
    return dirty_json_loads(s, as_list=True)
