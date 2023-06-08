"""Insert spaces, mypython/split_chinese.py."""
import re


def insert_spaces(text: str, method: int = None) -> str:
    r"""Insert space between Chinese characters.

    To speed up, first check text contains more latin letters or Chinese charas, if more latin letters use insert_spaces(text,, 3) else use insert_spaces(text, None)

    Args:
        text: string of latin and Chinese chars
        method:
            None: default, re.sub(r"(?<=[a-zA-Z\d]) (?=[a-zA-Z\d])", "", text.replace("", " "))  # NOQA
            1: re.sub(r"[一-龟]|[^ 一-龟]+", r"\g<0> ", text)

    >>> insert_spaces("test亨利it四世上").strip()
    'test 亨 利 it 四 世 上'
    >>> insert_spaces("test亨利it四世上").strip().__len__()
    17

    """
    if method is None:
        if re.findall(r"[a-zA-Z ]+", text).__len__() > len(text) // 2:  # more latin  # NOQA
            method = 3
        else:  # more Chinese
            method = 0

    if method == 0:
        return re.sub(r"(?<=[a-zA-Z\d]) (?=[a-zA-Z\d])", "", text.replace("", " "))
    elif method == 1:
        return re.sub(r"[一-龟]|[^ 一-龟]+", r"\g<0> ", text)
    elif method == 2:
        return re.sub(r"[一-龟]|\d+|\w+", r"\g<0> ", text)
    elif method == 3:
        return re.sub(r"(?<=[^a-zA-Z\d])|(?=[^a-zA-Z\d])", " ", text)
    else:
        return re.sub(
            r"(?<=[a-zA-Z\d]) (?=[a-zA-Z\d])", "", text.replace("", " ")
        )  # NOQA
