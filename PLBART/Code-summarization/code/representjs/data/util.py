import re
from typing import Optional, Tuple
import time
from subprocess import Popen, PIPE
from loguru import logger


class Timer:
    """from https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/"""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        logger.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


def EncodeAsIds(sp, alpha, prog):
    # Encode as ids with sentencepiece
    if alpha:
        # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
        # NOTE: what is the second argument here (-1)?
        return sp.SampleEncodeAsIds(prog, -1, alpha)

    # using the best decoding
    return sp.EncodeAsIds(prog)
