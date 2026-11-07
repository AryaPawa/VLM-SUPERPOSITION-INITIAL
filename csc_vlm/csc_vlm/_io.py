"""
csc_vlm._io  --  make console output encoding-safe on every platform.

On Windows, when stdout/stderr are redirected (pytest's subprocess capture,
``> log.txt``, CI), Python encodes them with the ANSI code page (cp1252). Our
outputs contain non-cp1252 characters (Gamma, ||a||, ~=, beta, and decoded
tokenizer strings such as CJK tokens from Qwen), so the first print raises
UnicodeEncodeError and the run dies. Linux/Kaggle/SLURM use UTF-8 and are not
affected. Call ``utf8_stdio()`` first in every entry point.
"""
import sys


def utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001  (non-reconfigurable stream, e.g. some IDE consoles)
            pass
