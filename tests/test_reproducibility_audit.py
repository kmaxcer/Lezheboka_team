from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reproducibility_audit import gap_mask, sha256  # noqa: E402


def test_gap_mask_handles_string_flags():
    s = pd.Series([True, False, "true", "False", "1", "0", None])
    got = gap_mask(s).tolist()
    assert got == [True, False, True, False, True, False, False]


def test_sha256_is_stable(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"agropulse")
    assert sha256(p) == "c989aaf369f817df556adccf6e2b8f6072176db40b36b72d2bbc93b2cc3e6b63"
