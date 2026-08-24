import os
import sys
from pathlib import Path

# Add src to python path so slurmate is importable
src_path = str(Path(__file__).resolve().parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Enable mock mode universally for tests
os.environ["SLURMATE_MOCK"] = "1"


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_cluster_cache():
    """Forget memoised cluster facts between tests.

    The fetchers for partition names, accounts, QoS, node features, SelectType
    and MaxArraySize are memoised per process (they cannot change during one
    slurmate run, and were being queried twice per invocation). Tests vary the
    mocked output, so the cache has to be dropped between them or a later test
    would read an earlier one's answer.
    """
    from slurmate.system_utils import reset_cluster_cache

    reset_cluster_cache()
    yield
    reset_cluster_cache()
