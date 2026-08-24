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


# Environment variables slurmate reads that would silently change a result if the
# shell running the suite happens to set them. This is not hypothetical: the
# config tests set HOME and wrote $HOME/.config/slurmate/config.toml, but
# load_config() honours XDG_CONFIG_HOME first — and GitHub's runners export it
# while a midway3 login shell does not. Seven tests passed locally and failed in
# CI for a reason that had nothing to do with the code under test.
#
# SLURMATE_MOCK is deliberately absent: the module sets it above, on purpose.
# TERM/USER/LOGNAME are left alone too — tests that depend on them patch them,
# and blanking them changes what prompt_toolkit and getpass do rather than
# isolating anything.
_AMBIENT_VARS = (
    "XDG_CONFIG_HOME",       # decides where the global config is looked for
    "SLURMATE_GPU_FORMAT",   # decides which GPU directive is emitted
    "SLURMATE_NO_SAVE",
    "SLURMATE_DEBUG",
    "SLURMATE_LOG_DIR",
    "NO_COLOR",              # both change rich's rendering, so both change
    "FORCE_COLOR",           # what an output assertion sees
    "EDITOR",
    "VISUAL",
    "LMOD_CMD",              # set on any Lmod login shell, and consulted by
    "MODULESHOME",           # _module_command() — so module checks differed by
)                            # whether the suite ran from a login shell


@pytest.fixture(autouse=True)
def _neutralise_ambient_env(monkeypatch):
    """Remove host-specific variables so a test's result is the code's, not the
    shell's. Tests that need one of these set it themselves; setenv inside the
    test still wins, because this runs first.
    """
    for name in _AMBIENT_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
