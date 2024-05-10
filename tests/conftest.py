import importlib
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["OUTLINES_DISABLE_CACHE"] = "1"
    import outlines.caching

    importlib.reload(outlines.caching)
