"""Ensures the repo root is on sys.path so `from src.cbf... import ...` and
`from eval_cbf_modes import ...` resolve under plain `pytest` (not just
`python -m pytest`) -- there's no pyproject.toml/setup.cfg installing this
project as a package, so pytest's default "prepend" import mode would
otherwise only add tests/ itself, not repo root.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    """--solver picks which CBFQPConfig.solver backend the solver-exercising
    tests run against. Pass --solver=scipy_slsqp to put the retained fallback
    backend through the same assertions. Deliberately explicit rather than an
    env var, so a run's solver is visible in the command line and can't be left
    set by accident between runs.

    The default is read off CBFQPConfig rather than hardcoded, so a plain
    no-flag run always tests whatever the library actually defaults to. Pinning
    a literal here would let the two drift silently -- and a stale default would
    be invisible, since a run on the wrong backend looks identical to a run on
    the right one until an assertion happens to disagree.
    """
    from src.cbf.qp_filter import CBFQPConfig

    parser.addoption(
        "--solver",
        action="store",
        default=CBFQPConfig.__dataclass_fields__["solver"].default,
        help="CBF-QP backend under test (see _SOLVERS in src/cbf/qp_filter.py)",
    )


@pytest.fixture(scope="session")
def solver_name(pytestconfig):
    return pytestconfig.getoption("--solver")
