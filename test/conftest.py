import pytest


@pytest.fixture(scope="session", autouse=True)
def increase_stack_depth():
    # Bump recursion limit.
    # Needed for Kronecker tests
    # As per https://note.nkmk.me/en/python-sys-recursionlimit/
    import resource
    import sys

    sys.setrecursionlimit(2000)
    resource.setrlimit(resource.RLIMIT_STACK, (-1, -1))  # Increase underlying C stack
