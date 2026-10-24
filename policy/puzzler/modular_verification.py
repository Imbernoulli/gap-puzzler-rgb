"""Test-rehearsal checkpoint bridge. Never imported by policy/planning modules."""
_PROVIDER=None

def install(provider):
    global _PROVIDER
    if _PROVIDER is not None:raise RuntimeError('One independent verification provider per native worker')
    _PROVIDER=provider


def capture(env,world):
    if _PROVIDER is None:raise RuntimeError('Independent module verification provider is not installed')
    return _PROVIDER.capture(env,world)


def verify(kind,world,outputs):
    if _PROVIDER is None:raise RuntimeError('Independent module verification provider is not installed')
    return _PROVIDER.verify(kind,world,outputs)
