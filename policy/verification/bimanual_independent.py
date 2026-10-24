"""Owner-only joint evaluator; original strict release and all12 finish remain unchanged.

Red adds two distinct four-checkpoint physical handoffs to the baseline session.
Cyan's extension shares exactly the same private event cache. No privileged
state is attached to a policy World or returned through a tool.
"""
from verification import independent as base
from verification.red_bimanual_goal_independent import _Session

# The transfer and unchanged final evaluators share these owner-only objects.
_LOCK = base._LOCK
_SESSIONS = base._SESSIONS
_EVENTS = base._EVENTS
_error = base._error
_json_safe = base._json_safe
capture = base.capture
validate = base.validate
verify = base.verify
diagnostics = base.diagnostics


def install(env, expected_order, initialized_prefix, output_dir):
    with _LOCK:
        if id(env) in _SESSIONS:
            raise RuntimeError('Independent verifier already installed for this Env')
        session = _Session(env, expected_order, initialized_prefix, output_dir)
        _SESSIONS[id(env)] = session
    return session


def finalize(env):
    # Full-chain acceptance uses the original finish/session summary. The
    # given-prefix Red qualification finalizer cannot claim full acceptance.
    session = _SESSIONS[id(env)]
    observer = getattr(session, "_cyan_transfer_observer", None)
    if observer is not None:
        env.remove_step_observer(observer)
    return base.finalize(env)
