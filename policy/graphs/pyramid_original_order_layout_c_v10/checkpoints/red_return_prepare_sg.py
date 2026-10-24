from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('return_handoff_prepared', world, outputs)
CHECKPOINTS = [Checkpoint(name='red_return_prepare_independent_postcondition', subgraph='red_return_prepare_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
