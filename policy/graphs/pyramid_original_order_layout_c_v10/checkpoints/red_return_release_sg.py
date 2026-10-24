from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('return_handoff_receiver_only', world, outputs)
CHECKPOINTS = [Checkpoint(name='red_return_release_independent_postcondition', subgraph='red_return_release_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
