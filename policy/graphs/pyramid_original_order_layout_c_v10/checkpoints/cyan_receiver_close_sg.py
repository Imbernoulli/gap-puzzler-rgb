from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('handoff_dualhold', world, outputs)
CHECKPOINTS = [Checkpoint(name='cyan_receiver_close_independent_postcondition', subgraph='cyan_receiver_close_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
