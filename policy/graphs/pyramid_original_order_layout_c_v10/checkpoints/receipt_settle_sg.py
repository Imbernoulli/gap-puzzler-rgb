from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('release_dispatch', world, outputs)
CHECKPOINTS = [Checkpoint(name='receipt_settle_independent_postcondition', subgraph='receipt_settle_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
