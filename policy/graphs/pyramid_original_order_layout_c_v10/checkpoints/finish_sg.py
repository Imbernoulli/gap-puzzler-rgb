from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('finish', world, outputs)
CHECKPOINTS = [Checkpoint(name='finish_independent_postcondition', subgraph='finish_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
