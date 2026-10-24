from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('grasp', world, outputs)
CHECKPOINTS = [Checkpoint(name='grasp_independent_postcondition', subgraph='grasp_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
