from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('transport', world, outputs)
CHECKPOINTS = [Checkpoint(name='red_goal_transport_independent_postcondition', subgraph='red_goal_transport_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
