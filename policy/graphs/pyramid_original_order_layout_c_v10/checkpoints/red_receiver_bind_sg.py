from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('handoff_receiver_bound', world, outputs)
CHECKPOINTS = [Checkpoint(name='red_receiver_bind_independent_postcondition', subgraph='red_receiver_bind_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
