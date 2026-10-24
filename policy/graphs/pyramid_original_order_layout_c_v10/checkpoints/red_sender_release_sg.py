from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('handoff_receiver_only', world, outputs)
CHECKPOINTS = [Checkpoint(name='red_sender_release_independent_postcondition', subgraph='red_sender_release_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
