from gap.runtime.verify import Checkpoint
from puzzler.modular_verification import verify
def verified(world, outputs):
    return verify('handoff_receiveronly', world, outputs)
CHECKPOINTS = [Checkpoint(name='cyan_donor_release_independent_postcondition', subgraph='cyan_donor_release_sg', predicate=verified, validate=True, rationale="Independent native physical verification outside policy inputs")]
