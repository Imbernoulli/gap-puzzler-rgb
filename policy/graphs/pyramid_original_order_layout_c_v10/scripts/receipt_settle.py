from puzzler.modular_controller import ModularState, ModularResult
from puzzler.unified_contract import settle_placement_evidence
def run(ctx, state: ModularState) -> ModularResult:
    return settle_placement_evidence(ctx, state)
