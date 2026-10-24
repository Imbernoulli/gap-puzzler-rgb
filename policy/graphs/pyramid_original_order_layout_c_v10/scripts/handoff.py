from puzzler.modular_controller import ModularState, ModularResult
from puzzler.unified_contract import verified_handoff
def run(ctx, state: ModularState, expected_piece: str) -> ModularResult:
    return verified_handoff(ctx, state, expected_piece)
