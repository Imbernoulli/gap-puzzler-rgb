from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_handoff import handoff_prepare
def run(ctx, state: ModularState) -> ModularResult:
    return handoff_prepare(ctx, state)
