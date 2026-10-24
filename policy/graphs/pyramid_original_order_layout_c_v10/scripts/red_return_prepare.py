from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_handoff import return_handoff_prepare
def run(ctx, state: ModularState) -> ModularResult:
    return return_handoff_prepare(ctx, state)
