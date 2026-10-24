from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_goal import carry
def run(ctx, state: ModularState) -> ModularResult:
    return carry(ctx, state)
