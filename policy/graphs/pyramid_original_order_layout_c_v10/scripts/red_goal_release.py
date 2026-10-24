from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_goal import release
def run(ctx, state: ModularState) -> ModularResult:
    return release(ctx, state)
