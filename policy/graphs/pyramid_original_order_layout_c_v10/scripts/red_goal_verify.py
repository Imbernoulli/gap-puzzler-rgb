from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_goal import verify_release
def run(ctx, state: ModularState) -> ModularResult:
    return verify_release(ctx, state)
