from puzzler.modular_controller import ModularState, ModularResult
from puzzler.modular_controller import grasp_and_bind
def run(ctx, state: ModularState) -> ModularResult:
    return grasp_and_bind(ctx, state)
