from puzzler.modular_controller import ModularState, ModularResult
from puzzler.modular_controller import observe_and_plan
def run(ctx, state: ModularState) -> ModularResult:
    return observe_and_plan(ctx, state)
