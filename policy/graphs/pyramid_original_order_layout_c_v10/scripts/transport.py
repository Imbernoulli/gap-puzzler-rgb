from puzzler.modular_controller import ModularState, ModularResult
from puzzler.modular_controller import transport
def run(ctx, state: ModularState) -> ModularResult:
    return transport(ctx, state)
