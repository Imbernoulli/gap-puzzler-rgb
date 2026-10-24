from puzzler.modular_controller import ModularState, ModularResult
from puzzler.unified_contract import finish_assembly
def run(ctx, state: ModularState) -> ModularResult:
    return finish_assembly(ctx, state)
