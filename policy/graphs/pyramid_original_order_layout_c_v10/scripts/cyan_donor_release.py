from puzzler.modular_controller import ModularState, ModularResult
from puzzler.cyan_vertical_transfer_v2_adapter import release_donor
def run(ctx, state: ModularState) -> ModularResult:
    return release_donor(ctx, state)
