from puzzler.modular_controller import ModularState, ModularResult
from puzzler.cyan_vertical_transfer_v2_adapter import approach_receiver
def run(ctx, state: ModularState) -> ModularResult:
    return approach_receiver(ctx, state)
