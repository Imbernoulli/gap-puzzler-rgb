from puzzler.modular_controller import ModularState, ModularResult
from puzzler.cyan_vertical_transfer_v2_adapter import close_receiver
def run(ctx, state: ModularState) -> ModularResult:
    return close_receiver(ctx, state)
