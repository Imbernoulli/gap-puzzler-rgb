from puzzler.modular_controller import ModularState, ModularResult
from puzzler.cyan_vertical_transfer_v2_adapter import verify_receiver
def run(ctx, state: ModularState) -> ModularResult:
    return verify_receiver(ctx, state)
