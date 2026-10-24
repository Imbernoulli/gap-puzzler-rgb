from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_handoff import receiver_close
def run(ctx, state: ModularState) -> ModularResult:
    return receiver_close(ctx, state)
