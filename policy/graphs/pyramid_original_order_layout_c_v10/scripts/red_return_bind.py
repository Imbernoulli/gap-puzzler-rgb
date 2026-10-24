from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_handoff import receiver_bind
def run(ctx, state: ModularState) -> ModularResult:
    return receiver_bind(ctx, state)
