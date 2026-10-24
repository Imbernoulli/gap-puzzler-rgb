from puzzler.modular_controller import ModularState, ModularResult
from puzzler.red_bimanual_handoff import sender_release
def run(ctx, state: ModularState) -> ModularResult:
    return sender_release(ctx, state)
