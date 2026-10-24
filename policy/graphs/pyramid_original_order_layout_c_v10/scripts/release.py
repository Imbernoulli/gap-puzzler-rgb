from puzzler.modular_controller import ModularState, ModularResult
from puzzler.modular_controller import carry_and_release
def run(ctx, state: ModularState) -> ModularResult:
    if state['piece'] == 'cyan':
        parameters = {**state['parameters'], 'release_module': 'puzzler.cyan_receiver_release_feedback_delivery'}
        state = {**state, 'parameters': parameters}
    return carry_and_release(ctx, state)
