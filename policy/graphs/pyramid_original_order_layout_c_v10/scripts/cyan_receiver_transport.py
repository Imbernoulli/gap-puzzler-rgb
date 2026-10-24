from puzzler.modular_controller import ModularState, ModularResult
from puzzler.cyan_receiver_release_feedback_delivery import execute_transport
def run(ctx, state: ModularState) -> ModularResult:
    return execute_transport(ctx, state)
