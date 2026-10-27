"""Gold-blind budget enforcement. Never crop, pad, or select by task accuracy."""
from dataclasses import asdict


def fit_reference_cap(produce, count_tokens, cap, initial_control, attempt_limit, *, integer_control=False):
    control = initial_control
    attempts = []
    for _ in range(attempt_limit):
        result = produce(control)
        tokens = count_tokens(result.text)
        if tokens < 1:
            raise ValueError("Budgeted compressor produced empty context")
        attempts.append({"control": control, "reference_tokens": tokens, "result": asdict(result)})
        if tokens <= cap:
            return result, sum(a["result"]["seconds"] for a in attempts), attempts
        next_control = control * (cap / tokens) * 0.95
        if integer_control:
            next_control = min(control - 1, int(next_control))
            if next_control < 1:
                break
        control = next_control
    raise ValueError(f"Cannot meet reference-token cap={cap} after {len(attempts)} attempts; "
                     "no truncation or padding performed. Use a fresh run after revising the protocol.")
