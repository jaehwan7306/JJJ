"""Verified LLMLingua tokenizer boundary compatibility for new runs."""

def prefix_boundary_from_offsets(tokenizer, prefix: str, text: str) -> int:
    """Locate a proven boundary in the actual combined token stream.

    Decoding is not used as an equality test: whitespace cleanup and tokenizer
    normalization can make decode(encode(text)) differ from the source string.
    A token crossing the prefix/context boundary is rejected, never guessed.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Prefix fallback requires a fast tokenizer with source offsets")
    combined = prefix + text
    encoded = tokenizer(combined, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if len(ids) != len(offsets):
        raise ValueError("Token IDs and source offsets disagree")
    if ids != tokenizer(combined, add_special_tokens=False)["input_ids"]:
        raise ValueError("Requesting source offsets changed tokenization")
    boundary = len(prefix)
    first_context = None
    for index, (start, end) in enumerate(offsets):
        if not 0 <= start <= end <= len(combined):
            raise ValueError("Tokenizer returned invalid source offsets")
        if start < boundary < end:
            raise ValueError("A token crosses the question/context boundary; no safe split exists")
        if end > boundary and first_context is None:
            first_context = index
        if first_context is not None and start < boundary:
            raise ValueError("Tokenizer offsets do not form a question/context partition")
    if first_context is None or first_context == 0:
        raise ValueError("Cannot verify both prefix and context tokens")
    return first_context
