"""Regression test for the Stage 2 classifier-loading bug (see PROGRESS.md).

Pins that vlm_safety_score.py's use of src.Decoder.SemanticDecoder(16, 256) with
strict=True can never again silently accept a mismatched state_dict the way the old
from-scratch nn.Linear reimplementation did. This checks class identity/constructor
args, not real trained weights — no real classifier.pth exists in this checkout (see
PROGRESS.md's Stage 2 findings).

NOT executed as of this writing: this sandbox has no torch installed and no CUDA, and
src.Decoder.SemanticDecoder.__init__ hardcodes .cuda(), so it cannot be constructed here
regardless of key correctness. Run this once a GPU-capable environment (the pinned
cudatoolkit-dev=11.7.0 conda env) is available.
"""

import torch

from src.Decoder import SemanticDecoder

SEMANTIC_IN_CHANNELS = 16
SEMANTIC_OUT_CHANNELS = 256


def test_semantic_decoder_state_dict_keys_match():
    reference = SemanticDecoder(SEMANTIC_IN_CHANNELS, SEMANTIC_OUT_CHANNELS)
    state_dict = reference.state_dict()

    target = SemanticDecoder(SEMANTIC_IN_CHANNELS, SEMANTIC_OUT_CHANNELS)
    incompatible = target.load_state_dict(state_dict, strict=True)

    assert not incompatible.missing_keys, f"missing keys: {incompatible.missing_keys}"
    assert not incompatible.unexpected_keys, f"unexpected keys: {incompatible.unexpected_keys}"


if __name__ == "__main__":
    test_semantic_decoder_state_dict_keys_match()
    print("OK: SemanticDecoder state_dict keys match with strict=True.")
