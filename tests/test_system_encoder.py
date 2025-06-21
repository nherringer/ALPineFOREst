from alpfore.encoder.system_encoder import SystemEncoder
import pytest

def test_system_encoder_encoding():
    encoder = SystemEncoder()
    encoded = encoder.encode(("A", "B", 1.0))
    assert isinstance(encoded, list)
    assert all(isinstance(x, (int, float)) for x in encoded)
