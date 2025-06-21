from alpfore.pipeline.pipeline import Pipeline
import pytest


def test_pipeline_interface_minimal():
    class DummyEncoder:
        def encode(self, x):
            return list(x)

    class DummyLoader:
        def load(self):
            return ["structure"]

    p = Pipeline(encoder=DummyEncoder(), loader=DummyLoader())
    encoded = p.encode_and_load([("A", "B", 1.0)])
    assert isinstance(encoded, list)
    assert "structure" in encoded


