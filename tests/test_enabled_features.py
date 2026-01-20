def test_torch_importable():
    import torch

    # Just ensure it doesn't crash in this environment.
    assert torch is not None