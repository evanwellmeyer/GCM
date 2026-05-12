import pytest
import torch


@pytest.fixture
def device():
    # Component tests also run as a standalone script; this fixture lets pytest
    # exercise the same device-selection path without changing those tests.
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
