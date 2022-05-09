from click.testing import CliRunner
import pytest
from forestml.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()
