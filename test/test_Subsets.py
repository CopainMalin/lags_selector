import pytest
from pandas import DataFrame
from mlforecast import MLForecast
from Subsets import Subsets


@pytest.fixture
def sample_dataset():
    # Create a sample dataset for testing
    dataset = DataFrame({"ds": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    return dataset


@pytest.fixture
def sample_forecaster():
    # Create a sample forecaster for testing
    forecaster = MLForecast()
    return forecaster


def test_split_dataset(sample_dataset):
    subsets = Subsets(sample_dataset, None)
    train, test = subsets._Subsets__split_dataset(sample_dataset, 0.2)
    assert len(train) == 3
    assert len(test) == 2


def test_extract_Xy(sample_dataset):
    subsets = Subsets(sample_dataset, None)
    X, y = subsets._Subsets__extract_Xy(sample_dataset)
    assert len(X) == 5
    assert len(y) == 5


def test_post_init(sample_dataset, sample_forecaster):
    subsets = Subsets(sample_dataset, sample_forecaster)
    subsets.__post_init__()
    assert len(subsets.train) == 4
    assert len(subsets.test) == 1
    assert len(subsets.subtrain) == 3
    assert len(subsets.validation) == 1
    assert len(subsets.X_train) == 3
    assert len(subsets.y_train) == 3
    assert len(subsets.X_eval) == 1
    assert len(subsets.y_eval) == 1
