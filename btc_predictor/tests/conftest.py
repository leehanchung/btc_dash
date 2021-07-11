import pytest

from btc_predictor.models import BasePredictor


class MockDataset:
    pass


class MockModel:
    pass


mock_dataset = MockDataset
mock_model = MockModel


# @pytest.fixture
# def model_fixture():
#     model = BaseModel(dataset=mock_dataset, model=mock_model)
#     yield model
