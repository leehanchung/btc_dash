import logging

from omegaconf import OmegaConf

from btc_predictor.models import BasePredictor, LSTMBTCPredictor

_logger = logging.getLogger(__name__)


def load_wo_hydra(*, config_file: str) -> BasePredictor:
    """Helper function to loadinitialize [summary]

    Args:
        config_file (str): [description]

    Returns:
        BasePredictor: [description]
    """
    _logger.info("Loading model without Hydra Config...")
    params = OmegaConf.load(config_file)
    model = LSTMBTCPredictor(
        model_args=params.model_params, train_args=params.train_params
    )
    model.load(model_name=params.model_params.model_name, origin_pwd=True)

    return model
