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
    model = LSTMBTCPredictor(params=params)
    model.load(model_file=params.model_file, origin_pwd=True)

    return model
