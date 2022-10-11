from cgi import test
from core.train import train
from core.test import test


class Scaffold(object):
    def __init__(self) -> None:
        pass

    def train(self, cfg_path: str = "config/hyps.yaml"):
        train(cfg_path)

    def test(
        self,
        cfg_path: str = "config/hyps.yaml",
        model_path: str = None,
    ):
        test(cfg_path, model_path)
