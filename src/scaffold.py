from core.train import train
from core.test import testA


class Scaffold(object):
    def __init__(self) -> None:
        pass

    def train(self, cfg_path: str = "config/hyps.yaml"):
        train(cfg_path)

    def testA(
        self,
        cfg_path: str = "config/hyps.yaml",
        model_path: str = None,
    ):
        testA(cfg_path, model_path)

    def test(
        self,
        cfg_path: str = "config/hyps.yaml",
        model_path: str = None,
    ):
        self.testA(cfg_path, model_path)
