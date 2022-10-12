from core.train import train, train_full
from core.test import testA
from core.pretrain import pretrain


class Scaffold(object):
    def __init__(self) -> None:
        pass

    def pretrain(
        self,
        cfg_path: str = "./../config/hyps.yaml",
        model_path: str = None,
    ):
        pretrain(cfg_path, model_path)

    def train(self, cfg_path: str = "./../config/hyps.yaml", model_path: str = None):
        train(cfg_path, model_path)

    def train_full(
        self, cfg_path: str = "./../config/hyps.yaml", model_path: str = None
    ):
        train_full(cfg_path, model_path)

    def testA(
        self,
        cfg_path: str = "./../config/hyps.yaml",
        model_path: str = None,
    ):
        testA(cfg_path, model_path)

    def test(
        self,
        cfg_path: str = "./../config/hyps.yaml",
        model_path: str = None,
    ):
        self.testA(cfg_path, model_path)

    def rush(self):
        self.pretrain()
        self.train()
        self.test()
