import time

from pymoo.core.callback import Callback


class ResCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["X"] = []
        self.data["F"] = []
        self.data["time"] = []

    def notify(self, algorithm):
        self.data["X"].append(algorithm.pop.get("X").tolist())
        self.data["F"].append(algorithm.pop.get("F").tolist())
        self.data["time"].append(time.time())
