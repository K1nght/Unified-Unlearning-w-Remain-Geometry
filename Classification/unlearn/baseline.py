from .unlearn_method import UnlearnMethod


class Baseline(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)


