from .localmodel import LocalModel


class GlobalModel(LocalModel):
    def __init__(self, args) -> None:
        super().__init__(args=args)
        pass