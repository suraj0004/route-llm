import abc

import torch

from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel


def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]

@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        # This is the model pair for scoring at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        hidden_size=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MFModel.from_pretrained(
            checkpoint_path,
            dim=hidden_size,
            num_models=num_models,
            text_dim=text_dim,
            num_classes=num_classes,
            use_proj=use_proj,
        )
        self.model = self.model.eval().to(device)
        self.strong_model_id = MODEL_IDS[strong_model]
        self.weak_model_id = MODEL_IDS[weak_model]

    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate



ROUTER_CLS = {
    "mf": MatrixFactorizationRouter,
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
