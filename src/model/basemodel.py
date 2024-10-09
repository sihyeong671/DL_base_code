import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # define DL layer

    def forward(self, x):
        # using module that you define in __init__ and check whether it's right sequence
        return x


if __name__ == "__main__":
    # Check the model
    # model = CustomModel()
    # summary(model, input_size=(1, 3, 255, 255))
    pass
