import torch

try:
    from tof_vo_ml.model import ToFOdometryCNN
except ModuleNotFoundError:
    from model import ToFOdometryCNN


if __name__ == "__main__":
    model = ToFOdometryCNN(input_channels=8, output_dim=3)

    dummy_input = torch.randn(1, 8, 8, 8)
    output = model(dummy_input)

    print(output.shape)  # torch.Size([1, 3])
