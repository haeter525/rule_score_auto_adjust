import pytest
import torch
from model import RuleAdjustmentModel


@pytest.fixture
def model():
    return RuleAdjustmentModel(num_of_rules=5)


def test_forward_with_passing_stages(model):
    # Test the forward method with a tensor of passing stages
    input_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    output = model.forward(input_tensor)
    assert output is not None
    assert output.size(dim=-1) == 1


def test_forward_with_multiple_dimensions(model):
    # Test the forward method with a tensor of passing stages with multiple dimensions
    input_tensor = torch.tensor(
        [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32
    )
    output = model.forward(input_tensor)
    assert output is not None
    assert output.size(dim=-1) == 1 and output.size(dim=-2) == 2


def test_is_apk_malicious():
    # Initialize the model and the model input
    model = RuleAdjustmentModel(num_of_rules=5)
    passing_stages = torch.tensor([1, 2, 3, 4, 5])

    # Call the model with the model input to get the output
    score_in_percentage = model.forward(passing_stages)

    # Determine if the APK is malicious based on the output
    if score_in_percentage > 0.5:
        print("The APK is malicious")
    else:
        print("The APK is not malicious")
