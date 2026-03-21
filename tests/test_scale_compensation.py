import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCALE_UTILS_PATH = ROOT / "rebasin" / "rebasinnet" / "scale_utils.py"
SPEC = importlib.util.spec_from_file_location("scale_utils", SCALE_UTILS_PATH)
assert SPEC is not None and SPEC.loader is not None
scale_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scale_utils)

apply_input_inv_scale_to_conv = scale_utils.apply_input_inv_scale_to_conv
apply_input_inv_scale_to_linear = scale_utils.apply_input_inv_scale_to_linear
apply_output_scale_to_conv = scale_utils.apply_output_scale_to_conv
apply_output_scale_to_linear = scale_utils.apply_output_scale_to_linear
get_inv_scale_vector = scale_utils.get_inv_scale_vector
get_scale_vector = scale_utils.get_scale_vector
transform_bias_with_scale = scale_utils.transform_bias_with_scale


def test_identity_scale_linear():
    weight = torch.randn(5, 4, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64)
    zeros = torch.zeros(5, dtype=torch.float64)
    scale = get_scale_vector(zeros)
    inv_scale = get_inv_scale_vector(zeros)

    scaled_weight = apply_output_scale_to_linear(weight, scale)
    recovered_weight = apply_input_inv_scale_to_linear(weight.t(), inv_scale).t()
    scaled_bias = transform_bias_with_scale(bias, scale)

    assert torch.allclose(weight, scaled_weight)
    assert torch.allclose(weight, recovered_weight)
    assert torch.allclose(bias, scaled_bias)


def test_compensated_linear_function_preservation():
    batch = 8
    in_features = 4
    hidden_features = 5
    out_features = 3

    x = torch.randn(batch, in_features, dtype=torch.float64)
    w1 = torch.randn(hidden_features, in_features, dtype=torch.float64)
    b1 = torch.randn(hidden_features, dtype=torch.float64)
    w2 = torch.randn(out_features, hidden_features, dtype=torch.float64)
    b2 = torch.randn(out_features, dtype=torch.float64)
    log_scale = torch.randn(hidden_features, dtype=torch.float64) * 0.1
    scale = get_scale_vector(log_scale)
    inv_scale = get_inv_scale_vector(log_scale)

    y_ref = F.linear(F.linear(x, w1, b1), w2, b2)

    w1_t = apply_output_scale_to_linear(w1, scale)
    b1_t = transform_bias_with_scale(b1, scale)
    w2_t = apply_input_inv_scale_to_linear(w2, inv_scale)

    y_scaled = F.linear(F.linear(x, w1_t, b1_t), w2_t, b2)

    max_diff = (y_ref - y_scaled).abs().max().item()
    assert max_diff < 1e-5, f"Linear compensated scaling changed outputs: max_diff={max_diff}"


def test_compensated_conv_function_preservation():
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    w1 = torch.randn(5, 3, 3, 3, dtype=torch.float64)
    b1 = torch.randn(5, dtype=torch.float64)
    w2 = torch.randn(4, 5, 3, 3, dtype=torch.float64)
    b2 = torch.randn(4, dtype=torch.float64)
    log_scale = torch.randn(5, dtype=torch.float64) * 0.1
    scale = get_scale_vector(log_scale)
    inv_scale = get_inv_scale_vector(log_scale)

    y_ref = F.conv2d(F.conv2d(x, w1, b1, padding=1), w2, b2, padding=1)

    w1_t = apply_output_scale_to_conv(w1, scale)
    b1_t = transform_bias_with_scale(b1, scale)
    w2_t = apply_input_inv_scale_to_conv(w2, inv_scale)

    y_scaled = F.conv2d(F.conv2d(x, w1_t, b1_t, padding=1), w2_t, b2, padding=1)

    max_diff = (y_ref - y_scaled).abs().max().item()
    assert max_diff < 1e-5, f"Conv compensated scaling changed outputs: max_diff={max_diff}"


if __name__ == "__main__":
    test_identity_scale_linear()
    test_compensated_linear_function_preservation()
    test_compensated_conv_function_preservation()
    print("scale compensation tests passed")
