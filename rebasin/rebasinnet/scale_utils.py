import torch


def get_scale_vector(log_scale):
    return torch.exp(log_scale)


def get_inv_scale_vector(log_scale):
    return torch.exp(-log_scale)


def apply_output_scale_to_linear(weight, scale):
    return weight * scale.view(-1, 1)


def apply_input_inv_scale_to_linear(weight, inv_scale):
    return weight * inv_scale.view(1, -1)


def apply_output_scale_to_conv(weight, scale):
    return weight * scale.view(-1, 1, 1, 1)


def apply_input_inv_scale_to_conv(weight, inv_scale):
    return weight * inv_scale.view(1, -1, 1, 1)


def transform_bias_with_scale(bias, scale):
    return bias * scale


def apply_output_scale_to_weight(weight, scale):
    if weight.ndim == 2:
        return apply_output_scale_to_linear(weight, scale)
    if weight.ndim >= 3:
        return weight * scale.view(-1, *([1] * (weight.ndim - 1)))
    raise ValueError(f"Unsupported weight rank for output scaling: shape={tuple(weight.shape)}")


def apply_input_inv_scale_to_weight(weight, inv_scale):
    if weight.ndim == 2:
        return apply_input_inv_scale_to_linear(weight, inv_scale)
    if weight.ndim >= 3:
        return weight * inv_scale.view(1, -1, *([1] * (weight.ndim - 2)))
    raise ValueError(f"Unsupported weight rank for input inverse scaling: shape={tuple(weight.shape)}")
