import torch
from itertools import chain
from copy import deepcopy
from .sinkhorn import Sinkhorn, matching
from .graph.auto_graph import solve_graph
from .scale_utils import (
    apply_input_inv_scale_to_weight,
    apply_output_scale_to_weight,
    get_inv_scale_vector,
    get_scale_vector,
    transform_bias_with_scale,
)


class ReparamNet(torch.nn.Module):
    def __init__(self, model, permutation_type="mat_mul"):
        super().__init__()
        _permutation_types = ["mat_mul", "broadcast"]
        assert (
            permutation_type in _permutation_types
        ), "Permutation type must be in {}".format(_permutation_types)
        self.permutation_type = permutation_type
        self.output = deepcopy(model)
        self.model = deepcopy(model)
        for p1, p2 in zip(self.model.parameters(), self.output.parameters()):
            p1.requires_grad = False
            p2.requires_grad = False

    def set_model(self, model):
        self.model = deepcopy(model)
        for p1 in self.model.parameters():
            p1.requires_grad = False

    def _unpack_transform(self, transforms, index):
        transform = transforms[index]
        if isinstance(transform, dict):
            return (
                transform["perm"],
                transform.get("scale", None),
                transform.get("inv_scale", None),
            )
        return transform, None, None

    def training_rebasin(self, transforms):
        for (name, p1), p2 in zip(
            self.output.named_parameters(), self.model.parameters()
        ):
            if (
                name not in self.map_param_index
                or name not in self.map_prev_param_index
            ):
                continue
            i = self.perm_dict[self.map_param_index[name]]
            j = (
                self.perm_dict[self.map_prev_param_index[name]]
                if self.map_prev_param_index[name] is not None
                else None
            )
            Pi, scale_i, _ = (
                self._unpack_transform(transforms, i) if i is not None else (None, None, None)
            )
            Pj, _, inv_scale_j = (
                self._unpack_transform(transforms, j) if j is not None else (None, None, None)
            )

            if "bias" in name[-4:]:
                if i is not None:
                    transformed = p2
                    if scale_i is not None:
                        transformed = transform_bias_with_scale(transformed, scale_i)
                    p1.copy_(Pi @ transformed)
                else:
                    continue

            # batchnorm
            elif len(p1.shape) == 1:
                if i is not None:
                    transformed = p2
                    if scale_i is not None:
                        transformed = transform_bias_with_scale(transformed, scale_i)
                    p1.copy_((Pi @ transformed.view(p1.shape[0], -1)).view(p2.shape))

            # mlp / cnn
            elif "weight" in name[-6:]:
                transformed = p2
                if i is not None and j is None:
                    if scale_i is not None:
                        transformed = apply_output_scale_to_weight(transformed, scale_i)
                    p1.copy_((Pi @ transformed.view(Pi.shape[0], -1)).view(p2.shape))

                if i is not None and j is not None:
                    if scale_i is not None:
                        transformed = apply_output_scale_to_weight(transformed, scale_i)
                    transformed = (Pi @ transformed.view(Pi.shape[0], -1)).view(p2.shape)
                    if inv_scale_j is not None:
                        transformed = apply_input_inv_scale_to_weight(
                            transformed, inv_scale_j
                        )
                    p1.copy_(
                        (
                            Pj.view(1, *Pj.shape)
                            @ transformed.view(p2.shape[0], Pj.shape[0], -1)
                        ).view(p2.shape)
                    )

                if i is None and j is not None:
                    if inv_scale_j is not None:
                        transformed = apply_input_inv_scale_to_weight(
                            transformed, inv_scale_j
                        )
                    p1.copy_(
                        (
                            Pj.view(1, *Pj.shape)
                            @ transformed.view(p2.shape[0], Pj.shape[0], -1)
                        ).view(p2.shape)
                    )

    def update_batchnorm(self, model):
        for m1, m2 in zip(self.model.modules(), model.modules()):
            if "BatchNorm" in str(type(m2)):
                if m2.running_mean is None:
                    m1.running_mean = None
                else:
                    m1.running_mean.copy_(m2.running_mean)
                if m2.running_var is None:
                    m1.running_var = None
                    m1.track_running_stats = False
                else:
                    m1.running_var.copy_(m2.running_var)

    def permute_batchnorm(self, transforms):
        for (name, m1), m2 in zip(self.output.named_modules(), self.model.modules()):
            if "BatchNorm" in str(type(m2)):
                if name + ".weight" in self.map_param_index:
                    if m2.running_mean is None and m2.running_var is None:
                        continue
                    i = self.perm_dict[self.map_param_index[name + ".weight"]]
                    Pi, _, _ = (
                        self._unpack_transform(transforms, i)
                        if i is not None
                        else (None, None, None)
                    )
                    index = (
                        torch.argmax(Pi, dim=1)
                        if i is not None
                        else torch.arange(m2.running_mean.shape[0])
                    )
                    m1.running_mean.copy_(m2.running_mean[index, ...])
                    m1.running_var.copy_(m2.running_var[index, ...])

    def eval_rebasin(self, transforms):
        for (name, p1), p2 in zip(
            self.output.named_parameters(), self.model.parameters()
        ):
            if (
                name not in self.map_param_index
                or name not in self.map_prev_param_index
            ):
                continue
            i = self.perm_dict[self.map_param_index[name]]
            j = (
                self.perm_dict[self.map_prev_param_index[name]]
                if self.map_prev_param_index[name] is not None
                else None
            )
            Pi, scale_i, _ = (
                self._unpack_transform(transforms, i) if i is not None else (None, None, None)
            )
            Pj, _, inv_scale_j = (
                self._unpack_transform(transforms, j) if j is not None else (None, None, None)
            )

            if "bias" in name[-4:]:
                if i is not None:
                    transformed = p2.data
                    if scale_i is not None:
                        transformed = transform_bias_with_scale(transformed, scale_i)
                    index = torch.argmax(Pi, dim=1)
                    p1.copy_(transformed[index, ...])
                else:
                    continue

            # batchnorm
            elif len(p1.shape) == 1:
                if i is not None:
                    transformed = p2.data
                    if scale_i is not None:
                        transformed = transform_bias_with_scale(transformed, scale_i)
                    index = torch.argmax(Pi, dim=1)
                    p1.copy_(transformed[index, ...])

            # mlp / cnn
            elif "weight" in name[-6:]:
                transformed = p2.data
                if i is not None and j is None:
                    if scale_i is not None:
                        transformed = apply_output_scale_to_weight(transformed, scale_i)
                    index = torch.argmax(Pi, dim=1)
                    p1.copy_(transformed.view(Pi.shape[0], -1)[index, ...].view(p2.shape))

                if i is not None and j is not None:
                    if scale_i is not None:
                        transformed = apply_output_scale_to_weight(transformed, scale_i)
                    index = torch.argmax(Pi, dim=1)
                    transformed = transformed[index, ...]
                    if inv_scale_j is not None:
                        transformed = apply_input_inv_scale_to_weight(
                            transformed, inv_scale_j
                        )
                    index = torch.argmax(Pj, dim=1)
                    p1.copy_(transformed[:, index, ...])

                if i is None and j is not None:
                    if inv_scale_j is not None:
                        transformed = apply_input_inv_scale_to_weight(
                            transformed, inv_scale_j
                        )
                    index = torch.argmax(Pj, dim=1)
                    p1.copy_(
                        (
                            transformed.view(p2.shape[0], Pj.shape[0], -1)[:, index, ...]
                        ).view(p2.shape)
                    )

    def forward(self, transforms):
        for p1, p2 in zip(self.output.parameters(), self.model.parameters()):
            p1.data = p2.data.clone()

        for p1 in self.output.parameters():
            p1._grad_fn = None

        if self.training or self.permutation_type == "mat_mul":
            self.training_rebasin(transforms)
        else:
            self.eval_rebasin(transforms)

        self.permute_batchnorm(transforms)

        return self.output

    def to(self, device):
        self.output.to(device)
        self.model.to(device)

        return self


class RebasinNet(torch.nn.Module):
    def __init__(
        self,
        model,
        input_shape,
        remove_nodes=list(),
        l=1.0,
        tau=1.0,
        n_iter=20,
        operator="implicit",
        permutation_type="mat_mul",
        scale_invariant=False,
        lambda_scale=1e-4,
    ):
        super().__init__()
        assert operator in [
            "implicit",
        ], "Operator must be either `implicit`"

        self.reparamnet = ReparamNet(model, permutation_type=permutation_type)
        self.param_precision = next(iter(model.parameters())).data.dtype
        input = torch.randn(input_shape, dtype=self.param_precision)
        perm_dict, n_perm, permutation_g, parameter_map = solve_graph(
            model, input, remove_nodes=remove_nodes
        )

        P_sizes = [None] * n_perm
        map_param_index = dict()
        map_prev_param_index = dict()
        nodes = list(permutation_g.nodes.keys())
        for name, p in model.named_parameters():
            if parameter_map[name] not in nodes:
                continue
            else:
                map_param_index[name] = permutation_g.naming[parameter_map[name]]
            parents = permutation_g.parents(parameter_map[name])
            map_prev_param_index[name] = (
                None if len(parents) == 0 else permutation_g.naming[parents[0]]
            )

            if "weight" in name[-6:]:
                if len(p.shape) == 1:  # batchnorm
                    pass  # no permutation : bn is "part" for the previous one like biais
                else:
                    if (
                        map_param_index[name] is not None
                        and perm_dict[map_param_index[name]] is not None
                    ):
                        perm_index = perm_dict[map_param_index[name]]
                        P_sizes[perm_index] = (p.shape[0], p.shape[0])

        self.reparamnet.map_param_index = map_param_index
        self.reparamnet.map_prev_param_index = map_prev_param_index
        self.reparamnet.perm_dict = perm_dict

        self.p = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.eye(ps[0], dtype=self.param_precision)
                    + torch.randn(ps, dtype=self.param_precision) * 0.1,
                    requires_grad=True,
                )
                if ps is not None
                else None
                for ps in P_sizes
            ]
        )

        self.l = l
        self.tau = tau
        self.n_iter = n_iter
        self.operator = operator
        self.scale_invariant = scale_invariant
        self.lambda_scale = lambda_scale
        self.u = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros((ps[0],), dtype=self.param_precision),
                    requires_grad=True,
                )
                if ps is not None
                else None
                for ps in P_sizes
            ]
        )

    def update_batchnorm(self, model):
        self.reparamnet.update_batchnorm(model)

    def random_init(self):
        for p in self.p:
            ci = torch.randperm(p.shape[0])
            p.data = (torch.eye(p.shape[0])[ci, :]).to(p.data.device)
        for u in self.u:
            if u is not None:
                u.data.zero_()

    def identity_init(self):
        for p in self.p:
            p.data = torch.eye(p.shape[0]).to(p.data.device)
        for u in self.u:
            if u is not None:
                u.data.zero_()

    def scale_regularizer(self):
        if not self.scale_invariant:
            return torch.tensor(0.0, dtype=self.param_precision, device=self.p[0].device)

        reg = torch.tensor(0.0, dtype=self.param_precision, device=self.p[0].device)
        for u in self.u:
            if u is not None:
                reg = reg + torch.pow(u, 2).sum()

        return self.lambda_scale * reg

    def scale_stats(self):
        if not self.scale_invariant:
            return None

        scales = torch.cat(
            [get_scale_vector(u.detach()).flatten() for u in self.u if u is not None]
        )
        inv_scales = torch.cat(
            [get_inv_scale_vector(u.detach()).flatten() for u in self.u if u is not None]
        )
        return {
            "scale_min": float(scales.min().item()),
            "scale_max": float(scales.max().item()),
            "scale_mean": float(scales.mean().item()),
            "inv_scale_min": float(inv_scales.min().item()),
            "inv_scale_max": float(inv_scales.max().item()),
            "inv_scale_mean": float(inv_scales.mean().item()),
        }

    def eval(self):
        self.reparamnet.eval()
        return super().eval()

    def train(self, mode: bool = True):
        self.reparamnet.train(mode)
        return super().train(mode)

    def forward(self, x=None):

        if self.training:
            gk = list()
            for i in range(len(self.p)):
                if self.operator == "implicit":
                    sk = Sinkhorn.apply(
                        -self.p[i] * self.l,
                        torch.ones((self.p[i].shape[0])).to(self.p[0].device),
                        torch.ones((self.p[i].shape[1])).to(self.p[0].device),
                        self.n_iter,
                        self.tau,
                    )
                if self.scale_invariant:
                    gk.append(
                        {
                            "perm": sk,
                            "scale": get_scale_vector(self.u[i]),
                            "inv_scale": get_inv_scale_vector(self.u[i]),
                        }
                    )
                else:
                    gk.append(sk)

        else:
            gk = list()
            for i, p in enumerate(self.p):
                hk = (
                    matching(p.cpu().detach().numpy())
                    .to(self.param_precision)
                    .to(self.p[0].device)
                )
                if self.scale_invariant:
                    gk.append(
                        {
                            "perm": hk,
                            "scale": get_scale_vector(self.u[i].detach()),
                            "inv_scale": get_inv_scale_vector(self.u[i].detach()),
                        }
                    )
                else:
                    gk.append(hk)

        m = self.reparamnet(gk)
        if x is not None and x.ndim == 1:
            x.unsqueeze_(0)

        if x is not None:
            return m(x)

        return m

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.reparamnet.output.zero_grad(set_to_none)
        return super().zero_grad(set_to_none)

    def parameters(self, recurse: bool = True):
        if self.scale_invariant:
            return chain(
                self.p.parameters(recurse),
                self.u.parameters(recurse),
            )
        return self.p.parameters(recurse)

    def to(self, device):
        for p in self.p:
            if p is not None:
                p.data = p.data.to(device)
        for u in self.u:
            if u is not None:
                u.data = u.data.to(device)

        return self
