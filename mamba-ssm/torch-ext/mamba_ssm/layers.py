import torch
import torch.nn as nn

from ._causal_conv1d import causal_conv1d_fn as cuda_causal_conv1d_fn
from ._causal_conv1d import causal_conv1d_update as cuda_causal_conv1d_update
from .ops import mamba_chunk_scan_combined as cuda_mamba_chunk_scan_combined
from .ops import mamba_split_conv1d_scan_combined as cuda_mamba_split_conv1d_scan_combined
from .ops import selective_state_update as cuda_selective_state_update
from .ops.selective_scan_interface import mamba_inner_fn as cuda_mamba_inner_fn
from .ops.selective_scan_interface import selective_scan_fn as cuda_selective_scan_fn


class mamba_inner_fn(nn.Module):
    def forward(
        self,
        xz: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: torch.Tensor | None,
        x_proj_weight: torch.Tensor,
        delta_proj_weight: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor | None,
        A: torch.Tensor,
        B: torch.Tensor | None = None,
        C: torch.Tensor | None = None,
        D: torch.Tensor | None = None,
        delta_bias: torch.Tensor | None = None,
        delta_softplus: bool = True,
        b_rms_weight: torch.Tensor | None = None,
        c_rms_weight: torch.Tensor | None = None,
        dt_rms_weight: torch.Tensor | None = None,
        b_c_dt_rms_eps: float = 1e-6,
        **kwargs,
    ):
        return cuda_mamba_inner_fn(
            xz,
            conv1d_weight,
            conv1d_bias,
            x_proj_weight,
            delta_proj_weight,
            out_proj_weight,
            out_proj_bias,
            A,
            B,
            C,
            D,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            b_rms_weight=b_rms_weight,
            c_rms_weight=c_rms_weight,
            dt_rms_weight=dt_rms_weight,
            b_c_dt_rms_eps=b_c_dt_rms_eps,
        )


class mamba_split_conv1d_scan_combined(nn.Module):
    def forward(
        self,
        zxbcdt: torch.Tensor,
        conv1d_weight: torch.Tensor,
        conv1d_bias: torch.Tensor | None,
        dt_bias: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        chunk_size: int,
        initial_states: torch.Tensor | None = None,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        return_final_states: bool = False,
        activation: str = "silu",
        rmsnorm_weight: torch.Tensor | None = None,
        rmsnorm_eps: float = 1e-6,
        outproj_weight: torch.Tensor | None = None,
        outproj_bias: torch.Tensor | None = None,
        headdim: int | None = None,
        ngroups: int = 1,
        norm_before_gate: bool = True,
        **kwargs,
    ):
        # For varlen
        seq_idx = kwargs.pop("seq_idx", None)

        return cuda_mamba_split_conv1d_scan_combined(
            zxbcdt,
            conv1d_weight,
            conv1d_bias,
            dt_bias,
            A,
            D=D,
            chunk_size=chunk_size,
            seq_idx=seq_idx,
            activation=activation,
            rmsnorm_weight=rmsnorm_weight,
            rmsnorm_eps=rmsnorm_eps,
            outproj_weight=outproj_weight,
            outproj_bias=outproj_bias,
            headdim=headdim,
            ngroups=ngroups,
            norm_before_gate=norm_before_gate,
            return_final_states=return_final_states,
            dt_limit=dt_limit,
            initial_states=initial_states,
        )


class mamba_chunk_scan_combined(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        chunk_size: int,
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        initial_states: torch.Tensor | None = None,
        dt_softplus: bool = False,
        dt_limit: tuple[float, float] = (0.0, float("inf")),
        return_final_states: bool = False,
        **kwargs,
    ):
        # For varlen
        seq_idx = kwargs.pop("seq_idx", None)

        return cuda_mamba_chunk_scan_combined(
            hidden_states,
            dt,
            A,
            B,
            C,
            D=D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            chunk_size=chunk_size,
            seq_idx=seq_idx,
            return_final_states=return_final_states,
            dt_limit=dt_limit,
            initial_states=initial_states,
        )


class selective_state_update(nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        hidden_states: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        z: torch.Tensor | None = None,
        **kwargs,
    ):
        return cuda_selective_state_update(
            state,
            hidden_states,
            dt,
            A,
            B,
            C,
            D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
        )


class selective_scan_fn(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        delta_bias: torch.Tensor | None = None,
        delta_softplus: bool = False,
        return_last_state: bool = False,
        # Unused here but fallbacks for torch only paths
        use_mambapy: bool = False,
        use_associative_scan: bool = False,
        **kwargs,
    ):
        return cuda_selective_scan_fn(
            hidden_states,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )


class causal_conv1d_fn(nn.Module):
    def forward(
        self,
        hidden_states,
        weight,
        bias=None,
        activation=None,
        **kwargs,
    ):
        # For varlen
        seq_idx = kwargs.pop("seq_idx", None)

        return cuda_causal_conv1d_fn(
            x=hidden_states,
            weight=weight,
            bias=bias,
            activation=activation,
            seq_idx=seq_idx,
        )


class causal_conv1d_update(nn.Module):
    def forward(
        self,
        hidden_states,
        conv_state,
        weight,
        bias=None,
        activation=None,
    ):
        return cuda_causal_conv1d_update(
            x=hidden_states,
            conv_state=conv_state,
            weight=weight,
            bias=bias,
            activation=activation,
        )


__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "mamba_inner_fn",
    "mamba_split_conv1d_scan_combined",
    "mamba_chunk_scan_combined",
    "selective_state_update",
    "selective_scan_fn",
]
