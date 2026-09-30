"""Blocked grid encoding and forecasting with fused inference epilogues."""

import torch
from torch import nn
import triton
import triton.language as tl

from .utils import device_context


GRID_CHUNK_SIZE = 32768


def _can_chunk(layer, weights):
    return (
        not layer.training
        and not torch.is_grad_enabled()
        and weights.dtype == torch.float32
        and weights.device.type in ("cpu", "cuda", "xpu")
        and not torch.is_autocast_enabled(weights.device.type)
    )


@triton.jit
def _conditioning(
    values,
    film,
    output,
    input_batch_stride,
    input_point_stride,
    output_batch_stride,
    output_point_stride,
    points,
    channels: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch = row // points
    point = row % points
    cols = tl.arange(0, BLOCK)
    valid = cols < channels
    x = tl.load(
        values + batch * input_batch_stride + point * input_point_stride + cols,
        valid,
        0,
    )
    scale = tl.load(film + batch * 2 * channels + cols, valid, 0)
    offset = tl.load(film + batch * 2 * channels + channels + cols, valid, 0)
    result = x * (1.0 + scale) + offset
    tl.store(
        output + batch * output_batch_stride + point * output_point_stride + cols,
        result,
        valid,
    )


@triton.jit
def _forecast_output(
    values,
    gate,
    shift,
    output,
    input_batch_stride,
    output_batch_stride,
    output_point_stride,
    points,
    channels: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(1)
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < points * channels
    channel = indices % channels
    x = tl.load(values + batch * input_batch_stride + indices, valid, 0)
    gated = tl.load(gate + channel, valid, 0)
    offset = tl.load(shift + channel, valid, 0)
    result = tl.where(gated, 1.0 / (1.0 + tl.exp(-(x - offset))), x)
    tl.store(
        output
        + batch * output_batch_stride
        + (indices // channels) * output_point_stride
        + channel,
        result,
        valid,
    )


class WeatherNext2GridEncoder(nn.Module):
    def forward(self, grid_features, spatial_features, conditioning):
        if not _can_chunk(self, self.fc1.weight):
            return type(self).forward(
                self, grid_features, spatial_features, conditioning
            )
        batch, points, _ = grid_features.shape
        channels = self.fc2.out_features
        output = self.fc2.weight.new_empty((batch, points, channels))
        film = self.norm.linear(conditioning) if output.device.type != "cpu" else None
        for start in range(0, points, GRID_CHUNK_SIZE):
            end = min(start + GRID_CHUNK_SIZE, points)
            spatial = spatial_features[start:end].unsqueeze(0).expand(batch, -1, -1)
            inputs = torch.cat(
                [
                    spatial.to(output.dtype),
                    grid_features[:, start:end].to(output.dtype),
                ],
                dim=-1,
            )
            values = self.fc2(self.activation_fn(self.fc1(inputs)))
            target = output[:, start:end]
            if output.device.type == "cpu":
                target.copy_(self.norm(values, conditioning))
            else:
                # Keep the original LayerNorm reduction: small rounding changes
                # here can be amplified by the downstream mesh transformer.
                values = self.norm.norm(values)
                with device_context(output.device):
                    _conditioning[(batch * (end - start),)](
                        values,
                        film,
                        target,
                        values.stride(0),
                        values.stride(1),
                        target.stride(0),
                        target.stride(1),
                        end - start,
                        channels,
                        triton.next_power_of_2(channels),
                        enable_fp_fusion=False,
                    )
        return output


class WeatherNext2ForecastHead(nn.Module):
    def forward(self, grid_states):
        if (
            not _can_chunk(self, self.decoder_proj.weight)
            or grid_states.dtype != torch.float32
        ):
            return type(self).forward(self, grid_states)
        batch, points, _ = grid_states.shape
        channels = self.output_proj.out_features
        output = grid_states.new_empty((batch, points, channels))
        for start in range(0, points, GRID_CHUNK_SIZE):
            end = min(start + GRID_CHUNK_SIZE, points)
            values = self.output_proj(
                self.act_fn(self.decoder_proj(grid_states[:, start:end]))
            )
            target = output[:, start:end]
            if output.device.type == "cpu":
                target.copy_(
                    torch.where(
                        self.sigmoid_gate,
                        torch.sigmoid(values - self.sigmoid_shift),
                        values,
                    )
                )
            else:
                with device_context(output.device):
                    _forecast_output[
                        (triton.cdiv((end - start) * channels, 1024), batch)
                    ](
                        values,
                        self.sigmoid_gate,
                        self.sigmoid_shift,
                        target,
                        values.stride(0),
                        target.stride(0),
                        target.stride(1),
                        end - start,
                        channels,
                        1024,
                    )
        return output.transpose(1, 2).reshape(
            batch, channels, self.config.grid_latitudes, self.config.grid_longitudes
        )
