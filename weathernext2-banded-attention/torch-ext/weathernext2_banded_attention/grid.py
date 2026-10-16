"""Tiled grid inference, following Faster-WeatherNext's blocked graph and decoder paths."""

import torch
import triton
import triton.language as tl
from torch import nn

from .utils import device_context


def _can_tile(layer, states):
    return not layer.training and not torch.is_grad_enabled() and not torch.is_autocast_enabled(states.device.type)


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
    batch, point = row // points, row % points
    cols = tl.arange(0, BLOCK)
    valid = cols < channels
    x = tl.load(values + batch * input_batch_stride + point * input_point_stride + cols, valid, 0)
    scale = tl.load(film + batch * 2 * channels + cols, valid, 0)
    offset = tl.load(film + batch * 2 * channels + channels + cols, valid, 0)
    tl.store(
        output + batch * output_batch_stride + point * output_point_stride + cols,
        x * (1.0 + scale) + offset,
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
    gated = tl.load(gate + channel, valid, 0).to(tl.int1)
    offset = tl.load(shift + channel, valid, 0)
    result = tl.where(gated, 1.0 / (1.0 + tl.exp(-(x - offset))), x)
    tl.store(
        output + batch * output_batch_stride + (indices // channels) * output_point_stride + channel,
        result,
        valid,
    )


class WeatherNext2ConditionedMlp(nn.Module):
    can_torch_compile = True

    def forward(self, hidden_states, conditioning, spatial_features=None):
        if not _can_tile(self, hidden_states) or self.chunk_size is None:
            return type(self).forward(self, hidden_states, conditioning, spatial_features=spatial_features)
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(1).expand(-1, spatial_features.shape[0], -1)
        batch, points, _ = hidden_states.shape
        channels = self.fc2.out_features
        output = hidden_states.new_empty((batch, points, channels), dtype=self.fc2.weight.dtype)
        fused = not output.is_cpu and output.dtype == torch.float32
        film = self.norm.linear(conditioning) if fused else None
        for start in range(0, points, self.chunk_size):
            end = min(start + self.chunk_size, points)
            states = hidden_states[:, start:end].to(dtype=self.fc1.weight.dtype)
            if spatial_features is not None:
                spatial = spatial_features[start:end].to(states).unsqueeze(0).expand(batch, -1, -1)
                states = torch.cat([spatial, states], dim=-1)
            values = self.fc2(self.activation_fn(self.fc1(states)))
            target = output[:, start:end]
            if not fused:
                target.copy_(self.norm(values, conditioning))
            else:
                # Keep LayerNorm's reduction and the reference's separate multiply/add rounding.
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


class WeatherNext2GridToMesh(nn.Module):
    can_torch_compile = True

    def forward(self, grid_states, mesh_states, edge_features, senders, receivers, conditioning):
        if not _can_tile(self, grid_states) or self.config.chunk_size is None:
            return type(self).forward(self, grid_states, mesh_states, edge_features, senders, receivers, conditioning)
        projected_senders = None
        aggregated = torch.zeros(mesh_states.shape, dtype=torch.float32, device=mesh_states.device)
        for start in range(0, senders.numel(), self.config.chunk_size):
            edges = slice(start, start + self.config.chunk_size)
            features = edge_features[edges].unsqueeze(0).expand(grid_states.shape[0], -1, -1)
            features = features.to(self.edge_encoder.fc1.weight.dtype)
            messages, projected_senders = self.edge_update(
                self.edge_encoder(features, conditioning),
                grid_states,
                senders[edges],
                conditioning,
                projected_senders=projected_senders,
            )
            aggregated.index_add_(1, receivers[edges], messages.float())
        del projected_senders
        if self.aggregate_normalization is not None:
            aggregated = aggregated / self.aggregate_normalization
        mesh_states = mesh_states + self.mesh_node_update(
            torch.cat([mesh_states, aggregated.to(mesh_states.dtype)], dim=-1), conditioning
        )
        grid_states.add_(self.grid_node_update(grid_states, conditioning))
        return grid_states, mesh_states


class WeatherNext2MeshToGrid(nn.Module):
    can_torch_compile = True

    def forward(self, grid_states, mesh_states, edge_features, senders, receivers, conditioning):
        if not _can_tile(self, grid_states) or self.config.chunk_size is None:
            return type(self).forward(self, grid_states, mesh_states, edge_features, senders, receivers, conditioning)
        points = grid_states.shape[1]
        if senders.numel() != 3 * points:
            raise ValueError("The mesh-to-grid graph needs exactly three edges per grid point.")
        projected_senders = None
        for start in range(0, points, self.config.chunk_size):
            block = grid_states[:, start : start + self.config.chunk_size]
            edges = slice(3 * start, 3 * (start + block.shape[1]))
            local_receivers = receivers[edges] - start
            features = edge_features[edges].unsqueeze(0).expand(grid_states.shape[0], -1, -1)
            features = features.to(self.edge_encoder.fc1.weight.dtype)
            messages, projected_senders = self.edge_update(
                self.edge_encoder(features, conditioning),
                mesh_states,
                block,
                senders[edges],
                local_receivers,
                conditioning,
                projected_senders=projected_senders,
            )
            aggregated = torch.zeros(block.shape, dtype=torch.float32, device=block.device)
            aggregated.index_add_(1, local_receivers, messages.float())
            block.add_(self.grid_node_update(torch.cat([block, aggregated.to(block.dtype)], dim=-1), conditioning))
        mesh_states = mesh_states + self.mesh_node_update(mesh_states, conditioning)
        return grid_states, mesh_states


class WeatherNext2ForecastHead(nn.Module):
    can_torch_compile = True

    def forward(self, grid_states):
        if not _can_tile(self, grid_states) or self.config.chunk_size is None:
            return type(self).forward(self, grid_states)
        batch, points, _ = grid_states.shape
        channels = self.fc2.out_features
        output = grid_states.new_empty((batch, points, channels))
        fused = not output.is_cpu and output.dtype == torch.float32
        for start in range(0, points, self.config.chunk_size):
            end = min(start + self.config.chunk_size, points)
            values = self.fc2(self.activation_fn(self.fc1(grid_states[:, start:end])))
            target = output[:, start:end]
            if not fused:
                target.copy_(torch.where(self.sigmoid_gate, torch.sigmoid(values - self.sigmoid_shift), values))
            else:
                with device_context(output.device):
                    _forecast_output[(triton.cdiv((end - start) * channels, 1024), batch)](
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
        return output.transpose(1, 2).reshape(batch, channels, self.config.grid_latitudes, self.config.grid_longitudes)
