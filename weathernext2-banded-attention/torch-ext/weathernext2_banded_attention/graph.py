"""Block bipartite graph execution without materializing all edge messages."""

import torch
from torch import nn


GRID_CHUNK_SIZE = 32768
EDGE_CHUNK_SIZE = 65536


def _grid_to_mesh(
    layer, grid_states, mesh_states, edge_features, senders, receivers, conditioning
):
    update = layer.edge_update
    projected_grid = update.sender_proj(grid_states)
    aggregated = torch.zeros_like(mesh_states, dtype=torch.float32)
    for start in range(0, senders.numel(), EDGE_CHUNK_SIZE):
        edge_slice = slice(start, start + EDGE_CHUNK_SIZE)
        encoded_edges = layer.edge_encoder(edge_features[:, edge_slice], conditioning)
        messages = (
            update.edge_proj(encoded_edges) + projected_grid[:, senders[edge_slice]]
        )
        if update.receiver_proj is not None:
            messages = (
                messages + update.receiver_proj(mesh_states)[:, receivers[edge_slice]]
            )
        messages = update.norm(update.out_proj(update.act_fn(messages)), conditioning)
        aggregated.index_add_(1, receivers[edge_slice], messages.float())
    del projected_grid
    if layer.aggregate_normalization is not None:
        aggregated = aggregated / layer.aggregate_normalization
    inputs = torch.cat([mesh_states, aggregated], dim=-1)
    mesh_states = mesh_states + layer.mesh_node_update(inputs, conditioning)
    grid_states = grid_states + layer.grid_node_update(grid_states, conditioning)
    return grid_states, mesh_states


class WeatherNext2BipartiteGraphNetwork(nn.Module):
    def forward(
        self,
        grid_states: torch.Tensor,
        mesh_states: torch.Tensor,
        edge_features: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_points = grid_states.shape[1]
        eligible = (
            not self.training
            and not torch.is_grad_enabled()
            and grid_states.dtype == torch.float32
            and not torch.is_autocast_enabled(grid_states.device.type)
        )
        # The decoder requires three consecutive edges per grid point; the encoder
        # accepts arbitrary connectivity and accumulates into the full mesh.
        if not self.grid_to_mesh:
            eligible = (
                eligible
                and senders.numel() == 3 * num_points
                and torch.equal(
                    receivers,
                    torch.arange(
                        num_points, device=receivers.device, dtype=receivers.dtype
                    ).repeat_interleave(3),
                )
            )
        if not eligible:
            return type(self).forward(
                self,
                grid_states,
                mesh_states,
                edge_features,
                senders,
                receivers,
                conditioning,
            )

        if self.grid_to_mesh:
            return _grid_to_mesh(
                self,
                grid_states,
                mesh_states,
                edge_features,
                senders,
                receivers,
                conditioning,
            )

        update = self.edge_update
        projected_mesh = update.sender_proj(mesh_states)
        output = torch.empty_like(grid_states)
        for start in range(0, num_points, GRID_CHUNK_SIZE):
            end = min(start + GRID_CHUNK_SIZE, num_points)
            grid_block = grid_states[:, start:end]
            edge_slice = slice(3 * start, 3 * end)
            local_receivers = receivers[edge_slice] - start
            encoded_edges = self.edge_encoder(
                edge_features[:, edge_slice], conditioning
            )
            messages = (
                update.edge_proj(encoded_edges) + projected_mesh[:, senders[edge_slice]]
            )
            if update.receiver_proj is not None:
                messages = (
                    messages + update.receiver_proj(grid_block)[:, local_receivers]
                )
            messages = update.norm(
                update.out_proj(update.act_fn(messages)), conditioning
            )
            aggregated = torch.zeros(
                grid_block.shape, device=messages.device, dtype=torch.float32
            )
            aggregated.index_add_(1, local_receivers, messages.float())
            if self.aggregate_normalization is not None:
                aggregated = aggregated / self.aggregate_normalization
            inputs = torch.cat([grid_block, aggregated.to(messages.dtype)], dim=-1)
            output[:, start:end] = grid_block + self.grid_node_update(
                inputs, conditioning
            )

        mesh_states = mesh_states + self.mesh_node_update(mesh_states, conditioning)
        return output, mesh_states
