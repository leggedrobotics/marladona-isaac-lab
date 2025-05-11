import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborNet(nn.Module):
    def __init__(self, neighbor_state_dim, expand_dim, max_num_teammate, max_num_opponent, global_encoding_dim) -> None:
        super().__init__()
        self.pose_dim = 4
        self.max_num_teammate = max_num_teammate
        self.max_num_opponent = max_num_opponent
        self.neighbor_state_dim = neighbor_state_dim
        # self.global_dim = global_encoding_dim
        self.global_dim = self.neighbor_state_dim
        self.global_encoding_dim = global_encoding_dim
        self.teammate_encoder = nn.Sequential(
            nn.Linear(neighbor_state_dim + expand_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, global_encoding_dim),
        )
        self.opponent_encoder = nn.Sequential(
            nn.Linear(neighbor_state_dim + expand_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, global_encoding_dim),
        )
        # self.teammate_encoder = nn.Identity()
        # self.opponent_encoder = nn.Identity()

    def forward(self, ego_states, neighbor_states):
        device = ego_states.device
        expanded_states = torch.cat(
            [
                neighbor_states.view(-1, self.max_num_teammate + self.max_num_opponent, self.neighbor_state_dim),
                ego_states.unsqueeze(1).expand(-1, self.max_num_teammate + self.max_num_opponent, -1),
            ],
            dim=-1,
        )
        active_agents = ~torch.isnan(expanded_states).any(dim=-1)
        active_teammates = active_agents[:, : self.max_num_teammate].reshape(-1)
        active_opponents = active_agents[:, self.max_num_teammate :].reshape(-1)
        teammate_feature = torch.empty(
            (expanded_states.shape[0] * self.max_num_teammate, self.global_encoding_dim),
            device=device,
        ).fill_(-float("inf"))
        opponent_feature = torch.empty(
            (expanded_states.shape[0] * self.max_num_opponent, self.global_encoding_dim),
            device=device,
        ).fill_(-float("inf"))

        flatten_teammate_states = expanded_states[:, : self.max_num_teammate, :].reshape(-1, expanded_states.size()[-1])
        flatten_opponent_states = expanded_states[:, self.max_num_teammate :, :].reshape(-1, expanded_states.size()[-1])

        teammate_feature[active_teammates] = self.teammate_encoder(flatten_teammate_states[active_teammates])
        opponent_feature[active_opponents] = self.opponent_encoder(flatten_opponent_states[active_opponents])

        teammate_global_encoding = teammate_feature.view(-1, self.max_num_teammate, self.global_encoding_dim).max(
            dim=1
        )[0]

        teammate_global_encoding[teammate_global_encoding.isinf()] = -2.0

        # needed for onnx export
        # torch.full(
        #     teammate_global_encoding[teammate_global_encoding.isinf()].shape, -2.0, device=device
        # )

        opponent_global_encoding = opponent_feature.view(-1, self.max_num_opponent, self.global_encoding_dim).max(
            dim=1
        )[0]

        return torch.cat([teammate_global_encoding, opponent_global_encoding], dim=1)
