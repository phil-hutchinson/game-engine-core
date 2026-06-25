import torch
import torch.nn as nn


class TicTacToeMLP(nn.Module):
    """Small MLP for TicTacToe: shared trunk with value and policy heads.

    Input:  9 floats (board squares from active player's perspective)
    Output: (value: Tensor[1], policy_logits: Tensor[9])
    """

    def __init__(self):
        super().__init__()
        # define model architecture, starting with common backbone
        self._backbone = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # define heads (different outputs) that use the results of this backbone
        self._value_head = nn.Linear(64, 1)
        self._policy_head = nn.Linear(64, 9)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # run input tensor through main model
        features = self._backbone(x)
        # calculate strength of current position (from current player's perspective)
        value = torch.tanh(self._value_head(features))
        # calculate strength of each possible move (from current player's perspective)
        policy_logits = self._policy_head(features)
        return value, policy_logits
