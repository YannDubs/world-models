""" Define controller """
import torch
import torch.nn as nn


class Controller(nn.Module):
    """ Controller """

    def __init__(self, latents, recurrents, actions, is_gate=False):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)
        self.is_gate = is_gate
        """
        if self.is_gate:
            self.ff_gater = nn.Linear(latents, 1)
        """

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        actions = self.fc(cat_in)

        if self.is_gate:
            """
            gate = torch.sigmoid(self.ff_gater)
            """
            gate = .8
            try:
                self.old_actions = (1 - gate) * self.old_actions + gate * actions
                actions = self.old_actions
            except AttributeError:
                # first input
                self.old_actions = actions

        return actions
