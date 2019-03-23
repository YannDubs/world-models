""" Define controller """
import torch
import torch.nn as nn


class Controller(nn.Module):
    """ Controller """

    def __init__(self, latents, recurrents, actions,
                 is_gate=False,
                 is_dynamic_gate=False):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)
        self.is_gate = is_gate
        self.is_dynamic_gate = is_dynamic_gate

        if self.is_dynamic_gate:
            self.ff_gater = nn.Linear(latents, actions)
        elif self.is_gate:
            self.gates = nn.Parameter(torch.tensor([0.5, 0.5, 0.5]))

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        actions = self.fc(cat_in)

        if self.is_dynamic_gate or self.is_gate:
            if self.is_dynamic_gate:
                gates = torch.sigmoid(self.ff_gater(inputs[0]))
            else:
                gates = self.gates

            try:
                self.old_actions = (1 - gates) * self.old_actions + gates * actions
                actions = self.old_actions
            except AttributeError:
                # first input
                self.old_actions = actions

        # only works for car racing!
        actions = actions.unbind(dim=1)
        actions = torch.stack((torch.tanh(actions[0]),  # steer
                               torch.sigmoid(actions[1]),  # accelerate
                               torch.clamp(torch.tanh(actions[2]), min=0, max=1)),  # brake
                              dim=1)

        return actions
