import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F


class A2C(object):
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(
                actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.super_optimizer = torch.optim.SGD(actor_critic.parameters(), lr=lr, momentum=0.9)

    def update(self, rollouts):
        """
        update model using a2c loss
        """
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # in torch .view() is used to reshape the tensor (batch, channels, height, width)
        values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.states[0].view(-1, self.actor_critic.state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # Bootstrapped n-step value
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef).backward()

        # clip the gradients (max 0.5)
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    def supervised_updates(self, action, values, real_actions, returns):
        """
        update model with supervised learning using imitation loss
        :param: actions: actions taken by the naive agent (level x)
        :param: values: value estimation for the naive agent
        :param real_actions: taken by the agent seen translations
        :param returns: discounter sum of rewards
        """

        # mean square error for the value
        value_loss = F.mse_loss(values, returns)
        # cross entropy between the actions taken by the new agent (level x) and the trained agent in level 1.
        # the real actions are based on the translations of UNIT and the behavior of the agent
        policy_loss = F.binary_cross_entropy(action, real_actions).mean()

        # for the supervised updates SGD is the optimizer
        self.super_optimizer.zero_grad()

        # in the paper the imitation loss for the policy is multiplied by 0.5
        (value_loss * self.value_loss_coef + policy_loss).backward()

        # clip the gradients (max 0.5)
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                 self.max_grad_norm)
        self.super_optimizer.step()

        return value_loss.item(), policy_loss.item()