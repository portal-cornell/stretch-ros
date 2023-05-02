import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR

from .misc_utils import *
from .img_js_net import *
from .buffer import *


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def load_iql_trainer(device):
    img_comp_dim = 64
    img_js_net = ImageJointStateNet(img_comp_dim=img_comp_dim)

    # constants
    state_dim = img_js_net.output_dim
    action_dim = KEYBOARD_ACTIONS
    n_hidden = 3
    max_action = 1

    # hyperparameters (will be overwritten by ckpt)
    discount = 0.99
    tau = 0.005
    beta = 3.0
    iql_tau = 0.7
    max_timesteps = 3000
    finetune_resnet = False

    q_network = TwinQ(state_dim, action_dim, n_hidden=n_hidden).to(device)
    v_network = ValueFunction(state_dim, n_hidden=n_hidden).to(device)
    actor = GaussianPolicy(state_dim, action_dim, max_action).to(device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "action_dim": action_dim,
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": discount,
        "tau": tau,
        "device": device,
        "beta": beta,
        "iql_tau": iql_tau,
        "max_steps": max_timesteps,
        "finetune_resnet": finetune_resnet,
    }

    for network in [img_js_net, q_network, v_network, actor]:
        network.eval()

    print("---------------------------------------")
    print(f"Loading IQL")
    print("---------------------------------------")

    return ImplicitQLearning(img_js_net, **kwargs)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     # nn.Dropout(p=0.5),
        #     nn.Linear(256, 256),
        #     nn.BatchNorm1d(256),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(256, act_dim),
        # )
        self.activation = nn.Tanh()
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> MultivariateNormal:
        if not deterministic:
            # mean = self.activation(self.net(obs))
            mean = self.net(obs)
            std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
            scale_tril = torch.diag(std)
            return MultivariateNormal(mean, scale_tril=scale_tril)
        else:
            # return self.activation(self.net(obs))
            return self.net(obs)

    @torch.no_grad()
    def act(
        self, state: torch.Tensor, deterministic: bool = False, device: str = "cpu"
    ):
        if not deterministic:
            dist = self(state)
            action = dist.mean if not self.training else dist.sample()
            action = torch.clamp(
                self.max_action * action, -self.max_action, self.max_action
            )
            return action
        else:
            return torch.clamp(
                self(state, deterministic) * self.max_action,
                -self.max_action,
                self.max_action,
            )
            # return self(state, deterministic)


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = True, device: str = "cpu"):
        return torch.clamp(
            self(state) * self.max_action, -self.max_action, self.max_action
        )


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
        self,
        img_js_net: nn.Module,
        action_dim: int,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,
        beta: float = 3.0,
        max_steps: int = 1000000,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
        finetune_resnet: bool = True,
    ):
        self.img_js_net = img_js_net
        self.action_dim = action_dim
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).eval().to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device
        self.finetune_resnet = finetune_resnet
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=action_dim, top_k=1
        ).to(device)
        self.CELoss = nn.CrossEntropyLoss()
        # if not finetune_resnet:
        #     print("==> Freezing Resnet")
        #     for name, param in self.img_js_net.named_parameters():
        #         param.requires_grad = False

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        log_dict["avg_v"] = torch.mean(v).item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v,
        observations,
        actions,
        rewards,
        terminals,
        log_dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        log_dict["avg_q"] = torch.mean(sum(q for q in qs)).item()
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph=True)
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(self, adv, observations, actions, log_dict):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        actions = torch.argmax(actions, dim=1).squeeze(0)
        policy_accuracy = self.accuracy(policy_out.mean, actions)
        log_dict["training_acc"] = policy_accuracy.item()
        ce_loss = self.CELoss(policy_out.mean, actions)
        log_dict["ce_loss"] = torch.mean(ce_loss).item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            images,
            joint_states,
            actions,
            rewards,
            next_images,
            next_joint_states,
            dones,
        ) = batch.values()
        log_dict = {}

        images, joint_states = images.to(self.device), joint_states.to(self.device)
        next_images, next_joint_states = next_images.to(
            self.device
        ), next_joint_states.to(self.device)

        observations = self.img_js_net(images, joint_states)
        next_observations = self.img_js_net(next_images, next_joint_states)
        actions = F.one_hot(actions.squeeze().long(), num_classes=self.action_dim)

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "img_js_net": self.img_js_net.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.img_js_net.load_state_dict(state_dict["img_js_net"])

        self.total_it = state_dict["total_it"]
