#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from .network_utils import *
from .network_bodies import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.config import Config

class VanillaNet(nn.Module, BaseNet):
    def __init__(self, output_dim, body):
        super(VanillaNet, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class VanillaNet_CL(nn.Module, BaseNet):
    def __init__(self, output_dim, task_label_dim, body):
        super(VanillaNet_CL, self).__init__()
        self.fc_head = layer_init(nn.Linear(body.feature_dim, output_dim))
        self.body = body
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, x, task_label=None, to_numpy=False):
        x = tensor(x)
        task_label = tensor(task_label)
        phi = self.body(x, task_label)
        y = self.fc_head(phi)
        if to_numpy:
            y = y.cpu().detach().numpy()
        return y

class DuelingNet(nn.Module, BaseNet):
    def __init__(self, action_dim, body):
        super(DuelingNet, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class DuelingNet_CL(nn.Module, BaseNet):
    def __init__(self, action_dim, task_label_dim, body):
        super(DuelingNet_CL, self).__init__()
        self.fc_value = layer_init(nn.Linear(body.feature_dim, 1))
        self.fc_advantage = layer_init(nn.Linear(body.feature_dim, action_dim))
        self.body = body
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, x, task_label=None, to_numpy=False):
        x = tensor(x)
        task_label = tensor(task_label)
        phi = self.body(x, task_label)
        value = self.fc_value(phi)
        advantange = self.fc_advantage(phi)
        q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        if to_numpy:
            return q.cpu().detach().numpy()
        return q

class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        if to_numpy:
            return prob.cpu().detach().numpy()
        return prob

class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x, to_numpy=False):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        if to_numpy:
            quantiles = quantiles.cpu().detach().numpy()
        return quantiles

class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def predict(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())

class ActorCriticNetSS(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, \
        new_task_mask, discrete_mask=True):
        super(ActorCriticNetSS, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = MultitaskMaskLinear(actor_body.feature_dim, action_dim, \
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.fc_critic = MultitaskMaskLinear(critic_body.feature_dim, 1, \
            discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)

        ap = [p for p in self.actor_body.parameters() if p.requires_grad is True]
        ap += [p for p in self.fc_action.parameters() if p.requires_grad is True]
        self.actor_params = ap

        cp = [p for p in self.critic_body.parameters() if p.requires_grad is True]
        cp += [p for p in self.fc_critic.parameters() if p.requires_grad is True]
        self.critic_params = cp

        self.phi_params = [p for p in self.phi_body.parameters() if p.requires_grad is True]

class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def predict(self, obs, to_numpy=False):
        phi = self.feature(obs)
        action = self.actor(phi)
        if to_numpy:
            return action.cpu().detach().numpy()
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))

class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.ones(1, action_dim))
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, to_numpy=False):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        mean = F.tanh(self.network.fc_action(phi_a))
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, self.std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        return action, log_prob, tensor(np.zeros((log_prob.size(0), 1))), v

# actor-critic net for continual learning where tasks are labelled using
# supermask superposition algorithm
class GaussianActorCriticNet_SS(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931 #-20.
    LOG_STD_MAX = 0.4055 #1.3
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'):
        super(GaussianActorCriticNet_SS, self).__init__()
        # continuous values mask is used for Gaussian (continuous control policies)
        discrete_mask = False
        self.network = ActorCriticNetSS(state_dim, action_dim, phi_body, actor_body, critic_body, \
            num_tasks, new_task_mask, discrete_mask=discrete_mask)
        self.task_label_dim = task_label_dim

        self.network.fc_log_std = MultitaskMaskLinear(self.network.actor_body.feature_dim, \
            action_dim, discrete=discrete_mask, num_tasks=num_tasks, new_mask_type=new_task_mask)
        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        #mean = F.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        log_std = self.network.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_SS.LOG_STD_MIN, \
            GaussianActorCriticNet_SS.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), \
                ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output

# actor-critic net for continual learning where tasks are labelled
class GaussianActorCriticNet_CL(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931 #-20.
    LOG_STD_MAX = 0.4055 #1.3
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim

        self.network.fc_log_std = layer_init(nn.Linear(self.network.actor_body.feature_dim, \
            action_dim), 1e-3)
        self.network.actor_params += [p for p in self.network.fc_log_std.parameters() if p.requires_grad is True]
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out
        #mean = F.tanh(self.network.fc_action(phi_a))
        mean = self.network.fc_action(phi_a)
        if to_numpy:
            return mean.cpu().detach().numpy()
        v = self.network.fc_critic(phi_v)
        log_std = self.network.fc_log_std(phi_a)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_CL.LOG_STD_MIN, \
            GaussianActorCriticNet_CL.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), \
                ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = entropy.sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output

class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.to(Config.DEVICE)

    def predict(self, obs, action=None):
        obs = tensor(obs)
        phi = self.network.phi_body(obs)
        phi_a = self.network.actor_body(phi)
        phi_v = self.network.critic_body(phi)
        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1), v

# actor-critic net for continual learning where tasks are labelled using
# supermask superposition algorithm
class CategoricalActorCriticNet_SS(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 num_tasks=3,
                 new_task_mask='random'):
        super(CategoricalActorCriticNet_SS, self).__init__()
        self.network = ActorCriticNetSS(state_dim, action_dim, phi_body, actor_body, critic_body, num_tasks, new_task_mask)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output

# actor-critic net for continual learning where tasks are labelled
class CategoricalActorCriticNet_CL(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(CategoricalActorCriticNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim
        self.to(Config.DEVICE)

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        logits = self.network.fc_action(phi_a)
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output


# ------------------------
# HyperNetwork-based heads
# ------------------------

class _TaskEncoder(nn.Module):
    def __init__(self, in_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class _HyperLinear(nn.Module):
    def __init__(self, z_dim, in_features, out_features, hidden=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net_W = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features * in_features)
        )
        self.net_b = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

        # Initialize the last layer of generators to be very small.
        # This ensures the generated W starts close to 0 (or small random noise),
        # preventing massive gradients at step 0.
        #self.net_W[-1].weight.data.normal_(0, 0.001)
        #self.net_W[-1].bias.data.normal_(0, 0.001)
        #self.net_b[-1].weight.data.normal_(0, 0.001)
        #self.net_b[-1].bias.data.zero_()

    def forward(self, z):
        W = self.net_W(z).view(-1, self.out_features, self.in_features)
        b = self.net_b(z).view(-1, self.out_features)
        return W, b

class CategoricalActorCriticNet_HNet_CL(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 embed_dim=64,
                 hyper_hidden=128):
        super(CategoricalActorCriticNet_HNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim
        self.encoder = _TaskEncoder(task_label_dim, embed_dim) if task_label_dim and task_label_dim > 0 else None
        self.h_action = _HyperLinear(embed_dim, self.network.actor_body.feature_dim, action_dim, hidden=hyper_hidden)
        self.h_critic = _HyperLinear(embed_dim, self.network.critic_body.feature_dim, 1, hidden=hyper_hidden)
        self.to(Config.DEVICE)

    def _lin_apply(self, phi, W, b):
        # phi: [B, F], W: [B, O, F], b: [B, O]
        return torch.bmm(W, phi.unsqueeze(-1)).squeeze(-1) + b

    def generate_heads(self, task_label, detach=True): # Add detach flag
        if not isinstance(task_label, torch.Tensor):
            task_label = torch.tensor(task_label).to(Config.DEVICE)
            
        z = self.encoder(task_label)
        Wa, ba = self.h_action(z)
        Wv, bv = self.h_critic(z)
        
        heads = {'Wa': Wa, 'ba': ba, 'Wv': Wv, 'bv': bv}
        
        if detach:
            return {k: v.detach() for k, v in heads.items()}
        return heads

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        z = self.encoder(task_label)
        Wa, ba = self.h_action(z)
        Wv, bv = self.h_critic(z)
        # broadcast parameters if single z for batch
        if Wa.size(0) == 1 and phi_a.size(0) > 1:
            Wa = Wa.expand(phi_a.size(0), -1, -1)
            ba = ba.expand(phi_a.size(0), -1)
            Wv = Wv.expand(phi_v.size(0), -1, -1)
            bv = bv.expand(phi_v.size(0), -1)

        logits = self._lin_apply(phi_a, Wa, ba)
        v = self._lin_apply(phi_v, Wv, bv)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output

class GaussianActorCriticNet_HNet_CL(nn.Module, BaseNet):
    LOG_STD_MIN = -0.6931
    LOG_STD_MAX = 0.4055
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None,
                 embed_dim=64,
                 hyper_hidden=128):
        super(GaussianActorCriticNet_HNet_CL, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.task_label_dim = task_label_dim
        self.encoder = _TaskEncoder(task_label_dim, embed_dim) if task_label_dim and task_label_dim > 0 else None
        self.h_action = _HyperLinear(embed_dim, self.network.actor_body.feature_dim, action_dim, hidden=hyper_hidden)
        self.h_logstd = _HyperLinear(embed_dim, self.network.actor_body.feature_dim, action_dim, hidden=hyper_hidden)
        self.h_critic = _HyperLinear(embed_dim, self.network.critic_body.feature_dim, 1, hidden=hyper_hidden)
        self.to(Config.DEVICE)

    def _lin_apply(self, phi, W, b):
        return torch.bmm(W, phi.unsqueeze(-1)).squeeze(-1) + b

    def generate_heads(self, task_label):
        if not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        z = self.encoder(task_label)
        Wa, ba = self.h_action(z)
        Wv, bv = self.h_critic(z)
        Wl, bl = self.h_logstd(z)
        return {'Wa': Wa.detach(), 'ba': ba.detach(), 'Wv': Wv.detach(), 'bv': bv.detach(), 'Wl': Wl.detach(), 'bl': bl.detach()}

    def predict(self, obs, action=None, task_label=None, return_layer_output=False, to_numpy=False):
        obs = tensor(obs)
        if task_label is not None and not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        layers_output = []
        phi, out = self.network.phi_body(obs, task_label, return_layer_output, 'network.phi_body')
        layers_output += out
        phi_a, out = self.network.actor_body(phi, None, return_layer_output, 'network.actor_body')
        layers_output += out
        phi_v, out = self.network.critic_body(phi, None, return_layer_output, 'network.critic_body')
        layers_output += out

        z = self.encoder(task_label)
        Wa, ba = self.h_action(z)
        Wl, bl = self.h_logstd(z)
        Wv, bv = self.h_critic(z)
        if Wa.size(0) == 1 and phi_a.size(0) > 1:
            Wa = Wa.expand(phi_a.size(0), -1, -1)
            ba = ba.expand(phi_a.size(0), -1)
            Wl = Wl.expand(phi_a.size(0), -1, -1)
            bl = bl.expand(phi_a.size(0), -1)
            Wv = Wv.expand(phi_v.size(0), -1, -1)
            bv = bv.expand(phi_v.size(0), -1)

        mean = self._lin_apply(phi_a, Wa, ba)
        v = self._lin_apply(phi_v, Wv, bv)
        log_std = self._lin_apply(phi_a, Wl, bl)
        log_std = torch.clamp(log_std, GaussianActorCriticNet_HNet_CL.LOG_STD_MIN, GaussianActorCriticNet_HNet_CL.LOG_STD_MAX)
        if to_numpy:
            return mean.cpu().detach().numpy()
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_mean', mean), ('policy_std', std), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return mean, action, log_prob, entropy, v, layers_output

# ------------------------
# Full-network HyperNetwork with learned task embeddings
# ------------------------

class _TaskEmbeddingTable(nn.Module):
    """
    Maintains a dictionary of learnable embeddings, one per task key.
    Keys are derived from task labels (one-hot or continuous). Previous
    task embeddings are frozen once training on that task finishes.
    """
    def __init__(self, embed_dim, init_std=0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_std = init_std
        self.embeddings = nn.ParameterDict()

    def _label_to_key(self, label: torch.Tensor) -> str:
        vec = label.detach().cpu().view(-1).numpy()
        # Prefer a stable, human-readable key for one-hot labels
        if vec.sum() > 0 and np.all(vec >= -1e-6) and np.isclose(vec.sum(), 1.0, atol=1e-3):
            idx = int(np.argmax(vec))
            if np.isclose(vec[idx], 1.0, atol=1e-3):
                return f"task_{idx}"
        # Fallback: rounded string of first components
        head = "_".join([f"{v:.3f}" for v in vec[:8]])
        return f"vec_{head}"

    def get(self, label: torch.Tensor, trainable: bool = True, freeze_others: bool = True):
        """
        Returns (embedding, key). If key is new, a fresh embedding is created.
        """
        key = self._label_to_key(label)
        if key not in self.embeddings:
            param = nn.Parameter(torch.zeros(self.embed_dim, device=label.device))
            nn.init.normal_(param, std=self.init_std)
            self.embeddings[key] = param
        if freeze_others:
            for k, p in self.embeddings.items():
                p.requires_grad = (k == key) and trainable
        else:
            self.embeddings[key].requires_grad = trainable
        return self.embeddings[key], key

    def freeze(self, key: str):
        if key in self.embeddings:
            self.embeddings[key].requires_grad = False

class CategoricalActorCriticNet_FullHNet_CL(nn.Module, BaseNet):
    """
    Hypernetwork that generates the full policy/value MLP (phi + actor + critic)
    from a learned task embedding. No task label is concatenated to observations.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 task_label_dim=None,  # kept for API symmetry; not concatenated to obs
                 hidden_units=(200, 200, 200),
                 embed_dim=64,
                 hyper_hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = list(hidden_units)
        self.embed_dim = embed_dim
        self.task_label_dim = task_label_dim

        # Build shapes and total parameter count for the solver network
        dims = [state_dim] + self.hidden_units
        shapes = []
        for idx, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            shapes.append((f'phi_W{idx}', (dout, din)))
            shapes.append((f'phi_b{idx}', (dout,)))
        shapes.append(('Wa', (action_dim, dims[-1])))
        shapes.append(('ba', (action_dim,)))
        shapes.append(('Wv', (1, dims[-1])))
        shapes.append(('bv', (1,)))
        self.shapes = shapes
        self.total_params = sum(np.prod(shape) for _, shape in shapes)
        self.num_phi_layers = len(dims) - 1

        self.generator = nn.Sequential(
            nn.Linear(embed_dim, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, hyper_hidden), nn.ReLU(),
            nn.Linear(hyper_hidden, self.total_params)
        )
        self.embeddings = _TaskEmbeddingTable(embed_dim)
        self.gate = F.relu
        self.to(Config.DEVICE)

    def _split_params(self, flat: torch.Tensor):
        params = {}
        idx = 0
        for name, shape in self.shapes:
            n = int(np.prod(shape))
            chunk = flat[..., idx:idx+n]
            params[name] = chunk.view(flat.size(0), *shape)
            idx += n
        return params

    def generate_weights(self, task_label, detach=True, trainable=True, freeze_others=True):
        if not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        if task_label.dim() == 1:
            task_label = task_label.unsqueeze(0)
        z, key = self.embeddings.get(task_label[0], trainable=trainable, freeze_others=freeze_others)
        z = z.unsqueeze(0)  # shape [1, embed_dim]
        theta = self.generator(z)
        params = self._split_params(theta)
        if detach:
            params = {k: v.detach() for k, v in params.items()}
        return params, key

    def predict(self, obs, action=None, task_label=None, return_layer_output=False):
        assert task_label is not None, 'task_label is required to select task embedding'
        obs = tensor(obs)
        if not isinstance(task_label, torch.Tensor):
            task_label = tensor(task_label)
        if task_label.dim() == 1:
            task_label = task_label.unsqueeze(0)

        params, key = self.generate_weights(task_label, detach=False)

        # If a single parameter set is generated, broadcast over the batch
        batch_size = obs.size(0)
        if next(iter(params.values())).size(0) == 1:
            params = {k: v.expand(batch_size, *v.shape[1:]) for k, v in params.items()}

        x = obs
        layers_output = []
        # Hidden MLP
        for i in range(self.num_phi_layers):
            W_i = params[f'phi_W{i}']  # [B, out, in]
            b_i = params[f'phi_b{i}']  # [B, out]
            x = self.gate(torch.bmm(W_i, x.unsqueeze(-1)).squeeze(-1) + b_i)
            if return_layer_output:
                layers_output.append((f'phi.{i}', x))

        Wa = params['Wa']
        ba = params['ba']
        Wv = params['Wv']
        bv = params['bv']
        logits = torch.bmm(Wa, x.unsqueeze(-1)).squeeze(-1) + ba
        v = torch.bmm(Wv, x.unsqueeze(-1)).squeeze(-1) + bv
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        if return_layer_output:
            layers_output += [('policy_logits', logits), ('policy_action', action), ('value_fn', v)]
        log_prob = dist.log_prob(action).unsqueeze(-1)
        return logits, action, log_prob, dist.entropy().unsqueeze(-1), v, layers_output
