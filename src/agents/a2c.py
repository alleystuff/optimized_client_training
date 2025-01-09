"""
Implement agent training and evaluation
"""
import os
import csv
import json
import ptan
import math
import json
import torch
import numpy as np
import pandas as pd
from torch import nn
import pandas as pd
import torch.nn.functional as F
from collections import namedtuple
from utils.agent_train_test_utils import (
    get_exp_source_first_last, unpack_batch_a2c, calc_logprob,
    get_training_batches, create_agent_training_data
)
from config_folder import agent_config_file

class ModelA2C(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(obs_size, agent_config_file.HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(agent_config_file.HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.var = nn.Sequential(
            nn.Linear(agent_config_file.HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(agent_config_file.HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)

class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states):
        #if state tensors are on gpu then transfer to cpu for numpy conversion
        if states.device.type == agent_config_file.DEVICE:
            states = states.cpu()
        states_v = ptan.agent.float32_preprocessor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()
        action = [np.random.normal(i, j) for i, j in zip(mu, sigma)]
        action = torch.clip(
            torch.tensor(
                action,
                dtype=torch.float32
            ), 
            agent_config_file.LBOUND, agent_config_file.UBOUND
        )
        action = action.to(self.device)
        mu, var_v = torch.tensor(mu_v, dtype=torch.float32), torch.tensor(var_v, dtype=torch.float32)
        return mu, var_v, action

class A2C:
    def __init__(
        self,
        obs_size, 
        act_size, 
        device=agent_config_file.DEVICE  if torch.cuda.is_available() else "cpu", 
        path=agent_config_file.BEHAVIOR_MODEL_SAVE_PATH
    ):
        """
        Initialize agent
        """
        self.device = device
        self.model_save_path = path

        self.agent_model = ModelA2C(
            obs_size,
            act_size
        ).to(self.device)
        
        self.agent_model_optimizer = torch.optim.Adam(
            params = self.agent_model.parameters(),                        
            lr = agent_config_file.LEARNING_RATE
        )

        print(f"Checking for saved agent model checkpoint")
        if os.path.isfile(self.model_save_path):
            print(f"Loading saved agent model checkpoint")
            checkpoint = torch.load(
                self.model_save_path,
                map_location=agent_config_file.DEVICE if torch.cuda.is_available() else "cpu"
            )
            self.agent_model.load_state_dict(checkpoint['model_state_dict'])
            self.agent_model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print(f"No agent model checkpoint found")

        self.agent = AgentA2C(
            self.agent_model,
            device = self.device
        )

    def get_agent_model(self):
        return self.agent_model
    
    def set_exp_source(self):
        """
        Set experience source to train the agent on. 
        """
        self.agent_training_data = create_agent_training_data(
            agent_training_data_file=agent_config_file.AGENT_TRAINING_DATA_FILE
        )
        self.exp_source = get_exp_source_first_last(
            self.agent_training_data, 
            step_count=agent_config_file.REWARD_STEPS
        )

    def train_agent(self, server_round):
        """
        Train Agent
        """
        self.set_exp_source() #set the source of experience/trajectories the agent will train on
        all_batches = get_training_batches(self.exp_source)
        self.server_round = server_round
        
        #agent training
        AgentLoss = namedtuple(
            'AgentLoss', field_names=['batch_id', 'loss']
        )
        batch_id = 0
        all_training_losses = []
        self.agent_model.train()
        for idx, batch in enumerate(all_batches):
            states_v, actions_v, ref_vals_v = unpack_batch_a2c(
                batch, 
                self.agent, 
                last_val_gamma=agent_config_file.GAMMA**agent_config_file.REWARD_STEPS, 
                device=self.device
            )
            
            batch_size = len(batch)
            batch.clear()

            batch_id += 1
            
            self.agent_model_optimizer.zero_grad()
            mu_v, var_v, value_v = self.agent_model(states_v)

            loss_value_v = F.mse_loss(
                value_v.squeeze(-1),
                ref_vals_v
            )

            adv_v = ref_vals_v.unsqueeze(dim=-1) - value_v.detach()
            log_prob_v = adv_v * calc_logprob(
                mu_v, 
                var_v,
                actions_v
            )
            loss_policy_v = -log_prob_v.mean()
            ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
            entropy_loss_v = agent_config_file.ENTROPY_BETA * ent_v.mean()

            loss_v = loss_policy_v + entropy_loss_v + loss_value_v
            # loss_v.requires_grad = True
            
            loss_v.backward()
            self.agent_model_optimizer.step()

            agent_loss= AgentLoss(
                batch_id=idx,
                loss=loss_v.item()
            )
            all_training_losses.append(agent_loss)
            print(f"Batch Size: {batch_size}| Batch {batch_id} | Loss {loss_v.item()}")
            
            if os.path.isfile(agent_config_file.AGENT_TRAINING_METRICS_FILE):
                print(f"Writing data to {agent_config_file.AGENT_TRAINING_METRICS_FILE}")
                with open(agent_config_file.AGENT_TRAINING_METRICS_FILE, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            batch_id,
                            self.server_round,
                            loss_v.item()
                        ]
                    )
            else:
                with open(agent_config_file.AGENT_TRAINING_METRICS_FILE, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            'batch_id',
                            'server_round',
                            'loss'
                        ]
                    )
                    writer.writerow(
                        [
                            batch_id,
                            self.server_round,
                            loss_v.item()
                        ]
                    )
                print(f"New agent metrics file created: {agent_config_file.AGENT_TRAINING_METRICS_FILE}")
            
            #save model checkpoint
            torch.save({
                'model_state_dict': self.agent_model.state_dict(),
                'optimizer_state_dict': self.agent_model_optimizer.state_dict(),
                'loss': loss_v.item()
            }, agent_config_file.BEHAVIOR_MODEL_SAVE_PATH)

            #save model checkpoint
            torch.save({
                'model_state_dict': self.agent_model.state_dict(),
                'optimizer_state_dict': self.agent_model_optimizer.state_dict(),
                'loss': loss_v.item()
            }, agent_config_file.TARGET_MODEL_SAVE_PATH)
