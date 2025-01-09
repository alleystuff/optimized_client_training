import os
import torch
import numpy as np
import pandas as pd
from torch import nn
import flwr as fl
from flwr.common import Context
from flwr.client import ClientApp
from flwr.simulation import run_simulation
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional

#data imports
from data.data_prepper import load_datasets

#client imports
from src.clients.ntp_client import NTPClient
from src.clients.otp_client import OTPClient, OTPFineTuneClient

#model imports
from src.models.server_models import initialize_model
from src.server_strategy import (fedavg, fedavgm, fedmedian, fedprox, fedcda)

#config and util imports
from utils.utils import set_torch_device
from config_folder import client_config_file, server_config_file
from config_folder.client_config_file import get_server_checkpoint_path

import random
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
# torch.manual_seed(42)

def run_server(
        num_cpus,
        num_gpus,
        dataset_id=0,
        optimize_metric="f1_scores",
        strategy_name="FedAvg",
        meta_action_type="take_epsilon_greedy_weighted_metric_action",
        seed=42
    ):
    """Run server"""
    
    DEVICE = set_torch_device(
        strategy=strategy_name,
        manual_seed=seed
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Using Device: {DEVICE}")
    if DEVICE=="mps":
        torch.mps.manual_seed(seed)
    elif "cuda" in str(DEVICE):
        torch.cuda.manual_seed(seed)
    
    dataset_name = client_config_file.DATASETS[dataset_id]['name']
    num_classes = client_config_file.DATASETS[dataset_id]['num_classes']
    feature_name = client_config_file.DATASETS[dataset_id]['feature_name']
    label_name = client_config_file.DATASETS[dataset_id]['label_name']
    server_rounds = client_config_file.DATASETS[dataset_id]['training_periods']
    optimizer_config = client_config_file.DATASETS[dataset_id]['optimizer_config']
    print(f"Dataset Name: {dataset_name} | Num Classes: {num_classes} | Feature Name: {feature_name} | Label Name: {label_name}")

    # initialize the server model
    server, input_size = initialize_model(
        model_name = "resnet",
        num_classes = num_classes,
        feature_extract = False,
        use_pretrained = None
    )
    first_parameter = next(server.parameters())
    # input_shape = client_config_file.DATASETS[dataset_id]['input_shape']
    client_batch_size = client_config_file.LOCAL_TRAINING_BATCH_SIZE
    print(f"Model Input Shape: {first_parameter.size()}| Input Shape: {input_size} | Batch Size: {client_batch_size}")

    # local model - copy of the server
    model = server.to(DEVICE)

    # client loss function
    client_loss_fn = nn.CrossEntropyLoss() 
    
    # client optimizer
    # client_optimizer = torch.optim.Adam( 
    #     params = model.parameters(),
    #     lr = client_config_file.LOCAL_LEARNING_RATE
    # ) 

    print(f"Checking for server checkpoint")
    SERVER_CHECKPOINT_PATH = get_server_checkpoint_path(f"{dataset_name}_{strategy_name}_{meta_action_type}")
    print(f"Server Model File Name: {SERVER_CHECKPOINT_PATH}")
    if os.path.isfile(SERVER_CHECKPOINT_PATH):
        print(f"Loading server checkpoint")
        model.load_state_dict(
            torch.load(
                SERVER_CHECKPOINT_PATH,
                map_location = torch.device(DEVICE)
            )
        )
        # else:
        #     torch.load()
    else:
        print(f"No server checkpoint found")
    # return model, local_train_dataloaders, local_val_dataloaders

    
    print(f"Experiment Type: {strategy_name} | {meta_action_type}")
    print(f"Total Server Rounds: {server_rounds}")
    def client_fn(context: Context):
        """
        Create a Flower client representing a single client.
        cid: client id - default argument expected by Flower and used by Flower during federated learning to pass index of the client to be selected
        """
        os.chdir("<replace with your root>") #replace this location with the full path to the root folder of the project
        print(os.getcwd())


        partition_id = context.node_config["partition-id"]
        cid = partition_id
        num_partitions = context.node_config["num-partitions"]
        train_dataloader, val_dataloader, neutral_dataloader, fds_partitioners = load_datasets(
            partition_id=partition_id,
            num_partitions=num_partitions,
            dataset_name = dataset_name,
            feature_name = feature_name,
            label_name = label_name,
            input_size = input_size,
            batch_size = client_config_file.LOCAL_TRAINING_BATCH_SIZE,
            seed=seed
        )

        print(f"Total Number of local train and validation partitions created: {num_partitions}")

        local_train_batch = next(iter(train_dataloader))
        local_val_batch = next(iter(val_dataloader))
        print(f"Local Train Batch Input Data Shape: {local_train_batch[feature_name].shape}")
        print(f"Local Test Batch Input Data Shape: {local_val_batch[feature_name].shape}")

        print(f"Client ID: {cid}")
        print(f"Client ID {cid} | Train Data Size| {len(train_dataloader)}")
        if int(cid)==0:
            return OTPClient(
                client_id = cid,
                model = model,
                train_dataloader = train_dataloader, 
                val_dataloader = val_dataloader,
                public_partition_dataloader = neutral_dataloader,
                neutral_dataloader = neutral_dataloader,
                dataset_name = dataset_name, 
                feature_name = feature_name, 
                label_name = label_name, 
                loss_fn = client_loss_fn, 
                optimizer_config = optimizer_config, 
                num_classes = len(pd.Series(
                    [i[label_name] for i in train_dataloader.dataset.to_list()]
                ).unique()),
                optimize_metric = optimize_metric,
                meta_action_type = meta_action_type,
                experiment_type = f"{strategy_name}_{meta_action_type}",
                dataset_id = dataset_id,
                seed=seed,
                device = DEVICE, 
                verbose = True
            ).to_client()
        elif int(cid)>=1:
            return NTPClient(
                client_id = cid,
                model = model,
                train_dataloader = train_dataloader, 
                val_dataloader = val_dataloader,
                public_partition_dataloader = neutral_dataloader,
                neutral_dataloader = neutral_dataloader, 
                dataset_name = dataset_name,
                feature_name = feature_name, 
                label_name = label_name, 
                loss_fn = client_loss_fn, 
                optimizer_config = optimizer_config, 
                experiment_type = f"{strategy_name}_{meta_action_type}",
                dataset_id = dataset_id,
                seed = seed,
                device = DEVICE, 
                verbose = True
            ).to_client()

    def client_config(server_round: int):
        """
        Config function used to pass variables and data to the local clients during federated learning
        The federated learning strategy will call this function every round.
        """
        config = {
            "server_round": server_round,
            "local_epochs": client_config_file.LOCAL_TRAINING_EPOCHS,
            "strategy": strategy_name
        }
        return config
    
    def server_fn(context: Context):
        if strategy_name=="FedAvg":
            strategy = fedavg.fedavg_strategy(
                model,
                client_config=client_config,
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                meta_action_type=meta_action_type,
                seed = seed
            )
        if strategy_name=="FedMedian":
            strategy = fedmedian.fedmedian_strategy(
                model,
                client_config=client_config,
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                meta_action_type=meta_action_type,
                seed = seed
            )
        if strategy_name=="FedProx":
            strategy = fedprox.fedprox_strategy(
                model,
                client_config=client_config,
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                meta_action_type=meta_action_type,
                seed = seed,
                proximal_mu=server_config_file.FEDPROX_PROXIMAL_MU
            )
        if strategy_name=="FedCDA":
            strategy = fedcda.fedcda_strategy(
                model,
                client_config=client_config,
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                meta_action_type=meta_action_type,
                seed = seed
            )
        if strategy_name=="FedAvgM":
            strategy = fedavgm.fedavgm_strategy(
                model,
                client_config=client_config,
                dataset_name=dataset_name,
                strategy_name=strategy_name,
                meta_action_type=meta_action_type,
                seed = seed,
                server_learning_rate=1.0,
                server_momentum=0.0
            )
        server_config = ServerConfig(
            num_rounds = server_rounds
        )
        return ServerAppComponents(strategy=strategy, config=server_config)
    
    ##define client resources and start simulations based on the current strategy
    backend_config = {
        "client_resources": {
            "num_cpus": num_cpus, 
            "num_gpus": num_gpus
        }
    } 

    server = ServerApp(
        server_fn=server_fn
    )

    client = ClientApp(
        client_fn=client_fn
    )
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=client_config_file.NUM_CLIENTS,
        backend_config=backend_config
    )

    return f"Completed simulation/experiment: {strategy_name} | Action type: {meta_action_type} | Dataset: {dataset_name}"
