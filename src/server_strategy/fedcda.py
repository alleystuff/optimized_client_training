import os
import copy
import torch
import random
import flwr as fl
import numpy as np
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, OrderedDict

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy import Strategy

from utils.utils import get_parameters
from config_folder import client_config_file, server_config_file
from config_folder.client_config_file import get_server_checkpoint_path


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class FedCDA(Strategy):
    """Federated CDA Strategy

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        current_round: int = 1, #this needs to be passed in by the user and incremented at each round
        warmup_period: int = 50,
        k: int=3,#default is 4
        memory_k: int=2, #default is 3
        batch_client_num: int=2, #default is 5
        server_model = None,
        client_model_save_location: str = "./",
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.current_round = current_round
        self.warmup_period = warmup_period
        self.k = k
        self.memory_k = memory_k
        self.batch_client_num = batch_client_num
        self.server_model = server_model #this is used to get model keys to store local client models at each round
        self.server_model_copy = self.server_model #create a copy to manipulate it
        self.client_model_save_location = client_model_save_location
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedCDA(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def save_client_models(self, results):
        """
        Save local models received from the participating clients.
        If the models already exist then replace with the most recently received models.
        """
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            
            client_parameters = parameters_to_ndarrays(fit_res.parameters)
            param_dict = zip(self.server_model.state_dict().keys(), client_parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
            self.server_model_copy.load_state_dict(state_dict, strict=True)
            torch.save(
                self.server_model_copy.state_dict(), 
                f"{self.client_model_save_location}_client_{client_id}.pth"
            )
            self.client_model = None

    def set_client_model(self, parameters):
        client_parameters = parameters_to_ndarrays(parameters)
        param_dict = zip(self.server_model.state_dict().keys(), client_parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.server_model_copy.load_state_dict(state_dict, strict=True)

    def model_aggregrate_new(self, num):  
        for name, data in self.server_model.state_dict().items(): 
            update_per_layer = self.weight_accumulator[name] * ( 1 / num) 
            if data.type() != update_per_layer.type():  
                data.add_(update_per_layer.to(torch.int64))  
            else:
                data.add_(update_per_layer)  

    def batch_list(self, len_data_list, interval):
        res_list = []
        for i in range(0,len_data_list,interval):
            if(i % interval == 0):
                begin,end = i, min(i + interval, len_data_list)
                res_list.append((begin,end))
        return res_list
    
    def get_block_i(self, data_list, interval):
        begin, end = interval
        return data_list[begin: end]
    
    def search(self, random_index_k, start_index, now_path, res):
        """
        search random batches to minimize local loss and select params
        return: weights for selected clients
        1. create random batches from participating clients
        2. minimize the objective using local losses
        3. select client models which minimize the objective
        4. aggregate and return parameters
        """
        if(start_index>=len(random_index_k)):
            weight_accumulator = {}
            for name, params in self.server_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)
            
            for i in range(len(self.all_weights)):
                if(len(self.all_weights[i]))!=0:
                    print(f"lenght of client models for each client in, All weights length list currently: {[len(i) for i in self.all_weights]}")
                    print(f"Now path: {now_path} | Now path element: {now_path[i]} | Now path index: {now_path[i]}")
                    temp = self.all_weights[i][now_path[i]]
                    for name, params in temp.items():
                        weight_accumulator[name] += params
            
            for name, params in weight_accumulator.items():
                weight_accumulator[name]  = weight_accumulator[name]/self.cal_num
            
            temp_res = 0
            for i in range(len(self.all_weights)):
                if(len(self.all_weights[i]))!=0:
                    print(f"Now path: {now_path[i]}")
                    temp = self.all_weights[i][now_path[i]]
                    for name, params in temp.items():
                        temp_res += np.linalg.norm((weight_accumulator[name]-params).cpu())  
            if(temp_res<res):
                print("Appear")
                for i in range(len(now_path)):
                    self.all_choose_k[i] = now_path[i] 
                res = temp_res
            return 
        print(f"Start index at: {start_index} | Random Index k: {random_index_k} | len of random index k: {len(random_index_k)}")
        print(f"Selected random ID: {random_index_k[start_index]}")
        
        temp_index = random_index_k.index(random_index_k[start_index])
        
        print(f"lenght of client models for each client in, All weights length list: {[len(i) for i in self.all_weights]}")
        print(f"Selected random ID: {random_index_k[start_index]}")
        print(f"Selected random ID's Index: {temp_index}")
        if len(self.all_weights[temp_index])==0:
            try:
                print(f"All weights length: {len(self.all_weights[temp_index])}")
                start_index = start_index + 1 #when start index is 0 but their are no weights at that index then increment the start index
                temp_index = random_index_k.index(random_index_k[start_index]) #update temp_index in case previous if statement was triggered
                print(f"All weights length: {len(self.all_weights[temp_index])}")
                for every in range(len(self.all_weights[temp_index])):
                    now_path[temp_index] = every
                    self.search(random_index_k, start_index+1, now_path, res)
            except:
                print(f"All weights length: {len(self.all_weights[temp_index])}")
                for every in range(len(self.all_weights[temp_index])):
                    now_path[temp_index] = every
                    self.search(random_index_k, start_index+1, now_path, res)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        
        self.current_round = server_round
        print(f"Current Round: {self.current_round} | Server Round: {server_round}")
        self.save_client_models(results) #save client models
        local_clients = list()
        for client_proxy, fit_res in results:
            client_dict = dict()
            client_dict["client_id"] = client_proxy.cid 
            client_dict["parameters"] = fit_res.parameters
            client_dict["loss"] = fit_res.metrics['local_loss']
            local_clients.append(client_dict)
            
        """fed cda aggregation"""
        all_client_num = len(local_clients)
        print(f"Total Number of clients: {all_client_num}")
        self.all_weights = [[] for i in range(all_client_num)]
        self.all_choose_k = [0 for i in range(all_client_num)]

        self.cal_num = all_client_num #number of clients participating in the current round
        print(f"Total Number of clients: {all_client_num}")
        random_index = [int(client["client_id"]) for client in local_clients]# IDs of the clients participating in the current round
        print(f"IDs of clients: {random_index}")
        random_index_k = random.sample(random_index, self.k)  # generate a random sample from the participating client of size k
        print(f"Random Sample: {random_index_k}")

        round_all_weights = [{} for i in range(all_client_num)] 
        self.weight_accumulator = {}
        for name, params in self.server_model.state_dict().items():
            self.weight_accumulator[name] = torch.zeros_like(params)  

        client_models = os.listdir(server_config_file.FEDCDA_SAVE_PATH)
        client_ids = [int(client_model.split('.pth')[0].split('_')[2]) for client_model in client_models]
        print(f"Client IDs: {client_ids}")
        for idx, c_index in enumerate(client_ids):
            print(f"Client ID: {c_index} | Random Index k: {random_index_k}")
            if c_index in random_index_k:
                client_params = [client["parameters"] for client in local_clients if int(client["client_id"])==c_index][0]
                self.set_client_model(client_params) 
                client_res = {} 
                for name, data in self.server_model_copy.state_dict().items(): 
                    client_res[name] = data

                weight_temp = {} 
                for name, params in self.server_model.state_dict().items():
                    weight_temp[name] = client_res[name] - params
                
                client_temp = copy.deepcopy(client_res)
                try:
                    if(len(self.all_weights[idx])>=self.memory_k):
                        del self.all_weights[idx][0] 
                        self.all_weights[idx].append({}) 
                        for name, data in self.server_model.state_dict().items():
                            self.all_weights[idx][self.memory_k-1][name] = client_temp[name] 
                    else:
                        self.all_weights[idx].append({}) 
                        for name, data in self.server_model.state_dict().items():
                            temp_idx = len(self.all_weights[idx])
                            if temp_idx>=1:
                                temp_idx = temp_idx - 1
                            self.all_weights[idx][temp_idx][name] = client_temp[name]  
                    round_all_weights[idx] = weight_temp 
                except:
                    continue
            else:
                try:
                    if(len(self.all_weights[idx])!=0):
                        for name, data in self.server_model.state_dict().items():
                            self.weight_accumulator[name].add_(self.all_weights[idx][self.all_choose_k[idx]][name] - data)
                    else:
                        if self.current_round < all_client_num:
                            self.cal_num = self.cal_num - self.current_round 
                except:
                    continue
        if self.current_round<self.warmup_period:
            """do fed avg during warm up period"""
            for i in range(all_client_num):
                if(len(self.all_weights[i])!=0):
                    self.all_choose_k[i] = len(self.all_weights[i])#-1
                    print(f"Warmup all_choose_k: {len(self.all_choose_k)}")

            if self.inplace:
                # Does in-place weighted average of results
                aggregated_ndarrays = aggregate_inplace(results)
            else:
                # Convert results
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)

            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

            # Aggregate custom metrics if aggregation fn was provided
            metrics_aggregated = {}
            return parameters_aggregated, metrics_aggregated
        else:
            random_split = self.batch_list(len(random_index_k), self.batch_client_num)
            print(f"Random Split from batch_list method: {random_split}")

            for i in range(len(random_split)):
                b, e = random_split[i]
                batch_i = self.get_block_i(random_index_k, (b, e ))
                now_path = copy.deepcopy(self.all_choose_k)
                result = 99999999999999999
                self.search(batch_i, 0, now_path, result) 
            
            for idx, random_c_index in enumerate(random_index_k):
                for name, data in self.server_model.state_dict().items():
                    print(f"Post search - all_choose_k[idx]: {self.all_choose_k[idx]}")
                    if len(self.all_weights[idx])!=0: 
                        round_all_weights[idx][name] = self.all_weights[idx][self.all_choose_k[idx]][name] - self.server_model.state_dict()[name] 
                try:
                    for name, params in self.server_model.state_dict().items():
                        self.weight_accumulator[name].add_(round_all_weights[idx][name])
                except:
                    continue
            self.model_aggregrate_new(self.cal_num)
            parameters_aggregated = ndarrays_to_parameters([val.cpu().numpy() for _, val in self.server_model.state_dict().items()])
            metrics_aggregated = {}
            return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
    
def fedcda_strategy(model, client_config, dataset_name, strategy_name, meta_action_type, seed=42):
    """
    Implement FedProx Strategy
    """
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model)) #set intial parameters
    class SaveFedCDAModelStrategy(FedCDA):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using FedCDA and store checkpoint"""
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)

                # Save the model
                SERVER_CHECKPOINT_PATH = get_server_checkpoint_path(f"{dataset_name}_{strategy_name}_{meta_action_type}_seed_{seed}")
                torch.save(
                    model.state_dict(), 
                    SERVER_CHECKPOINT_PATH
                )

            return aggregated_parameters, aggregated_metrics

    strategy = SaveFedCDAModelStrategy(
        fraction_fit = 0.8,
        fraction_evaluate = 1.0,
        min_fit_clients = int(0.8*client_config_file.NUM_CLIENTS),
        min_evaluate_clients = client_config_file.NUM_CLIENTS,
        min_available_clients = client_config_file.NUM_CLIENTS,
        current_round = 1,
        warmup_period = 10,
        k = 4,#default is 4
        memory_k = 3, #default is 3
        batch_client_num = 5, #default is 5
        server_model = model,
        on_fit_config_fn = client_config,
        on_evaluate_config_fn = client_config,
        initial_parameters = initial_parameters,
        client_model_save_location = server_config_file.FEDCDA_SAVE_PATH
    )
    return strategy
