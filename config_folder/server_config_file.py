FEDCDA_SAVE_PATH = "src/saved_models/server_models/clients/"
FEDPROX_PROXIMAL_MU = 1

DATASET_IDS = [2, 1] #0, 1, 2
STRATEGIES = ['FedAvg', 'FedAvgM', 'FedMedian', 'FedProx','FedCDA'] #'FedAvg', 'FedAvgM', 'FedMedian', 'FedProx', 
ACTION_TYPES = ['None', 'take_random_action', 'take_epsilon_greedy_normalized_action', 'take_epsilon_greedy_weighted_metric_action'] #'None', 
