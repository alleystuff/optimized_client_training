#!/bin/sh
ddpg_model_path="src/saved_models/agent_models/ddpg/"
server_model_path="src/saved_models/server_models/server/"
metrics_path="data/saved_metrics/ddpg/"
agent_data_path="data/agent_data/ddpg/"
client_models_at_the_server_path="src/saved_models/server_models/clients/"
optimal_model_at_client_side_path="src/saved_models/optimal_client/"

ddpg_model_files=`ls $ddpg_model_path`
server_model_files=`ls $server_model_path`
metrics_files=`ls $metrics_path`
agent_data_files=`ls $agent_data_path`
client_models_at_the_server_files=`ls $client_models_at_the_server_path`
optimal_model_at_client_side_path_files=`ls $optimal_model_at_client_side_path`

#clean all files before running new experiments
for file in $ddpg_model_files
do
   rm -r "$ddpg_model_path$file"
done

for file in $server_model_files
do
   rm -r "$server_model_path$file"
done

for file in $metrics_files
do
   rm -r "$metrics_path$file"
done

for file in $agent_data_files
do
   rm -r "$agent_data_path$file"
done

for file in $client_models_at_the_server_files
do
   rm -r "$client_models_at_the_server_path$file"
done

for file in $optimal_model_at_client_side_path_files
do
   rm -r "$optimal_model_at_client_side_path$file"
done

# touch "data/agent_data/ddpg/ddpg_agent_training_data.csv"
# touch "data/saved_metrics/ddpg/ddpg_agent_training_metrics.csv"
# touch "data/saved_metrics/ddpg/ddpg_otp_clone_metrics.csv"
# touch "data/saved_metrics/ddpg/ddpg_otp_metrics.csv"
# touch "data/saved_metrics/ddpg/ddpg_ntp_metrics.csv"

# #run experiments
# python3 -m run_experiments &
# echo $! > save_pid.txt
nohup python3 -m run_experiments > training.log 2>&1 &
echo $! > save_pid.txt