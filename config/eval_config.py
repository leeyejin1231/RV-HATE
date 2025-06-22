tuning_param  = ["dataset"]
dataset = ["hateval_c20"] # dataset for evaluation
dataset_name = "hateval"

train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "bert-base-uncased"
SEED = 20
gpu_id = 1
type = "base"

base_model = f"./save/{dataset_name}/base/{dataset[0]}/{SEED}"
ner_model_dir = f"./save/{dataset_name}/ner/{dataset[0]}/{SEED}"
outlier_model_dir = f"./save/{dataset_name}/outlier/{dataset[0]}/{SEED}"
hard_negative_model_dir = f"./save/{dataset_name}/hard_negative/Ours/{dataset[0]}/{SEED}"

## voting weight
weight = False
time_step = 100000
## voting threshold
# threshold = 0.65

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, 
         "ner_model_dir":ner_model_dir, "outlier_model_dir":outlier_model_dir, "hard_negative_model_dir":hard_negative_model_dir,
         "base_model":base_model, "weight":weight, "gpu_id":gpu_id, "time_step":time_step, "type":type}
