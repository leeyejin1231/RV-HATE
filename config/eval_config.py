tuning_param  = ["dataset"]
dataset = ["ihc-sim2_ihc_pure_c75"] # dataset for evaluation
dataset_name = "ihc_new"

train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "bert-base-uncased"
SEED = 0

base_model = f"./save/{dataset_name}/baseline/{dataset[0]}/{SEED}"
ner_model_dir = f"./save/{dataset_name}/ner-target/{dataset[0]}/{SEED}"
cosine_model_dir = f"./save/{dataset_name}/cosine/{dataset[0]}/{SEED}"
outlier_model_dir = f"./save/{dataset_name}/outlier/{dataset[0]}/{SEED}"
hard_negative_model_dir = f"./save/{dataset_name}/lahn/Ours/{dataset[0]}/{SEED}"

## voting weight
weight = False
## voting threshold
# threshold = 0.65

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, 
         "ner_model_dir":ner_model_dir, "cosine_model_dir":cosine_model_dir, "outlier_model_dir":outlier_model_dir, "hard_negative_model_dir":hard_negative_model_dir,
         "base_model":base_model, "weight":weight}
