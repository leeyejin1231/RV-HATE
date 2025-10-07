tuning_param  = ["dataset"]
dataset = ["ihc_pure_c20"]                # dataset for evaluation
dataset_name = "ihc_pure"

train_batch_size = 8
eval_batch_size = 8
hidden_size = 768
model_type = "bert-base-uncased"
SEED = 42                                 # random seed
gpu_id = 0                                # GPU id
type = "m0"                               # the type of module (default: m0) do not change this value

base_model = f"./save/{dataset_name}/m0/{dataset[0]}/{SEED}"                       #m0
ner_model_dir = f"./save/{dataset_name}/m1/{dataset[0]}/{SEED}"                    #m1
outlier_model_dir = f"./save/{dataset_name}/m2/{dataset[0]}/{SEED}"                #m2
hard_negative_model_dir = f"./save/{dataset_name}/m3/Ours/{dataset[0]}/{SEED}"     #m3


weight = False            # voting weight
time_step = 100000        # time step
# threshold = 0.65        # voting threshold

param = {"dataset":dataset,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"dataset":dataset,"SEED":SEED,"model_type":model_type, 
         "ner_model_dir":ner_model_dir, "outlier_model_dir":outlier_model_dir, "hard_negative_model_dir":hard_negative_model_dir,
         "base_model":base_model, "weight":weight, "gpu_id":gpu_id, "time_step":time_step, "type":type}
