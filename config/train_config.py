dataset = ["ihc_pure_c20"]            # dataset

tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay"] ## list of possible paramters to be tuned
train_batch_size = [8]                # batch size for training
eval_batch_size = [8]                 # batch size for evaluation
decay = [0.0]                         # default value of AdamW
hidden_size = 768                     # hidden size
nepoch = [6]                          # number of epochs
lambda_loss = [0.5]                   # scaling factor (CE vs. SCL)
temperature = [0.3]                   # temperature for contrastive loss
main_learning_rate = [3e-5]           # learning rate
type = "m0"                           # module type (m0 - m2)

run_name = "hateval/m0"               # save path
loss_type = ""                        # only for saving file name (default: "") DO NOT CHANGE THIS VALUE
model_type = "bert-base-uncased"      # model type
SEED = [42]                           # random seed
w_aug = True                          # default value (True) DO NOT CHANGE THIS VALUE
w_double = False                      # default value (False) DO NOT CHANGE THIS VALUE
w_separate = False                    # default value (False) DO NOT CHANGE THIS VALUE
w_sup = False                         # default value (False) DO NOT CHANGE THIS VALUE
cuda_id = 0                           # GPU id

## target tagging
use_ner = True if type == "ner" else False

save = True                           # saving model parameters

param = {"type":type,"cuda_id":cuda_id,"use_ner":use_ner,"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate}
