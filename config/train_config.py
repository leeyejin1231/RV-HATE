dataset = ["dynahate_c125"]

tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay"] ## list of possible paramters to be tuned
train_batch_size = [8]
eval_batch_size = [8]
decay = [0.0]               # default value of AdamW
hidden_size = 768
nepoch = [6]    
lambda_loss = [0.5]        # scaling factor (CE vs. SCL)
temperature = [0.3]
main_learning_rate = [3e-5]
type = "ner"

run_name = "dynahate/ner"
loss_type = ""              # only for saving file name
model_type = "bert-base-uncased"
SEED = [60]
w_aug = True                # using shared semantics as positive pairs
w_double = False
w_separate = False
w_sup = False
cuda_id = 0

## target tagging
use_ner = True

save = True             # saving model parameters

param = {"type":type,"cuda_id":cuda_id,"use_ner":use_ner,"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate}
