dataset = ["toxi-sim_toxigen_c20"]

tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay",
                 "loss_type", "model_type", "momentum", "queue_size", "hard_neg_k" , "aug_type", "aug_enque", "moco_weight"] ## list of possible paramters to be tuned


temperature = [0.3]
lambda_loss = [0.75]
momentum = [0.999]

aug_type = ["Augmentation"]
# aug_type = ["Dropout"]


queue_size = [1024]
hard_neg_k = [16]


aug_enque = ["False"]
moco_weight = ["True"]

train_batch_size = [8]
eval_batch_size = [8]

decay = [0.0] # default value of AdamW
main_learning_rate = [1e-05]

hidden_size = 768
nepoch = [6]
run_name = "toxi_new/hard_negative"

model_type = ["bert-base-uncased"]


SEED = [0]

# loss_type = ["CE"]
loss_type = ["Ours"]

dir_name = "toxigen"

w_aug = True
w_double = False
w_separate = False
w_sup = True

save = True
param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate,
         "loss_type":loss_type, "dir_name":dir_name, "momentum":momentum, "queue_size":queue_size, "hard_neg_k":hard_neg_k,
         "aug_type":aug_type, "aug_enque":aug_enque, "moco_weight":moco_weight,}
