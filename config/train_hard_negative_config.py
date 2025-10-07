dataset = ["ihc_pure_c20"]

tuning_param  = ["lambda_loss", "main_learning_rate","train_batch_size","eval_batch_size","nepoch","temperature","SEED","dataset", "decay",
                 "loss_type", "model_type", "momentum", "queue_size", "hard_neg_k" , "aug_type", "aug_enque", "moco_weight"] ## list of possible paramters to be tuned


temperature = [0.3]
lambda_loss = [0.5]
momentum = [0.999]
queue_size = [1024]
hard_neg_k = [16]

aug_type = ["Augmentation"]              # default value (Augmentation) DO NOT CHANGE THIS VALUE
aug_enque = ["False"]                    # default value (False) DO NOT CHANGE THIS VALUE
moco_weight = ["True"]                   # default value (True) DO NOT CHANGE THIS VALUE

train_batch_size = [8]                    # batch size for training
eval_batch_size = [8]                     # batch size for evaluation
decay = [0.0]                             # default value of AdamW
main_learning_rate = [3e-05]              # learning rate

hidden_size = 768                         # hidden size
nepoch = [6]                              # number of epochs
run_name = "ihc_pure/m3"                  # save path
type = "m0"                               # the type of module (default: m0) DO NOT CHANGE THIS VALUE
model_type = ["bert-base-uncased"]        # model type
gpu_num = 0                               # GPU id
SEED = [42]                               # random seed
loss_type = ["Ours"]                      # loss type (default: Ours) DO NOT CHANGE THIS VALUE
dir_name = "ihc_pure"                      # dataset name

w_aug = True                               # default value (True) DO NOT CHANGE THIS VALUE
w_double = False                           # default value (False) DO NOT CHANGE THIS VALUE
w_separate = False                         # default value (False) DO NOT CHANGE THIS VALUE
w_sup = True                               # default value (True) DO NOT CHANGE THIS VALUE

save = True                                # saving model parameters

param = {"temperature":temperature,"run_name":run_name,"dataset":dataset,"main_learning_rate":main_learning_rate,"train_batch_size":train_batch_size,"eval_batch_size":eval_batch_size,"hidden_size":hidden_size,"nepoch":nepoch,"dataset":dataset,"lambda_loss":lambda_loss,"loss_type":loss_type,"decay":decay,"SEED":SEED,"model_type":model_type,"w_aug":w_aug, "w_sup":w_sup, "save":save,"w_double":w_double, "w_separate":w_separate,
         "loss_type":loss_type, "dir_name":dir_name, "momentum":momentum, "queue_size":queue_size, "hard_neg_k":hard_neg_k,
         "aug_type":aug_type, "aug_enque":aug_enque, "moco_weight":moco_weight,"type":type}
