import numpy as np
import random
import os
from easydict import EasyDict as edict

import torch
import torch.utils.data

from config import eval_config as train_config
from utils import iter_product, get_dataloader
from sklearn.metrics import f1_score

from model import primary_encoder_v2_no_pooler_for_con


# Credits https://github.com/varsha33/LCL_loss
def test(test_loader, model_base, model_ner, model_cosine, model_outlier, model_hard_negative, log):
    model_base.eval()
    model_ner.eval()
    model_cosine.eval()
    model_outlier.eval()
    model_hard_negative.eval()
    
    total_pred_1,total_true,total_pred_prob_1 = [],[],[]
    save_pred = {"true":[],"pred_1":[],"pred_prob_1":[],"feature":[]}

    total_num_corrects = 0
    total_num = 0
    print(len(test_loader))
    with torch.no_grad():
        for batch in test_loader:
            if "ihc" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "dynahate" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            elif "sbic" in log.param.dataset or "hateval" in log.param.dataset or "toxigen" in log.param.dataset:
                text_name = "post"
                label_name = "label"
            else:
                text_name = "cause"
                label_name = "emotion"
                raise NotImplementedError

            text = batch[text_name]
            attn = batch[text_name+"_attn_mask"]

            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                text = text.cuda()
                attn = attn.cuda()
                label = label.cuda()

            last_layer_hidden_states, supcon_feature_base = model_base.get_cls_features_ptrnsp(text,attn) # #v2
            pred_base = model_base(last_layer_hidden_states)
            last_layer_hidden_states, supcon_feature_ner = model_ner.get_cls_features_ptrnsp(text,attn) # #v2
            pred_ner = model_ner(last_layer_hidden_states)
            last_layer_hidden_states, supcon_feature_cosine = model_cosine.get_cls_features_ptrnsp(text,attn) # #v2
            pred_cosine = model_cosine(last_layer_hidden_states)
            last_layer_hidden_states, supcon_feature_outlier = model_outlier.get_cls_features_ptrnsp(text,attn) # #v2
            pred_outlier = model_outlier(last_layer_hidden_states)
            last_layer_hidden_states, supcon_feature_hard_negative = model_hard_negative.get_cls_features_ptrnsp(text,attn) # #v2
            pred_hard_negative = model_hard_negative(last_layer_hidden_states)

            pred_base_list = torch.max(pred_base, 1)[1].view(label.size())
            pred_ner_list = torch.max(pred_ner, 1)[1].view(label.size())
            pred_cosine_list = torch.max(pred_cosine, 1)[1].view(label.size())
            pred_outlier_list = torch.max(pred_outlier, 1)[1].view(label.size())
            pred_hard_negative_list = torch.max(pred_hard_negative, 1)[1].view(label.size())

            pred_list = torch.stack([pred_base_list, pred_ner_list, pred_cosine_list, pred_outlier_list, pred_hard_negative_list])
            weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]).cuda() 

            ## hard voting
            final_preds = []
            for i in range(pred_list.size(1)):
                votes = pred_list[:, i]
                 # Calculate weighted sum for each class
                unique_classes, counts = votes.unique(return_counts=True)
                weighted_counts = torch.zeros_like(unique_classes, dtype=torch.float)

                for j, cls in enumerate(unique_classes):
                    mask = (votes == cls).float()
                    weighted_counts[j] = torch.sum(mask * weights)  # Sum of weights for the class

                # Check for ties
                max_weight = torch.max(weighted_counts)
                tied_classes = unique_classes[weighted_counts == max_weight]  # Classes with ties

                if len(tied_classes) > 1:
                    # When there is a tie, apply unweighted majority voting
                    majority_vote = unique_classes[torch.argmax(counts)].item()
                else:
                    majority_vote = tied_classes[0].item()

                final_preds.append(majority_vote)

            # Final prediction result
            pred_list_1 = final_preds

            device_label = label.device  # Assume `label` is on the target device
            final_preds_1 = torch.tensor(final_preds).to(device_label)

            num_corrects_1 = (torch.tensor(final_preds_1).data == label.data).float().sum()
            true_list = label.data.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            total_pred_prob_1.extend(torch.tensor(final_preds).data.detach().cpu().tolist())

    f1_score_1 = f1_score(total_true,total_pred_1, average="macro")
    f1_score_1_w = f1_score(total_true,total_pred_1, average="weighted")
    f1_score_1 = {"macro":f1_score_1,"weighted":f1_score_1_w}

    total_acc = 100 * total_num_corrects / total_num

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1

    save_pred["pred_prob_1"] = total_pred_prob_1

    return total_acc,f1_score_1,save_pred

##################################################################################################
def cl_test(log):

    np.random.seed(log.param.SEED)
    random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    torch.backends.cudnn.deterministic = True #
    torch.backends.cudnn.benchmark = False #



    print("#######################start run#######################")
    print("log:", log)

    _,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=False,w_double=False,label_list=None)

    model_base = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    model_ner = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type, use_ner=True) # v2
    model_cosine = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    model_outlier = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    model_hard_negative = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    
    #################################################################
    # load model
    model_base.load_state_dict(torch.load(os.path.join(log.param.base_model, "model.pt")))
    model_ner.load_state_dict(torch.load(os.path.join(log.param.ner_model_dir, "model.pt")))
    model_cosine.load_state_dict(torch.load(os.path.join(log.param.cosine_model_dir, "model.pt")))
    model_outlier.load_state_dict(torch.load(os.path.join(log.param.outlier_model_dir, "model.pt")), strict=False)
    model_hard_negative.load_state_dict(torch.load(os.path.join(log.param.hard_negative_model_dir, "model.pt")), strict=False)
    print(f"model is loaded from {log.param.ner_model_dir}")
    
    model_base.eval()
    model_ner.eval()
    model_cosine.eval()
    model_outlier.eval()
    model_hard_negative.eval()
    if torch.cuda.is_available():
        model_base.cuda()
        model_ner.cuda()
        model_cosine.cuda()
        model_outlier.cuda()
        model_hard_negative.cuda()
    ###################################################################
    
    test_acc_1,test_f1_1,test_save_pred = test(test_data, model_base, model_ner, model_cosine, model_outlier, model_hard_negative,log)
    

    print("Model 1")
    print(f'Test Accuracy: {test_acc_1:.2f} Test F1: {test_f1_1["macro"]*100:.2f}')

    log.test_f1_score_1 = test_f1_1
    log.test_accuracy_1 = test_acc_1
    

if __name__ == '__main__':

    tuning_param = train_config.tuning_param

    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        log.param.label_size = 2
        
        # assert log.param.load_dir is not None, "to load a model, log.param.load_dir should be given!!"
        cl_test(log)
