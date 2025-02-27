import numpy as np
import json
import random
import os
from easydict import EasyDict as edict
# from torch.nn import functional as F
import json

import torch
import torch.utils.data

from config import eval_config as train_config
from utils import iter_product, get_dataloader
from sklearn.metrics import f1_score

from model import primary_encoder_v2_no_pooler_for_con


# Credits https://github.com/varsha33/LCL_loss
def test(test_loader, model_base, model_ner, model_cosine, model_outlier, model_hard_negative, log, weights):
    model_base.eval()
    model_ner.eval()
    model_cosine.eval()
    model_outlier.eval()
    model_hard_negative.eval()
    
    total_pred_1,total_true,total_pred_prob_1, total_confidence = [],[],[],[]
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
            # probabilities_base = F.softmax(pred_base, dim=1)
            last_layer_hidden_states, supcon_feature_ner = model_ner.get_cls_features_ptrnsp(text,attn) # #v2
            pred_ner = model_ner(last_layer_hidden_states)
            # probabilities_ner = F.softmax(pred_ner, dim=1)
            last_layer_hidden_states, supcon_feature_cosine = model_cosine.get_cls_features_ptrnsp(text,attn) # #v2
            pred_cosine = model_cosine(last_layer_hidden_states)
            # probabilities_cosine = F.softmax(pred_cosine, dim=1)
            last_layer_hidden_states, supcon_feature_outlier = model_outlier.get_cls_features_ptrnsp(text,attn) # #v2
            pred_outlier = model_outlier(last_layer_hidden_states)
            # probabilities_outlier = F.softmax(pred_outlier, dim=1)
            last_layer_hidden_states, supcon_feature_hard_negative = model_hard_negative.get_cls_features_ptrnsp(text,attn) # #v2
            pred_hard_negative = model_hard_negative(last_layer_hidden_states)
            # probabilities_hard_negatives = F.softmax(pred_hard_negative, dim=1)

            pred_list = [pred_base, pred_cosine, pred_ner, pred_outlier, pred_hard_negative]
            # pred_list = torch.stack([probabilities_base, probabilities_ner, probabilities_cosine, probabilities_outlier, probabilities_hard_negatives])

            # probabilities = pred_list.mean(dim=0)

            # print(f"~~~~~~{weights}~~~~~~")
            # 로짓 가중 평균 계산
            with torch.no_grad():
                # 로짓 값 가중 합산
                logits_weighted_sum = torch.zeros_like(pred_list[0])
                for pred, weight in zip(pred_list, weights):
                    logits_weighted_sum += pred * weight
            
            # pred_list_1 = []
            # voting_pred_list_1 = torch.max(logits_weighted_sum, 1)[1].view(label.size()).data.detach().cpu().tolist()

            # probabilities = probabilities.cpu()
            # pred_labels = torch.argmax(pred_base, dim=1)  
            # pred_confidence, _ = torch.max(probabilities, dim=1)

            # for i in range(len(pred)):
            #     if pred_confidence[i] >= log.param.threshold:
            #         pred_list_1.append(int(pred_labels[i].item()))
            #     else:
            #         pred_list_1.append(voting_pred_list_1[i])

            # pred_list_1_tensor = torch.tensor(pred_list_1).to(label.device)
            # num_corrects_1 = (pred_list_1_tensor.data == label.data).float().sum()

            # probabilities = F.softmax(logits_weighted_sum, dim=1)
            # pred_labels = torch.argmax(pred, dim=1)  
            # pred_confidence, _ = torch.max(pred, dim=1)

            pred_list_1 = torch.max(logits_weighted_sum, 1)[1].view(label.size()).data.detach().cpu().tolist()
            num_corrects_1 = (torch.max(logits_weighted_sum, 1)[1].view(label.size()).data == label.data).float().sum()

            true_list = label.data.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            # total_feature.extend(supcon_feature_ner.data.detach().cpu().tolist())
            # total_pred_prob_1.extend(logits_weighted_sum.data.detach().cpu().tolist())
            # total_confidence.extend(pred_confidence.data.detach().cpu().tolist())

    f1_score_1 = f1_score(total_true,total_pred_1, average="macro")
    f1_score_1_w = f1_score(total_true,total_pred_1, average="weighted")
    f1_score_1 = {"macro":f1_score_1,"weighted":f1_score_1_w}

    total_acc = 100 * total_num_corrects / total_num

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1

    # save_pred["feature"] = total_feature
    # save_pred["pred_prob_1"] = total_pred_prob_1
    save_pred["pred_confidence"] = total_confidence

    return total_acc,f1_score_1,save_pred

##################################################################################################
def cl_test(log, weights):

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
    model_outlier.load_state_dict(torch.load(os.path.join(log.param.outlier_model_dir, "model.pt")))
    model_hard_negative.load_state_dict(torch.load(os.path.join(log.param.hard_negative_model_dir, "model.pt")))
    
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
    
    # val_acc_1,val_f1_1,val_save_pred = test(valid_data,model_main,log)
    test_acc_1,test_f1_1,test_save_pred = test(test_data, model_base, model_ner, model_cosine, model_outlier, model_hard_negative,log, weights)
    

    print("Model 1")
    print(f'Test Accuracy: {test_acc_1:.2f} Test F1: {test_f1_1["macro"]*100:.2f}')

    log.test_f1_score_1 = test_f1_1
    log.test_accuracy_1 = test_acc_1


def compute_weigths(log):
    temp = []
    with open(f"{log.param.base_model}/log.json", 'r', encoding="utf-8") as file:
        baseline_json = json.load(file)
        temp.append(baseline_json['valid_f1_score_1']['macro']*100)
    with open(f"{log.param.ner_model_dir}/log.json", 'r', encoding="utf-8") as file:
        ner_json = json.load(file)
        temp.append(ner_json['valid_f1_score_1']['macro']*100)
    with open(f"{log.param.cosine_model_dir}/log.json", 'r', encoding="utf-8") as file:
        cosine_json = json.load(file)
        temp.append(cosine_json['valid_f1_score_1']['macro']*100)
    with open(f"{log.param.outlier_model_dir}/log.json", 'r', encoding="utf-8") as file:
        outlier_jason = json.load(file)
        temp.append(outlier_jason['valid_f1_score_1']['macro']*100)
    with open(f"{log.param.hard_negative_model_dir}/log.json", 'r', encoding="utf-8") as file:
        hard_negative_json = json.load(file)
        temp.append(hard_negative_json['valid_f1_score_1']['macro']*100)

    f1_scores = np.array(temp)
    weights_exp = np.exp(f1_scores)  # 지수 함수 적용
    normalized_weights = weights_exp / weights_exp.sum()  # 정규화
    rounded_weights = np.round(normalized_weights, 4)
    print(f"==={rounded_weights}===")

    return list(rounded_weights)


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
        if log.param.weight == True:
            print("----- using weights -----")
            weights = compute_weigths(log)
        else:
            weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        
        # assert log.param.load_dir is not None, "to load a model, log.param.load_dir should be given!!"
        cl_test(log, weights)
