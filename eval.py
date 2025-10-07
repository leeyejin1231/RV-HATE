import numpy as np
import random
import os
from easydict import EasyDict as edict

import torch
import torch.utils.data

from config import eval_config as train_config
from utils import iter_product, get_dataloader
from sklearn.metrics import f1_score, confusion_matrix

from model import primary_encoder_v2_no_pooler_for_con
import pandas as pd  
from transformers import AutoTokenizer


# Credits https://github.com/varsha33/LCL_loss
def test(test_loader, model_base, log):
    model_base.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(log.param.model_type)
    
    total_pred_1,total_true,total_pred_prob_1 = [],[],[]
    save_pred = {"true":[],"pred_1":[],"pred_prob_1":[],"feature":[]}
    wrong_predictions = {"text": [], "true_label": [], "predicted_label": []}  # 틀린 예측 저장용 딕셔너리 추가

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

            decoded_texts = [tokenizer.decode(t[t.nonzero()].squeeze(), skip_special_tokens=True) for t in text]

            label = batch[label_name]
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()

            if torch.cuda.is_available():
                text = text.cuda(log.param.gpu_id)
                attn = attn.cuda(log.param.gpu_id)
                label = label.cuda(log.param.gpu_id)

            last_layer_hidden_states, supcon_feature_base = model_base.get_cls_features_ptrnsp(text,attn) # #v2
            pred_base = model_base(last_layer_hidden_states)

            pred_list_1 = torch.argmax(pred_base, dim=1).cpu().numpy().tolist()

            pred_tensor = torch.tensor(pred_list_1, device=label.device)
            num_corrects_1 = (pred_tensor == label).float().sum()
            true_list = label.detach().cpu().tolist()

            total_num_corrects += num_corrects_1.item()
            total_num += text.shape[0]

            total_pred_1.extend(pred_list_1)
            total_true.extend(true_list)
            total_pred_prob_1.extend(torch.tensor(pred_list_1).data.detach().cpu().tolist())

            for i in range(len(pred_list_1)):
                if pred_list_1[i] != true_list[i]:
                    wrong_predictions["text"].append(decoded_texts[i])
                    wrong_predictions["true_label"].append(true_list[i])
                    wrong_predictions["predicted_label"].append(pred_list_1[i])

    f1_score_1 = f1_score(total_true,total_pred_1, average="macro")
    f1_score_1_w = f1_score(total_true,total_pred_1, average="weighted")
    f1_score_1 = {"macro":f1_score_1,"weighted":f1_score_1_w}

    cm = confusion_matrix(total_true, total_pred_1)
    tn, fp, fn, tp = cm.ravel()
    
    total_samples = len(total_true)
    tn_rate = tn / total_samples * 100
    fp_rate = fp / total_samples * 100
    fn_rate = fn / total_samples * 100
    tp_rate = tp / total_samples * 100

    print("\nConfusion Matrix Results:")
    print(f"True Negatives: {tn} ({tn_rate:.2f}%)")
    print(f"False Positives: {fp} ({fp_rate:.2f}%)")
    print(f"False Negatives: {fn} ({fn_rate:.2f}%)")
    print(f"True Positives: {tp} ({tp_rate:.2f}%)")

    total_acc = 100 * total_num_corrects / total_num

    save_pred["true"] = total_true
    save_pred["pred_1"] = total_pred_1
    save_pred["pred_prob_1"] = total_pred_prob_1

    wrong_pred_df = pd.DataFrame(wrong_predictions)
    output_path = os.path.join(os.path.dirname(log.param.base_model), f"{log.param.dataset}_wrong_predictions.csv")
    wrong_pred_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nWrong predictions saved to: {output_path}")

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

    _,_,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=False,w_double=False,label_list=None, type=log.param.type)

    model_base = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    
    #################################################################
    # load model
    model_base.load_state_dict(torch.load(os.path.join(log.param.base_model, "model.pt")))
    print(f"model is loaded from {log.param.ner_model_dir}")
    
    model_base.eval()
    if torch.cuda.is_available():
        model_base.cuda(log.param.gpu_id)
    ###################################################################
    
    test_acc_1,test_f1_1,test_save_pred = test(test_data, model_base, log)
    

    print("Model 1")
    print(f'Test Accuracy: {test_acc_1:.2f} Test F1: {test_f1_1["macro"]*100:.2f}')

    log.test_f1_score_1 = test_f1_1
    log.test_accuracy_1 = test_acc_1
    

if __name__ == '__main__':

    tuning_param = train_config.tuning_param

    param_list = [train_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list))

    for param_com in param_list[1:]:

        log = edict()
        log.param = train_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        log.param.label_size = 2
        
        cl_test(log)
