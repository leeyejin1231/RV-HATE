import numpy as np
import gym
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import f1_score
from utils import get_dataloader, iter_product
from model import primary_encoder_v2_no_pooler_for_con
from config import eval_config as train_config
from easydict import EasyDict as edict
from stable_baselines3.common.callbacks import BaseCallback


def normalize_weights(weights):
    w = np.clip(weights, 1e-6, None)
    return w / w.sum()

class EnsembleEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, valid_dataset, models, f1_baseline, gpu_id):
        super().__init__()
        self.valid_dataset = valid_dataset
        self.models = models
        self.baseline = f1_baseline

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self.current_weights = np.array([0.25]*4, dtype=np.float32)

        self.all_logits = [[] for _ in range(4)]
        self.all_labels = []
        with torch.no_grad():
            for batch in self.valid_dataset:
                text = batch['post'].cuda(gpu_id)
                attn = batch['post_attn_mask'].cuda(gpu_id)
                label = batch['label']
                label = torch.tensor(label)
                label = torch.autograd.Variable(label).long()
                label = label.cuda(gpu_id)
                self.all_labels.extend(label.cpu().numpy().tolist())
                for i, model in enumerate(self.models):
                    hidden, _ = model.get_cls_features_ptrnsp(text, attn)
                    out = model(hidden)
                    self.all_logits[i].append(out.cpu())
        self.all_logits = [torch.cat(logit_list, dim=0) for logit_list in self.all_logits]
        self.all_labels = np.array(self.all_labels)

    def reset(self):
        self.current_weights = np.array([0.25]*4, dtype=np.float32)
        return self.current_weights.copy()  
        
    def step(self, action):
        weights = normalize_weights(action)
        logits_sum = None
        for logit, w in zip(self.all_logits, weights):
            if logits_sum is None:
                logits_sum = w * logit
            else:
                logits_sum += w * logit
        preds = torch.argmax(logits_sum, dim=1).cpu().numpy().tolist()
        trues = self.all_labels.tolist()
        # Compute reward
        f1 = f1_score(trues, preds, average='macro')
        reward = 2 * (f1 - self.baseline)
        self.current_weights = weights
        done = True
        info = {'f1': f1}
        return self.current_weights.copy(), reward, done, info
    

def test(test_loader, models, weights, gpu_id):
    weights = np.array(weights).flatten()
    total_true, total_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            text = batch['post'].cuda(gpu_id)
            attn = batch['post_attn_mask'].cuda(gpu_id)
            label = batch['label']
            label = torch.tensor(label)
            label = torch.autograd.Variable(label).long()
            label = label.cuda(gpu_id)

            # Collect logits
            logits_sum = None
            for model, w in zip(models, weights):
                hidden, _ = model.get_cls_features_ptrnsp(text, attn)
                out = model(hidden)
                if logits_sum is None:
                    logits_sum = float(w) * out
                else:
                    logits_sum += float(w) * out

            # Predictions
            preds = torch.argmax(logits_sum, dim=1).cpu().numpy().tolist()
            trues = label.cpu().numpy().tolist()
            total_true.extend(trues)
            total_pred.extend(preds)

    # Compute reward
    f1 = f1_score(total_true, total_pred, average='macro')

    return f1


class PrintStepCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # self.locals contains 'rewards', 'infos', etc.
        rewards = self.locals.get('rewards', None)
        infos = self.locals.get('infos', None)[0]['f1']
        if rewards is not None:
            print(f"Step {self.num_timesteps}: reward: {rewards[0]:.4f}, f1: {infos:.4f}")
        return True


def start_ppo(log):
    np.random.seed(log.param.SEED)
    torch.manual_seed(log.param.SEED)
    torch.cuda.manual_seed(log.param.SEED)
    torch.cuda.manual_seed_all(log.param.SEED)

    torch.backends.cudnn.deterministic = True #
    torch.backends.cudnn.benchmark = False #
    
    _,valid_data,test_data = get_dataloader(log.param.train_batch_size,log.param.eval_batch_size,log.param.dataset,w_aug=False,w_double=False,label_list=None, type=log.param.type)

    # Instantiate models and load weights
    model_base = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    model_ner = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type, use_ner=True) # v2
    model_outlier = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    model_hard_negative = primary_encoder_v2_no_pooler_for_con(log.param.hidden_size,log.param.label_size,log.param.model_type) # v2
    
    # load model
    model_base.load_state_dict(torch.load(os.path.join(log.param.base_model, "model.pt")))
    model_ner.load_state_dict(torch.load(os.path.join(log.param.ner_model_dir, "model.pt")))
    model_outlier.load_state_dict(torch.load(os.path.join(log.param.outlier_model_dir, "model.pt")))
    model_hard_negative.load_state_dict(torch.load(os.path.join(log.param.hard_negative_model_dir, "model.pt")))

    model_base.eval()
    model_ner.eval()
    model_outlier.eval()
    model_hard_negative.eval()

    if torch.cuda.is_available():
        model_base.cuda(log.param.gpu_id)
        model_ner.cuda(log.param.gpu_id)
        model_outlier.cuda(log.param.gpu_id)
        model_hard_negative.cuda(log.param.gpu_id)

    models = [model_base, model_ner, model_outlier, model_hard_negative]
    init_weights = np.array([0.25]*4, dtype=np.float32)
    f1_baseline_test = test(test_data, models, init_weights, log.param.gpu_id)
    f1_baseline = test(valid_data, models, init_weights, log.param.gpu_id)
    env = DummyVecEnv([lambda: EnsembleEnv(test_data, tuple(models), f1_baseline=f1_baseline, gpu_id=log.param.gpu_id)])

    # Instantiate PPO agent
    ppo_agent = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        tensorboard_log='./ppo_ensemble_logs'
    )

    # Train
    print("======== Training PPO agent =========")
    # callback = PrintStepCallback()
    ppo_agent.learn(total_timesteps=log.param.time_step)

    obs = env.reset()
    optimal_weights, _ = ppo_agent.predict(obs)
    optimal_weights = normalize_weights(optimal_weights)
    print(f"Optimal Ensemble Weights: {optimal_weights}")

    # Evaluate final performance
    f1_new = test(test_data, models, optimal_weights, log.param.gpu_id)
    # f1_new_valid = test(valid_data, models, optimal_weights, log.param.gpu_id)
    print(f"Test Macro F1: {f1_baseline_test:.4f}")
    # print(f"New Valid Macro F1: {f1_new_valid:.4f}")
    print(f"New Macro F1: {f1_new:.4f}")


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
        
        start_ppo(log)