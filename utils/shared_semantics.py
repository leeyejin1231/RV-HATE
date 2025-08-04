import pandas as pd
import os
import numpy as np
import random
import argparse
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
sys.path.append("/home/yejin/contents/V-HATE/")

from simcse import SimCSE
import torch
from utils import Clustering

device = torch.device('cuda')

np.random.seed(0)
random.seed(0)


# given a dataset, compute clustering and select the closest data from each cluster
def cluster_n_select(dataset, sent_emb_model, input_col, args, remove_outlier=False, is_not_hate=None, use_ner=False):
    
    if use_ner:
        simcse = AutoTokenizer.from_pretrained(f"{sent_emb_model}", device=device)
        new_tokens = ["[TARGET]"]
        simcse.add_tokens(new_tokens)
    else:
        simcse = SimCSE(f"{sent_emb_model}", device=device)
    
    clustering = Clustering(int(args.cluster_num), dataset, input_col, simcse, use_ner)

    print("======cosine=======")
    cluster_center_post = clustering.get_center_post_idx_with_cosine(k=1)
    ## use euclidean distance
    # closest_data = clustering.get_center_post_idx_with_uclidean()

    m_clusters = clustering.get_cluster_number()

    ### remove outlier
    if remove_outlier:
        before_dataset = len(dataset)
        dataset = clustering.remove_outlier_data()
        m_clusters = dataset['cluster_label'].tolist()
        dataset = dataset.drop(columns=['cluster_label'])
        print("=============================================================")
        print(f"=== before_data: {before_dataset}, after_data: {len(dataset)}, ({((before_dataset-(len(dataset)))/before_dataset)*100:.2f}% removed)")

    centroid_sample = [dataset[input_col][cluster_center_post[i]:cluster_center_post[i]+1].values[0] for i in m_clusters]
    
    ## use euclidean distance
    # centroid_sample = [dataset[input_col][closest_data[i]] for i in m_clusters]

    # ## distinguish the non-hate label
    if is_not_hate:
        m_clusters = [clu_num + 1000 for clu_num in m_clusters]
    
    dataset['cluster'] = m_clusters
    dataset['centroid_sample'] = centroid_sample
    
    return dataset

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_num', default=20,type=int, help='Enter the number of cluster')
    parser.add_argument('--load_dataset', default="ihc_pure",type=str, help='Enter the path of the dataset')
    parser.add_argument('--load_sent_emb_model', default="princeton-nlp/unsup-simcse-bert-base-uncased",type=str, help='Enter the path/type of the sentence embedding model')
    parser.add_argument('--module_type', default="m2",type=str, help='Enter the type of module')
    args = parser.parse_args() 

    if args.module_type == "m0":
        remove_outlier = False
        use_ner = False
    elif args.module_type == "m1":
        remove_outlier = False
        use_ner = True
    elif args.module_type == "m2":
        remove_outlier = True
        use_ner = False
    else:
        raise NotImplementedError


    # load raw dataset
    if args.load_dataset == "ihc_pure":
        train_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'train.tsv'), delimiter='\t', header=0)
        valid_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'valid.tsv'), delimiter='\t', header=0)
        test_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'test.tsv'), delimiter='\t', header=0)
        input_col = 'post'
        class_col = 'class'
        hate_class = "implicit_hate"
        not_hate_class = "not_hate"
    elif args.load_dataset == "sbic":
        train_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'dev.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'test.csv'), delimiter=',', header=0)
        input_col = 'post'
        class_col = 'offensiveLABEL'
        hate_class = "offensive"
        not_hate_class = "not_offensive"
    elif args.load_dataset == "dynahate":
        train_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'dev.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'test.csv'), delimiter=',', header=0)
        input_col = 'text'
        class_col = 'label'
        hate_class = "hate"
        not_hate_class = "nothate"
    elif args.load_dataset == "hateval" or args.load_dataset == "toxigen" or args.load_dataset == "white" or args.load_dataset == "union":
        train_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'train.csv'), delimiter=',', header=0)
        valid_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'valid.csv'), delimiter=',', header=0)
        test_dataset = pd.read_csv(os.path.join('./raw_dataset', args.load_dataset, 'test.csv'), delimiter=',', header=0)
        input_col = 'post'
        class_col = 'label'
        hate_class = 1
        not_hate_class = 0
    else:
        raise NotImplementedError
    
    # processing each classes
    print("processing each classes...")    

    ## 1) hate class
    mask_implicit_hate1 = train_dataset[class_col] == hate_class
    implicit_hate1 = train_dataset.loc[mask_implicit_hate1,:]
    implicit_hate1 = implicit_hate1.reset_index(drop=True)

    mask_implicit_hate2 = valid_dataset[class_col] == hate_class
    implicit_hate2 = valid_dataset.loc[mask_implicit_hate2,:]
    implicit_hate2 = implicit_hate2.reset_index(drop=True) 

    mask_implicit_hate3 = test_dataset[class_col] == hate_class
    implicit_hate3 = test_dataset.loc[mask_implicit_hate3,:]
    implicit_hate3 = implicit_hate3.reset_index(drop=True)

    ### cluster samples and and select the closest sample to the centroid per cluster
    implicit_hate1 = cluster_n_select(implicit_hate1, args.load_sent_emb_model, input_col, args=args, remove_outlier=remove_outlier, use_ner=use_ner)
    implicit_hate2 = cluster_n_select(implicit_hate2, args.load_sent_emb_model, input_col, args=args, use_ner=use_ner)
    implicit_hate3 = cluster_n_select(implicit_hate3, args.load_sent_emb_model, input_col, args=args, use_ner=use_ner)
    print(f"class: implicit_hate DONE")
    
    ## 2) not_hate
    mask_not_hate1 = train_dataset[class_col] == not_hate_class
    not_hate1 = train_dataset.loc[mask_not_hate1,:]
    not_hate1 = not_hate1.reset_index(drop=True)

    mask_not_hate2 = valid_dataset[class_col] == not_hate_class
    not_hate2 = valid_dataset.loc[mask_not_hate2,:]
    not_hate2 = not_hate2.reset_index(drop=True)

    mask_not_hate3 = test_dataset[class_col] == not_hate_class
    not_hate3 = test_dataset.loc[mask_not_hate3,:]
    not_hate3 = not_hate3.reset_index(drop=True)

    ### cluster samples and and select the closest sample to the centroid per cluster
    not_hate1 = cluster_n_select(not_hate1, args.load_sent_emb_model, input_col, args=args, remove_outlier=remove_outlier, is_not_hate=True, use_ner=use_ner)
    not_hate2 = cluster_n_select(not_hate2, args.load_sent_emb_model, input_col, args=args, is_not_hate=True, use_ner=use_ner)
    not_hate3 = cluster_n_select(not_hate3, args.load_sent_emb_model, input_col, args=args, is_not_hate=True, use_ner=use_ner)
    print(f"class: not_hate DONE")
    
    
    # concat the samples of each class
    total_train_dataset = pd.concat([implicit_hate1, not_hate1])    
    total_train_dataset = total_train_dataset.sample(frac=1).reset_index(drop=True)

    total_valid_dataset = pd.concat([implicit_hate2, not_hate2])    
    total_valid_dataset = total_valid_dataset.sample(frac=1).reset_index(drop=True)

    total_test_dataset = pd.concat([implicit_hate3, not_hate3])    
    total_test_dataset = total_test_dataset.sample(frac=1).reset_index(drop=True)
    
    print(f"class: implicit_hate DONE")
    
    # save the dataset
    if "princeton" in args.load_sent_emb_model:
        model_name = "princeton-nlp"
    else:
        model_name = args.load_sent_emb_model
    os.makedirs(f"./clustered_dataset/{model_name}/{args.load_dataset}_c{args.cluster_num}_{args.module_type}", exist_ok=True)
    if args.load_dataset == "ihc_pure":
        total_train_dataset.to_csv(os.path.join(f"./clustered_dataset/{model_name}/{args.load_dataset}_c{args.cluster_num}_{args.module_type}", "train.tsv"), sep="\t", index=False)
        total_valid_dataset.to_csv(os.path.join(f"./clustered_dataset/{model_name}/{args.load_dataset}_c{args.cluster_num}_{args.module_type}", "valid.tsv"), sep="\t", index=False)
        total_test_dataset.to_csv(os.path.join(f"./clustered_dataset/{model_name}/{args.load_dataset}_c{args.cluster_num}_{args.module_type}", "test.tsv"), sep="\t", index=False)
    else:
        total_train_dataset.to_csv(os.path.join(f"./clustered_dataset/{model_name}/{args.load_dataset}_c{args.cluster_num}_{args.module_type}", "train.csv"), sep=",", index=False)
        total_valid_dataset.to_csv(os.path.join(f"./clustered_dataset/{model_name}/{args.load_dataset}_c{args.cluster_num}_{args.module_type}", "valid.csv"), sep=",", index=False)
        total_test_dataset.to_csv(os.path.join(f"./clustered_dataset/{model_name}/{args.load_dataset}_c{args.cluster_num}_{args.module_type}", "test.csv"), sep=",", index=False)
