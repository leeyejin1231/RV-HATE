from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import torch
import pandas as pd

device = torch.device('cuda')

class Clustering():
    def __init__(self, n_clusters, dataset, input_col, emb_model, use_ner):
        self.n_clusters = n_clusters
        self.dataset = dataset
        self.post = dataset[input_col].to_list()
        self.use_ner = use_ner

        if use_ner == "True":
            self.tokenized_post = emb_model.batch_encode_plus(self.post, max_length=512, truncation=True, padding='max_length').input_ids
            self.tokenized_post = torch.tensor(self.tokenized_post)
        else:
            self.tokenized_post = emb_model.encode(self.post)

        print(f"--- k={self.n_clusters} clustering ---")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto")
        self.kmeans.fit(self.tokenized_post)
        self.centers = self.kmeans.cluster_centers_
        self.labels = self.kmeans.labels_
    
    def get_cluster_number(self):
        return self.labels
    
    def get_center_post_idx_with_cosine(self, k):
        # calculate cosine similarity between center and each post
        if k > 1:
            print(f"==> top_k is {k}")
        closest_sentences = []
        for c_idx, center in enumerate(self.centers):
            cluster_post = [self.tokenized_post[i].tolist() for i in range(len(self.tokenized_post)) if self.labels[i] == c_idx]
            similarities = cosine_similarity(center.reshape(1, -1), cluster_post).flatten()
            if k > 1:
                closet_sort = np.argsort(similarities)[::-1]
                closet_idx = closet_sort[:k]
            else:
                closet_idx = np.argmax(similarities)
            closest_sentences.append(closet_idx)

        return closest_sentences
    
    def get_center_post_idx_with_uclidean(self):
        # calculate euclidean distance between center and each post
        closest_data = []
        all_data = [i for i in range(len(self.tokenized_post))]
        
        for i in range(self.n_clusters):
            center_vec = self.centers[i]
            center_vec = center_vec.reshape(1, -1) 
            
            data_idx_within_i_cluster = [idx for idx, clu_num in enumerate(self.labels) if clu_num == i]

            one_cluster_tf_matrix = np.zeros((len(data_idx_within_i_cluster) , self.centers.shape[1]))
            for row_num, data_idx in enumerate(data_idx_within_i_cluster):
                one_row = self.tokenized_post[data_idx]
                one_cluster_tf_matrix[row_num] = one_row
            
            closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_tf_matrix)
            closest_idx_in_one_cluster_tf_matrix = closest[0]
            closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
            data_id = all_data[closest_data_row_num]

            closest_data.append(data_id)
        assert len(closest_data) == self.n_clusters

        return closest_data
    
    #  z-score
    # def outliers_modified_z_score(self, ys):
    #     median_y = np.median(ys)
    #     median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    #     modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y1
    #                         for y in ys]
    #     return modified_z_scores
    
    def remove_outliers(self, cluster_num, labels):
        embeddings_1 = self.tokenized_post[labels==cluster_num]
        dataset_1 = self.dataset[self.dataset['cluster_label']==cluster_num].copy()

        distances = np.linalg.norm(embeddings_1 - self.centers[cluster_num], axis=1)
        
        ## IQR
        ## calculate the interquartile range (IQR) of the distances
        quartiles = np.percentile(distances, [25, 75])
        iqr = quartiles[1] - quartiles[0]
        dataset_1['outlier'] = [True if distances[i] < (quartiles[1] + 1.5 * iqr) else False for i in range(len(dataset_1))]

        ## z-score
        # calculate the z-score of the distances
        # distance_mean = np.mean(distances)
        # distances_std = np.std(distances)
        # dataset_1['outlier'] = [False if (distance-distance_mean)/distances_std > 1.96 else True for distance in distances]

        dataset_2 = dataset_1[dataset_1['outlier']].copy()
        dataset_2 = dataset_2.drop(columns=['outlier'])

        return dataset_2
    
    def remove_outlier_data(self):
        self.dataset['cluster_label'] = self.labels
        after_dataset = self.remove_outliers(0, self.labels)

        for i in range(1, self.n_clusters):
            after_data = self.remove_outliers(i, self.labels)
            after_dataset = pd.concat([after_dataset, after_data], axis=0)

        return after_dataset
                
