# V-HATE: Voting-based Implicit Hate Speech Detection
## About V-HATE
**V-HATE** is a voting-based framework that selects the optimal combination of modules for each dataset to enhance implicit hate speech detection performance.
We develop five specialized modules designed to capture diverse dataset-specific characteristics.
</br>
<p align="center">
    <image src='./images/overview.png' width='700px'>
</p>

A key strength of this method is its modular design, where each component is fine-tuned to tackle specific challenges in hate speech detection, such as subtle target mentions, noisy data, and hard-to-detect "hard negatives." It can adapt to the unique characteristics of different datasets. Moreover, by using a voting system that integrates the strengths of all modules, it ensures more reliable and flexible detection results overall.

### ⚙️ Modules
#### M1. Clustering-based Contrastive Learning

We used **[SharedCon](https://github.com/hsannn/sharedcon)** as base model.
Groups similar sentences into clusters and selects the sample nearest the center as the anchor for contrastive learning. This approach helps capture shared semantic cues critical for detecting implicit hate speech.

#### M2. Using Target Special Token with NER Tagger
<image src='./images/M2.png' width='400px'>  

Tags explicit mentions of specific groups (e.g., organizations) with a **[TARGET]** token.
It helps the model distinguish hate speech from offensive but non-hateful remarks.

#### M3. Remove Outliers in a Clustering
<image src='./images/M3.png' width='400px'>  

Identifies and removes outlier sentences (e.g., broken or noisy text) within each cluster.
Reducing such noise enhances the clarity of each cluster’s representation.

#### M4. Using Cosine Similarity
<image src='./images/M4.png' width='400px'>  

Uses cosine similarity, instead of Euclidean distance, when selecting the cluster center.
Focusing on vector angles ensures more semantically coherent anchor selection.

#### M5. Contrastive Learning with Hard Negative Samples
<image src='./images/M5.png' width='400px'>  

Adds “hard negatives” that have high similarity yet different labels, tightening the model’s decision boundary.
This helps the model better differentiate subtle hate from non-hate content.


## 🛠️ Getting Start
### 📚 Datasets
Dataset file route: `./raw_datasets/{dataset_name}/`  
Dataset split: Train, Vaild, Test (8:1:1)  

We used the `IHC`, `SBIC`, `DYNA`, `Hateval` and `Toxigen` datasets.

### Install requirements
```bash
$ pip install -r requirements.txt
```
### Module Setting
Modify the `start.sh` file.
#### 1. SharedCon (default)
```sh
python shared_semantics.py \
    --cluster_num {num_of_clusters} \
    --load_dataset {dataset name} \
    --load_sent_emb_model {embedding model name} \
    --center_type euclidean \
    --threshold False
    --use_ner False
    
python preprocess_dataset.py \
    -m {embedding model name} \
    -d {data_name} \
    -t bert-base-uncased
    -n False
    ...

```
#### 2. Using NER
```sh
python shared_semantics.py \
    ...
    --use_ner True
    ...

python preprocess_dataset.py \
    ...
    -n True
```
#### 3. Remove Outlier
```sh
python shared_semantics.py \
    ...
    --threshold True
    ...
```
#### 4. Using Cosine Similarity
```sh
python shared_semantics.py \
    ...
    --center_type cosine
    ...
```
#### 5. Using Hard Negative Samples
```sh
python train_hard_negative.py
```

### Train
#### 1. Modify the `shart.sh` file
```sh
python shared_semantics.py \
    --cluster_num 20 \
    --load_dataset toxigen \
    --load_sent_emb_model toxi-sim \
    --center_type euclidean \
    --threshold False
    --use_ner False

python preprocess_dataset.py \
    -m toxi-sim \
    -d toxigen_c20 \
    -t bert-base-uncased
    -n False

python train.py
## python train_hard_negative.py

EOF
```
- `cluster_nun` = number of clusters  
- `laod_dataset` = dataset name  
- `load_sent_emb_model` = sentence imbedding model name  
- `center_type_euclidean` = `cosine` or `euclidean`  
- `threshold` = `True` or `False` (remove outliers)
- `use_ner` = `True` or `False`

#### 2. Modify the config file
Modify `train_config.py` or `train_hard_negative_config.py` file.


#### 3. Start train
```bash
$ chmod +x shart.sh 
$ ./start.sh
```

### Evaluation
#### 1. Modify the `eval_config.py` file
Set the model paths.
- base_model
- ner_model_dir
- cosine_model_dir
- outlier_model_dir
- hard_negative_model_dir

#### 2. Test start
```bash
$ chmod +x eval.sh 
$ ./eval.sh
```


## Acknowlegement
Our code is based on the code from https://github.com/hsannn/sharedcon.  
Also, hard negative smaple follows the code from https://github.com/Hanyang-HCC-Lab/LAHN.
