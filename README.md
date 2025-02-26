# V-HATE
### V-HATE: Voting-based Implicit Hate Speech Detection
We develop five specialized modules designed to capture diverse dataset-specific characteristics.  
**V-HATE** is a voting-based framework that selects the optimal combination of modules for each dataset to enhance detection performance.  
## Datasets
Datset file route: `./raw_datasets/{dataset_name}/`  
Dataset split: Train, Vaild, Test (8:1:1)  

We use the `IHC`, `SBIC`, `DYNA`, `Hateval` and `Toxigen` datasets.

## Install requirements
```bash
$ pip install -r requirements.txt
```
## Module Setting
Modify the `start.sh` file.
### 1. SharedCon (default)
The baseline code followed the sharedcon repository
https://github.com/hsannn/sharedcon
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
### 2. Using NER
```sh
python shared_semantics.py \
    ...
    --use_ner True
    ...

python preprocess_dataset.py \
    ...
    -n True
```
### 3. Remove Outlier
```sh
python shared_semantics.py \
    ...
    --threshold True
    ...
```
### 4. Using Cosine Similarity
```sh
python shared_semantics.py \
    ...
    --center_type cosine
    ...
```
### 5. Using Hard Negative Samples
The hard negative code followed the LAHN repository
https://github.com/Hanyang-HCC-Lab/LAHN
```sh
python train_hard_negative.py
```

## Train
### 1. Modify the `shart.sh` file
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

### 2. Modify the config file
Modify `train_config.py` or `train_config_lahn.py` file.


### 3. Start train
```bash
$ chmod +x shart.sh 
$ ./start.sh
```

## Evaluation
### 1. Modify the `eval_config.py` file
Set the model paths.
- base_model
- ner_model_dir
- cosine_model_dir
- outlier_model_dir
- hard_negative_model_dir

### 2. Test start
```bash
$ chmod +x eval.sh 
$ ./eval.sh
```