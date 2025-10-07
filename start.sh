echo "### job start"
echo "###"
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python utils/shared_semantics.py \
 	--cluster_num 20 \                                                          # number of clusters
 	--load_dataset ihc_pure \                                                   # dataset
 	--load_sent_emb_model princeton-nlp/unsup-simcse-bert-base-uncased \        # sentence embedding model
     --module_type m0                                                            # the type of module

python utils/preprocess_dataset.py \
     -e princeton-nlp \                # sentence embedding model
     -d ihc_pure_c20 \                 # preprocessed dataset
     -t bert-base-uncased \            # tokenizer type
     -m m0                             # the type of module

## m0 - m2 
## modify the train_config.py file
python train.py

## m3
## modify the train_hard_negative_config.py file
# python train_hard_negative.py

echo "###"
echo "### END DATE=$(date)"
