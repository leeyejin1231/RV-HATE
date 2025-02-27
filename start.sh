echo "### job start"
echo "###"
echo "### START DATE=$(date)"
echo "### HOSTNAME=$(hostname)"
echo "### CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python utils/shared_semantics.py \
 	--cluster_num 75 \
 	--load_dataset ihc_pure \
 	--load_sent_emb_model ssgyejin/ihc-sim2 \
   --center_type euclidean \
 	--cluster_togather False \
   --threshold False \
   --use_ner False

python utils/preprocess_dataset.py \
     -m ssgyejin/ihc-sim2 \
     -d ihc_pure_c75 \
     -t bert-base-uncased \
     -n False

# modify the train_config.py file
python train.py
# modify the train_hard_negative_config.py file
# python train_hard_negative.py

echo "###"
echo "### END DATE=$(date)"

EOF
