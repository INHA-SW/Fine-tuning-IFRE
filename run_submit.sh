ModelType=bert
#nodropPrototype-dropRelation-lr-1e-5
#dropPrototype-nodropRelation-lr-2e-5
#nodropPrototype-nodropRelation-lr-1e-5
#acl-camera-ready-$N-$K.pth.tar

N=5
K=1

python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --model proto --encoder bert --hidden_size 768 --val_step 1000 --test test_wiki_input-$N-$K \
    --batch_size 1 --test_online \
    --load_ckpt ./checkpoint/$ModelType/camery-ready-CP-$N-$K.pth.tar \
    --pretrain_ckpt ./bert-base-uncased \
    --test_output ./submit/$ModelType/pred-$N-$K.json \
    --cat_entity_rep \
    --backend_model bert
