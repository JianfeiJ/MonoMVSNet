export CUDA_VISIBLE_DEVICES=0,1,2,3
DTU_TRAINING="/space0/jiangjf/Data/DTU_Dataset/dtu_training/"
DTU_TRAINLIST="lists/dtu/train.txt"
DTU_TESTLIST="lists/dtu/test.txt"

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="./checkpoints/dtu"$exp
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIR
fi

python -m torch.distributed.launch --master_port 1412 --nproc_per_node=4 train_dtu.py --logdir $DTU_LOG_DIR --dataset=dtu_yao_vit --batch_size=4 --trainpath=$DTU_TRAINING --summary_freq 100 \
        --ndepths 8,8,4,4 --attn_temp 2 --wd 0.0001 --lr 0.001 --epochs 15 --lrepochs "10,12,14:2" --trainlist $DTU_TRAINLIST --testlist $DTU_TESTLIST $PY_ARGS | tee -a $DTU_LOG_DIR/log.txt
