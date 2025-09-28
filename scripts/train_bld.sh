export CUDA_VISIBLE_DEVICES=0,1,2,3
BLD_TRAINING="/space0/jiangjf/Data/low_res/"
BLD_TRAINLIST="lists/blendedmvs/train.txt"
BLD_TESTLIST="lists/blendedmvs/val.txt"
BLD_CKPT_FILE="checkpoints/dtu_train/dtu_best.ckpt"  # dtu pretrained model

exp=$1
PY_ARGS=${@:2}

BLD_LOG_DIR="./checkpoints/bld_ft"$exp
if [ ! -d $BLD_LOG_DIR ]; then
    mkdir -p $BLD_LOG_DIR
fi

python -m torch.distributed.launch --master_port 12345 --nproc_per_node=4 train_bld.py --logdir $BLD_LOG_DIR --dataset=blendedmvs --batch_size=4 --training_views 9 --trainpath=$BLD_TRAINING --summary_freq 100 --loadckpt $BLD_CKPT_FILE\
        --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5 --lr 0.001 --wd 0.0001 --lrepochs "6,8,10,12:2" --attn_temp 2 --trainlist $BLD_TRAINLIST --testlist $BLD_TESTLIST  $PY_ARGS | tee -a $BLD_LOG_DIR/log.txt
