export CUDA_VISIBLE_DEVICES=0
DTU_TESTPATH="/space0/jiangjf/Data/dtu_test/"
DTU_TESTLIST="lists/dtu/test.txt"
DTU_CKPT_FILE='checkpoints/dtu_train/dtu_best.ckpt' # dtu pretrained model

exp=$1
PY_ARGS=${@:2}

DTU_LOG_DIR="outputs/dtu_pcd"$exp
if [ ! -d $DTU_LOG_DIR ]; then
    mkdir -p $DTU_LOG_DIRm
fi
DTU_OUT_DIR="outputs/dtu_pcd"$exp
if [ ! -d $DTU_OUT_DIR ]; then
    mkdir -p $DTU_OUT_DIR
fi

python test_dtu_pcd.py --dataset=general_eval_vit --batch_size=1 --testpath=$DTU_TESTPATH  --testlist=$DTU_TESTLIST --loadckpt $DTU_CKPT_FILE --outdir $DTU_OUT_DIR\
             --thres_view 2 --ndepths 8,8,4,4 --conf 0.6 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $DTU_LOG_DIR/log_test.txt
