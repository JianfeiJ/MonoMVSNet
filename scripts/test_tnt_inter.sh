export CUDA_VISIBLE_DEVICES=0
TNT_TESTPATH="/space0/jiangjf/Data/tanksandtemples_1/intermediate/"
TNT_TESTLIST="lists/tnt/inter.txt"
TNT_CKPT_FILE="checkpoints/bld_ft/bld_best.ckpt" # fine-tuned model


exp=$1
PY_ARGS=${@:2}

TNT_LOG_DIR="outputs/tnt/"$exp
if [ ! -d $TNT_LOG_DIR ]; then
    mkdir -p $TNT_LOG_DIR
fi

TNT_OUT_DIR="outputs/tnt/"$exp
if [ ! -d $TNT_OUT_DIR ]; then
    mkdir -p $TNT_OUT_DIR
fi


python test_dypcd_tnt_inter.py --dataset=tanks --batch_size=1 --testpath=$TNT_TESTPATH  --testlist=$TNT_TESTLIST --loadckpt $TNT_CKPT_FILE --interval_scale 1.06 --outdir $TNT_OUT_DIR\
        --ndepths 8,8,4,4 --depth_inter_r 0.5,0.5,0.5,0.5  --num_view=21 --num_worker 2 --group_cor --attn_temp 2 --inverse_depth $PY_ARGS | tee -a $TNT_LOG_DIR/log_test.txt