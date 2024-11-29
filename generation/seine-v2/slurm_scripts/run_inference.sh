srun -p Gvlab-S1 -N1 -n1 --cpus-per-task=8 --quotatype=auto \
    python seine.py \
    --config='./configs/demo.yaml' \
    --demotext="#C C mixes the food in the pot." \
    --demoimage="/mnt/petrelfs/xujilan/newtools/seine-v2/input2/test18_mixthefoodinthepot.png" \
    --demosavepath="result.mp4" \
    --checkpoint="/mnt/hwfile/internvideo/share_data/huangyifei/model_weights/seine/finetune_seine_256p_15ep_60K.pt" \