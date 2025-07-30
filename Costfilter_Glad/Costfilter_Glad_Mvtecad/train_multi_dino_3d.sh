export INSTANCE_DIR='your_path/anomaly/CostFilterAD/Costfilter_Glad/Costfilter_Glad_Mvtecad'     # VisA   MPDD   MVTec-AD  PCBBank
export OUTPUT_DIR="/model/MVTec_AD_dino_multi/MVTec-AD_20000step_bs32_eps_anomaly2_multiclass_0"
export SEED=1234


python main_multi_glad_dino_3d.py \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=256 \
    --test_batch_size=1 \
    --pre_compute_text_embeddings \
    --seed=$SEED
 


# cd your_path/anomaly/CostFilterAD/Costfilter_Glad/Costfilter_Glad_Mvtecad

# nohup bash train_multi_dino_3d.sh > train_costfilter_glad_mvtecad.log 2>&1 &


