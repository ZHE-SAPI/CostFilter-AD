export INSTANCE_DIR='/home/customer/Desktop/ZZ/anomaly/GLAD-main/hdd/Datasets/MVTec-AD'     # VisA   MPDD   MVTec-AD  PCBBank
export OUTPUT_DIR="/model/MVTec_AD_dino_multi/MVTec-AD_20000step_bs32_eps_anomaly2_multiclass_0"
export SEED=1234

# # 生成 latent 等data_dict
# accelerate launch main_multi.py \
#     --instance_data_dir=$INSTANCE_DIR \
#     --output_dir=$OUTPUT_DIR \
#     --resolution=256 \
#     --test_batch_size=1 \
#     --pre_compute_text_embeddings \
#     --seed=$SEED


# # 正常test
python draw_logits_distribution.py \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=256 \
    --test_batch_size=1 \
    --pre_compute_text_embeddings \
    --seed=$SEED


# cd /home/customer/Desktop/ZZ/anomaly/GLAD-main
# bash draw_logits_dis.sh 

