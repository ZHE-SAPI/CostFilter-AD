export INSTANCE_DIR='/home/ZZ/anomaly/GLAD-main/hdd/Datasets/VisA_pytorch'     # VisA   MPDD   MVTec-AD  PCBBank
export OUTPUT_DIR="/model/Multi-class_VisA/VisA_20000step_bs32_eps_anomaly2_multiclass_0"
export SEED=15

# # 生成 latent 等data_dict
# accelerate launch main_multi.py \
#     --instance_data_dir=$INSTANCE_DIR \
#     --output_dir=$OUTPUT_DIR \
#     --resolution=256 \
#     --test_batch_size=1 \
#     --pre_compute_text_embeddings \
#     --seed=$SEED


# # 正常test
python main_multi_glad.py \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=256 \
    --test_batch_size=1 \
    --pre_compute_text_embeddings \
    --seed=$SEED

# # 训练test_volume
# # CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 
# python main_multi_cost2.py \
#     --instance_data_dir=$INSTANCE_DIR \
#     --output_dir=$OUTPUT_DIR \
#     --resolution=256 \
#     --test_batch_size=1 \
#     --pre_compute_text_embeddings \
#     --seed=$SEED

# cd /home/ZZ/anomaly/GLAD-main/2d_batch_1_shengchengshuju/GLAD-VISA
# nohup bash test_multi.sh > train_save_pt_14.log 2>&1 &