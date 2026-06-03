export INSTANCE_DIR='/home/sysmanager/customer/Desktop/ZZ/anomaly/GLAD-main/hdd/Datasets/VisA_pytorch'     # VisA   MPDD   MVTec-AD  PCBBank
export OUTPUT_DIR="/model/Multi-class_VisA/VisA_20000step_bs32_eps_anomaly2_multiclass_0"
export SEED=1234



# # 正常test
python main_multi_glad_dino_gauss_3d_qianghua.py \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=256 \
    --test_batch_size=1 \
    --pre_compute_text_embeddings \
    --seed=$SEED


# cd /home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_Glad/Costfilter_Glad_VisA

# nohup bash test_multi_dino_gauss_3d_qianghua.sh > test_costfilter_glad_visa_again.log 2>&1 &

