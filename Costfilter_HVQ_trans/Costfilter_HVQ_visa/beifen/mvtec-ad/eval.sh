# PYTHONPATH=$PYTHONPATH:../../ \
# srun --mpi=pmi2 -p$2 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --cpus-per-task=4 --job-name=mvtec \
# python -u ../../tools/train_val.py -e


PYTHONPATH=$PYTHONPATH:../../ \
CUDA_VISIBLE_DEVICES=1 python -u ../../tools/train_val.py -e 


# cd /home/customer/Desktop/ZZ/anomaly/MY_HVQeriment/experiments/MVTec-AD
#  nohup bash /home/customer/Desktop/ZZ/anomaly/HVQ-Trans-master/experiments/MVTec-AD/eval.sh > trans_val_3.log 2>&1 &