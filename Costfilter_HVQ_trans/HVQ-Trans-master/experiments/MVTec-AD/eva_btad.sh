# PYTHONPATH=$PYTHONPATH:../../ \
# srun --mpi=pmi2 -p$2 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --cpus-per-task=4 --job-name=mvtec \
# python -u ../../tools/train_val.py -e


PYTHONPATH=$PYTHONPATH:../../ python -u ../../tools/train_val_withOT_btad.py -e 
# cd /home/ZZ/anomaly/HVQ-Trans-master/experiments/MVTec-AD

# nohup bash /home/ZZ/anomaly/HVQ-Trans-master/experiments/MVTec-AD/eva_btad.sh > test_val_btad_withOT.log 2>&1 &
