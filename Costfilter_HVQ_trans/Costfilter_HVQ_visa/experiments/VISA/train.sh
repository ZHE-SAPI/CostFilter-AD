# PYTHONPATH=$PYTHONPATH:../../ \
# srun --mpi=pmi2 -p$2 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --cpus-per-task=4 --job-name=mvtec \
# python -u ../../tools/train_val.py





PYTHONPATH=$PYTHONPATH:../../  python -u ../../tools/train_val_woOT.py

# cd /home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_visa/experiments/VISA
 # nohup bash /home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_visa/experiments/VISA/train.sh > trans_val_hvqtran_my_visa.log 2>&1 &
# nohup bash ./train.sh > ./train_val_hvqtran_my_visa.log 2>&1 &

