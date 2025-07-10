# PYTHONPATH=$PYTHONPATH:../../ \
# srun --mpi=pmi2 -p$2 -n$1 --gres=gpu:$1 --ntasks-per-node=$1 --cpus-per-task=4 --job-name=mvtec \
# python -u ../../tools/train_val.py -e


PYTHONPATH=$PYTHONPATH:../../ python -u ../../tools/train_val_woOT.py -e 


# cd /home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_visa/experiments/VISA
# nohup bash ./eval.sh > ./test_val_hvqtran_my_visa.log 2>&1 &
