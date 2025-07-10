

export CUDA_VISIBLE_DEVICES=1


# cd /home/sysmanager/customer/Desktop/ZZ/anomaly/CostFilterAD/Costfilter_hvqtrans/Costfilter_HVQ_btad/experiments/MVTec-AD


PYTHONPATH=$PYTHONPATH:../../ python -u ../../tools/train_val_withOT_btad.py -e --lamda 0.02
# nohup bash ./eva_btad.sh > test_costfilter_hvqtrans_btad.log 2>&1 &

