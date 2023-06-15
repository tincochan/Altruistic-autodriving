for i in 1 2 10 20
do
    python3 train.py --device 2 --epoch 40 --data real-n --sampling xgb --train_from 20140101 --test_from 20170101 --test_length 365 --valid_length 90 --initial_inspection_rate ${i} --lr 0.05 --pos_weight 50 --pretrainstep 2 --sample random;
    python3 train.py --dim 16 --device 2 --epoch 40 --data real-c --sampling xgb --train_from 20160101 --test_from 20190101 --test_length 365 --valid_length 90 --initial_inspection_rate ${i} --lr 0.01 --pos_weight 1;
    python3 train.py --device 2 --epoch 40 --data real-t --sampling xgb --train_from 20170101 --test_from 20190101 --test_length 365 --valid_length 90 --initial_inspection_rate ${i} --lr 0.004 --pos_weight 20 --sample random;
done