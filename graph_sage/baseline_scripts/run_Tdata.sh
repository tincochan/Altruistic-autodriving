for rate in 5 10 20 30 40 50 60 70 80 90
do 
   python3 baseline_xgb.py --epoch 40 --data real-t --sampling xgb --train_from 20180101 --test_from 20190101 --test_length 365 --valid_length 90 --initial_inspection_rate $rate --lr 0.004 --pos_weight 1 --sample random
done

