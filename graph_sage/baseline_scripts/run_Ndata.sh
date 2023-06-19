for rate in 5 10 20 30 40 50 60 70 80 90
do 
   python3 baseline_xgb.py --device 2 --epoch 40 --data real-n --sampling xgb --train_from 20150101 --test_from 20170101 --test_length 365 --valid_length 90 --initial_inspection_rate $rate
done
