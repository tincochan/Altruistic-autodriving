import argparse
import time

curr_time = str(round(time.time(),3))

def get_parser():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters related to DATE
    parser.add_argument('--epoch', type=int, default=20, help="Number of epochs for DATE-related models")
    parser.add_argument('--pretrainstep', type=int, default=2, help="Number of epochs for pretraining")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for DATE-related models")
    parser.add_argument('--dim', type=int, default=32, help="Hidden layer dimension")
    parser.add_argument('--lr', type=float, default=0.04, help="learning rate")
    parser.add_argument('--l2', type=float, default=0.0001, help="l2 reg")
    parser.add_argument('--alpha', type=float, default=10, help="Regression loss weight")
    parser.add_argument('--pos_weight', type=float, default=2, help="Positive weight in XGB")
    parser.add_argument('--head_num', type=int, default=4, help="Number of heads for self attention")
    
    # Hyperparameters related to customs selection
    parser.add_argument('--device', type=int, default=0, help='select which device to run, choose gpu number in your devices or cpu') 
    parser.add_argument('--sampling', type=str, default = 'bATE', choices=['random', 'xgb', 'xgb_lr', 'DATE', 'diversity', 'badge', 'bATE', 'upDATE', 'enhanced_bATE', 'hybrid', 'tabnet', 'ssl_ae', 'noupDATE', 'randomupDATE'], help='Sampling strategy')
    parser.add_argument('--sample', type=str, default = 'random', choices=['random', 'imp', 'hs'], help='Sampling strategy')
    parser.add_argument('--initial_inspection_rate', type=float, default=10, help='Initial inspection rate in training data by percentile')
    parser.add_argument('--final_inspection_rate', type=float, default = 5, help='Percentage of test data need to query')
    parser.add_argument('--inspection_plan', type=str, default = 'direct_decay', choices=['direct_decay','linear_decay','fast_linear_decay'], help='Inspection rate decaying option for simulation time')
    parser.add_argument('--train_from', type=str, default = '20130101', help = 'Training period start from (YYYYMMDD)')
    parser.add_argument('--test_from', type=str, default = '20130201', help = 'Testing period start from (YYYYMMDD)')
    parser.add_argument('--test_length', type=int, default=7, help='Single testing period length (e.g., 7)')
    parser.add_argument('--valid_length', type=int, default=7, help='Validation period length (e.g., 7)')
    parser.add_argument('--data', type=str, default='synthetic', choices = ['synthetic', 'real-n', 'real-m', 'real-t', 'real-k', 'real-c'], help = 'Dataset')
    parser.add_argument('--numweeks', type=int, default=50, help='number of test weeks (week if test_length = 7)')
    parser.add_argument('--semi_supervised', type=int, default=0, help='Additionally using uninspected, unlabeled data (1=semi-supervised, 0=fully-supervised)')
    parser.add_argument('--save', type=int, default=0, help='Save intermediary files (1=save, 0=not save)')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for spliting dataset')
    
    return parser