import os
import argparse
from turtle import clear
from torch.backends import cudnn
from utils.utils import *
from solver import Solver


def str2bool(v):
    return v.lower() in 'true'


def main(config):
    # random seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=6)
    parser.add_argument('--input_c', type=int, default=55)  # 需要改为节点数
    parser.add_argument('--output_c', type=int, default=55)  # 需要改为节点数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='MSL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/processed/MSL/')
    parser.add_argument('--model_save_path', type=str, default='hy_checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)  # 冗余
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--alpha", type=float, default=0.01, choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
                        help="Predict weight")
    parser.add_argument("--beta", type=float, default=0.5, choices=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1],
                        help="DRscore weight")
    parser.add_argument("--seed", type=int, default=1, choices=[1, 2, 3, 4, 5], help="random seed")
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 2, 3, 4, 5], help="predict horizon")
    # ==========hy================
    parser.add_argument('-gat_inter_dim', help='dimension', type=int, default=128)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=64)
    parser.add_argument('-decay', help='decay', type=float, default=0)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    for i in range(1):
        config.seed = 3
        main(config)
