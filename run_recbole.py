import argparse
from recbole_gnn.quick_start import run_recbole_gnn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', type=str, default='rec', help='mode of training')
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--lr', '-l', type=str, default=0.001, help='learing rate')
    parser.add_argument('--weight_decay', '-w', type=str, default=1e-6, help='weight_decay')
    parser.add_argument('--gpu', '-g', type=str, default='0', help='gpu id')
    parser.add_argument('--suffix', '-s', type=str, default=None, help='log suffix')
    parser.add_argument('--config_files', type=str, default='config.yaml', help='config files')
    parser.add_argument('--tem', '-t', type=str, default=None, help='temperature')
    parser.add_argument('--mask_mode', type=str, default='all', choices=['all', 'diagonal', 'none'], help='masking mode for false negatives')
    parser.add_argument('--pop_beta', type=float, default=0.0, help='item popularity negative logQ correction coefficient (beta)')
    parser.add_argument('--pop_alpha', type=float, default=0.0, help='user popularity negative logQ correction coefficient (alpha)')
    parser.add_argument('--cross_weight', type=float, default=0.0, help='weight of cross-domain user-item negative contrast')
    parser.add_argument('--layer_weight_mode', type=str, default='mean', choices=['mean', 'learnable', 'decay'], help='GCN layer weighting mode')
    parser.add_argument('--epochs', type=int, default=None, help='number of training epochs')
    parser.add_argument('--resume_file', type=str, default=None, help='checkpoint path to resume training from')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    config_dict = {
        'mask_mode': args.mask_mode,
        'pop_beta': args.pop_beta,
        'pop_alpha': args.pop_alpha,
        'cross_weight': args.cross_weight,
        'layer_weight_mode': args.layer_weight_mode,
    }
    if args.seed is not None:
        config_dict['seed'] = args.seed
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.resume_file:
        config_dict['resume_file'] = args.resume_file

    run_recbole_gnn(
        model=args.model,
        dataset=args.dataset,
        gpu=args.gpu,
        suffix=args.suffix,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tem=args.tem,
        config_file_list=config_file_list,
        config_dict=config_dict
    )