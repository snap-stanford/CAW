import argparse
import sys


def get_args():
    parser = argparse.ArgumentParser('Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')

    # select dataset and training mode
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit',
                        choices=['wikipedia', 'reddit', 'socialevolve', 'uci', 'enron', 'socialevolve_1month', 'socialevolve_2weeks'],
                        default='wikipedia')
    parser.add_argument('--data_usage', default=1.0, type=float, help='fraction of data to use (0-1)')
    parser.add_argument('-m', '--mode', type=str, default='t', choices=['t', 'i'], help='transductive (t) or inductive (i)')

    # method-related hyper-parameters
    parser.add_argument('--n_degree', nargs='*', default=['64', '1'],
                        help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--bias', default=0.0, type=float, help='the hyperparameter alpha controlling sampling preference in recent time, default to 0 which is uniform sampling')
    parser.add_argument('--agg', type=str, default='walk', choices=['tree', 'walk'],
                        help='tree based hierarchical aggregation or walk-based flat lstm aggregation, we only use the default here')
    parser.add_argument('--pos_enc', type=str, default='lp', choices=['spd', 'lp', 'saw'], help='way to encode distances, shortest-path distance or landing probabilities, or self-based anonymous walk (baseline)')
    parser.add_argument('--pos_dim', type=int, default=172, help='dimension of the positional embedding')
    parser.add_argument('--pos_sample', type=str, default='binary', choices=['multinomial', 'binary'], help='two equivalent sampling method with empirically different running time')
    parser.add_argument('--walk_pool', type=str, default='attn', choices=['attn', 'sum'], help='how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other walk_ arguments')
    parser.add_argument('--walk_n_head', type=int, default=8, help="number of heads to use for walk attention")
    parser.add_argument('--walk_mutual', action='store_true', help="whether to do mutual query for source and target node random walks")
    parser.add_argument('--walk_linear_out', action='store_true', default=False, help="whether to linearly project each node's embedding")

    parser.add_argument('--attn_agg_method', type=str, default='attn', choices=['attn', 'lstm', 'mean'], help='local aggregation method, we only use the default here')
    parser.add_argument('--attn_mode', type=str, default='prod', choices=['prod', 'map'],
                        help='use dot product attention or mapping based, we only use the default here')
    parser.add_argument('--attn_n_head', type=int, default=2, help='number of heads used in tree-shaped attention layer, we only use the default here')
    parser.add_argument('--time', type=str, default='time', choices=['time', 'pos', 'empty'], help='how to use time information, we only use the default here')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=1e-3,
                        help='tolerated marginal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--cpu_cores', type=int, default=1, help='number of cpu_cores used for position encoding')
    parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv