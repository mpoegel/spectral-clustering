import argparse
import numpy as np
from cssp import CSSP


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Benchmark spectral clustering algorithms')
    parser.add_argument('dataset', metavar='d', type=str,
                        help='data set to use in benchmarking')
    parser.add_argument('algorithms', metavar='a', type=str, nargs='+', choices=['cssp'],
                        help='algorithms to run')
    parser.add_argument('--subset', '-s', type=int, nargs='*',
                        help='use only a subset of classes from the data set')
    parser.add_argument('--iterations', '-i', type=int, default=10,
                        help='number of iterations to average over')
    parser.add_argument('--columns', '-m', type=int, default=1000,
                        help='number of columns to sample')
    args = parser.parse_args()

    A = np.loadtxt(args.dataset, delimiter=',')
    if args.subset:
        inds = np.ma.masked_where(np.logical_or.reduce([A[:,-1] == v for v in args.subset]),
                                  np.empty(A.shape[0])).mask
        X = A[inds, :-2]
        Y = A[inds, -1].astype(int)
    else:
        X = A[:, :-2]
        Y = A[:, -1].astype(int)
    k = len(np.unique(Y))

    print()
    print('Number of data: {0}'.format(len(X)))
    print('k = {0}'.format(k))
    print('iterations = {0}'.format(args.iterations))
    print()

    models = dict()
    for algorithm in args.algorithms:
        for i in range(args.iterations):
            if algorithm == 'cssp':
                m = args.columns
                model = CSSP(k, m)
            model.fit(X)
            if not algorithm in models:
                models[algorithm] = []
            models[algorithm].append(model)

    max_algo_name = max([len(algo) for algo in models.keys()])
    print('Algorithm |  Time  | Accuracy')
    print('----------+--------+----------')
    for name, model in models.items():
        t = sum([m.time for m in model]) / args.iterations
        acc = sum([m.accuracy(Y) for m in model]) / args.iterations
        print('{:<{nl}} | {:<6} | {:<6}'.format(name, round(t, 3), round(acc, 3), nl=9))
