# Spectral Clustering

Project for CSCI 4971 Large Scale Matrix Computation and Machine Learning exploring spectral
clustering algorithms on very large data sets. Specifically, I examine the algorithm presented in
"Time and Space Efficient Spectral Clustering via Column Sampling" by Li *et al.*

Get started by running `make data` to download the data sets used in this experiment.

You can then run the benchmarking python script, `src\benchmark.py`:
```shell
$ python src\benchmark.py -h
usage: Benchmark spectral clustering algorithms [-h]
                                                [--subset [SUBSET [SUBSET ...]]]
                                                [--iterations ITERATIONS]
                                                [--columns COLUMNS]
                                                [--gamma GAMMA]
                                                d a [a ...]

positional arguments:
  d                     data set to use in benchmarking
  a                     algorithms to run

optional arguments:
  -h, --help            show this help message and exit
  --subset [SUBSET [SUBSET ...]], -s [SUBSET [SUBSET ...]]
                        use only a subset of classes from the data set
  --iterations ITERATIONS, -i ITERATIONS
                        number of iterations to average over
  --columns COLUMNS, -m COLUMNS
                        number of columns to sample
  --gamma GAMMA, -g GAMMA
                        KASP data reduction ratio
```