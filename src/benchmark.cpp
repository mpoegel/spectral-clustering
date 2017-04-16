#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>
#include <vector>

#include <Eigen/Sparse>

using namespace std;


int lineCount(const char* filename)
{
  int count = 0;
  string line;
  ifstream infile(filename);
  while (getline(infile, line))
    ++count;
  return count;
  infile.close();
}

vector<string> split(const char *str, char c=' ')
{
  vector<string> result;
  do {
    const char *begin = str;
    while (*str != c && *str)
      str++;
    result.push_back(string(begin, str));
  } while (0 != *str++);
  return result;
}

tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::VectorXi> loadSparseMatrix(const char* filename)
{
  // construct a sparse data matrix, X, and a class vector, Y
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  int numRows = lineCount(filename);
  int numCols = 0;
  tripletList.reserve(numRows * 256);
  ifstream infile(filename);
  string line;
  vector<string> bits;
  Eigen::VectorXi Y(numRows);
  int r = 0;
  // read each line of the input file, dynamically counting the number of columns
  while (getline(infile, line)) {
    // each row is of the format: y_i col:val col:val ...
    bits = split(line.c_str());
    Y[r] = atoi(bits[0].c_str());
    for (int c=1; c<bits.size(); c++) {
      vector<string> bit = split(bits[c].c_str(), ':');
      if (bit.size() != 2) {
        continue;
      }
      int col = atoi(bit[0].c_str());
      double val = atof(bit[1].c_str());
      // build the list of triplets used to construct the sparse matrix
      tripletList.push_back(T(r, col-1, val));
      if (col > numCols) {
        numCols = col;
      }
    }
    r++;
  }
  Eigen::SparseMatrix<double, Eigen::RowMajor> X(numRows, numCols);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  infile.close();
  return make_tuple(X, Y);
}

void filter(Eigen::SparseMatrix<double, Eigen::RowMajor> &X, Eigen::VectorXi &Y, const vector<int> &mask)
{
  int count = 0;
  for (int r=0; r<Y.rows(); r++) {
    for (int m=0; m<mask.size(); m++) {
      if (Y[r] == mask[m])
        count++;
    }
  }
  typedef Eigen::Triplet<int> T;
  std::vector<T> tripletList;
  tripletList.reserve(count);
  Eigen::VectorXi Y_prime(count);
  int k = 0;
  for (int r=0; r<Y.rows(); r++) {
    for (int m=0; m<mask.size(); m++) {
      if (Y[r] == mask[m]) {
        tripletList.push_back(T(k, r, 1));
        Y_prime[k] = Y[r];
        k++;
      }
    }
  }
  Eigen::SparseMatrix<double, Eigen::RowMajor> M(count, Y.rows());
  M.setFromTriplets(tripletList.begin(), tripletList.end());
  X = M * X;
  Y = Y_prime;
}

vector<int> vrange(int a, int b)
{
  int n = b - a;
  vector<int> res(n, 0);
  for (int i=0; i<n; i++) {
    res[i] = a + i;
  }
  return res;
}

vector<int> sampleRange(int a, int b, int k) {
  vector<int> res;
  res.reserve(k);
  vector<int> range = vrange(a, b);
  unsigned int seed = chrono::system_clock::now().time_since_epoch().count();
  shuffle(range.begin(), range.end(), default_random_engine(seed));
  for (int i=0; i<k; i++) {
    res[i] = range[i];
  }
  return res;
}

Eigen::VectorXd zeros(int n)
{
  Eigen::VectorXd res(n);
  for (int i=0; i<n; i++) {
    res[i] = 0;
  }
  return res;
}

Eigen::MatrixXd sparseToDense(const Eigen::SparseMatrix<double, Eigen::RowMajor> &X)
{
  int n = X.rows();
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  return I * X;
}

Eigen::VectorXi kMeans(const Eigen::MatrixXd &X, int k)
{
  int n = X.rows();
  int d = X.cols();
  Eigen::MatrixXd centroids(k, d);
  vector<int> centRows = sampleRange(0, n, k);
  for (int i=0; i<k; i++) {
    int r = centRows[i];
    centroids.row(i) = X.row(r);
  }
  Eigen::VectorXi Y_hat(n);
  // map each input to its closest centroid
  for (int i=0; i<n; i++) {
    double min_d = (centroids.row(0) - X.row(i)).norm();
    int min_c = 0;
    for (int j=1; j<k; j++) {
      double d = (centroids.row(j) - X.row(i)).norm();
      if (d < min_d) {
        min_d = d;
        min_c = j;
      }
    }
    Y_hat[i] = min_c;
  }
  Eigen::VectorXi Y_hat_copy(n);
  int num_updates = n;
  int iterations = 0;
  while (true) {
    // map each input to its closest centroid
    for (int i=0; i<n; i++) {
      double min_d = (centroids.row(0) - X.row(i)).norm();
      int min_c = 0;
      for (int j=1; j<k; j++) {
        double d = (centroids.row(j) - X.row(i)).norm();
        if (d < min_d) {
          min_d = d;
          min_c = j;
        }
      }
      Y_hat_copy[i] = min_c;
    }
    // check stopping condition
    num_updates = 0;
    Eigen::Matrix<bool, Eigen::Dynamic, 1> diff = Y_hat.cwiseEqual(Y_hat_copy);
    for (int i=0; i<n; i++) {
      if (!diff[i]) num_updates++;
    }
    Y_hat = Y_hat_copy;
    // if (num_updates > n * 0.1) {
    if (iterations > 100) {
      break;
    }
    // compute the new centers
    for (int i=0; i<k; i++) {
      Eigen::VectorXd new_mu = zeros(d);
      int count = 0;
      for (int j=0; j<n; j++) {
        if (Y_hat[j] == i) {
          new_mu += X.row(j);
          count++;
        }
      }
      centroids.row(i) = new_mu;
    }
    iterations++;
  }
  cout << "stopped after " << iterations << " iterations" << endl;
  return Y_hat;
}

Eigen::VectorXi CSSP(Eigen::SparseMatrix<double, Eigen::RowMajor> &X, int k, int m)
{
  int n = X.rows();
  Eigen::VectorXi Y_hat(n);
  
  
  return Y_hat;
}


int main(int argc, char* argv[])
{
  cout << "hello world!" << endl;
  string fn = "data/raw/usps";

  Eigen::SparseMatrix<double, Eigen::RowMajor> X;
  Eigen::VectorXi Y;
  tie(X, Y) = loadSparseMatrix(fn.c_str());
  
  vector<int> mask;
  mask.push_back(1);
  mask.push_back(2);
  filter(X, Y, mask);
  // // cout << Y.rows() << endl;
  // CSSP(X, 2, 300);

  Eigen::MatrixXd Xd = sparseToDense(X);
  
  cout << "starting k-means" << endl;
  Eigen::VectorXi Y_hat = kMeans(X, 2);
  Eigen::Matrix<bool, Eigen::Dynamic, 1> diff = Y.cwiseEqual(Y_hat);
  double accuracy = 0.0;
  for (int i=0; i<diff.rows(); i++) {
    if (diff[i]) accuracy++;
  }
  accuracy = (1 - accuracy / Y.rows()) * 100;
  cout << accuracy << endl;

  return 0;
}
