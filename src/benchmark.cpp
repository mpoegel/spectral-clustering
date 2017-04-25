#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <armadillo>

using namespace std;


arma::uvec CSSC(const arma::mat &X, int k, int m)
{
  arma::wall_clock timer;
  timer.tic();
  int n = X.n_rows;
  arma::uvec inds = arma::linspace<arma::uvec>(0, n-1, n);
  inds = arma::shuffle(inds);
  inds = inds.rows(0, m-1);

  arma::mat Z = X.rows(inds);
  double mu = 0.0;
  for (unsigned int i=0; i<m; i++) {
    for (unsigned int j=0; j<m; j++) {
      mu += pow(arma::norm(Z.row(i) - Z.row(j)), 2);
    }
  }
  mu /= pow(mu, 2);
  mu = 1 / mu;

  arma::mat A_11(m, m);
  for (unsigned int i=0; i<m; i++) {
    for (unsigned int j=i; j<m; j++) {
      double val = exp(-mu * pow(arma::norm(Z.row(i) - Z.row(j)), 2));
      A_11(i, j) = val;
      A_11(j, i) = val;
    }
  }

  arma::vec ww = A_11 * arma::ones<arma::vec>(m);
  arma::mat D_star = arma::diagmat(ww);
  arma::mat D_star_ = arma::diagmat(arma::pow(arma::sqrt(ww), -1));
  arma::mat M_star = D_star_ * A_11 * D_star_;
  // find the eigendecomposition of M_star
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, M_star);
  eigval = eigval.rows(m-k, m-1);
  eigvec = eigvec.cols(m-k, m-1);
  
  arma::mat Lam = arma::diagmat(eigval);
  arma::mat B = D_star_ * eigvec * arma::diagmat(arma::pow(eigval, -1));
  
  arma::mat Q(n, k);
  for (unsigned int i=0; i<n; i++) {
    arma::rowvec a(m);
    for (unsigned int j=0; j<m; j++) {
      a.col(j) = arma::norm(X.row(i) - Z.row(j));
    }
    Q.row(i) = a * B;
  }
  
  arma::vec dd = Q * Lam * Q.t() * arma::ones<arma::vec>(n);
  arma::mat D_hat = arma::diagmat(dd);
  arma::mat U = arma::diagmat(arma::pow(arma::sqrt(dd), -1)) * Q;
  // orthogonalize U
  arma::mat P = U.t() * U;
  arma::vec Sig;
  arma::mat Vp;
  arma::eig_sym(Sig, Vp, P);
  arma::mat Sig_ = arma::diagmat(arma::sqrt(Sig));
  B = Sig_ * Vp.t() * Lam * Vp * Sig_;
  arma::vec Lam_tilde;
  arma::mat V_tilde;
  arma::eig_sym(Lam_tilde, V_tilde, B);
  
  U = U * Vp * arma::diagmat(arma::pow(arma::sqrt(Sig), -1)) * V_tilde;
  // cluster the approximated eigenvectors, U
  arma::mat centroids;
  arma::uvec y_hat(n);  
  bool status = arma::kmeans(centroids, U.t(), k, arma::random_subset, 10, false);
  if (!status) {
    cout << "clustering failed!" << endl;
    return y_hat;
  }
  centroids = centroids.t();
  arma::vec d(k);
  double t = timer.toc();
  cout << "Finished after " << t << "s" << endl;
  for (unsigned int i=0; i<n; i++) {
    for (unsigned int j=0; j<k; j++) {
      d.row(j) = arma::norm(U.row(i) - centroids.row(j));
    }
    y_hat.row(i) = d.index_min();
  }
  return y_hat;
}


double accuracy(const arma::uvec &Y, const arma::uvec &y_hat)
{
  arma::uvec uniques = arma::unique(Y);
  unsigned int k = uniques.n_rows;
  unsigned int n = Y.n_rows;
  vector<unsigned int> perm(k, 0);
  for (unsigned int i=0; i<k; i++) perm[i] = i + 1;
  double res = 0.0;
  do {
    arma::uvec yy(y_hat);
    for (unsigned int i=0; i<k; i++) {
      arma::uvec inds = arma::find(y_hat == i);
      arma::uvec f(inds.n_rows);
      f.fill(perm[i]);
      yy.rows(inds) = f;
    }
    arma::uvec match = yy == Y;
    double r = (double)arma::sum(match) / (double)n;
    res = max(r, res);
  } while (next_permutation(perm.begin(), perm.end()));

  return res;
}


int main(int argc, char* argv[])
{
  string file_name = "data/processed/usps.csv";
  arma::arma_rng::set_seed_random();

  arma::mat A;
  A.load(file_name, arma::csv_ascii);
  int d = A.n_cols;
  arma::mat X = A.cols(0, d-2);
  arma::vec Y = A.col(d-1);
  arma::uvec uY = arma::conv_to<arma::uvec>::from(Y);

  arma::uvec inds = arma::find(uY == 1);
  inds = arma::join_cols(inds, arma::find(uY == 2));

  X = X.rows(inds);
  uY = uY.rows(inds);

  cout << X.n_rows << ", " << X.n_cols << endl;
  cout << uY.n_rows << ", " << uY.n_cols << endl;
  cout << arma::unique(uY).t() << endl;

  unsigned int m = 300;
  unsigned int k = 2;
  arma::uvec y_hat = CSSC(X, k, m);

  double accur = accuracy(uY, y_hat);
  cout << endl;
  cout << "Accuracy: " << accur << endl;
  cout << endl;

  return 0;
}
