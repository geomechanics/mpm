// Getting the quadratures for Tnquadratures = 1
template <>
inline Eigen::MatrixXd mpm::TetrahedronQuadrature<3, 1>::quadratures() const {
  Eigen::Matrix<double, 3, 1> quadratures;
  quadratures(0, 0) = 1.0 / 4;
  quadratures(1, 0) = 1.0 / 4;
  quadratures(2, 0) = 1.0 / 4;

  return quadratures;
}

// Getting the weights for Tnquadratures = 1
template <>
inline Eigen::VectorXd mpm::TetrahedronQuadrature<3, 1>::weights() const {
  Eigen::VectorXd weights(1);
  weights(0) = 1.0 / 6;

  return weights;
}

// Getting the quadratures for Tnquadratures = 4
template <>
inline Eigen::MatrixXd mpm::TetrahedronQuadrature<3, 4>::quadratures() const {
  Eigen::Matrix<double, 3, 4> quadratures;
  quadratures(0, 0) = 0.58541020;
  quadratures(1, 0) = 0.13819660;
  quadratures(2, 0) = 0.13819660;

  quadratures(0, 1) = 0.13819660;
  quadratures(1, 1) = 0.58541020;
  quadratures(2, 1) = 0.13819660;

  quadratures(0, 2) = 0.13819660;
  quadratures(1, 2) = 0.13819660;
  quadratures(2, 2) = 0.58541020;

  quadratures(0, 3) = 0.13819660;
  quadratures(1, 3) = 0.13819660;
  quadratures(2, 3) = 0.13819660;

  return quadratures;
}

// Getting the weights for Tnquadratures = 4
template <>
inline Eigen::VectorXd mpm::TetrahedronQuadrature<3, 4>::weights() const {
  Eigen::VectorXd weights(4);
  weights(0) = 1.0 / 24;
  weights(1) = 1.0 / 24;
  weights(2) = 1.0 / 24;
  weights(3) = 1.0 / 24;

  return weights;
}