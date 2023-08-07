//! Convert 2nd-order symmetric tensor from voigt notation to full matrix form
template <>
inline const Eigen::Matrix<double, 1, 1> mpm::math::matrix_form<1>(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor) {
  Eigen::Matrix<double, 1, 1> matrix_tensor;
  matrix_tensor(0, 0) = voigt_tensor(0);
  return matrix_tensor;
}

//! Convert 2nd-order symmetric tensor from voigt notation to full matrix form
template <>
inline const Eigen::Matrix<double, 2, 2> mpm::math::matrix_form<2>(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor) {
  Eigen::Matrix<double, 2, 2> matrix_tensor;
  matrix_tensor(0, 0) = voigt_tensor(0);
  matrix_tensor(1, 1) = voigt_tensor(1);
  matrix_tensor(0, 1) = voigt_tensor(3);
  matrix_tensor(1, 0) = matrix_tensor(0, 1);
  return matrix_tensor;
}

//! Convert 2nd-order symmetric tensor from voigt notation to full matrix form
template <>
inline const Eigen::Matrix<double, 3, 3> mpm::math::matrix_form<3>(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor) {
  return mpm::math::matrix_form(voigt_tensor);
}

//! Convert 2nd-order symmetric tensor from voigt notation to full matrix form
inline const Eigen::Matrix<double, 3, 3> mpm::math::matrix_form(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor) {
  Eigen::Matrix<double, 3, 3> matrix_tensor;
  matrix_tensor(0, 0) = voigt_tensor(0);
  matrix_tensor(1, 1) = voigt_tensor(1);
  matrix_tensor(2, 2) = voigt_tensor(2);
  matrix_tensor(0, 1) = voigt_tensor(3);
  matrix_tensor(1, 0) = matrix_tensor(0, 1);
  matrix_tensor(1, 2) = voigt_tensor(4);
  matrix_tensor(2, 1) = matrix_tensor(1, 2);
  matrix_tensor(0, 2) = voigt_tensor(5);
  matrix_tensor(2, 0) = matrix_tensor(0, 2);

  return matrix_tensor;
}

//! Convert 2nd-order symmetric tensor from full matrix form to voigt notation
template <>
inline const Eigen::Matrix<double, 6, 1> mpm::math::voigt_form<1>(
    const Eigen::Matrix<double, 1, 1>& matrix_tensor) {
  Eigen::Matrix<double, 6, 1> voigt_tensor =
      Eigen::Matrix<double, 6, 1>::Zero();
  voigt_tensor(0) = matrix_tensor(0, 0);
  return voigt_tensor;
}

//! Convert 2nd-order symmetric tensor from full matrix form to voigt notation
template <>
inline const Eigen::Matrix<double, 6, 1> mpm::math::voigt_form<2>(
    const Eigen::Matrix<double, 2, 2>& matrix_tensor) {
  Eigen::Matrix<double, 6, 1> voigt_tensor =
      Eigen::Matrix<double, 6, 1>::Zero();
  voigt_tensor(0) = matrix_tensor(0, 0);
  voigt_tensor(1) = matrix_tensor(1, 1);
  voigt_tensor(3) = matrix_tensor(0, 1);
  return voigt_tensor;
}

//! Convert 2nd-order symmetric tensor from full matrix form to voigt notation
template <>
inline const Eigen::Matrix<double, 6, 1> mpm::math::voigt_form<3>(
    const Eigen::Matrix<double, 3, 3>& matrix_tensor) {
  return mpm::math::voigt_form(matrix_tensor);
}

//! Convert 2nd-order symmetric tensor from full matrix form to voigt notation
inline const Eigen::Matrix<double, 6, 1> mpm::math::voigt_form(
    const Eigen::Matrix<double, 3, 3>& matrix_tensor) {
  Eigen::Matrix<double, 6, 1> voigt_tensor;
  voigt_tensor(0) = matrix_tensor(0, 0);
  voigt_tensor(1) = matrix_tensor(1, 1);
  voigt_tensor(2) = matrix_tensor(2, 2);
  voigt_tensor(3) = matrix_tensor(0, 1);
  voigt_tensor(4) = matrix_tensor(1, 2);
  voigt_tensor(5) = matrix_tensor(0, 2);
  return voigt_tensor;
}

//! Compute principal stress/strain from given stress/strain in matrix form
inline const Eigen::Matrix<double, 3, 1> mpm::math::principal_tensor(
    const Eigen::Matrix<double, 3, 3>& matrix_tensor) {

  Eigen::Matrix<double, 3, 1> principal_tensor;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(matrix_tensor);
  principal_tensor = es.eigenvalues();

  // Sort principal tensor, 0: major, 1: intermediate, 2: minor
  std::swap(principal_tensor[0], principal_tensor[2]);

  return principal_tensor;
}

//! Compute principal stress/strain from given stress/strain in matrix form
inline const Eigen::Matrix<double, 3, 1> mpm::math::principal_tensor(
    const Eigen::Matrix<double, 3, 3>& matrix_tensor,
    Eigen::Matrix<double, 3, 3>& directors) {

  Eigen::Matrix<double, 3, 1> principal_tensor;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(matrix_tensor);
  principal_tensor = es.eigenvalues();
  directors = es.eigenvectors();

  // Sort principal tensor and directors
  std::swap(principal_tensor[0], principal_tensor[2]);
  Eigen::Matrix<double, 3, 1> temp_vector = directors.col(0);
  directors.col(0) = directors.col(2);
  directors.col(2) = temp_vector;

  return principal_tensor;
}

//! Compute principal stress/strain from given stress/strain in voigt notation
inline const Eigen::Matrix<double, 3, 1> mpm::math::principal_tensor(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor) {
  const auto& matrix_tensor = mpm::math::matrix_form(voigt_tensor);
  const auto& principal_tensor = mpm::math::principal_tensor(matrix_tensor);
  return principal_tensor;
}

//! Compute principal stress/strain from given stress/strain in voigt notation
inline const Eigen::Matrix<double, 3, 1> mpm::math::principal_tensor(
    const Eigen::Matrix<double, 6, 1>& voigt_tensor,
    Eigen::Matrix<double, 3, 3>& directors) {
  const auto& matrix_tensor = mpm::math::matrix_form(voigt_tensor);
  const auto& principal_tensor =
      mpm::math::principal_tensor(matrix_tensor, directors);
  return principal_tensor;
}