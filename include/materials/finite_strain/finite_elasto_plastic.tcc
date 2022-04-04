//! Initialise state variables
template <unsigned Tdim>
mpm::dense_map mpm::FiniteElastoPlastic<Tdim>::initialise_state_variables() {
  // Base state variable
  mpm::dense_map state_vars = {// Left elastic Cauchy Green tensor
                               {"elastic_left_cauchy_green_00", 0.},
                               {"elastic_left_cauchy_green_11", 0.},
                               {"elastic_left_cauchy_green_22", 0.},
                               {"elastic_left_cauchy_green_01", 0.},
                               {"elastic_left_cauchy_green_12", 0.},
                               {"elastic_left_cauchy_green_02", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::FiniteElastoPlastic<Tdim>::state_variables()
    const {
  const std::vector<std::string> state_vars = {
      "elastic_left_cauchy_green_00", "elastic_left_cauchy_green_11",
      "elastic_left_cauchy_green_22", "elastic_left_cauchy_green_01",
      "elastic_left_cauchy_green_12", "elastic_left_cauchy_green_02"};
  return state_vars;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::FiniteElastoPlastic<Tdim>::compute_stress(
    const Vector6d& stress, const Matrix3x3& deformation_gradient,
    const Matrix3x3& deformation_gradient_increment,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Updated deformation gradient
  const Matrix3x3 updated_deformation_gradient =
      deformation_gradient_increment * deformation_gradient;
  const double updated_jacobian = updated_deformation_gradient.determinant();

  // Elastic left Cauchy-Green tensor
  Matrix3x3 elastic_left_cauchy_green = Matrix3x3::Zero();
  elastic_left_cauchy_green(0, 0) =
      (*state_vars).at("elastic_left_cauchy_green_00");
  elastic_left_cauchy_green(1, 1) =
      (*state_vars).at("elastic_left_cauchy_green_11");
  elastic_left_cauchy_green(2, 2) =
      (*state_vars).at("elastic_left_cauchy_green_22");
  elastic_left_cauchy_green(0, 1) =
      (*state_vars).at("elastic_left_cauchy_green_01");
  elastic_left_cauchy_green(1, 2) =
      (*state_vars).at("elastic_left_cauchy_green_12");
  elastic_left_cauchy_green(0, 2) =
      (*state_vars).at("elastic_left_cauchy_green_02");
  elastic_left_cauchy_green(1, 0) = elastic_left_cauchy_green(0, 1);
  elastic_left_cauchy_green(2, 1) = elastic_left_cauchy_green(1, 2);
  elastic_left_cauchy_green(2, 0) = elastic_left_cauchy_green(0, 2);

  // Trial elastic left Cauchy-Green tensor
  const Matrix3x3 trial_left_cauchy_green =
      deformation_gradient_increment * elastic_left_cauchy_green *
      deformation_gradient_increment.transpose();

  // Principal values of left Cauchy-Green tensor
  Matrix3x3 directors = Matrix3x3::Zero();
  const Vector3d principal_trial_left_cauchy_green =
      mpm::materials::principal_tensor(trial_left_cauchy_green, directors);

  // Principal values of trial Hencky (logarithmic) strain
  Vector3d principal_elastic_hencky_strain =
      0.5 * principal_trial_left_cauchy_green.array().log();

  // Compute kirchhoff stress and perform elasto-plastic return mapping
  const Vector3d principal_kirchhoff_stress = this->compute_return_mapping(
      principal_elastic_hencky_strain, ptr, state_vars);

  // Principal values of Cauchy stress
  const Matrix3x3 principal_cauchy_stress =
      1.0 / updated_jacobian * principal_kirchhoff_stress.asDiagonal();

  // Cauchy stress
  const Matrix3x3 cauchy_stress =
      directors * principal_cauchy_stress * directors.transpose();

  // Convert to Voigt notation
  Eigen::Matrix<double, 6, 1> cauchy_stress_vector;
  cauchy_stress_vector(0) = cauchy_stress(0, 0);
  cauchy_stress_vector(1) = cauchy_stress(1, 1);
  cauchy_stress_vector(2) = cauchy_stress(2, 2);
  cauchy_stress_vector(3) = cauchy_stress(0, 1);
  cauchy_stress_vector(4) = cauchy_stress(1, 2);
  cauchy_stress_vector(5) = cauchy_stress(2, 0);

  // New elastic left Cauchy-Green tensor
  elastic_left_cauchy_green.setZero();
  elastic_left_cauchy_green.diagonal() =
      (2.0 * principal_elastic_hencky_strain).array().exp();
  elastic_left_cauchy_green =
      directors * elastic_left_cauchy_green * directors.transpose();

  (*state_vars).at("elastic_left_cauchy_green_00") =
      elastic_left_cauchy_green(0, 0);
  (*state_vars).at("elastic_left_cauchy_green_11") =
      elastic_left_cauchy_green(1, 1);
  (*state_vars).at("elastic_left_cauchy_green_22") =
      elastic_left_cauchy_green(2, 2);
  (*state_vars).at("elastic_left_cauchy_green_01") =
      elastic_left_cauchy_green(0, 1);
  (*state_vars).at("elastic_left_cauchy_green_12") =
      elastic_left_cauchy_green(1, 2);
  (*state_vars).at("elastic_left_cauchy_green_02") =
      elastic_left_cauchy_green(0, 2);

  return cauchy_stress_vector;
}

//! Compute consistent tangent matrix
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::FiniteElastoPlastic<Tdim>::compute_consistent_tangent_matrix(
        const Vector6d& stress, const Vector6d& prev_stress,
        const Matrix3x3& deformation_gradient,
        const Matrix3x3& deformation_gradient_increment,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Elastic left Cauchy-Green tensor
  Matrix3x3 elastic_left_cauchy_green = Matrix3x3::Zero();
  elastic_left_cauchy_green(0, 0) =
      (*state_vars).at("elastic_left_cauchy_green_00");
  elastic_left_cauchy_green(1, 1) =
      (*state_vars).at("elastic_left_cauchy_green_11");
  elastic_left_cauchy_green(2, 2) =
      (*state_vars).at("elastic_left_cauchy_green_22");
  elastic_left_cauchy_green(0, 1) =
      (*state_vars).at("elastic_left_cauchy_green_01");
  elastic_left_cauchy_green(1, 2) =
      (*state_vars).at("elastic_left_cauchy_green_12");
  elastic_left_cauchy_green(0, 2) =
      (*state_vars).at("elastic_left_cauchy_green_02");
  elastic_left_cauchy_green(1, 0) = elastic_left_cauchy_green(0, 1);
  elastic_left_cauchy_green(2, 1) = elastic_left_cauchy_green(1, 2);
  elastic_left_cauchy_green(2, 0) = elastic_left_cauchy_green(0, 2);

  //! Consistent tangent matrix
  Matrix6x6 const_tangent = this->compute_elasto_plastic_tensor(
      stress, elastic_left_cauchy_green, ptr, state_vars, true);

  return const_tangent;
}