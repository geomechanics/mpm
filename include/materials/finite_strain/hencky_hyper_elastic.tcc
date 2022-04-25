//! Read material properties
template <unsigned Tdim>
mpm::HenckyHyperElastic<Tdim>::HenckyHyperElastic(
    unsigned id, const Json& material_properties)
    : Material<Tdim>(id, material_properties) {
  try {
    density_ = material_properties.at("density").template get<double>();
    youngs_modulus_ =
        material_properties.at("youngs_modulus").template get<double>();
    poisson_ratio_ =
        material_properties.at("poisson_ratio").template get<double>();

    // Calculate Lame's constants
    mu_ = youngs_modulus_ / (2.0 * (1. + poisson_ratio_));
    lambda_ = youngs_modulus_ * poisson_ratio_ /
              ((1. + poisson_ratio_) * (1. - 2. * poisson_ratio_));

    properties_ = material_properties;

    // Set elastic tensor
    this->compute_elastic_tensor();

  } catch (Json::exception& except) {
    console_->error("Material parameter not set: {} {}\n", except.what(),
                    except.id);
  }
}

//! Return elastic tensor
template <unsigned Tdim>
bool mpm::HenckyHyperElastic<Tdim>::compute_elastic_tensor() {
  const double a1 = lambda_ + 2. * mu_;
  const double a2 = lambda_;

  // clang-format off
  // compute elasticityTensor
  de_ = Eigen::Matrix<double, 6, 6>::Zero();
  de_(0,0)=a1;    de_(0,1)=a2;    de_(0,2)=a2;
  de_(1,0)=a2;    de_(1,1)=a1;    de_(1,2)=a2;
  de_(2,0)=a2;    de_(2,1)=a2;    de_(2,2)=a1;
  de_(3,3)=mu_;   de_(4,4)=mu_;   de_(5,5)=mu_;
  // clang-format on
  return true;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::HenckyHyperElastic<Tdim>::compute_stress(
    const Vector6d& stress,
    const Eigen::Matrix<double, 3, 3>& deformation_gradient,
    const Eigen::Matrix<double, 3, 3>& deformation_gradient_increment,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Updated deformation gradient
  const Eigen::Matrix<double, 3, 3> updated_deformation_gradient =
      deformation_gradient_increment * deformation_gradient;
  const double updated_jacobian = updated_deformation_gradient.determinant();

  // Left Cauchy-Green strain
  const Eigen::Matrix<double, 3, 3> left_cauchy_green_tensor =
      updated_deformation_gradient * updated_deformation_gradient.transpose();

  // Principal values of left Cauchy-Green strain
  Eigen::Matrix<double, 3, 3> directors = Eigen::Matrix<double, 3, 3>::Zero();
  const Eigen::Matrix<double, 3, 1> principal_left_cauchy_green_strain =
      mpm::math::principal_tensor(left_cauchy_green_tensor, directors);

  // Principal values of Hencky (logarithmic) strain
  const Eigen::Matrix<double, 3, 1> principal_hencky_strain =
      0.5 * principal_left_cauchy_green_strain.array().log();

  // Principal values of Kirchhoff stress
  const Eigen::Matrix<double, 3, 1> principal_kirchhoff_stress =
      de_.block(0, 0, 3, 3) * principal_hencky_strain;

  // Principal values of Cauchy stress
  const Eigen::Matrix<double, 3, 3> principal_cauchy_stress =
      1.0 / updated_jacobian * principal_kirchhoff_stress.asDiagonal();

  // Cauchy stress
  const Eigen::Matrix<double, 3, 3> cauchy_stress =
      directors * principal_cauchy_stress * directors.transpose();

  // Convert to Voigt notation
  Eigen::Matrix<double, 6, 1> cauchy_stress_vector;
  cauchy_stress_vector(0) = cauchy_stress(0, 0);
  cauchy_stress_vector(1) = cauchy_stress(1, 1);
  cauchy_stress_vector(2) = cauchy_stress(2, 2);
  cauchy_stress_vector(3) = cauchy_stress(0, 1);
  cauchy_stress_vector(4) = cauchy_stress(1, 2);
  cauchy_stress_vector(5) = cauchy_stress(2, 0);

  return cauchy_stress_vector;
}

//! Compute consistent tangent matrix
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::HenckyHyperElastic<Tdim>::compute_consistent_tangent_matrix(
        const Vector6d& stress, const Vector6d& prev_stress,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient_increment,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Updated deformation gradient
  const Eigen::Matrix<double, 3, 3> updated_deformation_gradient =
      deformation_gradient_increment * deformation_gradient;
  const double updated_jacobian = updated_deformation_gradient.determinant();

  const double modified_mu =
      (mu_ - lambda_ * std::log(updated_jacobian)) / updated_jacobian;
  const double modified_lambda = lambda_ / updated_jacobian;

  const double a1 = modified_lambda + 2. * modified_mu;
  const double a2 = modified_lambda;

  Matrix6x6 const_tangent = Matrix6x6::Zero();
  // clang-format off
  // compute elasticityTensor
  const_tangent(0,0)=a1;    const_tangent(0,1)=a2;    const_tangent(0,2)=a2;
  const_tangent(1,0)=a2;    const_tangent(1,1)=a1;    const_tangent(1,2)=a2;
  const_tangent(2,0)=a2;    const_tangent(2,1)=a2;    const_tangent(2,2)=a1;
  const_tangent(3,3)=modified_mu;
  const_tangent(4,4)=modified_mu;
  const_tangent(5,5)=modified_mu;
  // clang-format on

  return const_tangent;
}