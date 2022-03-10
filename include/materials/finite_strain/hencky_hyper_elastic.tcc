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

    // Calculate constrained and shear modulus
    double constrained_modulus =
        youngs_modulus_ * (1. - poisson_ratio_) /
        ((1. + poisson_ratio_) * (1. - 2. * poisson_ratio_));
    double shear_modulus = youngs_modulus_ / (2.0 * (1. + poisson_ratio_));

    // Calculate wave velocities
    vp_ = sqrt(constrained_modulus / density_);
    vs_ = sqrt(shear_modulus / density_);

    properties_ = material_properties;
    properties_["pwave_velocity"] = vp_;
    properties_["swave_velocity"] = vs_;

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
  de_(0,0)=a1;    de_(0,1)=a2;    de_(0,2)=a2;    de_(0,3)=0;    de_(0,4)=0;    de_(0,5)=0;
  de_(1,0)=a2;    de_(1,1)=a1;    de_(1,2)=a2;    de_(1,3)=0;    de_(1,4)=0;    de_(1,5)=0;
  de_(2,0)=a2;    de_(2,1)=a2;    de_(2,2)=a1;    de_(2,3)=0;    de_(2,4)=0;    de_(2,5)=0;
  de_(3,0)= 0;    de_(3,1)= 0;    de_(3,2)= 0;    de_(3,3)=mu_;  de_(3,4)=0;    de_(3,5)=0;
  de_(4,0)= 0;    de_(4,1)= 0;    de_(4,2)= 0;    de_(4,3)=0;    de_(4,4)=mu_;  de_(4,5)=0;
  de_(5,0)= 0;    de_(5,1)= 0;    de_(5,2)= 0;    de_(5,3)=0;    de_(5,4)=0;    de_(5,5)=mu_;
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
  // Voigt notation
  Eigen::Matrix<double, 6, 1> left_cauchy_green_vector =
      Eigen::Matrix<double, 6, 1>::Zero();
  left_cauchy_green_vector(0) = left_cauchy_green_tensor(0, 0);
  left_cauchy_green_vector(1) = left_cauchy_green_tensor(1, 1);
  left_cauchy_green_vector(2) = left_cauchy_green_tensor(2, 2);
  left_cauchy_green_vector(3) = left_cauchy_green_tensor(0, 1);
  left_cauchy_green_vector(4) = left_cauchy_green_tensor(1, 2);
  left_cauchy_green_vector(5) = left_cauchy_green_tensor(2, 0);

  // Principal values of left Cauchy-Green strain
  Eigen::Matrix<double, 3, 1> principal_left_cauchy_green_strain =
      Eigen::Matrix<double, 3, 1>::Zero();
  Eigen::Matrix<double, 3, 3> directors = Eigen::Matrix<double, 3, 3>::Zero();
  principal_left_cauchy_green_strain =
      mpm::materials::principal_tensor(left_cauchy_green_vector, directors);

  // Principal values of Hencky (logarithmic) strain
  Eigen::Matrix<double, 3, 1> principal_hencky_strain =
      Eigen::Matrix<double, 3, 1>::Zero();
  principal_hencky_strain(0) = 0.5 * log(principal_left_cauchy_green_strain(0));
  principal_hencky_strain(1) = 0.5 * log(principal_left_cauchy_green_strain(1));
  principal_hencky_strain(2) = 0.5 * log(principal_left_cauchy_green_strain(2));

  // Principal values of Kirchhoff stress
  this->compute_elastic_tensor();
  Eigen::Matrix<double, 3, 1> principal_kirchhoff_stress =
      Eigen::Matrix<double, 3, 1>::Zero();
  principal_kirchhoff_stress(0) = de_(0, 0) * principal_hencky_strain(0) +
                                  de_(0, 1) * principal_hencky_strain(1) +
                                  de_(0, 2) * principal_hencky_strain(2);
  principal_kirchhoff_stress(1) = de_(1, 0) * principal_hencky_strain(0) +
                                  de_(1, 1) * principal_hencky_strain(1) +
                                  de_(1, 2) * principal_hencky_strain(2);
  principal_kirchhoff_stress(2) = de_(2, 0) * principal_hencky_strain(0) +
                                  de_(2, 1) * principal_hencky_strain(1) +
                                  de_(2, 2) * principal_hencky_strain(2);

  // Principal values of Cauchy stress
  Eigen::Matrix<double, 3, 3> principal_cauchy_stress =
      Eigen::Matrix<double, 3, 3>::Zero();
  principal_cauchy_stress(0, 0) =
      principal_kirchhoff_stress(0) / updated_jacobian;
  principal_cauchy_stress(1, 1) =
      principal_kirchhoff_stress(1) / updated_jacobian;
  principal_cauchy_stress(2, 2) =
      principal_kirchhoff_stress(2) / updated_jacobian;

  // Cauchy stress
  Eigen::Matrix<double, 3, 3> cauchy_stress =
      directors * principal_cauchy_stress * directors.transpose();
  // Voigt notation
  Eigen::Matrix<double, 6, 1> cauchy_stress_vector =
      Eigen::Matrix<double, 6, 1>::Zero();
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
  Eigen::Matrix<double, 3, 3> updated_deformation_gradient =
      deformation_gradient_increment * deformation_gradient;
  const double updated_jacobian = updated_deformation_gradient.determinant();

  const double modified_mu =
      (mu_ - lambda_ * log(updated_jacobian)) / updated_jacobian;
  const double modified_lambda = lambda_ / updated_jacobian;

  const double a1 = modified_lambda + 2. * modified_mu;
  const double a2 = modified_lambda;

  // clang-format off
  // compute elasticityTensor
  de_(0,0)=a1;    de_(0,1)=a2;    de_(0,2)=a2;    de_(0,3)=0;           de_(0,4)=0;           de_(0,5)=0;
  de_(1,0)=a2;    de_(1,1)=a1;    de_(1,2)=a2;    de_(1,3)=0;           de_(1,4)=0;           de_(1,5)=0;
  de_(2,0)=a2;    de_(2,1)=a2;    de_(2,2)=a1;    de_(2,3)=0;           de_(2,4)=0;           de_(2,5)=0;
  de_(3,0)= 0;    de_(3,1)= 0;    de_(3,2)= 0;    de_(3,3)=modified_mu; de_(3,4)=0;           de_(3,5)=0;
  de_(4,0)= 0;    de_(4,1)= 0;    de_(4,2)= 0;    de_(4,3)=0;           de_(4,4)=modified_mu; de_(4,5)=0;
  de_(5,0)= 0;    de_(5,1)= 0;    de_(5,2)= 0;    de_(5,3)=0;           de_(5,4)=0;           de_(5,5)=modified_mu;
  // clang-format on            
  return de_;
}