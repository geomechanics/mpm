//! Read material properties
template <unsigned Tdim>
mpm::LinearElastic<Tdim>::LinearElastic(unsigned id,
                                        const Json& material_properties)
    : Material<Tdim>(id, material_properties) {
  try {

    // Set objective stress rate type
    if (material_properties.contains("stress_rate")) {
      auto stress_rate =
          material_properties["stress_rate"].template get<std::string>();
      if (stress_rate == "jaumann")
        stress_rate_ = mpm::StressRate::Jaumann;
      else if (stress_rate == "green_naghdi")
        stress_rate_ = mpm::StressRate::GreenNaghdi;
      else
        stress_rate_ = mpm::StressRate::None;
    }

    density_ = material_properties.at("density").template get<double>();
    youngs_modulus_ =
        material_properties.at("youngs_modulus").template get<double>();
    poisson_ratio_ =
        material_properties.at("poisson_ratio").template get<double>();

    // Calculate bulk modulus
    bulk_modulus_ = youngs_modulus_ / (3.0 * (1. - 2. * poisson_ratio_));

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
bool mpm::LinearElastic<Tdim>::compute_elastic_tensor() {
  // Shear modulus
  const double G = youngs_modulus_ / (2.0 * (1. + poisson_ratio_));

  const double a1 = bulk_modulus_ + (4.0 / 3.0) * G;
  const double a2 = bulk_modulus_ - (2.0 / 3.0) * G;

  // clang-format off
  // compute elasticityTensor
  de_ = Eigen::Matrix<double, 6, 6>::Zero();
  de_(0,0)=a1;    de_(0,1)=a2;    de_(0,2)=a2;
  de_(1,0)=a2;    de_(1,1)=a1;    de_(1,2)=a2;
  de_(2,0)=a2;    de_(2,1)=a2;    de_(2,2)=a1;
  de_(3,3)=G;     de_(4,4)=G;     de_(5,5)=G;
  // clang-format on
  return true;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::LinearElastic<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Compute new stress
  Vector6d new_stress;
  switch (stress_rate_) {
    case mpm::StressRate::None:
      new_stress = stress + this->de_ * dstrain;
      break;
    case mpm::StressRate::Jaumann:
      new_stress = this->compute_jaumann_stress(stress, dstrain, this->de_, ptr,
                                                state_vars);
      break;
    case mpm::StressRate::GreenNaghdi:
      new_stress = this->compute_green_naghdi_stress(stress, dstrain, this->de_,
                                                     ptr, state_vars);
      break;
  }

  return new_stress;
}

//! Compute stress using objective algorithm assuming Jaumann rate
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::LinearElastic<Tdim>::compute_jaumann_stress(
    const Vector6d& stress, const Vector6d& dstrain, const Matrix6x6& de,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Displacement gradient
  const Eigen::Matrix3d G =
      ptr->deformation_gradient_increment() - Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d G_skew = 0.5 * (G - G.transpose());

  // Rotation matrices by tensor exponential
  const Eigen::Matrix3d& Lambda_delta = (0.5 * G_skew).exp();
  const Eigen::Matrix3d Lambda_Delta = Lambda_delta * Lambda_delta;

  // Rotated stress strain
  const Eigen::Matrix3d rot_stress = Lambda_Delta *
                                     mpm::math::matrix_form(stress, 1.0) *
                                     Lambda_Delta.transpose();
  const Eigen::Matrix3d rot_strain_rate = Lambda_delta *
                                          mpm::math::matrix_form(dstrain, 0.5) *
                                          Lambda_delta.transpose();

  // Compute new stress
  Vector6d new_stress = mpm::math::voigt_form(rot_stress, 1.0) +
                        de * mpm::math::voigt_form(rot_strain_rate, 2.0);

  return new_stress;
}

//! Compute stress using objective algorithm assuming Green-Naghdi rate
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1>
    mpm::LinearElastic<Tdim>::compute_green_naghdi_stress(
        const Vector6d& stress, const Vector6d& dstrain, const Matrix6x6& de,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Displacement and deformation gradient
  const Eigen::Matrix3d& F_inc = ptr->deformation_gradient_increment();
  const Eigen::Matrix3d& F = ptr->deformation_gradient();
  const Eigen::Matrix3d F_new = F_inc * F;
  const Eigen::Matrix3d F_half = 0.5 * F + 0.5 * F_new;

  // Compute rotation matrix
  const Eigen::Matrix3d& R_current = mpm::materials::compute_rotation_tensor(F);
  const Eigen::Matrix3d& R_half =
      mpm::materials::compute_rotation_tensor(F_half);
  const Eigen::Matrix3d& R_new = mpm::materials::compute_rotation_tensor(F_new);

  // Rotation matrices by R * R'
  const Eigen::Matrix3d Lambda_delta = R_new * R_half.transpose();
  const Eigen::Matrix3d Lambda_Delta = R_new * R_current.transpose();

  // Rotated stress strain
  const Eigen::Matrix3d rot_stress = Lambda_Delta *
                                     mpm::math::matrix_form(stress, 1.0) *
                                     Lambda_Delta.transpose();
  const Eigen::Matrix3d rot_strain_rate = Lambda_delta *
                                          mpm::math::matrix_form(dstrain, 0.5) *
                                          Lambda_delta.transpose();

  // Compute new stress
  Vector6d new_stress = mpm::math::voigt_form(rot_stress, 1.0) +
                        de * mpm::math::voigt_form(rot_strain_rate, 2.0);

  return new_stress;
}

//! Compute consistent tangent matrix
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::LinearElastic<Tdim>::compute_consistent_tangent_matrix(
        const Vector6d& stress, const Vector6d& prev_stress,
        const Vector6d& dstrain, const ParticleBase<Tdim>* ptr,
        mpm::dense_map* state_vars) {
  return de_;
}