//! Read material properties
template <unsigned Tdim>
mpm::LinearElastic<Tdim>::LinearElastic(unsigned id,
                                        const Json& material_properties)
    : Material<Tdim>(id, material_properties) {
  try {
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

    // Set objective stress rate type
    if (properties_.contains("stress_rate")) {
      auto stress_rate = properties_["stress_rate"].template get<std::string>();
      if (stress_rate == "Jaumann")
        stress_rate_ = mpm::StressRate::Jaumann;
      else if (stress_rate == "GreenNaghdi")
        stress_rate_ = mpm::StressRate::GreenNaghdi;
      else
        stress_rate_ = mpm::StressRate::None;
    }

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
  const Vector6d dstress = this->de_ * dstrain;

  // Compute new stress
  // TODO: Implement stress rate
  Vector6d new_stress;
  switch (stress_rate_) {
    case mpm::StressRate::None:
      new_stress = stress + dstress;
      break;
    case mpm::StressRate::Jaumann:
      new_stress = stress + dstress;
      break;
    case mpm::StressRate::GreenNaghdi:
      new_stress = stress + dstress;
      break;
  }

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