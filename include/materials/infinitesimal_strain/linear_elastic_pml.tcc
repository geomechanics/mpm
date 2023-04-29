//! Read material properties
template <unsigned Tdim>
mpm::LinearElasticPML<Tdim>::LinearElasticPML(unsigned id,
                                              const Json& material_properties)
    : LinearElastic<Tdim>(id, material_properties) {
  try {
    density_ = material_properties.at("density").template get<double>();
    youngs_modulus_ =
        material_properties.at("youngs_modulus").template get<double>();
    poisson_ratio_ =
        material_properties.at("poisson_ratio").template get<double>();

    // Calculate Lame's constants
    lambda_ = youngs_modulus_ * poisson_ratio_ / (1. + poisson_ratio_) /
              (1. - 2. * poisson_ratio_);
    shear_modulus_ = youngs_modulus_ / (2.0 * (1. + poisson_ratio_));

    // Maximum damping ratio
    alpha_ =
        material_properties.at("maximum_damping_ratio").template get<double>();
    // Damping power
    dpower_ = material_properties.at("damping_power").template get<double>();

    // Properties
    properties_ = material_properties;
  } catch (Json::exception& except) {
    console_->error("Material parameter not set: {} {}\n", except.what(),
                    except.id);
  }
}

//! Initialise state variables
template <unsigned Tdim>
mpm::dense_map mpm::LinearElasticPML<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {// Distance functions
                               {"distance_function_x", 0.},
                               {"distance_function_y", 0.},
                               {"distance_function_z", 0.},
                               {"boundary_thickness", 0.},
                               // Damping functions
                               {"damping_function_x", 0.},
                               {"damping_function_y", 0.},
                               {"damping_function_z", 0.},
                               // Historical strain variables
                               {"prev_strain_function_x", 0.},
                               {"prev_strain_function_y", 0.},
                               {"prev_strain_function_z", 0.},
                               {"prev_strain_function_xy", 0.},
                               {"prev_strain_function_yz", 0.},
                               {"prev_strain_function_xz", 0.},
                               {"old_strain_function_x", 0.},
                               {"old_strain_function_y", 0.},
                               {"old_strain_function_z", 0.},
                               {"old_strain_function_xy", 0.},
                               {"old_strain_function_yz", 0.},
                               {"old_strain_function_xz", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::LinearElasticPML<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {
      "distance_function_x",     "distance_function_y",
      "distance_function_z",     "boundary_thickness",
      "damping_function_x",      "damping_function_y",
      "damping_function_z",      "prev_strain_function_x",
      "prev_strain_function_y",  "prev_strain_function_z",
      "prev_strain_function_xy", "prev_strain_function_yz",
      "prev_strain_function_xz", "old_strain_function_x",
      "old_strain_function_y",   "old_strain_function_z",
      "old_strain_function_xy",  "old_strain_function_yz",
      "old_strain_function_xz"};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
void mpm::LinearElasticPML<Tdim>::compute_damping_functions(
    mpm::dense_map* state_vars) {
  const double boundary_thickness = (*state_vars).at("boundary_thickness");
  const double multiplier =
      alpha_ * std::pow(1.0 / boundary_thickness, dpower_);
  (*state_vars).at("damping_function_x") =
      multiplier * std::pow((*state_vars).at("distance_function_x"), dpower_);
  (*state_vars).at("damping_function_y") =
      multiplier * std::pow((*state_vars).at("distance_function_y"), dpower_);
  (*state_vars).at("damping_function_z") =
      multiplier * std::pow((*state_vars).at("distance_function_z"), dpower_);
}

//! Return PML elastic tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6> mpm::LinearElasticPML<Tdim>::compute_elastic_tensor(
    mpm::dense_map* state_vars) {
  // Initialise tensor
  Matrix6x6 de;

  // Damping functions
  this->compute_damping_functions(state_vars);
  const double c_x = (*state_vars).at("damping_function_x");
  const double c_y = (*state_vars).at("damping_function_y");
  const double c_z = (*state_vars).at("damping_function_z");

  const double diag = lambda_ + 2.0 * shear_modulus_;

  // compute elasticityTensor
  de = Eigen::Matrix<double, 6, 6>::Zero();
  de(0, 0) = diag;
  de(0, 1) = (1. + c_x) * (lambda_ + shear_modulus_) -
             shear_modulus_ * (1. + c_y) * (1. + c_y);
  de(0, 2) = (1. + c_x) * (lambda_ + shear_modulus_) -
             shear_modulus_ * (1. + c_z) * (1. + c_z);

  de(1, 0) = (1. + c_y) * (lambda_ + shear_modulus_) -
             shear_modulus_ * (1. + c_x) * (1. + c_x);
  de(1, 1) = diag;
  de(1, 2) = (1. + c_y) * (lambda_ + shear_modulus_) -
             shear_modulus_ * (1. + c_z) * (1. + c_z);

  de(2, 0) = (1. + c_z) * (lambda_ + shear_modulus_) -
             shear_modulus_ * (1. + c_x) * (1. + c_x);
  de(2, 1) = (1. + c_z) * (lambda_ + shear_modulus_) -
             shear_modulus_ * (1. + c_y) * (1. + c_y);
  de(2, 2) = diag;

  de(3, 3) = shear_modulus_ * (1. + c_x) * (1. + c_y);
  de(4, 4) = shear_modulus_ * (1. + c_y) * (1. + c_z);
  de(5, 5) = shear_modulus_ * (1. + c_x) * (1. + c_z);

  return de;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::LinearElasticPML<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {
  const Matrix6x6& de = this->compute_elastic_tensor(state_vars);
  const Vector6d dstress = de * dstrain;
  return (stress + dstress);
}

//! Compute consistent tangent matrix
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::LinearElasticPML<Tdim>::compute_consistent_tangent_matrix(
        const Vector6d& stress, const Vector6d& prev_stress,
        const Vector6d& dstrain, const ParticleBase<Tdim>* ptr,
        mpm::dense_map* state_vars) {
  const Matrix6x6& de = this->compute_elastic_tensor(state_vars);
  return de;
}