//! Read material properties
template <unsigned Tdim>
mpm::UnsplitPML<Tdim>::UnsplitPML(unsigned id, const Json& material_properties)
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

    // Normal damping ratio
    beta_ =
        material_properties.at("maximum_damping_ratio").template get<double>();
    // Characteristic length
    h_char_ =
        material_properties.at("characteristic_length").template get<double>();
    // Tangential damping ratio
    alpha_ = std::pow(lambda_ + 2. * shear_modulus_ / density_, 0.5);
    // Damping power
    dpower_ = material_properties.at("damping_power").template get<double>();

    // Set elastic tensor
    this->compute_elastic_tensor();

    // Properties
    properties_ = material_properties;
  } catch (Json::exception& except) {
    console_->error("Material parameter not set: {} {}\n", except.what(),
                    except.id);
  }
}

//! Initialise state variables
template <unsigned Tdim>
mpm::dense_map mpm::UnsplitPML<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {// Distance functions
                               {"distance_function_x", 0.},
                               {"distance_function_y", 0.},
                               {"distance_function_z", 0.},
                               {"boundary_thickness", 0.},
                               // Damping functions
                               {"damping_function_x", 0.},
                               {"damping_function_y", 0.},
                               {"damping_function_z", 0.},
                               // Time integrated variables 12+4+6+3
                               {"strain_int_xx", 0.},
                               {"strain_int_yy", 0.},
                               {"strain_int_zz", 0.},
                               {"strain_int_xy", 0.},
                               {"strain_int_xz", 0.},
                               {"strain_int_yz", 0.},
                               {"stress_2int_xx", 0.},
                               {"stress_2int_yy", 0.},
                               {"stress_2int_zz", 0.},
                               {"stress_2int_xy", 0.},
                               {"stress_2int_xz", 0.},
                               {"stress_2int_yz", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::UnsplitPML<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {
      "distance_function_x", "distance_function_y", "distance_function_z",
      "boundary_thickness",  "damping_function_x",  "damping_function_y",
      "damping_function_z",  "strain_int_xx",       "strain_int_yy",
      "strain_int_zz",       "strain_int_xy",       "strain_int_xz",
      "strain_int_yz",       "stress_2int_xx",      "stress_2int_yy",
      "stress_2int_zz",      "stress_2int_xy",      "stress_2int_xz",
      "stress_2int_yz"};
  return state_vars;
}