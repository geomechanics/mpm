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
                               {"prev_disp_x_j1", 0.},
                               {"prev_disp_y_j1", 0.},
                               {"prev_disp_z_j1", 0.},
                               {"prev_disp_x_j2", 0.},
                               {"prev_disp_y_j2", 0.},
                               {"prev_disp_z_j2", 0.},
                               {"prev_disp_x_j3", 0.},
                               {"prev_disp_y_j3", 0.},
                               {"prev_disp_z_j3", 0.},
                               {"prev_disp_x_j4", 0.},
                               {"prev_disp_y_j4", 0.},
                               {"prev_disp_z_j4", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::LinearElasticPML<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {
      "distance_function_x", "distance_function_y", "distance_function_z",
      "boundary_thickness",  "damping_function_x",  "damping_function_y",
      "damping_function_z",  "prev_disp_x_j1",      "prev_disp_y_j1",
      "prev_disp_z_j1",      "prev_disp_x_j2",      "prev_disp_y_j2",
      "prev_disp_z_j2",      "prev_disp_x_j3",      "prev_disp_y_j3",
      "prev_disp_z_j3",      "prev_disp_x_j4",      "prev_disp_y_j4",
      "prev_disp_z_j4"};
  return state_vars;
}
