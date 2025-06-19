//! Constructor with material properties
template <unsigned Tdim>
mpm::Terracotta<Tdim>::Terracotta(unsigned id, const Json& material_properties)
    : Material<Tdim>(id, material_properties) {
  try {
    // Fluid density
    density_ = material_properties.at("density").template get<double>();
    // Young's modulus
    youngs_modulus_ =
        material_properties.at("youngs_modulus").template get<double>();
    // Poisson ratio
    poisson_ratio_ =
        material_properties.at("poisson_ratio").template get<double>();
    // Bulk modulus
    bulk_modulus_ = youngs_modulus_ / (3.0 * (1. - 2. * poisson_ratio_));
    // Shear modulus
    shear_modulus_ = youngs_modulus_ / (2.0 * (1 + poisson_ratio_));

    // Volumetric terms
    c_ = std::sqrt(bulk_modulus_ / density_);
    gamma_ = material_properties.at("volumetric_gamma").template get<double>();

    // Shear terms
    // Dynamic viscosity: [Pa*s]
    dynamic_viscosity_ =
        material_properties.at("dynamic_viscosity").template get<double>();
    // Dynamic yield stress: [Pa]
    tau0_ = material_properties.at("tau0").template get<double>();

    // Regularization shape factor m
    m_ = material_properties.at("regularization_parameter")
             .template get<double>();

    // Thixotropy terms
    // Flocculation state flocculation_state
    lambda0_ =
        material_properties.at("flocculation_state").template get<double>();
    // Flocculation rate: [Pa/s]"flocculation_state"
    // athix_ > 0.5: highly thixtropix; athix_ < 0.1: Non-thixtropix;
    athix_ =
        material_properties.at("flocculation_parameter").template get<double>();
    // Deflocculation parameter: order of 0.01
    alpha_ =
        material_properties.at("deflocculation_rate").template get<double>();

    // Properties
    properties_ = material_properties;

  } catch (Json::exception& except) {
    console_->error("Material parameter not set: {} {}\n", except.what(),
                    except.id);
  }
}

//! Initialise history variables
template <unsigned Tdim>
mpm::dense_map mpm::Terracotta<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {// Papanastasiou–Roussel Terracotta parameters
                               // Pressure
                               {"pressure", 0.0},
                               // Volumetric strain
                               {"volumetric_strain", 0.0},
                               // Thixotropic parameters
                               {"lambda", lambda0_},
                               // Shear rate
                               {"gamma_dot", 0}};
  return state_vars;
}

//! State variables
template <unsigned Tdim>
std::vector<std::string> mpm::Terracotta<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {"pressure", "volumetric_strain",
                                               "lambda", "gamma_dot"};
  return state_vars;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::Terracotta<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {

  // Get volumetric strain
  const double vol_strain =
      (*state_vars).at("volumetric_strain") + ptr->dvolumetric_strain();

  // Update bulk modulus and pressure from equation of state
  const double K = density_ * c_ * c_ * std::exp(-gamma_ * vol_strain);
  (*state_vars).at("pressure") = (K - bulk_modulus_) / gamma_;

  // Get strain rate
  auto strain_rate = ptr->strain_rate();
  // Convert strain rate to rate of deformation tensor
  strain_rate.tail(3) *= 0.5;

  // Compute deviatoric strain rate
  Eigen::Matrix<double, 6, 1> strain_rate_dev;
  strain_rate_dev = strain_rate;
  const double volumetric_strain_rate = strain_rate.head(3).sum() / 3.0;
  strain_rate_dev.head(3).noalias() +=
      -Eigen::Vector3d::Constant(volumetric_strain_rate);

  // Compute shear rate
  const double shear_rate =
      std::sqrt(2. * (strain_rate_dev.dot(strain_rate_dev) +
                      strain_rate_dev.tail(3).dot(strain_rate_dev.tail(3))));

  // Get thixotropic parameters
  double lambda = (*state_vars).at("lambda");
  double lambda_new = lambda;

  // Compute lambda
  if (lambda > 0)
    lambda_new += dt * (athix_ / tau0_ - alpha_ * lambda * shear_rate);
  // Lambda cannot be negative
  else
    lambda_new = 0;

  // Compute apparent viscosity
  double apparent_viscosity = 0.;
  if (shear_rate > 0)
    apparent_viscosity =
        dynamic_viscosity_ + (tau0_ / shear_rate) * (1 + lambda_new) *
                                 (1. - std::exp(-m_ * shear_rate));

  // Compute shear stress
  const Eigen::Matrix<double, 6, 1> tau =
      2 * apparent_viscosity * strain_rate_dev;

  // Identity in voigt
  Vector6d m_voigt;
  m_voigt << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

  // Update stress
  const Eigen::Matrix<double, 6, 1> updated_stress =
      -(*state_vars).at("pressure") * m_voigt + tau;

  (*state_vars).at("volumetric_strain") = vol_strain;
  (*state_vars).at("lambda") = lambda_new;
  (*state_vars).at("gamma_dot") = shear_rate;

  return updated_stress;
}