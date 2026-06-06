//! Constructor with id and material properties
template <unsigned Tdim>
mpm::HerschelBulkleyViscoPlastic<Tdim>::HerschelBulkleyViscoPlastic(
    unsigned id, const Json& material_properties)
    : InfinitesimalElastoPlastic<Tdim>(id, material_properties) {
  try {
    // Density
    density_ = material_properties.at("density").template get<double>();
    // Young's modulus
    double youngs_modulus =
        material_properties.at("youngs_modulus").template get<double>();
    // Poisson ratio
    double poisson_ratio =
        material_properties.at("poisson_ratio").template get<double>();
    // Bulk modulus
    bulk_modulus_ = youngs_modulus / (3.0 * (1. - 2. * poisson_ratio));
    // Shear modulus
    shear_modulus_ = youngs_modulus / (2.0 * (1 + poisson_ratio));

    // Volumetric terms
    c_ = std::sqrt(bulk_modulus_ / density_);
    gamma_ = material_properties.at("volumetric_gamma").template get<double>();

    // Herschel-Bulkley parameters
    k_ = material_properties.at("consistency_parameter").template get<double>();
    n_ = material_properties.at("flow_index").template get<double>();

    // Peak and residual undrained shear strength
    if (material_properties.contains("tau_peak")) {
      tau_peak_ = material_properties.at("tau_peak").template get<double>();
    } else if (material_properties.contains("su_peak")) {
      tau_peak_ = 2.0 *
                  material_properties.at("su_peak").template get<double>() /
                  sqrt(3.0);
    } else {
      console_->error(
          "Herschel-Bulkley model requires either tau_peak or su_peak to be "
          "specified.");
    }
    if (material_properties.contains("tau_residual")) {
      tau_residual_ =
          material_properties.at("tau_residual").template get<double>();
    } else if (material_properties.contains("su_residual")) {
      tau_residual_ =
          2.0 * material_properties.at("su_residual").template get<double>() /
          sqrt(3.0);
    } else {
      console_->error(
          "Herschel-Bulkley model requires either tau_residual or su_residual "
          "to be specified.");
    }
    beta_ = material_properties.at("exponential_degradation")
                .template get<double>();

    // Parameters for return mapping algorithm
    if (material_properties.contains("rmap_absolute_tolerance")) {
      abs_tol_ = material_properties.at("rmap_absolute_tolerance")
                     .template get<double>();
    }
    if (material_properties.contains("rmap_relative_tolerance")) {
      rel_tol_ = material_properties.at("rmap_relative_tolerance")
                     .template get<double>();
    }
    if (material_properties.contains("rmap_max_iteration")) {
      max_iter_ =
          material_properties.at("rmap_max_iteration").template get<unsigned>();
    }

    // Properties
    properties_ = material_properties;
  } catch (std::exception& except) {
    console_->error("Material parameter not set: {}\n", except.what());
  }
}

//! Initialise state variables
template <unsigned Tdim>
mpm::dense_map
    mpm::HerschelBulkleyViscoPlastic<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {
      // Yield state: 0: elastic, 1: shear
      {"yield_state", 0},
      // Pressure
      {"pressure", 0.},
      // Volumetric strain
      {"volumetric_strain", 0.},
      // Shear stress ratio
      {"shear_stress_ratio", 0.},
      // Yield stress
      {"tau_yield", tau_peak_},
      // Plastic deviatoric strain rate
      {"pgamma_dot", 0.},
      // Plastic deviatoric strain (following p-q stress framework)
      {"pdstrain", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string>
    mpm::HerschelBulkleyViscoPlastic<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {
      "yield_state", "pressure",   "volumetric_strain", "shear_stress_ratio",
      "tau_yield",   "pgamma_dot", "pdstrain"};
  return state_vars;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1>
    mpm::HerschelBulkleyViscoPlastic<Tdim>::compute_stress(
        const Vector6d& stress, const Vector6d& dstrain,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {

  // Get volumetric strain
  const double vol_strain =
      (*state_vars).at("volumetric_strain") + ptr->dvolumetric_strain();

  //-------------------------------------------------------------------------
  // Elastic-predictor stage: compute the trial stress
  (*state_vars).at("yield_state") = 0;
  Matrix6x6 de = this->compute_elastic_tensor(vol_strain);
  Vector6d trial_stress =
      this->compute_trial_stress(stress, dstrain, de, ptr, state_vars);

  //-------------------------------------------------------------------------
  // Identity in voigt
  Vector6d m_voigt;
  m_voigt << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

  // Compute trial invariants
  const double p_tr = -mpm::materials::p(trial_stress);
  const Vector6d& deviatoric_stress_tr =
      mpm::materials::deviatoric_stress(trial_stress);
  const double q_tr = mpm::materials::q(trial_stress);
  const double tau_tr = q_tr / sqrt(3.0);

  // Compute static yield stress
  const double tau_yield_n = (*state_vars).at("tau_yield");

  // Compute new stress
  Vector6d updated_stress = Vector6d::Zero();
  double gamma_dot = 0.0;

  // Elastic state: stress point is less than yield surface
  const double p_new = p_tr;
  const double f_tr = tau_tr - tau_yield_n;
  if (f_tr <= 0.0) {
    updated_stress = trial_stress;
    gamma_dot = 0.0;
    (*state_vars).at("yield_state") = 0;
    (*state_vars).at("shear_stress_ratio") = 1.0;
    (*state_vars).at("tau_yield") = tau_yield_n;
  }
  // Plastic state: stress point is outside yield surface
  else {
    // Return mapping for Herschel-Bulkley viscoplastic model
    gamma_dot = std::max(tolerance_, f_tr / dt / shear_modulus_);
    double tau_m = tau_tr - shear_modulus_ * dt * gamma_dot;
    double tau_yield_m =
        tau_residual_ + (tau_yield_n - tau_residual_) *
                            std::exp(-beta_ * dt * gamma_dot / std::sqrt(3.0));

    // Start Newton-Raphson iteration
    unsigned iter = 0;
    double initial_res_norm;
    double residual = 0.0;
    double jacobian = 0.0;
    while (iter < max_iter_) {
      // Compute residual and jacobian
      residual = tau_m - tau_yield_m - k_ * std::pow(gamma_dot, n_);
      jacobian = -(shear_modulus_ * dt -
                   beta_ * dt / std::sqrt(3.0) * (tau_yield_n - tau_residual_) *
                       std::exp(-beta_ * dt * gamma_dot / std::sqrt(3.0)) +
                   k_ * n_ * std::pow(gamma_dot, n_ - 1.0));

      // Check residual convergence
      if (iter == 0) initial_res_norm = std::abs(residual);
      if (std::abs(residual) < abs_tol_ ||
          std::abs(residual) / initial_res_norm < rel_tol_) {
        break;
      }

      // Update gamma_dot
      const double delta_gamma_dot = residual / jacobian;
      gamma_dot -= delta_gamma_dot;

      // Update tau_m and tau_yield_m
      tau_m = tau_tr - shear_modulus_ * dt * gamma_dot;
      tau_yield_m = tau_residual_ +
                    (tau_yield_n - tau_residual_) *
                        std::exp(-beta_ * dt * gamma_dot / std::sqrt(3.0));

      // If delta_gamma_dot is too small, break to avoid numerical issues
      if (std::abs(delta_gamma_dot) < abs_tol_) break;

      // Increment iteration counter
      iter++;
    }

    // Update parameters
    const double tau_new = tau_m;
    const double tau_yield_new = tau_yield_m;

    const double shear_stress_ratio = tau_new / tau_tr;
    updated_stress =
        shear_stress_ratio * deviatoric_stress_tr - p_new * m_voigt;
    (*state_vars).at("yield_state") = 1;
    (*state_vars).at("shear_stress_ratio") = shear_stress_ratio;
    (*state_vars).at("tau_yield") = tau_yield_new;
  }

  // Update state variables
  (*state_vars).at("volumetric_strain") = vol_strain;
  (*state_vars).at("pgamma_dot") = gamma_dot;
  (*state_vars).at("pdstrain") += gamma_dot * dt / std::sqrt(3.0);
  (*state_vars).at("pressure") = p_new;

  return updated_stress;
}

//! Compute elastic tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::HerschelBulkleyViscoPlastic<Tdim>::compute_elastic_tensor(
        double vol_strain) {
  // Compute bulk modulus from equation of state
  const double K = density_ * c_ * c_ * std::exp(-gamma_ * vol_strain);
  // Shear modulus
  const double G = shear_modulus_;
  const double a1 = K + (4.0 / 3.0) * G;
  const double a2 = K - (2.0 / 3.0) * G;
  // compute elastic stiffness matrix
  // clang-format off
  Matrix6x6 de = Matrix6x6::Zero();
  de(0,0)=a1;    de(0,1)=a2;    de(0,2)=a2;
  de(1,0)=a2;    de(1,1)=a1;    de(1,2)=a2;
  de(2,0)=a2;    de(2,1)=a2;    de(2,2)=a1;
  de(3,3)=G;     de(4,4)=G;     de(5,5)=G;
  // clang-format on

  return de;
}

//! Compute constitutive relations matrix for elasto-plastic material
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::HerschelBulkleyViscoPlastic<Tdim>::compute_elasto_plastic_tensor(
        const Vector6d& stress, const Vector6d& dstrain,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt,
        double lin_v, double lin_a, bool hardening) {

  // Get yield type after return mapping algorithm
  mpm::hb_viscoplastic::FailureState yield_type =
      yield_type_.at(int((*state_vars).at("yield_state")));

  // Return the elastic consitutive tensor in elastic state
  if (yield_type == mpm::hb_viscoplastic::FailureState::Elastic) {
    const Matrix6x6 de =
        this->compute_elastic_tensor((*state_vars).at("volumetric_strain"));
    return de;
  }

  // Prepare necessary stress parameters
  const Vector6d& deviatoric_stress = mpm::materials::deviatoric_stress(stress);
  const double q = mpm::materials::q(stress);
  const double tau = q / sqrt(3.0);
  const double tau_ratio = (*state_vars).at("shear_stress_ratio");
  Vector6d director_n = Vector6d::Zero();
  if (tau > std::numeric_limits<double>::epsilon()) {
    director_n = deviatoric_stress / std::sqrt(2.0) / tau;
  }

  // Prepare fourth_order_identity and identity_cross
  Matrix6x6 fourth_order_identity = Matrix6x6::Zero();
  for (unsigned i = 0; i < 3; ++i) fourth_order_identity(i, i) = 1.0;
  for (unsigned i = 3; i < 6; ++i) fourth_order_identity(i, i) = 0.50;

  Matrix6x6 identity_cross = Matrix6x6::Zero();
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      identity_cross(i, j) = 1.0;
    }
  }

  // Prepare tensor_NxN, tensor_1xN, and tensor_Nx1
  Matrix6x6 tensor_NxN = Matrix6x6::Zero();
  for (unsigned i = 0; i < 6; ++i) {
    for (unsigned j = 0; j < 6; ++j) {
      tensor_NxN(i, j) = director_n[i] * director_n[j];
    }
  }

  // Compute the elasto-plastic stiffness matrix
  const double vol_strain = (*state_vars).at("volumetric_strain");
  const double K = density_ * c_ * c_ * std::exp(-gamma_ * vol_strain);
  const double tau_yield = (*state_vars).at("tau_yield");
  const double gamma_dot = (*state_vars).at("pgamma_dot");
  const double den =
      -(shear_modulus_ * dt -
        beta_ * dt / std::sqrt(3.0) * (tau_yield - tau_residual_) *
            std::exp(-beta_ * dt * gamma_dot / std::sqrt(3.0)) +
        k_ * n_ * std::pow(gamma_dot, n_ - 1.0));
  const double d_1 = 2.0 * shear_modulus_ * tau_ratio;
  const double d_2 = K - 2.0 / 3.0 * shear_modulus_ * tau_ratio;
  const double d_3 =
      2.0 * shear_modulus_ * (1.0 - tau_ratio + shear_modulus_ * dt / den);

  //! Elasto-plastic stiffness matrix
  Matrix6x6 d_ep =
      d_1 * fourth_order_identity + d_2 * identity_cross + d_3 * tensor_NxN;
  return d_ep;
}