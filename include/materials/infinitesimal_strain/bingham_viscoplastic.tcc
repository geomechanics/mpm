//! Constructor with id and material properties
template <unsigned Tdim>
mpm::BinghamViscoPlastic<Tdim>::BinghamViscoPlastic(
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

    // Shear terms
    dynamic_viscosity_ =
        material_properties.at("dynamic_viscosity").template get<double>();
    tau0_ = material_properties.at("tau0").template get<double>();
    lambda0_ =
        material_properties.at("flocculation_state").template get<double>();
    athix_ =
        material_properties.at("flocculation_rate").template get<double>();
    alpha_ =
        material_properties.at("deflocculation_rate").template get<double>();

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
mpm::dense_map mpm::BinghamViscoPlastic<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {
      // Viscoplastic Bingham parameters
      // Yield state: 0: elastic, 1: shear, 2: tensile
      {"yield_state", 0},
      // Pressure
      {"pressure", 0.},
      // Number of iteration in return mapping algorithm
      {"rmap_niteration", 0},
      // Volumetric strain
      {"volumetric_strain", 0.},
      // Shear stress ratio
      {"shear_stress_ratio", 0.},
      // Thixotropic parameters
      {"lambda", lambda0_},
      // Plastic deviatoric strain rate
      {"pgamma_dot", 0.},
      // Plastic deviatoric strain (following p-q stress framework)
      {"pdstrain", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::BinghamViscoPlastic<Tdim>::state_variables()
    const {
  const std::vector<std::string> state_vars = {
      "yield_state",        "pressure", "rmap_niteration", "volumetric_strain",
      "shear_stress_ratio", "lambda",   "pgamma_dot",      "pdstrain"};
  return state_vars;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::BinghamViscoPlastic<Tdim>::compute_stress(
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

  // Compute static yield stress considering thixotropic effects
  const double lambda_tr = (*state_vars).at("lambda") + athix_ / tau0_ * dt;
  const double static_yield_stress = (1.0 + lambda_tr) * tau0_;

  // Compute new stress
  Vector6d updated_stress = Vector6d::Zero();
  double gamma_dot = 0.0;

  // Elastic state: stress point is less than yield surface
  const double p_new = p_tr;
  if (tau_tr <= static_yield_stress) {
    updated_stress = trial_stress;
    gamma_dot = 0.0;
    (*state_vars).at("yield_state") = 0;
    (*state_vars).at("rmap_niteration") = 1;
    (*state_vars).at("shear_stress_ratio") = 1.0;
    (*state_vars).at("lambda") = lambda_tr;
  }
  // Plastic state: stress point is outside yield surface
  else {
    // Initialize variables
    double tau_new = tau_tr - shear_modulus_ * dt * gamma_dot;
    double lambda_new = lambda_tr;

    // Start Newton-Raphson iteration
    unsigned iter = 0;
    double initial_res;
    Eigen::Matrix<double, 2, 1> res_m;
    Eigen::Matrix<double, 2, 2> jac_m;
    while (iter < max_iter_) {
      // Compute residuals
      res_m(0) =
          tau_new - dynamic_viscosity_ * gamma_dot - tau0_ * (1.0 + lambda_new);
      res_m(1) = (1.0 + alpha_ * gamma_dot * dt) * lambda_new - lambda_tr;

      // Check convergence based on residual
      if (res_m.norm() < abs_tol_) break;
      if (iter == 0)
        initial_res = res_m.norm();
      else {
        if (res_m.norm() / initial_res < rel_tol_) break;
      }

      // Compute Jacobian
      jac_m(0, 0) = -(dt * shear_modulus_ + dynamic_viscosity_);
      jac_m(0, 1) = -tau0_;
      jac_m(1, 0) = alpha_ * lambda_new * dt;
      jac_m(1, 1) = (1.0 + alpha_ * dt * gamma_dot);

      // Update gamma_dot
      const Eigen::Matrix<double, 2, 1> delta_unknown =
          jac_m.inverse() * (-res_m);
      gamma_dot += delta_unknown(0);
      lambda_new += delta_unknown(1);

      // Update parameters
      tau_new = tau_tr - shear_modulus_ * dt * gamma_dot;

      // Check convergence based on solution
      if (delta_unknown.norm() < abs_tol_) break;

      // Increment iteration counter
      iter++;
    }
    (*state_vars).at("rmap_niteration") = iter + 1;

    const double shear_stress_ratio = tau_new / tau_tr;
    updated_stress =
        shear_stress_ratio * deviatoric_stress_tr - p_new * m_voigt;
    gamma_dot = ((tau_tr - tau_new) / shear_modulus_) / dt;
    (*state_vars).at("yield_state") = 1;
    (*state_vars).at("shear_stress_ratio") = shear_stress_ratio;
    (*state_vars).at("lambda") = lambda_new;
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
    mpm::BinghamViscoPlastic<Tdim>::compute_elastic_tensor(double vol_strain) {
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
    mpm::BinghamViscoPlastic<Tdim>::compute_elasto_plastic_tensor(
        const Vector6d& stress, const Vector6d& dstrain,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt,
        bool hardening) {

  // Get yield type after return mapping algorithm
  mpm::bingham_viscoplastic::FailureState yield_type =
      yield_type_.at(int((*state_vars).at("yield_state")));

  // Return the elastic consitutive tensor in elastic state
  if (yield_type == mpm::bingham_viscoplastic::FailureState::Elastic) {
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
  const double den = -(dynamic_viscosity_ + dt * shear_modulus_) +
                     dt * tau0_ * alpha_ * (*state_vars).at("lambda");
  const double d_1 = 2.0 * shear_modulus_ * tau_ratio;
  const double d_2 = K - 2.0 / 3.0 * shear_modulus_ * tau_ratio;
  const double d_3 =
      2.0 * shear_modulus_ * (1.0 - tau_ratio + shear_modulus_ * dt / den);

  //! Elasto-plastic stiffness matrix
  Matrix6x6 d_ep =
      d_1 * fourth_order_identity + d_2 * identity_cross + d_3 * tensor_NxN;
  return d_ep;
}