//! Constructor with id and material properties
template <unsigned Tdim>
mpm::MohrCoulomb<Tdim>::MohrCoulomb(unsigned id,
                                    const Json& material_properties)
    : InfinitesimalElastoPlastic<Tdim>(id, material_properties) {
  try {
    // General parameters
    // Density
    density_ = material_properties.at("density").template get<double>();

    // Initial Packing Fraction
    double initial_packing_fraction = 1.;
    if (material_properties.contains("packing_fraction"))
      initial_packing_fraction =
          material_properties.at("packing_fraction").template get<double>();
    else if (material_properties.contains("porosity"))
      initial_packing_fraction =
          1. - material_properties.at("porosity").template get<double>();

    // Solid grain density
    bool is_wet = false;
    // Check if two-phase simulation is considered
    if (material_properties.contains("porosity") &&
        (material_properties.contains("k_x") ||
         material_properties.contains("k_y") ||
         material_properties.contains("k_z")))
      is_wet = true;
    if (!is_wet)
      grain_density_ = density_ / initial_packing_fraction;
    else
      grain_density_ = density_;

    // Minimum packing fraction
    minimum_packing_fraction_ = 0.0;
    if (material_properties.contains("packing_fraction_minimum"))
      minimum_packing_fraction_ =
          material_properties.at("packing_fraction_minimum")
              .template get<double>();

    // Young's modulus
    youngs_modulus_ =
        material_properties.at("youngs_modulus").template get<double>();
    // Poisson ratio
    poisson_ratio_ =
        material_properties.at("poisson_ratio").template get<double>();
    // Softening status
    softening_ = material_properties.at("softening").template get<bool>();
    // Peak friction, dilation and cohesion
    phi_peak_ =
        material_properties.at("friction").template get<double>() * M_PI / 180.;
    psi_peak_ =
        material_properties.at("dilation").template get<double>() * M_PI / 180.;
    cohesion_peak_ = material_properties.at("cohesion").template get<double>();
    // Residual friction, dilation and cohesion
    phi_residual_ =
        material_properties.at("residual_friction").template get<double>() *
        M_PI / 180.;
    psi_residual_ =
        material_properties.at("residual_dilation").template get<double>() *
        M_PI / 180.;
    cohesion_residual_ =
        material_properties.at("residual_cohesion").template get<double>();
    // Peak plastic deviatoric strain
    pdstrain_peak_ =
        material_properties.at("peak_pdstrain").template get<double>();
    // Residual plastic deviatoric strain
    pdstrain_residual_ =
        material_properties.at("residual_pdstrain").template get<double>();
    // Tensile strength
    tension_cutoff_ =
        material_properties.at("tension_cutoff").template get<double>();
    // Properties
    properties_ = material_properties;
    // Bulk modulus
    bulk_modulus_ = youngs_modulus_ / (3.0 * (1. - 2. * poisson_ratio_));
    // Shear modulus
    shear_modulus_ = youngs_modulus_ / (2.0 * (1 + poisson_ratio_));
  } catch (std::exception& except) {
    console_->error("Material parameter not set: {}\n", except.what());
  }
}

//! Initialise state variables
template <unsigned Tdim>
mpm::dense_map mpm::MohrCoulomb<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {
      // MC parameters
      // Yield state: 0: elastic, 1: shear, 2: tensile
      {"yield_state", 0},
      // Friction (phi)
      {"phi", this->phi_peak_},
      // Dilation (psi)
      {"psi", this->psi_peak_},
      // Cohesion
      {"cohesion", this->cohesion_peak_},
      // Tensile cutoff (automatically adjusted according to the apex value)
      {"tension_cutoff",
       check_low((tension_cutoff_ < cohesion_peak_ / std::tan(phi_peak_))
                     ? tension_cutoff_
                     : cohesion_peak_ / std::tan(phi_peak_))},
      // Stress invariants
      // Pressure
      {"pressure", 0.},
      // Tau
      {"tau", 0.},
      // Theta
      {"theta", 0.},
      // Plastic deviatoric strain
      {"pdstrain", 0.},
      // Plastic strain components
      {"plastic_strain0", 0.},
      {"plastic_strain1", 0.},
      {"plastic_strain2", 0.},
      {"plastic_strain3", 0.},
      {"plastic_strain4", 0.},
      {"plastic_strain5", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::MohrCoulomb<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {"yield_state",
                                               "phi",
                                               "psi",
                                               "cohesion",
                                               "tension_cutoff",
                                               "pressure",
                                               "tau",
                                               "theta",
                                               "pdstrain",
                                               "plastic_strain0",
                                               "plastic_strain1",
                                               "plastic_strain2",
                                               "plastic_strain3",
                                               "plastic_strain4",
                                               "plastic_strain5"};
  return state_vars;
}

//! Compute stress invariants
template <unsigned Tdim>
bool mpm::MohrCoulomb<Tdim>::compute_stress_invariants(
    const Vector6d& stress, mpm::dense_map* state_vars) {
  // Compute the mean pressure
  (*state_vars).at("pressure") = -mpm::materials::p(stress);
  // Compute theta value
  (*state_vars).at("theta") = mpm::materials::lode_angle(stress);
  // Compute tau
  (*state_vars).at("tau") = std::sqrt(mpm::materials::j2(stress));

  return true;
}

//! Compute yield function and yield state
template <unsigned Tdim>
typename mpm::mohrcoulomb::FailureState
    mpm::MohrCoulomb<Tdim>::compute_yield_state(
        Eigen::Matrix<double, 2, 1>* yield_function,
        const mpm::dense_map& state_vars) {

  // Get stress invariants
  const double epsilon = -state_vars.at("pressure") * std::sqrt(3.);
  const double rho = std::sqrt(2.0) * state_vars.at("tau");
  const double theta = state_vars.at("theta");

  // Get MC parameters
  const double phi = state_vars.at("phi");
  const double cohesion = state_vars.at("cohesion");
  const double tension_cutoff = state_vars.at("tension_cutoff");

  // Compute yield functions (tension & shear)
  // Tension
  (*yield_function)(0) = std::sqrt(2. / 3.) * cos(theta) * rho +
                         epsilon / std::sqrt(3.) - tension_cutoff;
  // Shear
  (*yield_function)(1) =
      std::sqrt(1.5) * rho *
          ((sin(theta + M_PI / 3.) / (std::sqrt(3.) * cos(phi))) +
           (cos(theta + M_PI / 3.) * tan(phi) / 3.)) +
      (epsilon / std::sqrt(3.)) * tan(phi) - cohesion;

  // Initialise yield status (0: elastic, 1: shear failure, 2: tensile failure)
  auto yield_type = mpm::mohrcoulomb::FailureState::Elastic;
  // Check for tension and shear
  if ((*yield_function)(0) > 0.0 && (*yield_function)(1) > 0.0)
    yield_type = mpm::mohrcoulomb::FailureState::ShearTensile;
  // Shear failure
  if ((*yield_function)(0) <= 0.0 && (*yield_function)(1) > 0.0)
    yield_type = mpm::mohrcoulomb::FailureState::Shear;
  // Tension failure
  if ((*yield_function)(0) > 0.0 && (*yield_function)(1) <= 0.0)
    yield_type = mpm::mohrcoulomb::FailureState::Tensile;

  return yield_type;
}

//! Compute dF/dSigma and dP/dSigma
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::compute_df_dp(
    mpm::mohrcoulomb::FailureState yield_type, const mpm::dense_map* state_vars,
    const Vector6d& stress, Vector6d* df_dsigma, Vector6d* dp_dsigma,
    double* dp_dq, double* softening, double pdstrain) {
  // Get stress invariants
  const double rho = std::sqrt(2.0) * (*state_vars).at("tau");
  const double theta = (*state_vars).at("theta");
  // Get MC parameters
  const double phi = (*state_vars).at("phi");
  const double psi = (*state_vars).at("psi");
  const double tension_cutoff = (*state_vars).at("tension_cutoff");

  // Compute dF / dEpsilon,  dF / dRho, dF / dTheta
  double df_depsilon, df_drho, df_dtheta;
  // Values in tension yield
  if (yield_type == mpm::mohrcoulomb::FailureState::Tensile) {
    df_depsilon = 1. / std::sqrt(3.);
    df_drho = std::sqrt(2. / 3.) * cos(theta);
    df_dtheta = -std::sqrt(2. / 3.) * rho * sin(theta);
  }
  // Values in shear yield / elastic
  else if (yield_type == mpm::mohrcoulomb::FailureState::Shear) {
    df_depsilon = tan(phi) / std::sqrt(3.);
    df_drho = std::sqrt(1.5) *
              ((sin(theta + M_PI / 3.) / (std::sqrt(3.) * cos(phi))) +
               (cos(theta + M_PI / 3.) * tan(phi) / 3.));
    df_dtheta = std::sqrt(1.5) * rho *
                ((cos(theta + M_PI / 3.) / (std::sqrt(3.) * cos(phi))) -
                 (sin(theta + M_PI / 3.) * tan(phi) / 3.));
  } else {
    // Throw error for other yield types
    console_->error("Invalid yield type for compute_df_dp: {}\n",
                    static_cast<int>(yield_type));
  }

  // Compute dEpsilon / dSigma
  Vector6d depsilon_dsigma = mpm::materials::dp_dsigma() * std::sqrt(3.);
  // Initialise dRho / dSigma
  Vector6d drho_dsigma = mpm::materials::dq_dsigma(stress) * std::sqrt(2. / 3.);
  // Compute dtheta / dsigma
  Vector6d dtheta_dsigma = mpm::materials::dtheta_dsigma(
      stress, std::numeric_limits<double>::epsilon());
  // Compute dF/dSigma
  (*df_dsigma) = (df_depsilon * depsilon_dsigma) + (df_drho * drho_dsigma) +
                 (df_dtheta * dtheta_dsigma);

  // Compute dp/dsigma and dp/dj in tension yield
  if (yield_type == mpm::mohrcoulomb::FailureState::Tensile) {
    // Define deviatoric eccentricity
    const double et_value = 0.6;
    // Define meridional eccentricity
    const double xit = 0.1;
    // Compute Rt
    double sqpart = 4. * (1 - et_value * et_value) * cos(theta) * cos(theta) +
                    5. * et_value * et_value - 4. * et_value;
    if (sqpart < std::numeric_limits<double>::epsilon()) sqpart = 1.E-5;
    double rt_den = 2. * (1 - et_value * et_value) * cos(theta) +
                    (2. * et_value - 1) * std::sqrt(sqpart);
    const double rt_num =
        4. * (1 - et_value * et_value) * cos(theta) * cos(theta) +
        (2. * et_value - 1) * (2. * et_value - 1);
    if (fabs(rt_den) < std::numeric_limits<double>::epsilon()) rt_den = 1.E-5;
    const double rt = rt_num / (3. * rt_den);
    // Compute dP/dRt
    const double dp_drt =
        1.5 * rho * rho * rt /
        check_low(std::sqrt(xit * xit * tension_cutoff * tension_cutoff +
                            1.5 * rt * rt * rho * rho));
    // Compute dP/dRho
    const double dp_drho =
        1.5 * rho * rt * rt /
        check_low(std::sqrt(xit * xit * tension_cutoff * tension_cutoff +
                            1.5 * rt * rt * rho * rho));
    // Compute dP/dEpsilon
    const double dp_depsilon = 1. / std::sqrt(3.);
    // Compute dRt/dThera
    const double drtden_dtheta =
        -2. * (1 - et_value * et_value) * sin(theta) -
        (2. * et_value - 1) * 4. * (1 - et_value * et_value) * cos(theta) *
            sin(theta) /
            std::sqrt(4. * (1 - et_value * et_value) * cos(theta) * cos(theta) +
                      5. * et_value * et_value - 4. * et_value);
    const double drtnum_dtheta =
        -8. * (1 - et_value * et_value) * cos(theta) * sin(theta);
    const double drt_dtheta =
        (drtnum_dtheta * rt_den - drtden_dtheta * rt_num) /
        (3. * rt_den * rt_den);
    // Compute dP/dSigma
    (*dp_dsigma) = (dp_depsilon * depsilon_dsigma) + (dp_drho * drho_dsigma) +
                   (dp_drt * drt_dtheta * dtheta_dsigma);
    // Compute dP/dJ
    (*dp_dq) = dp_drho * std::sqrt(2. / 3.);
  }
  // Compute dp/dsigma and dp/dj in shear yield
  else if (yield_type == mpm::mohrcoulomb::FailureState::Shear) {
    // Compute Rmc
    const double r_mc = (3. - sin(phi)) / (6 * cos(phi));
    // Compute deviatoric eccentricity
    double e_val = (3. - sin(phi)) / (3. + sin(phi));
    if (e_val <= 0.5) e_val = 0.5 + 1.E-10;
    if (e_val > 1.) e_val = 1.;
    // Compute Rmw
    double sqpart = (4. * (1 - e_val * e_val) * std::pow(cos(theta), 2)) +
                    (5 * e_val * e_val) - (4. * e_val);
    if (sqpart < std::numeric_limits<double>::epsilon()) sqpart = 1.E-5;
    double m = (2. * (1 - e_val * e_val) * cos(theta)) +
               ((2. * e_val - 1) * std::sqrt(sqpart));
    if (fabs(m) < std::numeric_limits<double>::epsilon()) m = 1.E-5;
    const double l = (4. * (1. - e_val * e_val) * std::pow(cos(theta), 2)) +
                     std::pow((2. * e_val - 1.), 2);
    const double r_mw = (l / m) * r_mc;
    // Initialise meridional eccentricity
    const double xi = 0.1;
    double omega = std::pow((xi * cohesion_peak_ * tan(psi)), 2) +
                   std::pow((r_mw * std::sqrt(1.5) * rho), 2);
    if (omega < std::numeric_limits<double>::epsilon()) omega = 1.E-5;
    const double dl_dtheta =
        -8. * (1. - e_val * e_val) * cos(theta) * sin(theta);
    const double dm_dtheta =
        (-2. * (1. - e_val * e_val) * sin(theta)) +
        (0.5 * (2. * e_val - 1.) * dl_dtheta) / std::sqrt(sqpart);
    const double drmw_dtheta = ((m * dl_dtheta) - (l * dm_dtheta)) / (m * m);
    const double dp_depsilon = tan(psi) / std::sqrt(3.);
    const double dp_drho = 3. * rho * r_mw * r_mw / (2. * std::sqrt(omega));
    const double dp_dtheta =
        (3. * rho * rho * r_mw * r_mc * drmw_dtheta) / (2. * std::sqrt(omega));
    // compute the value of dp/dsigma and dp/dj in shear yield
    (*dp_dsigma) = (dp_depsilon * depsilon_dsigma) + (dp_drho * drho_dsigma) +
                   (dp_dtheta * dtheta_dsigma);
    (*dp_dq) = dp_drho * std::sqrt(2. / 3.);
  }

  // Compute softening part
  (*softening) = 0.;
  if (softening_ && pdstrain > pdstrain_peak_ &&
      pdstrain < pdstrain_residual_) {
    // Compute dPhi/dPstrain
    double dphi_dpstrain =
        (phi_residual_ - phi_peak_) / (pdstrain_residual_ - pdstrain_peak_);
    // Compute dc/dPstrain
    double dc_dpstrain = (cohesion_residual_ - cohesion_peak_) /
                         (pdstrain_residual_ - pdstrain_peak_);
    // Compute dF/dPstrain
    double df_dphi =
        std::sqrt(1.5) * rho *
            ((sin(phi) * sin(theta + M_PI / 3.) /
              (std::sqrt(3.) * cos(phi) * cos(phi))) +
             (cos(theta + M_PI / 3.) / (3. * cos(phi) * cos(phi)))) +
        (mpm::materials::p(stress) / (cos(phi) * cos(phi)));
    double df_dc = -1.;
    (*softening) =
        (-1.) * ((df_dphi * dphi_dpstrain) + (df_dc * dc_dpstrain)) * (*dp_dq);
  }
}

template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::MohrCoulomb<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {

  // Check density criterion for tensile separation.
  const double current_packing_density = ptr->mass_density();
  const double critical_density = minimum_packing_fraction_ * grain_density_;

  // Compute elastic stiffness.
  (*state_vars).at("yield_state") = 0;
  Matrix6x6 de = this->compute_elastic_tensor(state_vars);
  Vector6d trial_stress =
      this->compute_trial_stress(stress, dstrain, de, ptr, state_vars);

  // Separated state: current packing density is less than critical density
  if (current_packing_density <= critical_density) {
    (*state_vars).at("pdstrain") +=
        mpm::materials::q(trial_stress) / 3.0 / shear_modulus_;
    (*state_vars).at("yield_state") = 4;
    return Vector6d::Zero();
  }

  // Compute stress invariants based on trial stress
  this->compute_stress_invariants(trial_stress, state_vars);

  // Evaluate yield functions at trial stress.
  Eigen::Matrix<double, 2, 1> yield_function;
  auto yield_type = this->compute_yield_state(&yield_function, (*state_vars));

  // Elastic admissibility: if both surfaces are satisfied, accept trial stress.
  if (yield_type == mpm::mohrcoulomb::FailureState::Elastic) {
    (*state_vars).at("yield_state") = 0;
    return trial_stress;
  }

  // If plastic, do return mapping
  bool converged = false;
  Vector6d current_stress = trial_stress;
  Vector6d dstrain_p_tensile = Vector6d::Zero();
  Vector6d dstrain_p_shear = Vector6d::Zero();
  Eigen::Matrix<double, 2, 1> unknowns =
      Eigen::Matrix<double, 2, 1>::Constant(tolerance_);
  Eigen::Matrix<double, 2, 1> dlambda = Vector2d::Zero();

  // Updated state variables for return mapping iterations
  Vector6d plastic_strain = Vector6d::Zero();
  plastic_strain(0) = (*state_vars).at("plastic_strain0");
  plastic_strain(1) = (*state_vars).at("plastic_strain1");
  plastic_strain(2) = (*state_vars).at("plastic_strain2");
  plastic_strain(3) = (*state_vars).at("plastic_strain3");
  plastic_strain(4) = (*state_vars).at("plastic_strain4");
  plastic_strain(5) = (*state_vars).at("plastic_strain5");
  double pdstrain = (*state_vars).at("pdstrain");

  // Start newton raphson return mapping iterations
  for (unsigned itr = 0; itr < max_iter_; ++itr) {
    Vector6d stress_correction = Vector6d::Zero();

    // Return mapping in the shear-tensile region
    if (yield_type == mpm::mohrcoulomb::FailureState::ShearTensile) {
      auto [corner_correction, dstrain_p_tensile, dstrain_p_shear] =
          this->compute_corner_return(current_stress, de, state_vars,
                                      yield_function, unknowns, dlambda,
                                      pdstrain);
      stress_correction = corner_correction;
    } else {
      double yield_function_single =
          (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
              ? yield_function(0)
              : yield_function(1);
      double u_single = (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
                            ? unknowns(0)
                            : unknowns(1);
      double dlambda_single =
          (yield_type == mpm::mohrcoulomb::FailureState::Tensile) ? dlambda(0)
                                                                  : dlambda(1);

      auto [single_correction, dstrain_p] = this->compute_single_surface_return(
          yield_type, yield_function_single, current_stress, de, state_vars,
          u_single, dlambda_single, pdstrain);
      stress_correction = single_correction;
      if (yield_type == mpm::mohrcoulomb::FailureState::Tensile) {
        dstrain_p_tensile = dstrain_p;
        unknowns(0) = u_single;
        dlambda(0) = dlambda_single;
      } else {
        dstrain_p_shear = dstrain_p;
        unknowns(1) = u_single;
        dlambda(1) = dlambda_single;
      }
    }

    // Update stress and plastic strain
    current_stress = trial_stress + stress_correction;
    new_plastic_strain = plastic_strain + dstrain_p_shear + dstrain_p_tensile;
    pdstrain = mpm::materials::pdstrain(new_plastic_strain);

    // Update softening-dependent strength parameters.
    if (softening_) {
      this->update_softening_parameters(state_vars);
    }

    // Re-evaluate invariants and active surface at current stress.
    this->compute_stress_invariants(current_stress, state_vars);
    yield_type = this->compute_yield_state(&yield_function, (*state_vars));

    // ---------------------------------------------------------------------
    // B. Convergence check
    // ---------------------------------------------------------------------
    if (yield_function(0) <= Tolerance && yield_function(1) <= Tolerance) {
      converged = true;
      // Assign final yield state near the converged point.
      if (yield_function(0) > -Tolerance && yield_function(1) <= -Tolerance) {
        (*state_vars).at("yield_state") = 2;  // Tensile
      } else if (yield_function(1) > -Tolerance &&
                 yield_function(0) <= -Tolerance) {
        (*state_vars).at("yield_state") = 1;  // Shear
      } else {
        (*state_vars).at("yield_state") = 0;  // Elastic
      }
      break;
    }
  }

  // =========================================================================
  // 4. Post-convergence handling
  // =========================================================================
  if (!converged) {
    // Warn when return mapping fails to converge within the iteration budget.
    console_->warn(
        "MohrCoulomb::compute_stress: Return mapping did not converge "
        "after {} iterations. Final yield functions: f_t = {}, f_s = {}",
        max_iter_, yield_function(0), yield_function(1));
  }

  // Store final stress invariants.
  this->compute_stress_invariants(current_stress, state_vars);

  return current_stress;
}

//! Single-surface return mapping
//! Returns: (stress correction, plastic deviatoric strain increment, success
//! flag)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>>
    mpm::MohrCoulomb<Tdim>::compute_single_surface_return(
        mpm::mohrcoulomb::FailureState yield_type, double yield_function,
        const Vector6d& current_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars, double& u_single, double& dlambda_single,
        double pdstrain) {

  // Select active yield-function value on the current surface.
  double f_current = yield_function;

  // Compute flow/yield gradients and softening contribution.
  Vector6d df_dsigma = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  double dp_dq = 0.0;
  double softening_modulus = 0.0;

  this->compute_df_dp(yield_type, state_vars, current_stress, &df_dsigma,
                      &dp_dsigma, &dp_dq, &softening_modulus, pdstrain);

  // Consistency denominator:
  // n^T De m + H, where n = dF/dsigma, m = dP/dsigma, H = softening term.
  double denominator =
      2.0 * u_single * (df_dsigma.transpose() * de).dot(dp_dsigma) +
      softening_modulus;

  if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
    console_->warn(
        "MohrCoulomb: Consistency denominator is near zero ({:.6e}). "
        "This may indicate numerical instability. Applying regularization.",
        denominator);
    denominator = check_low(denominator, 1e-5);
  }

  // Plastic multiplier increment from first-order consistency.
  double d_u = -f_current / denominator;
  u_single += d_u;
  dlambda_single = u_single * u_single;

  // Stress update: Delta sigma = -dlambda * De * dP/dsigma.
  Vector6d stress_correction = -dlambda * de * dp_dsigma;
  // Plastic strain increment
  Vector6d dstrain_p = dlambda * dp_dsigma;

  return {stress_correction, dstrain_p};
}

//! Corner return mapping (multi-surface)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>,
           Eigen::Matrix<double, 6, 1>>
    mpm::MohrCoulomb<Tdim>::compute_corner_return(
        const Vector6d& current_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars,
        const Eigen::Matrix<double, 2, 1>& yield_function,
        Eigen::Matrix<double, 2, 1>& unknowns,
        Eigen::Matrix<double, 2, 1>& dlambda, double pdstrain) {

  // Compute gradients for both active surfaces.
  Vector6d df_dsigma_t = Vector6d::Zero();  // Tension
  Vector6d dp_dsigma_t = Vector6d::Zero();
  double dp_dq_t = 0.0;
  double softening_t = 0.0;

  Vector6d df_dsigma_s = Vector6d::Zero();  // Shear
  Vector6d dp_dsigma_s = Vector6d::Zero();
  double dp_dq_s = 0.0;
  double softening_s = 0.0;

  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars,
                      current_stress, &df_dsigma_t, &dp_dsigma_t, &dp_dq_t,
                      &softening_t, pdstrain);
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars,
                      current_stress, &df_dsigma_s, &dp_dsigma_s, &dp_dq_s,
                      &softening_s, pdstrain);

  // Recompute yield-function values at the current stress.
  const double f_t = yield_function(0);  // Tension
  const double f_s = yield_function(1);  // Shear

  // =========================================================================
  // Solve 2x2 consistency system for the two plastic multipliers:
  // [A11 A12] [lambda_t]   [f_t]
  // [A21 A22] [lambda_s] = [f_s]
  // with A_ij = n_i^T De m_j + H_i delta_ij.
  // =========================================================================
  Eigen::Matrix<double, 2, 2> A;
  Eigen::Vector2d b;

  // Assemble system matrix.
  Vector6d de_dp_t = de * dp_dsigma_t;
  Vector6d de_dp_s = de * dp_dsigma_s;

  A(0, 0) = df_dsigma_t.dot(de_dp_t) + softening_t;
  A(0, 1) = df_dsigma_t.dot(de_dp_s);
  A(1, 0) = df_dsigma_s.dot(de_dp_t);
  A(1, 1) = df_dsigma_s.dot(de_dp_s) + softening_s;

  b(0) = f_t;
  b(1) = f_s;

  // Check determinant for singular/ill-conditioned corner system.
  double det_A = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);

  if (std::abs(det_A) < std::numeric_limits<double>::epsilon()) {
    // Singular or near-singular system: corner return is not reliable.
    console_->error(
        "MohrCoulomb: Corner return matrix is singular (det = {:.6e}). "
        "Falling back to single surface return.",
        det_A);
    return {Vector6d::Zero(), Vector6d::Zero(), Vector6d::Zero()};
  }

  // Quadratic transformation to ensure positive lambdas
  A(0, 0) *= 2.0 * unknowns(0);
  A(0, 1) *= 2.0 * unknowns(1);
  A(1, 0) *= 2.0 * unknowns(0);
  A(1, 1) *= 2.0 * unknowns(1);

  // Solve explicitly via inverse (closed form for 2x2).
  Eigen::Matrix<double, 2, 2> A_inv;
  A_inv(0, 0) = A(1, 1) / det_A;
  A_inv(0, 1) = -A(0, 1) / det_A;
  A_inv(1, 0) = -A(1, 0) / det_A;
  A_inv(1, 1) = A(0, 0) / det_A;

  Eigen::Vector2d delta_u = -A_inv * b;
  unknowns += delta_u;
  dlambda(0) = unknowns(0) * unknowns(0);
  dlambda(1) = unknowns(1) * unknowns(1);

  // =========================================================================
  // Plastic multiplier admissibility checks
  // =========================================================================
  bool success = true;

  // =========================================================================
  // Stress correction by Koiter's rule:
  // Delta sigma = -De * (lambda_t * m_t + lambda_s * m_s)
  // =========================================================================
  Vector6d stress_correction =
      -de * (dlambda(0) * dp_dsigma_t + dlambda(1) * dp_dsigma_s);

  // Plastic strain increment from each mechanism.
  Vector6d dstrain_p_tensile = dlambda(0) * dp_dsigma_t;
  Vector6d dstrain_p_shear = dlambda(1) * dp_dsigma_s;

  return {stress_correction, dstrain_p_tensile, dstrain_p_shear, success};
}

//! Update softening parameters
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::update_softening_parameters(
    mpm::dense_map* state_vars, double pdstrain) {
  if (pdstrain <= pdstrain_peak_) {
    // Pre-peak: keep peak strength parameters.
    (*state_vars).at("phi") = phi_peak_;
    (*state_vars).at("psi") = psi_peak_;
    (*state_vars).at("cohesion") = cohesion_peak_;
  } else if (pdstrain >= pdstrain_residual_) {
    // Post-residual: keep residual strength parameters.
    (*state_vars).at("phi") = phi_residual_;
    (*state_vars).at("psi") = psi_residual_;
    (*state_vars).at("cohesion") = cohesion_residual_;
  } else {
    // Linear softening law between peak and residual states.
    const double ratio =
        (pdstrain - pdstrain_peak_) / (pdstrain_residual_ - pdstrain_peak_);
    (*state_vars).at("phi") = phi_peak_ + ratio * (phi_residual_ - phi_peak_);
    (*state_vars).at("psi") = psi_peak_ + ratio * (psi_residual_ - psi_peak_);
    (*state_vars).at("cohesion") =
        cohesion_peak_ + ratio * (cohesion_residual_ - cohesion_peak_);
  }

  // Enforce apex condition: sigma_t <= c / tan(phi) for MC cone closure.
  const double phi = (*state_vars).at("phi");
  const double cohesion = (*state_vars).at("cohesion");

  // Guard against near-zero tan(phi).
  const double tan_phi = std::tan(phi);
  const double min_tan_phi = 1.0e-10;

  double apex;
  if (std::abs(tan_phi) < min_tan_phi) {
    // For phi ~= 0, apex tends to a very large value.
    apex = std::numeric_limits<double>::max();
  } else {
    apex = cohesion / tan_phi;
  }

  // Clamp tension cutoff by apex.
  if ((*state_vars).at("tension_cutoff") > apex) {
    (*state_vars).at("tension_cutoff") = apex;
  }

  // Keep tensile cutoff non-negative.
  if ((*state_vars).at("tension_cutoff") < 0.0) {
    (*state_vars).at("tension_cutoff") = 0.0;
  }
}

//! Compute elastic tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6> mpm::MohrCoulomb<Tdim>::compute_elastic_tensor(
    mpm::dense_map* state_vars) {
  // Shear modulus
  const double G = shear_modulus_;
  const double a1 = bulk_modulus_ + (4.0 / 3.0) * G;
  const double a2 = bulk_modulus_ - (2.0 / 3.0) * G;
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
    mpm::MohrCoulomb<Tdim>::compute_elasto_plastic_tensor(
        const Vector6d& stress, const Vector6d& dstrain,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt,
        bool hardening) {

  mpm::mohrcoulomb::FailureState yield_type =
      yield_type_.at(int((*state_vars).at("yield_state")));
  // Return the updated stress in elastic state
  const Matrix6x6 de = this->compute_elastic_tensor(state_vars);
  if (yield_type == mpm::mohrcoulomb::FailureState::Elastic) {
    return de;
  }

  // Return zero tensor matrix in separated state
  if (yield_type == mpm::mohrcoulomb::FailureState::Separated) {
    return Matrix6x6::Zero();
  }

  //! Elasto-plastic stiffness matrix
  Matrix6x6 d_ep;
  // Compute df_dsigma dp_dsigma
  double softening = 0.;
  double dp_dq = 0.;
  Vector6d df_dsigma = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  // Compute stress invariants based on trial stress
  this->compute_stress_invariants(stress, state_vars);
  this->compute_df_dp(yield_type, state_vars, stress, &df_dsigma, &dp_dsigma,
                      &dp_dq, &softening);

  // Compute the d_ep tensor
  Eigen::Matrix<double, 6, 1> de_dpdsigma = de * dp_dsigma;
  double dfdsigma_de_dpdsigma = df_dsigma.dot(de_dpdsigma);
  Eigen::Matrix<double, 6, 1> de_dfdsigma = de * df_dsigma;

  if (!hardening) softening = 0.;
  d_ep = de - 1. / (dfdsigma_de_dpdsigma + softening) *
                  (de_dpdsigma * de_dfdsigma.transpose());

  return d_ep;
}
