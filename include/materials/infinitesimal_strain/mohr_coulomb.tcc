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
      {"pdstrain", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::MohrCoulomb<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {
      "yield_state", "phi", "psi",   "cohesion", "tension_cutoff",
      "pressure",    "tau", "theta", "pdstrain"};
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
  // Tolerance for yield function
  const double Tolerance = 1E-7;
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
  if ((*yield_function)(0) > Tolerance && (*yield_function)(1) > Tolerance) {
    // Compute tension and shear edge parameters
    const double n_phi = (1. + sin(phi)) / (1. - sin(phi));
    const double sigma_p =
        tension_cutoff * n_phi - 2. * cohesion * std::sqrt(n_phi);
    const double alpha_p = std::sqrt(1. + n_phi * n_phi) + n_phi;
    // Compute the shear-tension edge
    const double h =
        (*yield_function)(0) +
        alpha_p * (std::sqrt(2. / 3.) * cos(theta - 4. * M_PI / 3.) * rho +
                   epsilon / std::sqrt(3.) - sigma_p);
    // Tension
    if (h > std::numeric_limits<double>::epsilon())
      yield_type = mpm::mohrcoulomb::FailureState::Tensile;
    // Shear
    else
      yield_type = mpm::mohrcoulomb::FailureState::Shear;
  }
  // Shear failure
  if ((*yield_function)(0) < Tolerance && (*yield_function)(1) > Tolerance)
    yield_type = mpm::mohrcoulomb::FailureState::Shear;
  // Tension failure
  if ((*yield_function)(0) > Tolerance && (*yield_function)(1) < Tolerance)
    yield_type = mpm::mohrcoulomb::FailureState::Tensile;

  return yield_type;
}

//! Compute dF/dSigma and dP/dSigma
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::compute_df_dp(
    mpm::mohrcoulomb::FailureState yield_type, const mpm::dense_map* state_vars,
    const Vector6d& stress, Vector6d* df_dsigma, Vector6d* dp_dsigma,
    double* dp_dq, double* softening) {
  // Get stress invariants
  const double rho = std::sqrt(2.0) * state_vars.at("tau");
  const double theta = (*state_vars).at("theta");
  // Get MC parameters
  const double phi = (*state_vars).at("phi");
  const double psi = (*state_vars).at("psi");
  const double tension_cutoff = (*state_vars).at("tension_cutoff");
  // Get equivalent plastic deviatoric strain
  const double pdstrain = (*state_vars).at("pdstrain");
  // Compute dF / dEpsilon,  dF / dRho, dF / dTheta
  double df_depsilon, df_drho, df_dtheta;
  // Values in tension yield
  if (yield_type == mpm::mohrcoulomb::FailureState::Tensile) {
    df_depsilon = 1. / std::sqrt(3.);
    df_drho = std::sqrt(2. / 3.) * cos(theta);
    df_dtheta = -std::sqrt(2. / 3.) * rho * sin(theta);
  }
  // Values in shear yield / elastic
  else {
    df_depsilon = tan(phi) / std::sqrt(3.);
    df_drho = std::sqrt(1.5) *
              ((sin(theta + M_PI / 3.) / (std::sqrt(3.) * cos(phi))) +
               (cos(theta + M_PI / 3.) * tan(phi) / 3.));
    df_dtheta = std::sqrt(1.5) * rho *
                ((cos(theta + M_PI / 3.) / (std::sqrt(3.) * cos(phi))) -
                 (sin(theta + M_PI / 3.) * tan(phi) / 3.));
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
  else {
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
  double dphi_dpstrain = 0.;
  double dc_dpstrain = 0.;
  (*softening) = 0.;
  if (softening_ && pdstrain > pdstrain_peak_ &&
      pdstrain < pdstrain_residual_) {
    // Compute dPhi/dPstrain
    dphi_dpstrain =
        (phi_residual_ - phi_peak_) / (pdstrain_residual_ - pdstrain_peak_);
    // Compute dc/dPstrain
    dc_dpstrain = (cohesion_residual_ - cohesion_peak_) /
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

  // =========================================================================
  // 1. Constants and initialization
  // =========================================================================
  const double Tolerance = 1E-7;  // Yield-function tolerance
  const unsigned itr_max = 50;    // Maximum return-mapping iterations

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
    (*state_vars).at("yield_state") = 3;
    return Vector6d::Zero();
  }
  // Compute stress invariants based on trial stress
  this->compute_stress_invariants(trial_stress, state_vars);

  // Evaluate yield functions at trial stress.
  Eigen::Matrix<double, 2, 1> yield_function;
  auto yield_type = this->compute_yield_state(&yield_function, (*state_vars));

  // Elastic admissibility: if both surfaces are satisfied, accept trial stress.
  if (yield_function(0) <= Tolerance && yield_function(1) <= Tolerance) {
    (*state_vars).at("yield_state") = 0;
    return trial_stress;
  }

  // =========================================================================
  // 3. Plastic corrector (cutting-plane + corner return)
  //
  // The stress is projected back to the admissible domain by incremental
  // return mapping. A single violated surface uses one plastic multiplier;
  // simultaneous tension/shear violation uses a multi-surface corner return.
  // =========================================================================

  Vector6d current_stress = trial_stress;
  bool converged = false;

  for (unsigned itr = 0; itr < itr_max; ++itr) {
    // ---------------------------------------------------------------------
    // A. Re-evaluate invariants and active surface at current stress.
    // ---------------------------------------------------------------------
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

    // ---------------------------------------------------------------------
    // C. Determine active mechanism and choose return strategy.
    // ---------------------------------------------------------------------
    const bool tension_violated = (yield_function(0) > Tolerance);
    const bool shear_violated = (yield_function(1) > Tolerance);

    Vector6d stress_correction = Vector6d::Zero();
    double dpdstrain_inc = 0.0;

    if (tension_violated && shear_violated) {
      // =================================================================
      // Corner return mapping (multi-surface plasticity)
      // Used when both tension and shear surfaces are violated.
      // =================================================================
      auto [corner_correction, corner_dpdstrain, corner_success] =
          this->compute_corner_return(current_stress, de, state_vars);

      if (corner_success) {
        stress_correction = corner_correction;
        dpdstrain_inc = corner_dpdstrain;
      } else {
        // If corner return fails, fall back to single-surface return on the
        // most violated surface.
        if (yield_function(0) > yield_function(1)) {
          yield_type = mpm::mohrcoulomb::FailureState::Tensile;
        } else {
          yield_type = mpm::mohrcoulomb::FailureState::Shear;
        }
        const auto single_return = this->compute_single_surface_return(
            yield_type, yield_function, current_stress, de, state_vars);
        stress_correction = std::get<0>(single_return);
        dpdstrain_inc = std::get<1>(single_return);
      }
    } else {
      // =================================================================
      // Single-surface return mapping
      // =================================================================
      if (tension_violated) {
        yield_type = mpm::mohrcoulomb::FailureState::Tensile;
        (*state_vars).at("yield_state") = 2;
      } else {
        yield_type = mpm::mohrcoulomb::FailureState::Shear;
        (*state_vars).at("yield_state") = 1;
      }

      const auto single_return = this->compute_single_surface_return(
          yield_type, yield_function, current_stress, de, state_vars);
      stress_correction = std::get<0>(single_return);
      dpdstrain_inc = std::get<1>(single_return);
    }

    // ---------------------------------------------------------------------
    // D. Update stress and internal variable.
    // ---------------------------------------------------------------------
    current_stress += stress_correction;
    (*state_vars).at("pdstrain") += dpdstrain_inc;

    // Update softening-dependent strength parameters.
    if (softening_) {
      this->update_softening_parameters(state_vars);
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
        itr_max, yield_function(0), yield_function(1));
  }

  // Store final stress invariants.
  this->compute_stress_invariants(current_stress, state_vars);

  return current_stress;
}
//! Single-surface return mapping
//! Returns: (stress correction, plastic deviatoric strain increment, success
//! flag)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, double, bool>
    mpm::MohrCoulomb<Tdim>::compute_single_surface_return(
        mpm::mohrcoulomb::FailureState yield_type,
        const Eigen::Matrix<double, 2, 1>& yield_function,
        const Vector6d& current_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars) {

  const double Min_Denominator = 1E-15;

  // Select active yield-function value on the current surface.
  double f_current = (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
                         ? yield_function(0)
                         : yield_function(1);

  // Compute flow/yield gradients and softening contribution.
  Vector6d df_dsigma = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  double dp_dq = 0.0;
  double softening_modulus = 0.0;

  this->compute_df_dp(yield_type, state_vars, current_stress, &df_dsigma,
                      &dp_dsigma, &dp_dq, &softening_modulus);

  // Consistency denominator:
  // n^T De m + H, where n = dF/dsigma, m = dP/dsigma, H = softening term.
  double denominator =
      (df_dsigma.transpose() * de).dot(dp_dsigma) + softening_modulus;

  // =========================================================================
  // Denominator regularization
  // =========================================================================
  bool success = true;
  if (denominator < Min_Denominator) {
    if (denominator < -Min_Denominator) {
      // Negative denominator indicates local instability due to strong
      // softening.
      console_->warn(
          "MohrCoulomb: Negative denominator detected ({:.6e}). "
          "This indicates material instability (strong softening). "
          "Using regularization.",
          denominator);
      // Regularize to a small positive value.
      denominator = Min_Denominator;
      success = false;  // Mark partially successful correction.
    } else {
      // Near-zero denominator implies numerical singularity.
      console_->debug(
          "MohrCoulomb: Near-zero denominator detected ({:.6e}). "
          "Applying regularization.",
          denominator);
      denominator = Min_Denominator;
    }
  }

  // Plastic multiplier increment from first-order consistency.
  double dlambda = f_current / denominator;

  // Physical admissibility check.
  if (dlambda < 0.0) {
    // Negative plastic multiplier is non-admissible for this active set.
    console_->warn(
        "MohrCoulomb: Negative plastic multiplier ({:.6e}). Setting to zero.",
        dlambda);
    dlambda = 0.0;
    success = false;
  }

  // Clamp very large multiplier increments for numerical robustness.
  const double max_dlambda = 1.0;  // Can be tuned per problem class.
  if (dlambda > max_dlambda) {
    console_->debug(
        "MohrCoulomb: Large plastic multiplier ({:.6e}). Clamping to {:.6e}.",
        dlambda, max_dlambda);
    dlambda = max_dlambda;
  }

  // Stress update: Delta sigma = -dlambda * De * dP/dsigma.
  Vector6d stress_correction = -dlambda * de * dp_dsigma;
  // Internal variable increment from equivalent plastic deviatoric measure.
  double dpdstrain = dlambda * dp_dq;

  return {stress_correction, dpdstrain, success};
}

//! Corner return mapping (multi-surface)
//! Applies Koiter's rule when both tension and shear surfaces are active.
//! Returns: (stress correction, plastic deviatoric strain increment, success
//! flag)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, double, bool>
    mpm::MohrCoulomb<Tdim>::compute_corner_return(
        const Vector6d& current_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars) {

  const double Min_Denominator = 1E-15;

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
                      &softening_t);
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars,
                      current_stress, &df_dsigma_s, &dp_dsigma_s, &dp_dq_s,
                      &softening_s);

  // Recompute yield-function values at the current stress.
  Eigen::Matrix<double, 2, 1> yield_function;
  this->compute_yield_state(&yield_function, (*state_vars));
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

  if (std::abs(det_A) < Min_Denominator) {
    // Singular or near-singular system: corner return is not reliable.
    console_->debug(
        "MohrCoulomb: Corner return matrix is singular (det = {:.6e}). "
        "Falling back to single surface return.",
        det_A);
    return {Vector6d::Zero(), 0.0, false};
  }

  // Solve explicitly via inverse (closed form for 2x2).
  Eigen::Matrix<double, 2, 2> A_inv;
  A_inv(0, 0) = A(1, 1) / det_A;
  A_inv(0, 1) = -A(0, 1) / det_A;
  A_inv(1, 0) = -A(1, 0) / det_A;
  A_inv(1, 1) = A(0, 0) / det_A;

  Eigen::Vector2d lambdas = A_inv * b;
  double lambda_t = lambdas(0);
  double lambda_s = lambdas(1);

  // =========================================================================
  // Plastic multiplier admissibility checks
  // =========================================================================
  bool success = true;

  // Check for negative multipliers.
  if (lambda_t < 0.0 || lambda_s < 0.0) {
    if (lambda_t < 0.0 && lambda_s < 0.0) {
      // Both negative: active-set assumption is invalid.
      console_->debug(
          "MohrCoulomb: Both plastic multipliers negative (λ_t={:.6e}, "
          "λ_s={:.6e}). "
          "Corner return failed.",
          lambda_t, lambda_s);
      return {Vector6d::Zero(), 0.0, false};
    } else if (lambda_t < 0.0) {
      // Negative tension multiplier: prefer single shear-surface return.
      console_->debug(
          "MohrCoulomb: Tension plastic multiplier negative. "
          "Should use single shear surface return.");
      return {Vector6d::Zero(), 0.0, false};
    } else {
      // Negative shear multiplier: prefer single tension-surface return.
      console_->debug(
          "MohrCoulomb: Shear plastic multiplier negative. "
          "Should use single tension surface return.");
      return {Vector6d::Zero(), 0.0, false};
    }
  }

  // Clamp excessively large multipliers.
  const double max_lambda = 1.0;
  if (lambda_t > max_lambda) {
    lambda_t = max_lambda;
    success = false;
  }
  if (lambda_s > max_lambda) {
    lambda_s = max_lambda;
    success = false;
  }

  // =========================================================================
  // Stress correction by Koiter's rule:
  // Delta sigma = -De * (lambda_t * m_t + lambda_s * m_s)
  // =========================================================================
  Vector6d stress_correction =
      -de * (lambda_t * dp_dsigma_t + lambda_s * dp_dsigma_s);

  // Equivalent plastic deviatoric strain increment from both mechanisms.
  double dpdstrain = lambda_t * dp_dq_t + lambda_s * dp_dq_s;

  return {stress_correction, dpdstrain, success};
}

//! Update softening parameters
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::update_softening_parameters(
    mpm::dense_map* state_vars) {
  const double pdstrain = (*state_vars).at("pdstrain");

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
