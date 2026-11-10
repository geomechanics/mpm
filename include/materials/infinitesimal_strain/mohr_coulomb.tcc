//! Constructor with id and material properties
template <unsigned Tdim>
mpm::MohrCoulomb<Tdim>::MohrCoulomb(unsigned id,
                                    const Json& material_properties)
    : InfinitesimalElastoPlastic<Tdim>(id, material_properties) {
  try {
    // Density
    density_ = material_properties.at("density").template get<double>();

    // Initial packing fraction
    double initial_packing_fraction = 1.;
    if (material_properties.contains("packing_fraction"))
      initial_packing_fraction =
          material_properties.at("packing_fraction").template get<double>();
    else if (material_properties.contains("porosity"))
      initial_packing_fraction =
          1. - material_properties.at("porosity").template get<double>();

    // Solid grain density (two-phase if porosity and a permeability are given)
    const bool is_wet = material_properties.contains("porosity") &&
                        (material_properties.contains("k_x") ||
                         material_properties.contains("k_y") ||
                         material_properties.contains("k_z"));
    grain_density_ = is_wet ? density_ : density_ / initial_packing_fraction;

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
    // Minimum hyperbolic smoothing of the plastic potentials near the apex
    // (stress units). Needed for low / zero cohesion and tension cutoff.
    apex_smoothing_ = 0.;
    if (material_properties.contains("apex_smoothing"))
      apex_smoothing_ =
          material_properties.at("apex_smoothing").template get<double>();
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
      // Yield state: 0: elastic, 1: shear, 2: tensile, 3: shear-tensile,
      // 4: separated
      {"yield_state", 0},
      // Friction (phi)
      {"phi", this->phi_peak_},
      // Dilation (psi)
      {"psi", this->psi_peak_},
      // Cohesion
      {"cohesion", this->cohesion_peak_},
      // Tensile cutoff (limited by the apex value c / tan(phi))
      {"tension_cutoff",
       check_low((tension_cutoff_ < cohesion_peak_ / std::tan(phi_peak_))
                     ? tension_cutoff_
                     : cohesion_peak_ / std::tan(phi_peak_))},
      // Stress invariants
      {"pressure", 0.},
      {"tau", 0.},
      {"theta", 0.},
      // Plastic deviatoric strain
      {"pdstrain", 0.},
      // Plastic strain components
      {"plastic_strain_tensile0", 0.},
      {"plastic_strain_tensile1", 0.},
      {"plastic_strain_tensile2", 0.},
      {"plastic_strain_tensile3", 0.},
      {"plastic_strain_tensile4", 0.},
      {"plastic_strain_tensile5", 0.},
      {"plastic_strain_shear0", 0.},
      {"plastic_strain_shear1", 0.},
      {"plastic_strain_shear2", 0.},
      {"plastic_strain_shear3", 0.},
      {"plastic_strain_shear4", 0.},
      {"plastic_strain_shear5", 0.},
      // Number of return mapping iterations used in the last step
      {"max_itr", 0}};
  return state_vars;
}

//! State variable names
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
                                               "plastic_strain_tensile0",
                                               "plastic_strain_tensile1",
                                               "plastic_strain_tensile2",
                                               "plastic_strain_tensile3",
                                               "plastic_strain_tensile4",
                                               "plastic_strain_tensile5",
                                               "plastic_strain_shear0",
                                               "plastic_strain_shear1",
                                               "plastic_strain_shear2",
                                               "plastic_strain_shear3",
                                               "plastic_strain_shear4",
                                               "plastic_strain_shear5"};
  return state_vars;
}

//! Compute stress invariants
template <unsigned Tdim>
bool mpm::MohrCoulomb<Tdim>::compute_stress_invariants(
    const Vector6d& stress, mpm::dense_map* state_vars) {
  // Mean pressure
  (*state_vars).at("pressure") = -mpm::materials::p(stress);
  // Lode angle
  (*state_vars).at("theta") = mpm::materials::lode_angle(stress);
  // Tau = sqrt(J2)
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

  // Tension yield function
  (*yield_function)(0) = std::sqrt(2. / 3.) * cos(theta) * rho +
                         epsilon / std::sqrt(3.) - tension_cutoff;
  // Shear yield function
  (*yield_function)(1) =
      std::sqrt(1.5) * rho *
          ((sin(theta + M_PI / 3.) / (std::sqrt(3.) * cos(phi))) +
           (cos(theta + M_PI / 3.) * tan(phi) / 3.)) +
      (epsilon / std::sqrt(3.)) * tan(phi) - cohesion;

  // Determine yield type
  auto yield_type = mpm::mohrcoulomb::FailureState::Elastic;
  if ((*yield_function)(0) > 0.0 && (*yield_function)(1) > 0.0)
    yield_type = mpm::mohrcoulomb::FailureState::ShearTensile;
  else if ((*yield_function)(0) <= 0.0 && (*yield_function)(1) > 0.0)
    yield_type = mpm::mohrcoulomb::FailureState::Shear;
  else if ((*yield_function)(0) > 0.0 && (*yield_function)(1) <= 0.0)
    yield_type = mpm::mohrcoulomb::FailureState::Tensile;

  return yield_type;
}

//! Compute dF/dSigma and dP/dSigma
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::compute_df_dp(
    mpm::mohrcoulomb::FailureState yield_type, const mpm::dense_map* state_vars,
    const Vector6d& stress, Vector6d* df_dsigma, Vector6d* dp_dsigma,
    double* dp_dq, Matrix6x6* d2p_dsigma2, double* softening, double pdstrain) {
  // Get stress invariants
  const double rho = std::sqrt(2.0) * (*state_vars).at("tau");
  const double theta = (*state_vars).at("theta");
  // Get MC parameters
  const double phi = (*state_vars).at("phi");
  const double psi = (*state_vars).at("psi");
  const double cohesion = (*state_vars).at("cohesion");
  const double tension_cutoff = (*state_vars).at("tension_cutoff");
  const double d_theta = 1e-6;

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
    console_->error(
        "compute df dp\n"
        "Invalid yield type for compute_df_dp: {}\n",
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

  Matrix6x6 P_dev = Matrix6x6::Identity();
  P_dev.topLeftCorner<3, 3>() -= Matrix3x3::Constant(1.0 / 3.0);
  Vector6d n_rho = drho_dsigma;
  // Compute d2rho/dsigma2
  Vector6d T_diag;
  T_diag << 1.0, 1.0, 1.0, 2.0, 2.0, 2.0;
  Matrix6x6 T = T_diag.asDiagonal();

  // Correct d2rho/dsigma2
  Matrix6x6 d2rho_dsigma2 =
      (T * P_dev - drho_dsigma * drho_dsigma.transpose()) / rho;

  double J2 = mpm::materials::j2(stress);
  double J3 = mpm::materials::j3(stress);
  Vector6d s = mpm::materials::deviatoric_stress(stress);

  Vector6d v2;
  v2 << s(0), s(1), s(2), 2.0 * s(3), 2.0 * s(4), 2.0 * s(5);

  // 3. Second derivative of J2 w.r.t stress (d2J2) -> purely analytical
  Matrix6x6 d2J2 = Matrix6x6::Zero();
  d2J2.topLeftCorner<3, 3>().setConstant(-1.0 / 3.0);
  d2J2(0, 0) = d2J2(1, 1) = d2J2(2, 2) = 2.0 / 3.0;
  d2J2(3, 3) = d2J2(4, 4) = d2J2(5, 5) = 2.0;

  // Lambda to compute exact analytical 1st derivative of J3 w.r.t full stress
  auto eval_v3 = [](const Vector6d& sig) -> Vector6d {
    double I1 = sig(0) + sig(1) + sig(2);
    Vector6d dev = sig;
    dev.head(3).array() -= I1 / 3.0;

    double j2 = 0.5 * (dev(0) * dev(0) + dev(1) * dev(1) + dev(2) * dev(2)) +
                dev(3) * dev(3) + dev(4) * dev(4) + dev(5) * dev(5);

    double s2_11 = dev(0) * dev(0) + dev(3) * dev(3) + dev(5) * dev(5);
    double s2_22 = dev(3) * dev(3) + dev(1) * dev(1) + dev(4) * dev(4);
    double s2_33 = dev(5) * dev(5) + dev(4) * dev(4) + dev(2) * dev(2);
    double s2_12 = dev(0) * dev(3) + dev(3) * dev(1) + dev(5) * dev(4);
    double s2_23 = dev(3) * dev(5) + dev(1) * dev(4) + dev(4) * dev(2);
    double s2_13 = dev(0) * dev(5) + dev(3) * dev(4) + dev(5) * dev(2);

    Vector6d v;
    v << s2_11 - 2.0 / 3.0 * j2, s2_22 - 2.0 / 3.0 * j2, s2_33 - 2.0 / 3.0 * j2,
        2.0 * s2_12, 2.0 * s2_23, 2.0 * s2_13;
    return v;
  };

  // 4. Evaluate v3 at current state
  Vector6d v3 = eval_v3(stress);

  // 5. Evaluate 2nd derivative of J3 (d2J3) using exact finite difference of
  // analytical gradient
  Matrix6x6 d2J3_raw = Matrix6x6::Zero();
  d2J3_raw(0, 1) = d2J3_raw(1, 0) = s(2);
  d2J3_raw(0, 2) = d2J3_raw(2, 0) = s(1);
  d2J3_raw(1, 2) = d2J3_raw(2, 1) = s(0);

  d2J3_raw(0, 4) = d2J3_raw(4, 0) = -2.0 * s(4);
  d2J3_raw(1, 5) = d2J3_raw(5, 1) = -2.0 * s(5);
  d2J3_raw(2, 3) = d2J3_raw(3, 2) = -2.0 * s(3);

  d2J3_raw(3, 3) = -2.0 * s(2);
  d2J3_raw(4, 4) = -2.0 * s(0);
  d2J3_raw(5, 5) = -2.0 * s(1);

  d2J3_raw(3, 4) = d2J3_raw(4, 3) = 2.0 * s(5);
  d2J3_raw(3, 5) = d2J3_raw(5, 3) = 2.0 * s(4);
  d2J3_raw(4, 5) = d2J3_raw(5, 4) = 2.0 * s(3);

  Matrix6x6 d2J3 = P_dev.transpose() * d2J3_raw * P_dev;

  // 6. Chain rule constructions
  double c = 1.5 * std::sqrt(3.0);
  double J2_15 = std::pow(J2, 1.5);
  double J2_25 = std::pow(J2, 2.5);
  double J2_35 = std::pow(J2, 3.5);

  // Gradient of x w.r.t stress
  Vector6d dx_dsig = c * ((1.0 / J2_15) * v3 - (1.5 * J3 / J2_25) * v2);

  // Hessian of x w.r.t stress
  Matrix6x6 d2x_dsig2 =
      c *
      ((1.0 / J2_15) * d2J3 -
       (1.5 / J2_25) * (v3 * v2.transpose() + v2 * v3.transpose()) -
       (1.5 * J3 / J2_25) * d2J2 + (3.75 * J3 / J2_35) * (v2 * v2.transpose()));

  // 7. Final assembly of d2(theta)/dsig2
  double cos_3theta = std::cos(3.0 * theta);
  double sin_3theta = std::sin(3.0 * theta);

  // Singularity safeguard at meridians (theta = +/- 30 degrees)
  if (std::abs(sin_3theta) < 1e-6) {
    sin_3theta = (sin_3theta < 0) ? -1e-6 : 1e-6;
  }

  // Compute d2rho/dsigma2
  double dth_dx = -1.0 / (3.0 * sin_3theta);
  double d2th_dx2 = -cos_3theta / (3.0 * std::pow(sin_3theta, 3));

  Matrix6x6 d2theta_dsigma2 =
      d2th_dx2 * (dx_dsig * dx_dsig.transpose()) + dth_dx * d2x_dsig2;

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

    double drt = drt_dtheta;

    // Second derivatives of rt's numerator and denominator components
    const double d2rtnum_dtheta2 =
        -8. * (1. - et_value * et_value) *
        (cos(theta) * cos(theta) - sin(theta) * sin(theta));
    const double dsqpart_dtheta =
        drtnum_dtheta;  // Note: d(sqpart)/d(theta) equals drtnum_dtheta
                        // mathematically
    const double d2sqpart_dtheta2 = d2rtnum_dtheta2;
    const double d2rtden_dtheta2 =
        -2. * (1. - et_value * et_value) * cos(theta) +
        (2. * et_value - 1.) * ((d2sqpart_dtheta2) / (2. * std::sqrt(sqpart)) -
                                (dsqpart_dtheta * dsqpart_dtheta) /
                                    (4. * sqpart * std::sqrt(sqpart)));

    // Quotient rule for the second derivative of rt
    double d2rt = (d2rtnum_dtheta2 * rt_den - d2rtden_dtheta2 * rt_num) /
                      (3. * rt_den * rt_den) -
                  (2. * drtden_dtheta * drt_dtheta) / rt_den;

    // 3. Compute scalar derivatives of H_t
    double C0_t = std::pow(xit * tension_cutoff, 2);
    double H_t_val = std::sqrt(C0_t + 1.5 * rt * rt * rho * rho);

    double dHt_drho = (1.5 * rt * rt * rho) / H_t_val;
    double dHt_dtheta = (1.5 * rho * rho * rt * drt) / H_t_val;
    double d2Ht_drho2 = (1.5 * rt * rt * C0_t) / std::pow(H_t_val, 3);
    double d2Ht_drhodtheta =
        (3.0 * rho * rt * drt * (C0_t + 0.75 * rt * rt * rho * rho)) /
        std::pow(H_t_val, 3);
    double d2Ht_dtheta2 = (0.75 * rho * rho / std::pow(H_t_val, 3)) *
                          (2.0 * H_t_val * H_t_val * (drt * drt + rt * d2rt) -
                           3.0 * rho * rho * rt * rt * drt * drt);

    // Compute d2P/dSigma2
    (*d2p_dsigma2) =
        d2Ht_drho2 * (drho_dsigma * drho_dsigma.transpose()) +
        d2Ht_drhodtheta * (drho_dsigma * dtheta_dsigma.transpose() +
                           dtheta_dsigma * drho_dsigma.transpose()) +
        d2Ht_dtheta2 * (dtheta_dsigma * dtheta_dsigma.transpose()) +
        dHt_drho * d2rho_dsigma2 + dHt_dtheta * d2theta_dsigma2;

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
    double m = (2. * (1 - e_val * e_val) * cos(theta)) +
               ((2. * e_val - 1) * std::sqrt(sqpart));
    const double l = (4. * (1. - e_val * e_val) * std::pow(cos(theta), 2)) +
                     std::pow((2. * e_val - 1.), 2);
    double r_mw = (l / m) * r_mc;
    // Initialise meridional eccentricity
    const double xi = 0.1;
    const double a0 = 1e-2;
    double C0_s_min = std::pow(a0 * cohesion_peak_ * std::cos(phi), 2);
    double C0_s =
        std::max(std::pow(xi * cohesion_peak_ * std::tan(psi), 2), C0_s_min);

    // Force axisymmetric circle near apex for low cohesion and tension cutoff
    if (rho * rho < C0_s) {
      r_mw = r_mc;  // Force axisymmetric circle near apex
      drmw = 0.0;   // Kill dr_mw/dtheta derivative noise
      d2rmw = 0.0;  // Kill d2r_mw/dtheta2 derivative noise
    }

    double omega = C0_s + std::pow((r_mw * std::sqrt(1.5) * rho), 2);
    if (omega < std::numeric_limits<double>::epsilon()) omega = 1.E-5;
    const double dl_dtheta =
        -8. * (1. - e_val * e_val) * cos(theta) * sin(theta);
    const double dm_dtheta =
        (-2. * (1. - e_val * e_val) * sin(theta)) +
        (0.5 * (2. * e_val - 1.) * dl_dtheta) / std::sqrt(sqpart);
    const double dp_depsilon = tan(psi) / std::sqrt(3.);
    const double dp_drho = 3. * rho * r_mw * r_mw / (2. * std::sqrt(omega));
    double drmw = ((m * dl_dtheta) - (l * dm_dtheta)) / (m * m) * r_mc;

    double H_s_val = std::sqrt(omega);
    const double dp_dtheta = (1.5 * rho * rho * r_mw * drmw) / H_s_val;
    // compute the value of dp/dsigma and dp/dj in shear yield
    (*dp_dsigma) = (dp_depsilon * depsilon_dsigma) + (dp_drho * drho_dsigma) +
                   (dp_dtheta * dtheta_dsigma);

    (*dp_dq) = dp_drho * std::sqrt(2. / 3.);

    const double d2l_dtheta2 =
        -8.0 * (1.0 - e_val * e_val) *
        (cos(theta) * cos(theta) - sin(theta) * sin(theta));
    const double term1 = d2l_dtheta2 / std::sqrt(sqpart);
    const double term2 =
        (dl_dtheta * dl_dtheta) / (2.0 * sqpart * std::sqrt(sqpart));
    const double d2m_dtheta2 = -2.0 * (1.0 - e_val * e_val) * cos(theta) +
                               0.5 * (2.0 * e_val - 1.0) * (term1 - term2);
    // Exact analytical second derivative
    const double num2 = m * (m * d2l_dtheta2 - l * d2m_dtheta2) -
                        2.0 * dm_dtheta * (m * dl_dtheta - l * dm_dtheta);
    double d2rmw = (num2 / (m * m * m)) * r_mc;

    double dHs_drho = (1.5 * r_mw * r_mw * rho) / H_s_val;
    double dHs_dtheta = (1.5 * rho * rho * r_mw * drmw) / H_s_val;
    double d2Hs_drho2 = (1.5 * r_mw * r_mw * C0_s) / std::pow(H_s_val, 3);
    double d2Hs_drhodtheta =
        (3.0 * rho * r_mw * drmw * (C0_s + 0.75 * r_mw * r_mw * rho * rho)) /
        std::pow(H_s_val, 3);
    double d2Hs_dtheta2 =
        (0.75 * rho * rho / std::pow(H_s_val, 3)) *
        (2.0 * H_s_val * H_s_val * (drmw * drmw + r_mw * d2rmw) -
         3.0 * rho * rho * r_mw * r_mw * drmw * drmw);

    // Compute d2P/dSigma2
    (*d2p_dsigma2) =
        d2Hs_drho2 * (drho_dsigma * drho_dsigma.transpose()) +
        d2Hs_drhodtheta * (drho_dsigma * dtheta_dsigma.transpose() +
                           dtheta_dsigma * drho_dsigma.transpose()) +
        d2Hs_dtheta2 * (dtheta_dsigma * dtheta_dsigma.transpose()) +
        dHs_drho * d2rho_dsigma2 + dHs_dtheta * d2theta_dsigma2;
  }

  // Compute softening part
  (*softening) = 0.;
  if (yield_type == mpm::mohrcoulomb::FailureState::Shear) {
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
      const double dkappa_dlambda = mpm::materials::pdstrain(*dp_dsigma);
      (*softening) = (-1.) *
                     ((df_dphi * dphi_dpstrain) + (df_dc * dc_dpstrain)) *
                     dkappa_dlambda;
    }
  }
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::MohrCoulomb<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {
  const unsigned max_substeps = 64;
  const mpm::dense_map state_vars_n = *state_vars;

  for (unsigned nsub = 1; nsub <= max_substeps; nsub *= 2) {
    const bool last_attempt = (2 * nsub > max_substeps);
    // Restart from the start-of-step state
    *state_vars = state_vars_n;
    Vector6d sub_stress = stress;
    bool success = true;
    for (unsigned k = 0; k < nsub && success; ++k)
      sub_stress = this->compute_stress_increment(sub_stress, dstrain / nsub,
                                                  ptr, state_vars, dt / nsub,
                                                  last_attempt, &success);
    if (success || last_attempt) return sub_stress;
  }
  return stress;
}

//! Stress update for a single strain increment
//! allow_apex: permit the direct apex return as a last resort
//! success: set to false if the return mapping did not converge
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::MohrCoulomb<Tdim>::compute_stress_increment(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt,
    bool allow_apex, bool* success) {
  *success = true;

  // Density criterion for tensile separation
  const double current_packing_density = ptr->mass_density();
  const double critical_density = minimum_packing_fraction_ * grain_density_;

  // Elastic stiffness and elastic trial stress
  (*state_vars).at("yield_state") = 0;
  const Matrix6x6 de = this->compute_elastic_tensor(state_vars);
  const Vector6d trial_stress =
      this->compute_trial_stress(stress, dstrain, de, ptr, state_vars);

  // Characteristic normalisation stress (prevents division by zero at origin)
  const double sig_trial_norm = std::max(trial_stress.norm(), 1.0e-8);

  // Separated state: current packing density is less than critical density
  if (current_packing_density <= critical_density) {
    (*state_vars).at("pdstrain") +=
        mpm::materials::q(trial_stress) / 3.0 / shear_modulus_;
    (*state_vars).at("yield_state") = 4;
    return Vector6d::Zero();
  }

  // Stress invariants and yield functions at the trial stress
  this->compute_stress_invariants(trial_stress, state_vars);
  Eigen::Matrix<double, 2, 1> yield_function;
  auto yield_type = this->compute_yield_state(&yield_function, (*state_vars));
  const bool started_in_corner =
      (yield_type == mpm::mohrcoulomb::FailureState::ShearTensile);

  // Elastic admissibility: if both surfaces are satisfied, accept trial stress
  if (yield_type == mpm::mohrcoulomb::FailureState::Elastic) {
    (*state_vars).at("yield_state") = 0;
    return trial_stress;
  }

  // =========================================================================
  // Plastic: return mapping
  // =========================================================================
  bool converged = false;
  Vector6d current_stress = trial_stress;
  Vector6d dstrain_p_tensile = Vector6d::Zero();
  Vector6d dstrain_p_shear = Vector6d::Zero();
  Eigen::Matrix<double, 2, 1> dlambda = Vector2d::Zero();

  // Plastic strains at the start of the step
  const auto read_plastic_strain = [state_vars](const std::string& prefix) {
    Vector6d strain;
    for (unsigned i = 0; i < 6; ++i)
      strain(i) = (*state_vars).at(prefix + std::to_string(i));
    return strain;
  };
  const Vector6d plastic_strain_tensile =
      read_plastic_strain("plastic_strain_tensile");
  const Vector6d plastic_strain_shear =
      read_plastic_strain("plastic_strain_shear");
  // Accumulated equivalent plastic shear strain (softening variable) at the
  // start of the increment
  const double kappa_n = (*state_vars).at("pdstrain");

  // Store the updated plastic strains (start-of-step value + current
  // increment) in the state variables and update the plastic deviatoric
  // strains. The shear value is accumulated (kappa_n + increment), so
  // softening is irreversible.
  const auto commit_plastic_strain = [&](double& pdstrain_tensile,
                                         double& pdstrain_shear) {
    const Vector6d total_tensile = plastic_strain_tensile + dstrain_p_tensile;
    const Vector6d total_shear = plastic_strain_shear + dstrain_p_shear;
    pdstrain_tensile = mpm::materials::pdstrain(total_tensile);
    pdstrain_shear = kappa_n + mpm::materials::pdstrain(dstrain_p_shear);
    (*state_vars).at("pdstrain") = pdstrain_shear;
    for (unsigned i = 0; i < 6; ++i) {
      (*state_vars).at("plastic_strain_tensile" + std::to_string(i)) =
          total_tensile(i);
      (*state_vars).at("plastic_strain_shear" + std::to_string(i)) =
          total_shear(i);
    }
  };

  (*state_vars).at("max_itr") = 0;

  // -------------------------------------------------------------------------
  // 1. Shear-tensile (corner) region
  // -------------------------------------------------------------------------
  if (yield_type == mpm::mohrcoulomb::FailureState::ShearTensile) {
    Eigen::Matrix<double, 2, 1> dlambda_double = Vector2d::Zero();
    double pdstrain_tensile = mpm::materials::pdstrain(plastic_strain_tensile);
    double pdstrain_shear = kappa_n;

    for (unsigned itr = 0; itr < max_iter_; ++itr) {
      (*state_vars).at("max_itr") = itr + 1;

      // A. Corner return step
      auto [corrected_stress, dstrain_p_t, dstrain_p_s] =
          this->compute_corner_return(
              current_stress, trial_stress, de, state_vars, yield_function,
              dlambda_double, pdstrain_tensile, pdstrain_shear);
      current_stress = corrected_stress;
      dstrain_p_tensile = dstrain_p_t;
      dstrain_p_shear = dstrain_p_s;
      this->compute_stress_invariants(current_stress, state_vars);
      commit_plastic_strain(pdstrain_tensile, pdstrain_shear);

      // Update softening-dependent strength parameters
      if (softening_)
        this->update_softening_parameters(state_vars, pdstrain_tensile,
                                          pdstrain_shear);

      // Re-evaluate yield functions and stress residual at current stress
      this->compute_yield_state(&yield_function, (*state_vars));
      const Vector6d R_sigma = current_stress - trial_stress +
                               de * (dstrain_p_tensile + dstrain_p_shear);
      const double err_sigma = R_sigma.norm() / sig_trial_norm;

      // B. Convergence check
      // The signs of the multipliers are only checked once the corner solve
      // has finished (below): intermediate Newton iterates, especially the
      // first one from dlambda = 0, are not a reliable active-set indicator
      if (std::abs(yield_function(0)) <= tolerance_ &&
          std::abs(yield_function(1)) <= tolerance_ && err_sigma < rel_tol_) {
        if (dlambda_double(0) >= 0.0 && dlambda_double(1) >= 0.0) {
          dlambda = dlambda_double;
          converged = true;
          (*state_vars).at("yield_state") = 3;
        }
        break;
      }
    }

    if (dlambda_double(0) < 0.0 || dlambda_double(1) < 0.0) {
      // Negative multiplier at the end of the corner solve: the corner is not
      // the active set; revert to trial stress and pick the single surface
      current_stress = trial_stress;
      this->compute_stress_invariants(current_stress, state_vars);
      this->compute_yield_state(&yield_function, (*state_vars));
      if (dlambda_double(0) < 0.0 && dlambda_double(1) >= 0.0)
        yield_type = mpm::mohrcoulomb::FailureState::Shear;
      else if (dlambda_double(0) >= 0.0 && dlambda_double(1) < 0.0)
        yield_type = mpm::mohrcoulomb::FailureState::Tensile;
      else if (dlambda_double(0) < 0.0 && dlambda_double(1) < 0.0)
        yield_type = mpm::mohrcoulomb::FailureState::Elastic;
    } else if (!converged && allow_apex) {
      // Corner iterations failed on the final attempt: direct apex return
      auto [corrected_stress, dstrain_p_t, dstrain_p_s] =
          this->enforce_corner_return(trial_stress, de, state_vars,
                                      yield_function, dlambda, pdstrain_tensile,
                                      pdstrain_shear);
      current_stress = corrected_stress;
      dstrain_p_tensile = dstrain_p_t;
      dstrain_p_shear = dstrain_p_s;
      this->compute_stress_invariants(current_stress, state_vars);
      commit_plastic_strain(pdstrain_tensile, pdstrain_shear);
      converged = true;
    }
  }

  // Both multipliers negative in the corner region: trial stress is returned
  if (yield_type == mpm::mohrcoulomb::FailureState::Elastic) {
    console_->error(
        "Invalid elastic type for compute_df_dp: {}; Tensile yield function: "
        "{}; Shear yield function: {}\n",
        static_cast<int>(yield_type), yield_function(0), yield_function(1));
    *success = false;
    return current_stress;
  }

  // -------------------------------------------------------------------------
  // 2. Single surface (tension or shear), with a corner fallback if the
  //    other surface is violated after convergence
  // -------------------------------------------------------------------------
  if (yield_type == mpm::mohrcoulomb::FailureState::Tensile ||
      yield_type == mpm::mohrcoulomb::FailureState::Shear) {
    const bool is_tensile =
        (yield_type == mpm::mohrcoulomb::FailureState::Tensile);
    // Index of the active and the other surface (0: tension, 1: shear)
    const unsigned active = is_tensile ? 0 : 1;
    const unsigned other = is_tensile ? 1 : 0;

    double pdstrain_tensile = mpm::materials::pdstrain(plastic_strain_tensile);
    double pdstrain_shear = kappa_n;

    for (unsigned itr = 0; itr < max_iter_; ++itr) {
      (*state_vars).at("max_itr") = itr + 1;

      // A. Single surface return step
      double dlambda_single = dlambda(active);
      const double pdstrain_single =
          is_tensile ? pdstrain_tensile : pdstrain_shear;
      auto [corrected_stress, dp_dsigma_corrected] =
          this->compute_single_surface_return(
              yield_type, yield_function(active), current_stress, de,
              state_vars, dlambda_single, pdstrain_single, trial_stress);
      current_stress = corrected_stress;

      dlambda(active) = dlambda_single;
      if (is_tensile)
        dstrain_p_tensile = dlambda(0) * dp_dsigma_corrected;
      else
        dstrain_p_shear = dlambda(1) * dp_dsigma_corrected;

      commit_plastic_strain(pdstrain_tensile, pdstrain_shear);

      // Update softening-dependent strength parameters
      if (softening_)
        this->update_softening_parameters(state_vars, pdstrain_tensile,
                                          pdstrain_shear);

      // Re-evaluate yield functions and stress residual at current stress
      this->compute_yield_state(&yield_function, (*state_vars));
      const Vector6d R_sigma = current_stress - trial_stress +
                               dlambda(active) * (de * dp_dsigma_corrected);
      const double yield_function_single = yield_function(active);
      const double other_yield_function = yield_function(other);
      const double err_sigma = R_sigma.norm() / sig_trial_norm;

      // B. Convergence check
      if (std::abs(yield_function_single) <= tolerance_ &&
          err_sigma < rel_tol_) {
        if (other_yield_function <= tolerance_) {
          // Converged on the single surface; assign final yield state
          converged = true;
          if (yield_function(0) > -tolerance_ &&
              yield_function(1) <= tolerance_)
            (*state_vars).at("yield_state") = 2;  // Tensile
          else if (yield_function(1) > -tolerance_ &&
                   yield_function(0) <= tolerance_)
            (*state_vars).at("yield_state") = 1;  // Shear
          else
            (*state_vars).at("yield_state") = 0;  // Elastic
        } else {
          // Other surface violated: continue with corner return mapping
          for (unsigned citr = 0; citr < max_iter_; ++citr) {
            (*state_vars).at("max_itr") = citr + 1;

            auto [corner_stress, dstrain_p_t, dstrain_p_s] =
                this->compute_corner_return(current_stress, trial_stress, de,
                                            state_vars, yield_function, dlambda,
                                            pdstrain_tensile, pdstrain_shear);
            current_stress = corner_stress;
            dstrain_p_tensile = dstrain_p_t;
            dstrain_p_shear = dstrain_p_s;
            this->compute_stress_invariants(current_stress, state_vars);
            commit_plastic_strain(pdstrain_tensile, pdstrain_shear);

            // Update softening-dependent strength parameters
            if (softening_)
              this->update_softening_parameters(state_vars, pdstrain_tensile,
                                                pdstrain_shear);

            // Re-evaluate yield functions and stress residual
            this->compute_yield_state(&yield_function, (*state_vars));
            const Vector6d R_sigma_corner =
                current_stress - trial_stress +
                de * (dstrain_p_tensile + dstrain_p_shear);
            const double err_sigma_corner =
                R_sigma_corner.norm() / sig_trial_norm;

            // Convergence check
            if (yield_function(0) <= tolerance_ &&
                yield_function(1) <= tolerance_ &&
                err_sigma_corner < rel_tol_) {
              converged = true;
              (*state_vars).at("yield_state") = 3;
              break;
            }
          }
        }
        break;
      }
    }
  }

  // -------------------------------------------------------------------------
  // 3. Apex fallback (final substepping attempt only): a return that started
  //    in the corner region and did not converge on the corner or a single
  //    surface lies in the apex region
  // -------------------------------------------------------------------------
  if (!converged && started_in_corner && allow_apex) {
    double pdstrain_tensile = 0., pdstrain_shear = 0.;
    auto [apex_stress, dstrain_p_t, dstrain_p_s] = this->enforce_corner_return(
        trial_stress, de, state_vars, yield_function, dlambda, pdstrain_tensile,
        pdstrain_shear);
    current_stress = apex_stress;
    dstrain_p_tensile = dstrain_p_t;
    dstrain_p_shear = dstrain_p_s;
    this->compute_stress_invariants(current_stress, state_vars);
    commit_plastic_strain(pdstrain_tensile, pdstrain_shear);
    this->compute_yield_state(&yield_function, (*state_vars));
    converged = true;
  }

  // =========================================================================
  // Post-convergence handling
  // =========================================================================
  *success = converged;
  if (!converged && allow_apex) {
    console_->warn(
        "MohrCoulomb::compute_stress: Return mapping did not converge "
        "after {} iterations. Final yield functions: f_t = {}, f_s = {}",
        max_iter_, yield_function(0), yield_function(1));
  }

  return current_stress;
}

//! Single-surface return mapping step: Newton step on the stress residual and
//! consistency condition, with backtracking line search
//! Returns: (corrected stress, flow direction dP/dSigma at corrected stress);
//! dlambda_single is updated in place
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>>
    mpm::MohrCoulomb<Tdim>::compute_single_surface_return(
        mpm::mohrcoulomb::FailureState yield_type, double yield_function,
        const Vector6d& current_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars, double& dlambda_single, double pdstrain,
        const Vector6d& trial_stress) {
  // Flow / yield gradients, flow Hessian and softening at the current stress
  Vector6d df_dsigma = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  double dp_dq = 0.0;
  Matrix6x6 d2p_dsigma2 = Matrix6x6::Zero();
  double softening_modulus = 0.0;
  this->compute_df_dp(yield_type, state_vars, current_stress, &df_dsigma,
                      &dp_dsigma, &dp_dq, &d2p_dsigma2, &softening_modulus,
                      pdstrain);

  // Stress residual
  const Vector6d de_dp = de * dp_dsigma;
  const Vector6d R_sigma =
      current_stress - trial_stress + dlambda_single * de_dp;

  // Curvature operator M = I + dlambda De d2P/dSigma2 and its inverse
  const Matrix6x6 M =
      Matrix6x6::Identity() + dlambda_single * (de * d2p_dsigma2);
  const Matrix6x6 M_inv = M.partialPivLu().inverse();

  // Consistency denominator: n^T M^-1 De m + H
  // (n = dF/dSigma, m = dP/dSigma, H = softening modulus)
  double denominator = df_dsigma.dot(M_inv * de_dp) + softening_modulus;
  if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
    console_->warn(
        "MohrCoulomb: Consistency denominator is near zero ({:.6e}). "
        "This may indicate numerical instability. Applying regularization.",
        denominator);
    denominator = check_low(denominator);
  }

  // Newton increments of the plastic multiplier and of the stress
  const double numerator = yield_function - df_dsigma.dot(M_inv * R_sigma);
  const double d_dlambda = numerator / denominator;
  const Vector6d d_sigma = -d_dlambda * M_inv * de_dp - M_inv * R_sigma;

  // Merit function at the start of the step
  const double initial_merit =
      0.5 *
      (R_sigma.squaredNorm() + std::pow(std::max(0.0, yield_function), 2));

  // Backtracking line search: shrink the step until the merit function does
  // not increase
  double alpha = 1.0;
  Vector6d trial_corrected_stress = current_stress;
  double trial_dlambda = dlambda_single;
  Eigen::Matrix<double, 2, 1> trial_yield_function;
  Vector6d df_dsigma_trial = Vector6d::Zero();
  Vector6d dp_dsigma_trial = Vector6d::Zero();
  double dp_dq_trial = 0.0;
  Matrix6x6 d2p_dsigma2_trial = Matrix6x6::Zero();
  double softening_modulus_trial = 0.0;
  bool step_accepted = false;

  for (int line_iter = 0; line_iter < 10; ++line_iter) {
    // Propose step scaled by alpha
    trial_corrected_stress = current_stress + alpha * d_sigma;
    trial_dlambda = dlambda_single + alpha * d_dlambda;

    // Re-evaluate flow direction and yield function at the trial point
    this->compute_stress_invariants(trial_corrected_stress, state_vars);
    this->compute_df_dp(yield_type, state_vars, trial_corrected_stress,
                        &df_dsigma_trial, &dp_dsigma_trial, &dp_dq_trial,
                        &d2p_dsigma2_trial, &softening_modulus_trial, pdstrain);
    const Vector6d de_dp_trial = de * dp_dsigma_trial;
    const Vector6d R_sigma_trial =
        trial_corrected_stress - trial_stress + trial_dlambda * de_dp_trial;
    this->compute_yield_state(&trial_yield_function, (*state_vars));
    const double f_trial =
        (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
            ? trial_yield_function(0)
            : trial_yield_function(1);

    // Accept the step if the merit function did not increase
    const double trial_merit = 0.5 * (R_sigma_trial.squaredNorm() +
                                      std::pow(std::max(0.0, f_trial), 2));
    if (trial_merit <= initial_merit) {
      step_accepted = true;
      break;
    }

    alpha *= 0.8;  // Shrink the step by 20%
  }

  if (!step_accepted)
    console_->debug("Single surface: line search did not reduce the residual");

  // Apply the (possibly damped) update
  const Vector6d corrected_stress = trial_corrected_stress;
  dlambda_single = trial_dlambda;

  // Flow direction at the corrected stress
  Vector6d df_dsigma_after = Vector6d::Zero();
  Vector6d dp_dsigma_after = Vector6d::Zero();
  double dp_dq_after = 0.0;
  Matrix6x6 d2p_dsigma2_after = Matrix6x6::Zero();
  double softening_modulus_after = 0.0;
  this->compute_stress_invariants(corrected_stress, state_vars);
  this->compute_df_dp(yield_type, state_vars, corrected_stress,
                      &df_dsigma_after, &dp_dsigma_after, &dp_dq_after,
                      &d2p_dsigma2_after, &softening_modulus_after, pdstrain);

  return {corrected_stress, dp_dsigma_after};
}

//! Corner (two-surface) return mapping step: Newton step on the stress
//! residual and both consistency conditions, with backtracking line search
//! Returns: (corrected stress, tensile plastic strain increment, shear plastic
//! strain increment); dlambda is updated in place
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>,
           Eigen::Matrix<double, 6, 1>>
    mpm::MohrCoulomb<Tdim>::compute_corner_return(
        const Vector6d& current_stress, const Vector6d& trial_stress,
        const Matrix6x6& de, mpm::dense_map* state_vars,
        const Eigen::Matrix<double, 2, 1>& yield_function,
        Eigen::Matrix<double, 2, 1>& dlambda, double pdstrain_tensile,
        double pdstrain_shear) {
  // Gradients for both active surfaces at the current stress
  Vector6d df_dsigma_t = Vector6d::Zero();  // Tension
  Vector6d dp_dsigma_t = Vector6d::Zero();
  double dp_dq_t = 0.0;
  Matrix6x6 d2p_dsigma2_t = Matrix6x6::Zero();
  double softening_t = 0.0;

  Vector6d df_dsigma_s = Vector6d::Zero();  // Shear
  Vector6d dp_dsigma_s = Vector6d::Zero();
  double dp_dq_s = 0.0;
  Matrix6x6 d2p_dsigma2_s = Matrix6x6::Zero();
  double softening_s = 0.0;

  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars,
                      current_stress, &df_dsigma_t, &dp_dsigma_t, &dp_dq_t,
                      &d2p_dsigma2_t, &softening_t, pdstrain_tensile);
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars,
                      current_stress, &df_dsigma_s, &dp_dsigma_s, &dp_dq_s,
                      &d2p_dsigma2_s, &softening_s, pdstrain_shear);

  // Yield-function values at the current stress
  const double f_t = yield_function(0);  // Tension
  const double f_s = yield_function(1);  // Shear

  // Curvature operator M = I + De (dlambda_t d2P_t + dlambda_s d2P_s)
  const Matrix6x6 M = Matrix6x6::Identity() + de * (dlambda(0) * d2p_dsigma2_t +
                                                    dlambda(1) * d2p_dsigma2_s);
  const Matrix6x6 M_inv = M.partialPivLu().inverse();

  // =========================================================================
  // 2x2 consistency system for the plastic multiplier increments:
  // [A11 A12] [d_lambda_t]   [b1]
  // [A21 A22] [d_lambda_s] = [b2]
  // with A_ij = n_i^T M^-1 De m_j + H_i delta_ij and
  //      b_i  = f_i - n_i^T M^-1 R_sigma
  // =========================================================================
  const Vector6d de_dp_t = de * dp_dsigma_t;
  const Vector6d de_dp_s = de * dp_dsigma_s;

  const Vector6d R_sigma =
      current_stress - trial_stress +
      de * (dlambda(0) * dp_dsigma_t + dlambda(1) * dp_dsigma_s);

  Eigen::Matrix<double, 2, 2> A;
  A(0, 0) = df_dsigma_t.dot(M_inv * de_dp_t) + softening_t;
  A(0, 1) = df_dsigma_t.dot(M_inv * de_dp_s);
  A(1, 0) = df_dsigma_s.dot(M_inv * de_dp_t);
  A(1, 1) = df_dsigma_s.dot(M_inv * de_dp_s) + softening_s;

  Eigen::Vector2d b;
  b(0) = f_t - df_dsigma_t.dot(M_inv * R_sigma);
  b(1) = f_s - df_dsigma_s.dot(M_inv * R_sigma);

  // Singular / ill-conditioned corner system (relative threshold): keep the
  // current state unchanged (the caller's convergence check then fails and
  // the increment is retried with substepping)
  const double det_A = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  const double norm_A_sq = A.squaredNorm();
  if (std::abs(det_A) < 1e-12 * norm_A_sq || std::abs(det_A) < 1e-14) {
    console_->debug("MohrCoulomb: Singular corner system; step skipped.");
    return {current_stress, Vector6d(dlambda(0) * dp_dsigma_t),
            Vector6d(dlambda(1) * dp_dsigma_s)};
  }

  // Closed-form 2x2 inverse
  Eigen::Matrix<double, 2, 2> A_inv;
  A_inv(0, 0) = A(1, 1) / det_A;
  A_inv(0, 1) = -A(0, 1) / det_A;
  A_inv(1, 0) = -A(1, 0) / det_A;
  A_inv(1, 1) = A(0, 0) / det_A;

  // Newton increments of the plastic multipliers and of the stress
  const Eigen::Vector2d delta_dlambda = A_inv * b;
  const Vector6d d_sigma =
      -M_inv * de *
          (delta_dlambda(0) * dp_dsigma_t + delta_dlambda(1) * dp_dsigma_s) -
      M_inv * R_sigma;

  // Merit function at the start of the step
  const double initial_merit =
      0.5 * (R_sigma.squaredNorm() + 0.5 * (std::pow(std::max(0.0, f_t), 2) +
                                            std::pow(std::max(0.0, f_s), 2)));

  // Backtracking line search: shrink the step until the merit function does
  // not increase
  double alpha = 1.0;
  Vector6d trial_corrected_stress = current_stress;
  Eigen::Matrix<double, 2, 1> trial_dlambda = dlambda;
  Eigen::Matrix<double, 2, 1> trial_yield_function;

  Vector6d df_dsigma_t_trial = Vector6d::Zero();
  Vector6d dp_dsigma_t_trial = Vector6d::Zero();
  double dp_dq_t_trial = 0.0;
  Matrix6x6 d2p_dsigma2_t_trial = Matrix6x6::Zero();
  double softening_modulus_t_trial = 0.0;

  Vector6d df_dsigma_s_trial = Vector6d::Zero();
  Vector6d dp_dsigma_s_trial = Vector6d::Zero();
  double dp_dq_s_trial = 0.0;
  Matrix6x6 d2p_dsigma2_s_trial = Matrix6x6::Zero();
  double softening_modulus_s_trial = 0.0;

  bool step_accepted = false;

  for (int line_iter = 0; line_iter < 20; ++line_iter) {
    // Propose step scaled by alpha
    trial_corrected_stress = current_stress + alpha * d_sigma;
    trial_dlambda = dlambda + alpha * delta_dlambda;

    // Re-evaluate flow directions and yield functions at the trial point
    this->compute_stress_invariants(trial_corrected_stress, state_vars);
    this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars,
                        trial_corrected_stress, &df_dsigma_t_trial,
                        &dp_dsigma_t_trial, &dp_dq_t_trial,
                        &d2p_dsigma2_t_trial, &softening_modulus_t_trial,
                        pdstrain_tensile);
    this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars,
                        trial_corrected_stress, &df_dsigma_s_trial,
                        &dp_dsigma_s_trial, &dp_dq_s_trial,
                        &d2p_dsigma2_s_trial, &softening_modulus_s_trial,
                        pdstrain_shear);
    if (softening_)
      this->update_softening_parameters(state_vars, pdstrain_tensile,
                                        pdstrain_shear);

    const Vector6d de_dp_t_trial = de * dp_dsigma_t_trial;
    const Vector6d de_dp_s_trial = de * dp_dsigma_s_trial;
    const Vector6d R_sigma_trial =
        trial_corrected_stress - trial_stress +
        (trial_dlambda(0) * de_dp_t_trial + trial_dlambda(1) * de_dp_s_trial);
    this->compute_yield_state(&trial_yield_function, (*state_vars));

    // Accept the step if the merit function did not increase
    const double trial_merit =
        0.5 * (R_sigma_trial.squaredNorm() +
               0.5 * (std::pow(std::max(0.0, trial_yield_function(0)), 2) +
                      std::pow(std::max(0.0, trial_yield_function(1)), 2)));
    if (trial_merit <= initial_merit) {
      step_accepted = true;
      break;
    }

    alpha *= 0.8;  // Shrink the step by 20%
  }

  if (!step_accepted)
    console_->debug("Double surface: line search did not reduce the residual");

  // Apply the (possibly damped) update
  const Vector6d corrected_stress = trial_corrected_stress;
  dlambda = trial_dlambda;

  // Flow directions at the corrected stress
  Vector6d corrected_df_dsigma_t = Vector6d::Zero();  // Tension
  Vector6d corrected_dp_dsigma_t = Vector6d::Zero();
  double corrected_dp_dq_t = 0.0;
  Matrix6x6 corrected_d2p_dsigma2_t = Matrix6x6::Zero();
  double corrected_softening_t = 0.0;

  Vector6d corrected_df_dsigma_s = Vector6d::Zero();  // Shear
  Vector6d corrected_dp_dsigma_s = Vector6d::Zero();
  double corrected_dp_dq_s = 0.0;
  Matrix6x6 corrected_d2p_dsigma2_s = Matrix6x6::Zero();
  double corrected_softening_s = 0.0;

  this->compute_df_dp(
      mpm::mohrcoulomb::FailureState::Tensile, state_vars, corrected_stress,
      &corrected_df_dsigma_t, &corrected_dp_dsigma_t, &corrected_dp_dq_t,
      &corrected_d2p_dsigma2_t, &corrected_softening_t, pdstrain_tensile);
  this->compute_df_dp(
      mpm::mohrcoulomb::FailureState::Shear, state_vars, corrected_stress,
      &corrected_df_dsigma_s, &corrected_dp_dsigma_s, &corrected_dp_dq_s,
      &corrected_d2p_dsigma2_s, &corrected_softening_s, pdstrain_shear);

  // Plastic strain increments = multipliers x flow directions
  const Vector6d dstrain_p_t = dlambda(0) * corrected_dp_dsigma_t;
  const Vector6d dstrain_p_s = dlambda(1) * corrected_dp_dsigma_s;

  return {corrected_stress, dstrain_p_t, dstrain_p_s};
}

//! Direct apex return: snap the stress to the apex of the yield surface and
//! back-calculate the plastic strain increment from the elastic overstress
//! Returns: (apex stress, hydrostatic plastic strain increment, deviatoric
//! plastic strain increment)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>,
           Eigen::Matrix<double, 6, 1>>
    mpm::MohrCoulomb<Tdim>::enforce_corner_return(
        const Vector6d& trial_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars,
        const Eigen::Matrix<double, 2, 1>& yield_function,
        Eigen::Matrix<double, 2, 1>& dlambda, double pdstrain_tensile,
        double pdstrain_shear) {
  // Apex stress sigma_apex = [p, p, p, 0, 0, 0] with p = tension cutoff
  // (tension positive)
  const double p_apex = (*state_vars).at("tension_cutoff");
  Vector6d apex_stress = Vector6d::Zero();
  apex_stress(0) = p_apex;
  apex_stress(1) = p_apex;
  apex_stress(2) = p_apex;

  // Overstress to be dissipated by plastic strain:
  // delta_sigma = De * delta_eps_p  =>  delta_eps_p = De^-1 * delta_sigma
  const Vector6d delta_sigma = trial_stress - apex_stress;
  const Vector6d total_plastic_strain = de.fullPivLu().solve(delta_sigma);

  // Split into hydrostatic (tensile) and deviatoric (shear) parts
  const double vol_strain = total_plastic_strain(0) + total_plastic_strain(1) +
                            total_plastic_strain(2);
  Vector6d dstrain_p_t = Vector6d::Zero();
  dstrain_p_t(0) = vol_strain / 3.0;
  dstrain_p_t(1) = vol_strain / 3.0;
  dstrain_p_t(2) = vol_strain / 3.0;
  const Vector6d dstrain_p_s = total_plastic_strain - dstrain_p_t;

  console_->warn("MohrCoulomb: Direct apex return used (last resort).");

  return {apex_stress, dstrain_p_t, dstrain_p_s};
}

//! Update softening parameters
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::update_softening_parameters(
    mpm::dense_map* state_vars, double pdstrain_tensile,
    double pdstrain_shear) {
  if (pdstrain_shear <= pdstrain_peak_) {
    // Pre-peak: peak strength parameters
    (*state_vars).at("phi") = phi_peak_;
    (*state_vars).at("psi") = psi_peak_;
    (*state_vars).at("cohesion") = cohesion_peak_;
  } else if (pdstrain_shear >= pdstrain_residual_) {
    // Post-residual: residual strength parameters
    (*state_vars).at("phi") = phi_residual_;
    (*state_vars).at("psi") = psi_residual_;
    (*state_vars).at("cohesion") = cohesion_residual_;
  } else {
    // Linear softening between peak and residual states
    const double ratio = (pdstrain_shear - pdstrain_peak_) /
                         (pdstrain_residual_ - pdstrain_peak_);
    (*state_vars).at("phi") = phi_peak_ + ratio * (phi_residual_ - phi_peak_);
    (*state_vars).at("psi") = psi_peak_ + ratio * (psi_residual_ - psi_peak_);
    (*state_vars).at("cohesion") =
        cohesion_peak_ + ratio * (cohesion_residual_ - cohesion_peak_);
  }

  // Keep the tension cutoff at or below the Mohr-Coulomb apex c / tan(phi):
  // otherwise softening exposes a shear-cone apex inside the tension region
  // that neither the single-surface nor the corner return can reach
  const double tan_phi = std::tan((*state_vars).at("phi"));
  if (tan_phi > 1.0e-10) {
    const double apex = (*state_vars).at("cohesion") / tan_phi;
    if ((*state_vars).at("tension_cutoff") > apex)
      (*state_vars).at("tension_cutoff") = apex;
  }
  // Keep the tension cutoff non-negative
  if ((*state_vars).at("tension_cutoff") < 0.0)
    (*state_vars).at("tension_cutoff") = 0.0;
}

//! Compute elastic tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6> mpm::MohrCoulomb<Tdim>::compute_elastic_tensor(
    mpm::dense_map* state_vars) {
  // Shear modulus
  const double G = shear_modulus_;
  const double a1 = bulk_modulus_ + (4.0 / 3.0) * G;
  const double a2 = bulk_modulus_ - (2.0 / 3.0) * G;
  // Elastic stiffness matrix
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
  // Plastic deviatoric strains (shear: accumulated softening variable)
  Vector6d plastic_strain_tensile;
  for (unsigned i = 0; i < 6; ++i)
    plastic_strain_tensile(i) =
        (*state_vars).at("plastic_strain_tensile" + std::to_string(i));
  const double pdstrain_tensile =
      mpm::materials::pdstrain(plastic_strain_tensile);
  const double pdstrain_shear = (*state_vars).at("pdstrain");

  const mpm::mohrcoulomb::FailureState yield_type =
      yield_type_.at(int((*state_vars).at("yield_state")));

  // Elastic state
  const Matrix6x6 de = this->compute_elastic_tensor(state_vars);
  if (yield_type == mpm::mohrcoulomb::FailureState::Elastic) return de;

  // Separated state
  if (yield_type == mpm::mohrcoulomb::FailureState::Separated)
    return Matrix6x6::Zero();

  // -------------------------------------------------------------------------
  // Single surface: d_ep = De - (De m)(De n)^T / (n^T De m + H)
  // -------------------------------------------------------------------------
  if (yield_type == mpm::mohrcoulomb::FailureState::Tensile ||
      yield_type == mpm::mohrcoulomb::FailureState::Shear) {
    double softening = 0.;
    Matrix6x6 d2p_dsigma2 = Matrix6x6::Zero();
    double dp_dq = 0.;
    Vector6d df_dsigma = Vector6d::Zero();
    Vector6d dp_dsigma = Vector6d::Zero();
    const double pdstrain_single =
        (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
            ? pdstrain_tensile
            : pdstrain_shear;

    this->compute_stress_invariants(stress, state_vars);
    this->compute_df_dp(yield_type, state_vars, stress, &df_dsigma, &dp_dsigma,
                        &dp_dq, &d2p_dsigma2, &softening, pdstrain_single);

    const Vector6d de_dpdsigma = de * dp_dsigma;
    const double dfdsigma_de_dpdsigma = df_dsigma.dot(de_dpdsigma);
    const Vector6d de_dfdsigma = de * df_dsigma;

    if (!hardening) softening = 0.;
    const Matrix6x6 d_ep = de - 1. / (dfdsigma_de_dpdsigma + softening) *
                                    (de_dpdsigma * de_dfdsigma.transpose());
    return d_ep;
  }

  // -------------------------------------------------------------------------
  // Corner (shear-tensile):
  // d_ep = De - [De m_s, De m_t] h^-1 [De n_s, De n_t]^T
  // -------------------------------------------------------------------------
  // Shear surface
  double softening_shear = 0.;
  Matrix6x6 d2p_dsigma2_shear = Matrix6x6::Zero();
  double dp_dq_shear = 0.;
  Vector6d df_dsigma_shear = Vector6d::Zero();
  Vector6d dp_dsigma_shear = Vector6d::Zero();
  // Tension surface
  double softening_tension = 0.;
  Matrix6x6 d2p_dsigma2_tension = Matrix6x6::Zero();
  double dp_dq_tension = 0.;
  Vector6d df_dsigma_tension = Vector6d::Zero();
  Vector6d dp_dsigma_tension = Vector6d::Zero();

  this->compute_stress_invariants(stress, state_vars);
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars,
                      stress, &df_dsigma_tension, &dp_dsigma_tension,
                      &dp_dq_tension, &d2p_dsigma2_tension, &softening_tension,
                      pdstrain_tensile);
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars, stress,
                      &df_dsigma_shear, &dp_dsigma_shear, &dp_dq_shear,
                      &d2p_dsigma2_shear, &softening_shear, pdstrain_shear);

  const Vector6d de_dpdsigma_shear = de * dp_dsigma_shear;
  const Vector6d de_dpdsigma_tension = de * dp_dsigma_tension;
  const Vector6d de_dfdsigma_shear = de * df_dsigma_shear;
  const Vector6d de_dfdsigma_tension = de * df_dsigma_tension;

  if (!hardening) {
    softening_shear = 0.;
    softening_tension = 0.;
  }

  // 2x2 consistency matrix h_ij = n_i^T De m_j (+ softening on the diagonal);
  // cross-hardening between shear and tension is neglected
  Eigen::Matrix2d dfdsigma_de_dpdsigma;
  dfdsigma_de_dpdsigma(0, 0) =
      df_dsigma_shear.dot(de_dpdsigma_shear) + softening_shear;
  dfdsigma_de_dpdsigma(0, 1) = df_dsigma_shear.dot(de_dpdsigma_tension);
  dfdsigma_de_dpdsigma(1, 0) = df_dsigma_tension.dot(de_dpdsigma_shear);
  dfdsigma_de_dpdsigma(1, 1) =
      df_dsigma_tension.dot(de_dpdsigma_tension) + softening_tension;
  const Eigen::Matrix2d inv_dfdsigma_de_dpdsigma =
      dfdsigma_de_dpdsigma.inverse();

  Eigen::Matrix<double, 6, 2> de_dpdsigma;
  de_dpdsigma.col(0) = de_dpdsigma_shear;
  de_dpdsigma.col(1) = de_dpdsigma_tension;
  Eigen::Matrix<double, 2, 6> de_dfdsigma_transpose;
  de_dfdsigma_transpose.row(0) = de_dfdsigma_shear.transpose();
  de_dfdsigma_transpose.row(1) = de_dfdsigma_tension.transpose();

  const Matrix6x6 d_ep =
      de - de_dpdsigma * inv_dfdsigma_de_dpdsigma * de_dfdsigma_transpose;
  return d_ep;
}
