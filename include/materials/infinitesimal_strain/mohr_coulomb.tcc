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
    // Max number of iterations of return mapping
    if (material_properties.contains("rmap_max_iteration")) {
      max_iter_ =
          material_properties.at("rmap_max_iteration").template get<unsigned>();
    }
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
      {"plastic_strain_t0", 0.},
      {"plastic_strain_t1", 0.},
      {"plastic_strain_t2", 0.},
      {"plastic_strain_t3", 0.},
      {"plastic_strain_t4", 0.},
      {"plastic_strain_t5", 0.},
      {"plastic_strain_s0", 0.},
      {"plastic_strain_s1", 0.},
      {"plastic_strain_s2", 0.},
      {"plastic_strain_s3", 0.},
      {"plastic_strain_s4", 0.},
      {"plastic_strain_s5", 0.},
      // Maximum number of iteration
      {"max_itr", 0}};
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
                                               "plastic_strain_t0",
                                               "plastic_strain_t1",
                                               "plastic_strain_t2",
                                               "plastic_strain_t3",
                                               "plastic_strain_t4",
                                               "plastic_strain_t5",
                                               "plastic_strain_s0",
                                               "plastic_strain_s1",
                                               "plastic_strain_s2",
                                               "plastic_strain_s3",
                                               "plastic_strain_s4",
                                               "plastic_strain_s5",};
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
  // Compute epsilon
  const double epsilon = -(*state_vars).at("pressure") * std::sqrt(3.);
  // Compute rho
  const double rho = std::sqrt(2.0) * (*state_vars).at("tau");

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
    double* dp_dq, Matrix6x6* d2p_dsigma2, double* softening, double pdstrain) {
  // Get stress invariants
  const double rho = std::sqrt(2.0) * (*state_vars).at("tau");
  const double theta = (*state_vars).at("theta");
  // Get MC parameters
  const double phi = (*state_vars).at("phi");
  const double psi = (*state_vars).at("psi");
  const double cohesion = (*state_vars).at("cohesion");
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
    console_->error("compute df dp\n"
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
  P_dev.topLeftCorner<3,3>() -= Matrix3x3::Constant(1.0/3.0);
  Vector6d n_rho = drho_dsigma;
  // Compute d2rho/dsigma2 
  Vector6d T_diag;
  T_diag << 1.0, 1.0, 1.0, 2.0, 2.0, 2.0;
  Matrix6x6 T = T_diag.asDiagonal();

  // Correct d2rho/dsigma2
  Matrix6x6 d2rho_dsigma2 = (T * P_dev - drho_dsigma * drho_dsigma.transpose()) / rho;

  double J2 = mpm::materials::j2(stress);
  double J3 = mpm::materials::j3(stress);
  Vector6d s = mpm::materials::deviatoric_stress(stress);
  
  Vector6d v2;
  v2 << s(0), s(1), s(2), 2.0*s(3), 2.0*s(4), 2.0*s(5);
  
  // 3. Second derivative of J2 w.r.t stress (d2J2) -> purely analytical
  Matrix6x6 d2J2 = Matrix6x6::Zero();
  d2J2.topLeftCorner<3,3>().setConstant(-1.0/3.0);
  d2J2(0,0) = d2J2(1,1) = d2J2(2,2) = 2.0/3.0;
  d2J2(3,3) = d2J2(4,4) = d2J2(5,5) = 2.0;

  // Lambda to compute exact analytical 1st derivative of J3 w.r.t full stress
  auto eval_v3 = [](const Vector6d& sig) -> Vector6d {
      double I1 = sig(0) + sig(1) + sig(2);
      Vector6d dev = sig;
      dev.head(3).array() -= I1 / 3.0;
      
      double j2 = 0.5 * (dev(0)*dev(0) + dev(1)*dev(1) + dev(2)*dev(2)) 
                + dev(3)*dev(3) + dev(4)*dev(4) + dev(5)*dev(5);
      
      double s2_11 = dev(0)*dev(0) + dev(3)*dev(3) + dev(5)*dev(5);
      double s2_22 = dev(3)*dev(3) + dev(1)*dev(1) + dev(4)*dev(4);
      double s2_33 = dev(5)*dev(5) + dev(4)*dev(4) + dev(2)*dev(2);
      double s2_12 = dev(0)*dev(3) + dev(3)*dev(1) + dev(5)*dev(4);
      double s2_23 = dev(3)*dev(5) + dev(1)*dev(4) + dev(4)*dev(2);
      double s2_13 = dev(0)*dev(5) + dev(3)*dev(4) + dev(5)*dev(2);
      
      Vector6d v;
      v << s2_11 - 2.0/3.0*j2,
            s2_22 - 2.0/3.0*j2,
            s2_33 - 2.0/3.0*j2,
            2.0 * s2_12,
            2.0 * s2_23,
            2.0 * s2_13;
      return v;
  };

  // 4. Evaluate v3 at current state
  Vector6d v3 = eval_v3(stress);

  // 5. Evaluate 2nd derivative of J3 (d2J3) using exact finite difference of analytical gradient
  Matrix6x6 d2J3_raw = Matrix6x6::Zero();
  d2J3_raw(0,1) = d2J3_raw(1,0) = s(2);
  d2J3_raw(0,2) = d2J3_raw(2,0) = s(1);
  d2J3_raw(1,2) = d2J3_raw(2,1) = s(0);

  d2J3_raw(0,4) = d2J3_raw(4,0) = -2.0 * s(4); 
  d2J3_raw(1,5) = d2J3_raw(5,1) = -2.0 * s(5); 
  d2J3_raw(2,3) = d2J3_raw(3,2) = -2.0 * s(3);

  d2J3_raw(3,3) = -2.0 * s(2);
  d2J3_raw(4,4) = -2.0 * s(0);
  d2J3_raw(5,5) = -2.0 * s(1);

  d2J3_raw(3,4) = d2J3_raw(4,3) = 2.0 * s(5);
  d2J3_raw(3,5) = d2J3_raw(5,3) = 2.0 * s(4);
  d2J3_raw(4,5) = d2J3_raw(5,4) = 2.0 * s(3);

  Matrix6x6 d2J3 = P_dev.transpose() * d2J3_raw * P_dev;

  // 6. Chain rule constructions
  double c = 1.5 * std::sqrt(3.0);
  double J2_15 = std::pow(J2, 1.5);
  double J2_25 = std::pow(J2, 2.5);
  double J2_35 = std::pow(J2, 3.5);

  // Gradient of x w.r.t stress
  Vector6d dx_dsig = c * ( (1.0 / J2_15) * v3 - (1.5 * J3 / J2_25) * v2 );

  // Hessian of x w.r.t stress
  Matrix6x6 d2x_dsig2 = c * (
      (1.0 / J2_15) * d2J3 
      - (1.5 / J2_25) * (v3 * v2.transpose() + v2 * v3.transpose())
      - (1.5 * J3 / J2_25) * d2J2
      + (3.75 * J3 / J2_35) * (v2 * v2.transpose())
  );

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

  Matrix6x6 d2theta_dsigma2 = d2th_dx2 * (dx_dsig * dx_dsig.transpose()) 
                          + dth_dx * d2x_dsig2;
  
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
    const double d2rtnum_dtheta2 = -8. * (1. - et_value * et_value) * (cos(theta) * cos(theta) - sin(theta) * sin(theta));
    const double dsqpart_dtheta = drtnum_dtheta; // Note: d(sqpart)/d(theta) equals drtnum_dtheta mathematically
    const double d2sqpart_dtheta2 = d2rtnum_dtheta2;
    const double d2rtden_dtheta2 = -2. * (1. - et_value * et_value) * cos(theta) +
                                   (2. * et_value - 1.) * ( (d2sqpart_dtheta2) / (2. * std::sqrt(sqpart)) -
                                   (dsqpart_dtheta * dsqpart_dtheta) / (4. * sqpart * std::sqrt(sqpart)) );

    // Quotient rule for the second derivative of rt
    double d2rt = (d2rtnum_dtheta2 * rt_den - d2rtden_dtheta2 * rt_num) / (3. * rt_den * rt_den) -
                  (2. * drtden_dtheta * drt_dtheta) / rt_den;

    // 3. Compute scalar derivatives of H_t
    double C0_t = std::pow(xit * tension_cutoff, 2);
    double H_t_val = std::sqrt(C0_t + 1.5 * rt * rt * rho * rho);
    
    double dHt_drho = (1.5 * rt * rt * rho) / H_t_val;
    double dHt_dtheta = (1.5 * rho * rho * rt * drt) / H_t_val;
    double d2Ht_drho2 = (1.5 * rt * rt * C0_t) / std::pow(H_t_val, 3);
    double d2Ht_drhodtheta = (3.0 * rho * rt * drt * (C0_t + 0.75 * rt * rt * rho * rho)) / std::pow(H_t_val, 3);
    double d2Ht_dtheta2 = (0.75 * rho * rho / std::pow(H_t_val, 3)) * 
                          (2.0 * H_t_val * H_t_val * (drt * drt + rt * d2rt) - 3.0 * rho * rho * rt * rt * drt * drt);

    //Compute d2P/dSigma2
    (*d2p_dsigma2) = d2Ht_drho2 * (drho_dsigma * drho_dsigma.transpose()) +
                     d2Ht_drhodtheta * (drho_dsigma * dtheta_dsigma.transpose() + dtheta_dsigma * drho_dsigma.transpose()) +
                     d2Ht_dtheta2 * (dtheta_dsigma * dtheta_dsigma.transpose()) +
                     dHt_drho * d2rho_dsigma2 + 
                     dHt_dtheta * d2theta_dsigma2;

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
    // if (sqpart < std::numeric_limits<double>::epsilon()) sqpart = 1.E-5;
    double m = (2. * (1 - e_val * e_val) * cos(theta)) +
               ((2. * e_val - 1) * std::sqrt(sqpart));
    // if (fabs(m) < std::numeric_limits<double>::epsilon()) m = 1.E-5;
    const double l = (4. * (1. - e_val * e_val) * std::pow(cos(theta), 2)) +
                     std::pow((2. * e_val - 1.), 2);
    double r_mw = (l / m) * r_mc;
    // Initialise meridional eccentricity
    const double xi = 0.1;
    const double a0 = 1e-2;
    double C0_s_min = std::pow(a0 * cohesion_peak_ * std::cos(phi), 2);
    double C0_s = std::max(std::pow(xi * cohesion_peak_ * std::tan(psi), 2), C0_s_min);
    double omega = C0_s +
                   std::pow((r_mw * std::sqrt(1.5) * rho), 2);
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
    const double dp_dtheta =
        (1.5 * rho * rho * r_mw * drmw) / H_s_val;
    // compute the value of dp/dsigma and dp/dj in shear yield
    (*dp_dsigma) = (dp_depsilon * depsilon_dsigma) + (dp_drho * drho_dsigma) +
                   (dp_dtheta * dtheta_dsigma);
    
    (*dp_dq) = dp_drho * std::sqrt(2. / 3.);

    const double d2l_dtheta2 = -8.0 * (1.0 - e_val * e_val) * (cos(theta)*cos(theta) - sin(theta)*sin(theta));
    const double term1 = d2l_dtheta2 / std::sqrt(sqpart);
    const double term2 = (dl_dtheta * dl_dtheta) / (2.0 * sqpart * std::sqrt(sqpart));
    const double d2m_dtheta2 = -2.0 * (1.0 - e_val * e_val) * cos(theta) + 
                               0.5 * (2.0 * e_val - 1.0) * (term1 - term2);
    // Exact analytical second derivative
    const double num2 = m * (m * d2l_dtheta2 - l * d2m_dtheta2) - 
                        2.0 * dm_dtheta * (m * dl_dtheta - l * dm_dtheta);
    double d2rmw = (num2 / (m * m * m)) * r_mc;
    if (rho*rho < C0_s) {
      r_mw = r_mc;   // Force axisymmetric circle near apex
      drmw = 0.0;    // Kill dr_mw/dtheta derivative noise
      d2rmw = 0.0;   // Kill d2r_mw/dtheta2 derivative noise
    }

    double dHs_drho = (1.5 * r_mw * r_mw * rho) / H_s_val;
    double dHs_dtheta = (1.5 * rho * rho * r_mw * drmw) / H_s_val;
    double d2Hs_drho2 = (1.5 * r_mw * r_mw * C0_s) / std::pow(H_s_val, 3);
    double d2Hs_drhodtheta = (3.0 * rho * r_mw * drmw * (C0_s + 0.75 * r_mw * r_mw * rho * rho)) / std::pow(H_s_val, 3);
    double d2Hs_dtheta2 = (0.75 * rho * rho / std::pow(H_s_val, 3)) * 
                          (2.0 * H_s_val * H_s_val * (drmw * drmw + r_mw * d2rmw) - 3.0 * rho * rho * r_mw * r_mw * drmw * drmw);

    //Compute d2P/dSigma2
    (*d2p_dsigma2) = d2Hs_drho2 * (drho_dsigma * drho_dsigma.transpose()) +
                     d2Hs_drhodtheta * (drho_dsigma * dtheta_dsigma.transpose() + dtheta_dsigma * drho_dsigma.transpose()) +
                     d2Hs_dtheta2 * (dtheta_dsigma * dtheta_dsigma.transpose()) +
                     dHs_drho * d2rho_dsigma2 + 
                     dHs_dtheta * d2theta_dsigma2;
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
      (*softening) =
          (-1.) * ((df_dphi * dphi_dpstrain) + (df_dc * dc_dpstrain)) * (*dp_dq);
    }
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

  // Characteristic normalization stress (prevents division by zero at origin)
  const double sig_trial_norm = std::max(trial_stress.norm(), 1.0e-8);

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
  bool local_converged = false;
  Vector6d current_stress = trial_stress;
  Vector6d dstrain_p_t = Vector6d::Zero();
  Vector6d dstrain_p_s = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  Eigen::Matrix<double, 2, 1> dlambda = Vector2d::Zero();

  // Updated state variables for return mapping iterations
  Vector6d plastic_strain_t = Vector6d::Zero();
  Vector6d plastic_strain_s = Vector6d::Zero();
  plastic_strain_t(0) = (*state_vars).at("plastic_strain_t0");
  plastic_strain_t(1) = (*state_vars).at("plastic_strain_t1");
  plastic_strain_t(2) = (*state_vars).at("plastic_strain_t2");
  plastic_strain_t(3) = (*state_vars).at("plastic_strain_t3");
  plastic_strain_t(4) = (*state_vars).at("plastic_strain_t4");
  plastic_strain_t(5) = (*state_vars).at("plastic_strain_t5");
  plastic_strain_s(0) = (*state_vars).at("plastic_strain_s0");
  plastic_strain_s(1) = (*state_vars).at("plastic_strain_s1");
  plastic_strain_s(2) = (*state_vars).at("plastic_strain_s2");
  plastic_strain_s(3) = (*state_vars).at("plastic_strain_s3");
  plastic_strain_s(4) = (*state_vars).at("plastic_strain_s4");
  plastic_strain_s(5) = (*state_vars).at("plastic_strain_s5");

  // Start newton raphson return mapping iterations
  (*state_vars).at("max_itr") = 0;

  // Return mapping in the shear-tensile region
  if (yield_type == mpm::mohrcoulomb::FailureState::ShearTensile) {
      Eigen::Matrix<double, 2, 1> dlambda_double = Vector2d::Zero();
      Eigen::Matrix<double, 2, 1> pdstrain = Vector2d::Zero();
      pdstrain(0) = mpm::materials::pdstrain(plastic_strain_t);
      pdstrain(1) = mpm::materials::pdstrain(plastic_strain_s);

    for (unsigned itr = 0; itr < max_iter_; ++itr) {
      (*state_vars).at("max_itr") = itr + 1;
      Vector6d stress_correction = Vector6d::Zero();

      std::tie(current_stress, dstrain_p_t, dstrain_p_s, local_converged) =
            this->compute_corner_return(current_stress, trial_stress, de, state_vars,
                                        yield_function, dlambda_double, pdstrain);
      if (!local_converged) 
          break;

      this->compute_stress_invariants(current_stress, state_vars);
      
      Vector6d current_plastic_strain_t = plastic_strain_t + dstrain_p_t;
      Vector6d current_plastic_strain_s = plastic_strain_s + dstrain_p_s;
      pdstrain(0) = mpm::materials::pdstrain(current_plastic_strain_t);
      pdstrain(1) = mpm::materials::pdstrain(current_plastic_strain_s);
      (*state_vars).at("plastic_strain_t0") = current_plastic_strain_t(0);
      (*state_vars).at("plastic_strain_t1") = current_plastic_strain_t(1);
      (*state_vars).at("plastic_strain_t2") = current_plastic_strain_t(2);
      (*state_vars).at("plastic_strain_t3") = current_plastic_strain_t(3);
      (*state_vars).at("plastic_strain_t4") = current_plastic_strain_t(4);
      (*state_vars).at("plastic_strain_t5") = current_plastic_strain_t(5);
      (*state_vars).at("plastic_strain_s0") = current_plastic_strain_s(0);
      (*state_vars).at("plastic_strain_s1") = current_plastic_strain_s(1);
      (*state_vars).at("plastic_strain_s2") = current_plastic_strain_s(2);
      (*state_vars).at("plastic_strain_s3") = current_plastic_strain_s(3);
      (*state_vars).at("plastic_strain_s4") = current_plastic_strain_s(4);
      (*state_vars).at("plastic_strain_s5") = current_plastic_strain_s(5);


      // Update softening-dependent strength parameters.
      if (softening_) {
        this->update_softening_parameters(state_vars, pdstrain);
      }

      // Re-evaluate invariants and active surface at current stress.
      this->compute_yield_state(&yield_function, (*state_vars));
      Vector6d R_sigma = current_stress - trial_stress + de * (dstrain_p_t + dstrain_p_s);
      double err_sigma = R_sigma.norm() / sig_trial_norm;

      // ---------------------------------------------------------------------
      // B. Convergence check
      // ---------------------------------------------------------------------

      if (dlambda_double(0) < -mandatory_tolerance_ || dlambda_double(1) < -mandatory_tolerance_) {
        current_stress = trial_stress;
        this->compute_stress_invariants(current_stress, state_vars);
        this->compute_yield_state(&yield_function, (*state_vars));
        if (dlambda_double(0) < -mandatory_tolerance_ && dlambda_double(1) >= -mandatory_tolerance_)
          yield_type = mpm::mohrcoulomb::FailureState::Shear;
        else if (dlambda_double(0) >= -mandatory_tolerance_ && dlambda_double(1) < -mandatory_tolerance_)
          yield_type = mpm::mohrcoulomb::FailureState::Tensile;
        else if (dlambda_double(0) < -mandatory_tolerance_ && dlambda_double(1) < -mandatory_tolerance_)
          yield_type = mpm::mohrcoulomb::FailureState::Elastic;
        break;
      }

      if (std::abs(yield_function(0)) <= tolerance_ && std::abs(yield_function(1)) <= tolerance_ && err_sigma < rel_tol_) {
        if (dlambda_double(0) >= 0.0 && dlambda_double(1) >= 0.0) {
          dlambda = dlambda_double;
          converged = true;
          (*state_vars).at("yield_state") = 3;
        }
        break;
        }
      }
      if (dlambda_double(0) < 0.0 || dlambda_double(1) < 0.0) {
        current_stress = trial_stress;
        this->compute_stress_invariants(current_stress, state_vars);
        this->compute_yield_state(&yield_function, (*state_vars));
        if (dlambda_double(0) < 0.0 && dlambda_double(1) >= 0.0)
          yield_type = mpm::mohrcoulomb::FailureState::Shear;
        else if (dlambda_double(0) >= 0.0 && dlambda_double(1) < 0.0)
          yield_type = mpm::mohrcoulomb::FailureState::Tensile;
        else if (dlambda_double(0) < 0.0 && dlambda_double(1) < 0.0)
          yield_type = mpm::mohrcoulomb::FailureState::Elastic;
      }
      else {
        if (!converged) {
          std::tie(current_stress, dstrain_p_t, dstrain_p_s) =
                this->enforce_corner_return(trial_stress, de, state_vars, yield_function, dlambda, pdstrain);

          this->compute_stress_invariants(current_stress, state_vars);
      
          Vector6d current_plastic_strain_t = plastic_strain_t + dstrain_p_t;
          Vector6d current_plastic_strain_s = plastic_strain_s + dstrain_p_s;
          pdstrain(0) = mpm::materials::pdstrain(current_plastic_strain_t);
          pdstrain(1) = mpm::materials::pdstrain(current_plastic_strain_s);
          (*state_vars).at("plastic_strain_t0") = current_plastic_strain_t(0);
          (*state_vars).at("plastic_strain_t1") = current_plastic_strain_t(1);
          (*state_vars).at("plastic_strain_t2") = current_plastic_strain_t(2);
          (*state_vars).at("plastic_strain_t3") = current_plastic_strain_t(3);
          (*state_vars).at("plastic_strain_t4") = current_plastic_strain_t(4);
          (*state_vars).at("plastic_strain_t5") = current_plastic_strain_t(5);
          (*state_vars).at("plastic_strain_s0") = current_plastic_strain_s(0);
          (*state_vars).at("plastic_strain_s1") = current_plastic_strain_s(1);
          (*state_vars).at("plastic_strain_s2") = current_plastic_strain_s(2);
          (*state_vars).at("plastic_strain_s3") = current_plastic_strain_s(3);
          (*state_vars).at("plastic_strain_s4") = current_plastic_strain_s(4);
          (*state_vars).at("plastic_strain_s5") = current_plastic_strain_s(5);
          converged = true;
        }
      }
    }

  if (yield_type == mpm::mohrcoulomb::FailureState::Elastic) {
            std::cout << ptr->id() << std::endl;
            console_->error("Invalid elastic type for compute_df_dp: {}; Tensile yield function: {}; Shear yield function: {}\n",
                    static_cast<int>(yield_type), yield_function(0), yield_function(1));
            return current_stress;
    }
    else if (yield_type == mpm::mohrcoulomb::FailureState::Tensile or yield_type == mpm::mohrcoulomb::FailureState::Shear) {

    Eigen::Matrix<double, 2, 1> pdstrain = Vector2d::Zero();
    pdstrain(0) = mpm::materials::pdstrain(plastic_strain_t);
    pdstrain(1) = mpm::materials::pdstrain(plastic_strain_s);
    for (unsigned itr = 0; itr < max_iter_; ++itr) {
        (*state_vars).at("max_itr") = itr + 1;

        double yield_function_single =
        (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
              ? yield_function(0)
              : yield_function(1);
        double dlambda_single =
        (yield_type == mpm::mohrcoulomb::FailureState::Tensile) ? dlambda(0)
                                                                  : dlambda(1);

        auto [corrected_stress, dp_dsigma_corrected, local_converged] = this->compute_single_surface_return(
          yield_type, yield_function_single, current_stress, de, state_vars,
          dlambda_single, pdstrain, trial_stress);
        if (!local_converged) 
            break;

        current_stress = corrected_stress;

        if (yield_type == mpm::mohrcoulomb::FailureState::Tensile) {
          dlambda(0) = dlambda_single;
          dstrain_p_t = dlambda(0) * dp_dsigma_corrected;
        } else {
          dlambda(1) = dlambda_single;
          dstrain_p_s = dlambda(1) * dp_dsigma_corrected;
        }

      Vector6d current_plastic_strain_t = plastic_strain_t + dstrain_p_t;
      Vector6d current_plastic_strain_s = plastic_strain_s + dstrain_p_s;
      pdstrain(0) = mpm::materials::pdstrain(current_plastic_strain_t);
      pdstrain(1) = mpm::materials::pdstrain(current_plastic_strain_s);
      (*state_vars).at("plastic_strain_t0") = current_plastic_strain_t(0);
      (*state_vars).at("plastic_strain_t1") = current_plastic_strain_t(1);
      (*state_vars).at("plastic_strain_t2") = current_plastic_strain_t(2);
      (*state_vars).at("plastic_strain_t3") = current_plastic_strain_t(3);
      (*state_vars).at("plastic_strain_t4") = current_plastic_strain_t(4);
      (*state_vars).at("plastic_strain_t5") = current_plastic_strain_t(5);
      (*state_vars).at("plastic_strain_s0") = current_plastic_strain_s(0);
      (*state_vars).at("plastic_strain_s1") = current_plastic_strain_s(1);
      (*state_vars).at("plastic_strain_s2") = current_plastic_strain_s(2);
      (*state_vars).at("plastic_strain_s3") = current_plastic_strain_s(3);
      (*state_vars).at("plastic_strain_s4") = current_plastic_strain_s(4);
      (*state_vars).at("plastic_strain_s5") = current_plastic_strain_s(5);
      (*state_vars).at("pdstrain") = pdstrain(1);

    // Update softening-dependent strength parameters.
    if (softening_) {
      this->update_softening_parameters(state_vars, pdstrain);
    }

    // Re-evaluate invariants and active surface at current stress.
    this->compute_yield_state(&yield_function, (*state_vars));

    Vector6d R_sigma = Vector6d::Zero();
    double other_yield_function = 0.0;

    if (yield_type == mpm::mohrcoulomb::FailureState::Tensile) {
        R_sigma = current_stress - trial_stress + dlambda(0) * (de * dp_dsigma_corrected);
        yield_function_single = yield_function(0);
        other_yield_function = yield_function(1);
      } else {
        R_sigma = current_stress - trial_stress + dlambda(1) * (de * dp_dsigma_corrected);
        yield_function_single = yield_function(1);
        other_yield_function = yield_function(0);
      }
    double err_sigma = R_sigma.norm() / sig_trial_norm;

    // ---------------------------------------------------------------------
    // B. Convergence check
    // ---------------------------------------------------------------------
    if (std::abs(yield_function_single) <= tolerance_ && err_sigma < rel_tol_) {
      if (other_yield_function <= tolerance_) {
        converged = true;
        // Assign final yield state near the converged point.
        if (yield_function(0) > -tolerance_ && yield_function(1) <= tolerance_) {
          (*state_vars).at("yield_state") = 2;  // Tensile
        } else if (yield_function(1) > -tolerance_ &&
                  yield_function(0) <= tolerance_) {
          (*state_vars).at("yield_state") = 1;  // Shear
        } else {
          (*state_vars).at("yield_state") = 0;  // Elastic
        }
      }
      else {
            for (unsigned itr = 0; itr < max_iter_; ++itr) {
              (*state_vars).at("max_itr") = itr + 1;
              Vector6d stress_correction = Vector6d::Zero();

              std::tie(current_stress, dstrain_p_t, dstrain_p_s, local_converged) =
                    this->compute_corner_return(current_stress, trial_stress, de, state_vars,
                                                yield_function, dlambda,
                                                pdstrain);
              if (!local_converged) 
                  break;

              this->compute_stress_invariants(current_stress, state_vars);
              
              Vector6d current_plastic_strain_t = plastic_strain_t + dstrain_p_t;
              Vector6d current_plastic_strain_s = plastic_strain_s + dstrain_p_s;
              pdstrain(0) = mpm::materials::pdstrain(current_plastic_strain_t);
              pdstrain(1) = mpm::materials::pdstrain(current_plastic_strain_s);
              (*state_vars).at("plastic_strain_t0") = current_plastic_strain_t(0);
              (*state_vars).at("plastic_strain_t1") = current_plastic_strain_t(1);
              (*state_vars).at("plastic_strain_t2") = current_plastic_strain_t(2);
              (*state_vars).at("plastic_strain_t3") = current_plastic_strain_t(3);
              (*state_vars).at("plastic_strain_t4") = current_plastic_strain_t(4);
              (*state_vars).at("plastic_strain_t5") = current_plastic_strain_t(5);
              (*state_vars).at("plastic_strain_s0") = current_plastic_strain_s(0);
              (*state_vars).at("plastic_strain_s1") = current_plastic_strain_s(1);
              (*state_vars).at("plastic_strain_s2") = current_plastic_strain_s(2);
              (*state_vars).at("plastic_strain_s3") = current_plastic_strain_s(3);
              (*state_vars).at("plastic_strain_s4") = current_plastic_strain_s(4);
              (*state_vars).at("plastic_strain_s5") = current_plastic_strain_s(5);
              (*state_vars).at("pdstrain") = pdstrain(1);

              // Update softening-dependent strength parameters.
              if (softening_) {
                this->update_softening_parameters(state_vars, pdstrain);
              }

              // Re-evaluate invariants and active surface at current stress.
              this->compute_yield_state(&yield_function, (*state_vars));
              Vector6d R_sigma = current_stress - trial_stress + de * (dstrain_p_t + dstrain_p_s);
              double err_sigma = R_sigma.norm() / sig_trial_norm;

              // ---------------------------------------------------------------------
              // B. Convergence check
              // ---------------------------------------------------------------------
              if ((yield_function(0)) <= tolerance_ && (yield_function(1)) <= tolerance_ && err_sigma < rel_tol_) {
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

  // =========================================================================
  // 4. Post-convergence handling
  // =========================================================================
  if (!converged ) {
    // Warn when return mapping fails to converge within the iteration budget.
    console_->warn(
        "MohrCoulomb::compute_stress: Return mapping did not converge "
        "after {} iterations. Final yield functions: f_t = {}, f_s = {}",
        max_iter_, yield_function(0), yield_function(1));
  }

  return current_stress;
}

//! Single-surface return mapping
//! Returns: (stress correction, plastic deviatoric strain increment, success
//! flag)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>, bool>
    mpm::MohrCoulomb<Tdim>::compute_single_surface_return(
        mpm::mohrcoulomb::FailureState yield_type, double yield_function,
        const Vector6d& current_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars, double& dlambda_single,
        Eigen::Matrix<double, 2, 1>& pdstrain, const Vector6d& trial_stress) {

  // Select active yield-function value on the current surface.
  double f_current = yield_function;

  // Compute flow/yield gradients and softening contribution.
  Vector6d df_dsigma = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  double dp_dq = 0.0;
  Matrix6x6 d2p_dsigma2 = Matrix6x6::Zero();
  double softening_modulus = 0.0;

  // Updated state variables for return mapping iterations
  Vector6d plastic_strain_t = Vector6d::Zero();
  Vector6d plastic_strain_s = Vector6d::Zero();
  plastic_strain_t(0) = (*state_vars).at("plastic_strain_t0");
  plastic_strain_t(1) = (*state_vars).at("plastic_strain_t1");
  plastic_strain_t(2) = (*state_vars).at("plastic_strain_t2");
  plastic_strain_t(3) = (*state_vars).at("plastic_strain_t3");
  plastic_strain_t(4) = (*state_vars).at("plastic_strain_t4");
  plastic_strain_t(5) = (*state_vars).at("plastic_strain_t5");
  plastic_strain_s(0) = (*state_vars).at("plastic_strain_s0");
  plastic_strain_s(1) = (*state_vars).at("plastic_strain_s1");
  plastic_strain_s(2) = (*state_vars).at("plastic_strain_s2");
  plastic_strain_s(3) = (*state_vars).at("plastic_strain_s3");
  plastic_strain_s(4) = (*state_vars).at("plastic_strain_s4");
  plastic_strain_s(5) = (*state_vars).at("plastic_strain_s5");

  double pdstrain_single =
        (yield_type == mpm::mohrcoulomb::FailureState::Tensile) ? pdstrain(0)
                                                                  : pdstrain(1);

  this->compute_df_dp(yield_type, state_vars, current_stress, &df_dsigma,
                      &dp_dsigma, &dp_dq, &d2p_dsigma2, &softening_modulus, pdstrain_single);

  // Assemble system matrix.
  Vector6d de_dp = de * dp_dsigma;

  Vector6d R_sigma = current_stress - trial_stress + dlambda_single * de_dp;

  // Construct the 6x6 curvature operator M
  Matrix6x6 M = Matrix6x6::Identity() + dlambda_single * (de * d2p_dsigma2);

  // Invert the 6x6 curvature operator M
  Matrix6x6 M_inv = M.partialPivLu().inverse();

  // Consistency denominator:
  // n^T De m + H, where n = dF/dsigma, m = dP/dsigma, H = softening term.
  double denominator =
      df_dsigma.dot(M_inv * de_dp) +
      softening_modulus;

  if (std::abs(denominator) < std::numeric_limits<double>::epsilon()) {
    console_->warn(
        "MohrCoulomb: Consistency denominator is near zero ({:.6e}). "
        "This may indicate numerical instability. Applying regularization.",
        denominator);
    denominator = check_low(denominator);
  }

  // Consistency numerator:
  double numerator = f_current - df_dsigma.dot(M_inv * R_sigma);

  // Plastic multiplier increment from first-order consistency.
  double d_dlambda = numerator / denominator;

  // Stress update
  Vector6d d_sigma = -d_dlambda * M_inv * de_dp - M_inv * R_sigma;

  // --- 2. Calculate initial error (Merit Function) ---
  double initial_merit = 0.5 * (R_sigma.squaredNorm() + std::pow(std::max(0.0, f_current), 2));

  // --- 3. Line Search Backtracking ---
  double alpha = 1.0; // Start assuming full Newton step (100%)
  double c1 = 5e-2;   // Armijo decrease threshold

  Vector6d trial_corrected_stress = current_stress;
  double trial_dlambda = dlambda_single;
  double trial_merit = 0.0;

  Eigen::Matrix<double, 2, 1> trial_yield_function;
  Vector6d df_dsigma_trial = Vector6d::Zero();
  Vector6d dp_dsigma_trial = Vector6d::Zero();
  double dp_dq_trial = 0.0;
  Matrix6x6 d2p_dsigma2_trial = Matrix6x6::Zero();
  double softening_modulus_trial = 0.0;

  bool converged = false;

  double trial_merit_1 = 0.0;
  double r_norm_1 = 0.0;

  for (int line_iter = 0; line_iter < 10; ++line_iter) {
      // Propose step scaled by alpha
      trial_corrected_stress = current_stress + alpha * d_sigma;
      trial_dlambda = dlambda_single + alpha * d_dlambda;
      
      // Call your function that re-evaluates the yield function at trial_stress
      
      this->compute_stress_invariants(trial_corrected_stress, state_vars);

      this->compute_df_dp(yield_type, state_vars, trial_corrected_stress, &df_dsigma_trial,
                      &dp_dsigma_trial, &dp_dq_trial, &d2p_dsigma2_trial, &softening_modulus_trial, pdstrain_single);

      Vector6d trial_dstrain_p = trial_dlambda * dp_dsigma_trial;
      if (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
      {
        Vector6d trial_plastic_strain_t = plastic_strain_t + trial_dstrain_p;
        pdstrain(0) = mpm::materials::pdstrain(trial_plastic_strain_t);
      } else {
        Vector6d trial_plastic_strain_s = plastic_strain_s + trial_dstrain_p;
        pdstrain(1) = mpm::materials::pdstrain(trial_plastic_strain_s);
      }

      if (softening_) {
        this->update_softening_parameters(state_vars, pdstrain);
      }
      // Recalculate R_sigma and f at this temporary trial position
      Vector6d de_dp_trial = de * dp_dsigma_trial; // Update flow direction if necessary
      Vector6d R_sigma_trial = trial_corrected_stress - trial_stress + trial_dlambda * de_dp_trial;
      auto trial_yield_type = this->compute_yield_state(&trial_yield_function, (*state_vars));
      double f_trial=
          (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
              ? trial_yield_function(0)
              : trial_yield_function(1);

      // Compute trial merit error score
      trial_merit = 0.5 * (R_sigma_trial.squaredNorm() + std::pow(std::max(0.0, f_trial), 2));

      if (line_iter == 0) {
        trial_merit_1 = 0.5 * (R_sigma_trial.squaredNorm() + std::pow(std::max(0.0, f_trial), 2));
        r_norm_1 = R_sigma_trial.squaredNorm();
      }

      // Accept step if total error strictly decreased
      if (trial_merit <= initial_merit * (1-c1*alpha)) {
          converged = true;
          break; 
      }

      alpha *= 0.8; // Cut step size with a ratio of 0.8
  }

        if (!converged) {
          // Warn when return mapping fails to converge within the iteration budget.
          console_->warn(
              "Single surface: Return mapping did not converge ");
          trial_corrected_stress = current_stress;
          trial_dlambda = dlambda_single;
        }

    // --- 4. Apply damped state update ---
    Vector6d corrected_stress = trial_corrected_stress;
    dlambda_single = trial_dlambda;

    Vector6d df_dsigma_after = Vector6d::Zero();
    Vector6d dp_dsigma_after = Vector6d::Zero();
    double dp_dq_after = 0.0;
    Matrix6x6 d2p_dsigma2_after = Matrix6x6::Zero();
    double softening_modulus_after = 0.0;

    this->compute_stress_invariants(corrected_stress, state_vars);

    this->compute_df_dp(yield_type, state_vars, corrected_stress, &df_dsigma_after,
                        &dp_dsigma_after, &dp_dq_after, &d2p_dsigma2_after, &softening_modulus_after, pdstrain_single);

    Vector6d de_dp_after = de * dp_dsigma_after;
    Vector6d after_R_sigma = corrected_stress - trial_stress + dlambda_single * de_dp_after;
    Vector6d stress_correction = corrected_stress - current_stress;
    Vector6d theoretic_dp_dsigma_change = d2p_dsigma2 * stress_correction;

  return {corrected_stress, dp_dsigma_after, converged};
}

//! Corner return mapping (multi-surface)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>,
           Eigen::Matrix<double, 6, 1>, bool>
    mpm::MohrCoulomb<Tdim>::compute_corner_return(
        const Vector6d& current_stress, const Vector6d& trial_stress, const Matrix6x6& de,
        mpm::dense_map* state_vars,
        const Eigen::Matrix<double, 2, 1>& yield_function,
        Eigen::Matrix<double, 2, 1>& dlambda, Eigen::Matrix<double, 2, 1>& pdstrain) {

  // Compute gradients for both active surfaces.
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

  // Updated state variables for return mapping iterations
  Vector6d plastic_strain_t = Vector6d::Zero();
  Vector6d plastic_strain_s = Vector6d::Zero();
  plastic_strain_t(0) = (*state_vars).at("plastic_strain_t0");
  plastic_strain_t(1) = (*state_vars).at("plastic_strain_t1");
  plastic_strain_t(2) = (*state_vars).at("plastic_strain_t2");
  plastic_strain_t(3) = (*state_vars).at("plastic_strain_t3");
  plastic_strain_t(4) = (*state_vars).at("plastic_strain_t4");
  plastic_strain_t(5) = (*state_vars).at("plastic_strain_t5");
  plastic_strain_s(0) = (*state_vars).at("plastic_strain_s0");
  plastic_strain_s(1) = (*state_vars).at("plastic_strain_s1");
  plastic_strain_s(2) = (*state_vars).at("plastic_strain_s2");
  plastic_strain_s(3) = (*state_vars).at("plastic_strain_s3");
  plastic_strain_s(4) = (*state_vars).at("plastic_strain_s4");
  plastic_strain_s(5) = (*state_vars).at("plastic_strain_s5");

  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars,
                      current_stress, &df_dsigma_t, &dp_dsigma_t, &dp_dq_t,
                      &d2p_dsigma2_t, &softening_t, pdstrain(0));
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars,
                      current_stress, &df_dsigma_s, &dp_dsigma_s, &dp_dq_s,
                      &d2p_dsigma2_s, &softening_s, pdstrain(1));

  // Recompute yield-function values at the current stress.
  const double f_t = yield_function(0);  // Tension
  const double f_s = yield_function(1);  // Shear

  // Compute intermediate matrix M
  Matrix6x6 M = Matrix6x6::Identity() + de * (dlambda(0) * d2p_dsigma2_t + dlambda(1) * d2p_dsigma2_s);
  Matrix6x6 M_inv = M.partialPivLu().inverse();
  // =========================================================================
  // Solve 2x2 consistency system for the two plastic multipliers:
  // [A11 A12] [lambda_t]   [f_t]
  // [A21 A22] [lambda_s] = [f_s]
  // =========================================================================
  Eigen::Matrix<double, 2, 2> A;
  Eigen::Vector2d b;

  // Assemble system matrix.
  Vector6d de_dp_t = de * dp_dsigma_t;
  Vector6d de_dp_s = de * dp_dsigma_s;

  Vector6d R_sigma = current_stress - trial_stress + de * (dlambda(0) * dp_dsigma_t + dlambda(1) * dp_dsigma_s);

  A(0, 0) = df_dsigma_t.dot(M_inv*de_dp_t) + softening_t;
  A(0, 1) = df_dsigma_t.dot(M_inv*de_dp_s);
  A(1, 0) = df_dsigma_s.dot(M_inv*de_dp_t);
  A(1, 1) = df_dsigma_s.dot(M_inv*de_dp_s) + softening_s;

  b(0) = f_t-df_dsigma_t.dot(M_inv * R_sigma);
  b(1) = f_s-df_dsigma_s.dot(M_inv * R_sigma);

  // Check determinant for singular/ill-conditioned corner system.
  double det_A = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  double norm_A_sq = A.squaredNorm();

  // Relative ill-conditioning threshold instead of machine epsilon
  if (std::abs(det_A) < 1e-12 * norm_A_sq || std::abs(det_A) < 1e-14) {
      console_->warn(
          "MohrCoulomb: Corner return matrix is singular or ill-conditioned (det = {:.6e}). "
          "Falling back to single surface return.",
          det_A);
    return {Vector6d::Zero(), Vector6d::Zero(), Vector6d::Zero(), false};
  }

  // Solve explicitly via inverse (closed form for 2x2).
  Eigen::Matrix<double, 2, 2> A_inv;
  A_inv(0, 0) = A(1, 1) / det_A;
  A_inv(0, 1) = -A(0, 1) / det_A;
  A_inv(1, 0) = -A(1, 0) / det_A;
  A_inv(1, 1) = A(0, 0) / det_A;

  Eigen::Vector2d delta_dlambda = A_inv * b;

  // Stress update
  Vector6d d_sigma = - M_inv * de * (delta_dlambda(0) * dp_dsigma_t + delta_dlambda(1) * dp_dsigma_s) - M_inv * R_sigma;

  // --- 2. Calculate initial error (Merit Function) ---
  double initial_merit = 0.5 * (R_sigma.squaredNorm() + 0.5 * (std::pow(std::max(0.0, f_t), 2) + std::pow(std::max(0.0, f_s), 2)));

  // --- 3. Line Search Backtracking ---
  double alpha = 1.0; // Start assuming full Newton step (100%)
  double c1 = 5e-2;   // Armijo decrease threshold

  Vector6d trial_corrected_stress = current_stress;
  Eigen::Matrix<double, 2, 1> trial_dlambda = dlambda;
  double trial_merit = 0.0;

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

  double trial_merit_1 = 0.0;
  double r_norm_1 = 0.0;

  bool converged = false;


  for (int line_iter = 0; line_iter < 10; ++line_iter) {
      // Propose step scaled by alpha
      trial_corrected_stress = current_stress + alpha * d_sigma;
      trial_dlambda = dlambda + alpha * delta_dlambda;
      
      // Call your function that re-evaluates the yield function at trial_stress
      
      this->compute_stress_invariants(trial_corrected_stress, state_vars);

      this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars, trial_corrected_stress, &df_dsigma_t_trial,
                      &dp_dsigma_t_trial, &dp_dq_t_trial, &d2p_dsigma2_t_trial, &softening_modulus_t_trial, pdstrain(0));
      this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars, trial_corrected_stress, &df_dsigma_s_trial,
                      &dp_dsigma_s_trial, &dp_dq_s_trial, &d2p_dsigma2_s_trial, &softening_modulus_s_trial, pdstrain(1));

      Vector6d trial_dstrain_p_t = trial_dlambda(0) * dp_dsigma_t_trial;
      Vector6d trial_dstrain_p_s = trial_dlambda(1) * dp_dsigma_s_trial;
      Vector6d trial_plastic_strain_t = plastic_strain_t + trial_dstrain_p_t;
      Vector6d trial_plastic_strain_s = plastic_strain_s + trial_dstrain_p_s;
      pdstrain(0) = mpm::materials::pdstrain(trial_plastic_strain_t);
      pdstrain(1) = mpm::materials::pdstrain(trial_plastic_strain_s);
      if (softening_) {
        this->update_softening_parameters(state_vars, pdstrain);
      }
      // Recalculate R_sigma and f at this temporary trial position
      Vector6d de_dp_t_trial = de * dp_dsigma_t_trial;
      Vector6d de_dp_s_trial = de * dp_dsigma_s_trial;
      Vector6d R_sigma_trial = trial_corrected_stress - trial_stress + (trial_dlambda(0) * de_dp_t_trial + trial_dlambda(1) * de_dp_s_trial);
      auto trial_yield_type = this->compute_yield_state(&trial_yield_function, (*state_vars));

      // Compute trial merit error score
      trial_merit = 0.5 * (R_sigma_trial.squaredNorm()+ 0.5 * (std::pow(std::max(0.0, trial_yield_function(0)), 2) + std::pow(std::max(0.0, trial_yield_function(1)), 2)));

      if (line_iter == 0) {
        trial_merit_1 = 0.5 * (R_sigma_trial.squaredNorm() + 0.5 * (std::pow(std::max(0.0, trial_yield_function(0)), 2) + std::pow(std::max(0.0, trial_yield_function(1)), 2)));
        r_norm_1 = R_sigma_trial.squaredNorm();
      }

      // Accept step if total error strictly decreased
      if (trial_merit <= initial_merit * (1-c1*alpha)) {
          converged = true;
          break; 
      }

      alpha *= 0.8; // Cut step size with a ratio of 0.8
  }
        if (!converged) {
          // Warn when return mapping fails to converge within the iteration budget.
          console_->warn(
              "Double surface: Return mapping did not converge ");
          trial_corrected_stress = current_stress;
          trial_dlambda = dlambda;
        }

    // --- 4. Apply damped state update ---
    Vector6d corrected_stress = trial_corrected_stress;
    dlambda = trial_dlambda;

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

  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars,
                      corrected_stress, &corrected_df_dsigma_t, &corrected_dp_dsigma_t, &corrected_dp_dq_t,
                      &corrected_d2p_dsigma2_t, &corrected_softening_t, pdstrain(0));
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars,
                      corrected_stress, &corrected_df_dsigma_s, &corrected_dp_dsigma_s, &corrected_dp_dq_s,
                      &corrected_d2p_dsigma2_s, &corrected_softening_s, pdstrain(1));
  
  Vector6d corrected_de_dp_t = de * corrected_dp_dsigma_t;
  Vector6d corrected_de_dp_s = de * corrected_dp_dsigma_s;

  Vector6d R_sigma_after = corrected_stress - trial_stress + de * (dlambda(0) * corrected_dp_dsigma_t + dlambda(1) * corrected_dp_dsigma_s);

  // Multiply flow directions by plastic multipliers to return true strain increments
  Vector6d dstrain_p_t = dlambda(0) * corrected_dp_dsigma_t;
  Vector6d dstrain_p_s = dlambda(1) * corrected_dp_dsigma_s;

  return {corrected_stress, dstrain_p_t, dstrain_p_s, converged};
}


template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 1>,
           Eigen::Matrix<double, 6, 1>>
mpm::MohrCoulomb<Tdim>::enforce_corner_return(
    const Vector6d& trial_stress, const Matrix6x6& de,
    mpm::dense_map* state_vars,
    const Eigen::Matrix<double, 2, 1>& yield_function,
    Eigen::Matrix<double, 2, 1>& dlambda, Eigen::Matrix<double, 2, 1>& pdstrain) {

  // 1. Identify the Apex Pressure
  // Retrieve the tensile strength limit. Adjust the sign based on your code's 
  // convention (e.g., if compression is positive in your solver, this should be negative).
  double p_apex = (*state_vars).at("tension_cutoff"); 

  // 2. Construct the Target Apex Stress in Voigt notation 
  // sigma_apex = [p, p, p, 0, 0, 0]^T
  Vector6d apex_stress = Vector6d::Zero();
  apex_stress(0) = p_apex;
  apex_stress(1) = p_apex;
  apex_stress(2) = p_apex;

  // 3. Compute the Overstress Vector
  // This is the stress that must be completely dissipated by plastic strain.
  Vector6d delta_sigma = trial_stress - apex_stress;

  // 4. Back-calculate the Total Plastic Strain Increment
  // delta_sigma = D_e * delta_eps_p  =>  delta_eps_p = D_e^{-1} * delta_sigma
  // Using fullPivLu() for robust solving without explicitly assembling the inverse matrix.
  Vector6d total_plastic_strain = de.fullPivLu().solve(delta_sigma);

  // 5. Decompose Plastic Strain into Hydrostatic and Deviatoric Components
  // Calculate volumetric plastic strain: tr(eps_p) = eps_xx + eps_yy + eps_zz
  double vol_strain = total_plastic_strain(0) + total_plastic_strain(1) + total_plastic_strain(2);

  // dstrain_p_t is the purely hydrostatic component
  Vector6d dstrain_p_t = Vector6d::Zero();
  dstrain_p_t(0) = vol_strain / 3.0;
  dstrain_p_t(1) = vol_strain / 3.0;
  dstrain_p_t(2) = vol_strain / 3.0;

  // dstrain_p_s is the purely deviatoric component (total minus hydrostatic)
  Vector6d dstrain_p_s = total_plastic_strain - dstrain_p_t;


  // Return the fixed apex stress and the decomposed plastic strain vectors
  return {apex_stress, dstrain_p_t, dstrain_p_s};
}

//! Update softening parameters
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::update_softening_parameters(
    mpm::dense_map* state_vars, Eigen::Matrix<double, 2, 1>& pdstrain) {
  if (pdstrain(1) <= pdstrain_peak_) {
    // Pre-peak: keep peak strength parameters.
    (*state_vars).at("phi") = phi_peak_;
    (*state_vars).at("psi") = psi_peak_;
    (*state_vars).at("cohesion") = cohesion_peak_;
  } else if (pdstrain(1) >= pdstrain_residual_) {
    // Post-residual: keep residual strength parameters.
    (*state_vars).at("phi") = phi_residual_;
    (*state_vars).at("psi") = psi_residual_;
    (*state_vars).at("cohesion") = cohesion_residual_;
  } else {
    // Linear softening law between peak and residual states.
    const double ratio =
        (pdstrain(1) - pdstrain_peak_) / (pdstrain_residual_ - pdstrain_peak_);
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

  Vector6d plastic_strain_t = Vector6d::Zero();
  Vector6d plastic_strain_s = Vector6d::Zero();
  plastic_strain_t(0) = (*state_vars).at("plastic_strain_t0");
  plastic_strain_t(1) = (*state_vars).at("plastic_strain_t1");
  plastic_strain_t(2) = (*state_vars).at("plastic_strain_t2");
  plastic_strain_t(3) = (*state_vars).at("plastic_strain_t3");
  plastic_strain_t(4) = (*state_vars).at("plastic_strain_t4");
  plastic_strain_t(5) = (*state_vars).at("plastic_strain_t5");
  plastic_strain_s(0) = (*state_vars).at("plastic_strain_s0");
  plastic_strain_s(1) = (*state_vars).at("plastic_strain_s1");
  plastic_strain_s(2) = (*state_vars).at("plastic_strain_s2");
  plastic_strain_s(3) = (*state_vars).at("plastic_strain_s3");
  plastic_strain_s(4) = (*state_vars).at("plastic_strain_s4");
  plastic_strain_s(5) = (*state_vars).at("plastic_strain_s5");
  
  Eigen::Matrix<double, 2, 1> pdstrain = Vector2d::Zero();
  pdstrain(0) = mpm::materials::pdstrain(plastic_strain_t);
  pdstrain(1) = mpm::materials::pdstrain(plastic_strain_s);

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

  if (yield_type == mpm::mohrcoulomb::FailureState::Tensile || yield_type == mpm::mohrcoulomb::FailureState::Shear) {
  //! Elasto-plastic stiffness matrix
  Matrix6x6 d_ep;
  // Compute df_dsigma dp_dsigma
  double softening = 0.;
  Matrix6x6 d2p_dsigma2 = Matrix6x6::Zero();
  double dp_dq = 0.;
  Vector6d df_dsigma = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  double pdstrain_single =
        (yield_type == mpm::mohrcoulomb::FailureState::Tensile) ? pdstrain(0)
                                                                  : pdstrain(1);
  // Compute stress invariants based on trial stress
  this->compute_stress_invariants(stress, state_vars);
  this->compute_df_dp(yield_type, state_vars, stress, &df_dsigma, &dp_dsigma,
                      &dp_dq, &d2p_dsigma2, &softening, pdstrain_single);

  // Compute the d_ep tensor
  Eigen::Matrix<double, 6, 1> de_dpdsigma = de * dp_dsigma;
  double dfdsigma_de_dpdsigma = df_dsigma.dot(de_dpdsigma);
  Eigen::Matrix<double, 6, 1> de_dfdsigma = de * df_dsigma;

  if (!hardening) softening = 0.;
  d_ep = de - 1. / (dfdsigma_de_dpdsigma + softening) *
                  (de_dpdsigma * de_dfdsigma.transpose());

  return d_ep;
  }

  //! Elasto-plastic stiffness matrix
  Matrix6x6 d_ep;
  // Compute df_dsigma dp_dsigma for the shear surface
  double softening_shear = 0.;
  Matrix6x6 d2p_dsigma2_shear = Matrix6x6::Zero();
  double dp_dq_shear = 0.;
  Vector6d df_dsigma_shear = Vector6d::Zero();
  Vector6d dp_dsigma_shear = Vector6d::Zero();
  // Compute df_dsigma dp_dsigma for the tension surface
  double softening_tension = 0.;
  Matrix6x6 d2p_dsigma2_tension = Matrix6x6::Zero();
  double dp_dq_tension = 0.;
  Vector6d df_dsigma_tension = Vector6d::Zero();
  Vector6d dp_dsigma_tension = Vector6d::Zero();

  // Compute stress invariants based on trial stress
  this->compute_stress_invariants(stress, state_vars);
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars,
                      stress, &df_dsigma_tension, &dp_dsigma_tension,
                      &dp_dq_tension, &d2p_dsigma2_tension, &softening_tension,
                      pdstrain(0));
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars, stress,
                      &df_dsigma_shear, &dp_dsigma_shear, &dp_dq_shear,
                      &d2p_dsigma2_shear, &softening_shear, pdstrain(1));


  // Compute the d_ep tensor
  Eigen::Matrix<double, 6, 1> de_dpdsigma_shear = de * dp_dsigma_shear;
  Eigen::Matrix<double, 6, 1> de_dpdsigma_tension = de * dp_dsigma_tension;
  Eigen::Matrix<double, 6, 1> de_dfdsigma_shear = de * df_dsigma_shear;
  Eigen::Matrix<double, 6, 1> de_dfdsigma_tension = de * df_dsigma_tension;

  if (!hardening) {
    softening_shear = 0.;
    softening_tension = 0.;
  }

  // 2x2 consistency matrix: df_i : de : dp_j (+ softening on the diagonal)
  // Cross-hardening between shear and tension is neglected
  Eigen::Matrix2d dfdsigma_de_dpdsigma;
  dfdsigma_de_dpdsigma(0, 0) =
      df_dsigma_shear.dot(de_dpdsigma_shear) + softening_shear;
  dfdsigma_de_dpdsigma(0, 1) = df_dsigma_shear.dot(de_dpdsigma_tension);
  dfdsigma_de_dpdsigma(1, 0) = df_dsigma_tension.dot(de_dpdsigma_shear);
  dfdsigma_de_dpdsigma(1, 1) =
      df_dsigma_tension.dot(de_dpdsigma_tension) + softening_tension;

  // Loading check (Kuhn-Tucker): both plastic multipliers must be positive,
  // otherwise only one surface is active and the single surface tensor applies
  const Eigen::Matrix2d inv_dfdsigma_de_dpdsigma =
      dfdsigma_de_dpdsigma.inverse();
  Eigen::Vector2d df_de_dstrain;
  df_de_dstrain << de_dfdsigma_shear.dot(dstrain),
      de_dfdsigma_tension.dot(dstrain);
  const Eigen::Vector2d dlambda = inv_dfdsigma_de_dpdsigma * df_de_dstrain;

  // // Shear multiplier negative: tension only
  // if (dlambda(0) < 0.) {
  //   d_ep = de - 1. / dfdsigma_de_dpdsigma(1, 1) *
  //                   (de_dpdsigma_tension * de_dfdsigma_tension.transpose());
  //   return d_ep;
  // }
  // // Tension multiplier negative: shear only
  // if (dlambda(1) < 0.) {
  //   d_ep = de - 1. / dfdsigma_de_dpdsigma(0, 0) *
  //                   (de_dpdsigma_shear * de_dfdsigma_shear.transpose());
  //   return d_ep;
  // }

  // Corner: d_ep = de - [de_dp_s de_dp_t] * h^-1 * [de_df_s de_df_t]^T
  Eigen::Matrix<double, 6, 2> de_dpdsigma;
  de_dpdsigma.col(0) = de_dpdsigma_shear;
  de_dpdsigma.col(1) = de_dpdsigma_tension;
  Eigen::Matrix<double, 2, 6> de_dfdsigma_transpose;
  de_dfdsigma_transpose.row(0) = de_dfdsigma_shear.transpose();
  de_dfdsigma_transpose.row(1) = de_dfdsigma_tension.transpose();

  d_ep = de - de_dpdsigma * inv_dfdsigma_de_dpdsigma * de_dfdsigma_transpose;

  return d_ep;
}
