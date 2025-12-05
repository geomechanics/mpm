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
      // Epsilon
      {"epsilon", 0.},
      // Rho
      {"rho", 0.},
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
      "epsilon",     "rho", "theta", "pdstrain"};
  return state_vars;
}

//! Compute stress invariants
template <unsigned Tdim>
bool mpm::MohrCoulomb<Tdim>::compute_stress_invariants(
    const Vector6d& stress, mpm::dense_map* state_vars) {
  // Compute the mean pressure
  (*state_vars).at("epsilon") = mpm::materials::p(stress) * std::sqrt(3.);
  // Compute theta value
  (*state_vars).at("theta") = mpm::materials::lode_angle(stress);
  // Compute rho
  (*state_vars).at("rho") = std::sqrt(2. * mpm::materials::j2(stress));

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
  const double epsilon = state_vars.at("epsilon");
  const double rho = state_vars.at("rho");
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
  const double rho = (*state_vars).at("rho");
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
  // 1. 定数・初期設定
  // =========================================================================
  const double Tolerance = 1E-7;     // 降伏判定の許容値
  const double Min_Denominator = 1E-15;  // 分母の最小値
  const unsigned itr_max = 50;       // 最大反復回数
  const double G = shear_modulus_;
  
  // 密度チェック（粒子が分離状態にあるか）
  const double current_packing_density = ptr->mass_density();
  const double critical_density = minimum_packing_fraction_ * grain_density_;

  // 弾性テンソルの計算
  (*state_vars).at("yield_state") = 0;
  Matrix6x6 de = this->compute_elastic_tensor(state_vars);

  // =========================================================================
  // 2. Elastic Predictor (試行応力の計算)
  // =========================================================================
  Vector6d trial_stress =
      this->compute_trial_stress(stress, dstrain, de, ptr, state_vars);

  // Separated state (引張による分離): 応力をゼロにしてリターン
  if (current_packing_density <= critical_density) {
    (*state_vars).at("pdstrain") += mpm::materials::q(trial_stress) / (3.0 * G);
    (*state_vars).at("yield_state") = 3; 
    return Vector6d::Zero();
  }

  // 試行応力での不変量計算
  this->compute_stress_invariants(trial_stress, state_vars);

  // 試行応力での降伏関数値の確認
  Eigen::Matrix<double, 2, 1> yield_function;
  auto yield_type = this->compute_yield_state(&yield_function, (*state_vars));

  // 弾性判定: 両方の降伏関数が許容値以下なら試行応力をそのまま返す
  if (yield_function(0) <= Tolerance && yield_function(1) <= Tolerance) {
    (*state_vars).at("yield_state") = 0;
    return trial_stress;
  }

  // =========================================================================
  // 3. Plastic Corrector (Cutting Plane Method + Corner Return)
  // =========================================================================
  
  Vector6d current_stress = trial_stress;
  double lambda_total = 0.0;
  bool converged = false;
  unsigned final_itr = 0;

  for (unsigned itr = 0; itr < itr_max; ++itr) {
    final_itr = itr;
    
    // ---------------------------------------------------------------------
    // A. 現在の応力状態での降伏状態・不変量を再評価
    // ---------------------------------------------------------------------
    this->compute_stress_invariants(current_stress, state_vars);
    yield_type = this->compute_yield_state(&yield_function, (*state_vars));

    // ---------------------------------------------------------------------
    // B. 収束判定
    // ---------------------------------------------------------------------
    if (yield_function(0) <= Tolerance && yield_function(1) <= Tolerance) {
      converged = true;
      // 最終的なyield_stateを設定
      if (yield_function(0) > -Tolerance && yield_function(1) <= -Tolerance) {
        (*state_vars).at("yield_state") = 2;  // Tensile
      } else if (yield_function(1) > -Tolerance && yield_function(0) <= -Tolerance) {
        (*state_vars).at("yield_state") = 1;  // Shear
      } else {
        (*state_vars).at("yield_state") = 0;  // Elastic
      }
      break;
    }

    // ---------------------------------------------------------------------
    // C. 降伏タイプの判定とコーナー処理
    // ---------------------------------------------------------------------
    const bool tension_violated = (yield_function(0) > Tolerance);
    const bool shear_violated = (yield_function(1) > Tolerance);

    Vector6d stress_correction = Vector6d::Zero();
    double dpdstrain_inc = 0.0;

    if (tension_violated && shear_violated) {
      // =================================================================
      // コーナーリターンマッピング（マルチサーフェス法）
      // 両方の降伏関数が違反している場合
      // =================================================================
      auto [corner_correction, corner_dpdstrain, corner_success] = 
          this->compute_corner_return(current_stress, de, state_vars);
      
      if (corner_success) {
        stress_correction = corner_correction;
        dpdstrain_inc = corner_dpdstrain;
      } else {
        // コーナーリターンが失敗した場合、より違反が大きい方で単一面リターン
        if (yield_function(0) > yield_function(1)) {
          yield_type = mpm::mohrcoulomb::FailureState::Tensile;
        } else {
          yield_type = mpm::mohrcoulomb::FailureState::Shear;
        }
        auto [single_correction, single_dpdstrain, single_success] =
            this->compute_single_surface_return(
                yield_type, yield_function, current_stress, de, state_vars);
        stress_correction = single_correction;
        dpdstrain_inc = single_dpdstrain;
      }
    } else {
      // =================================================================
      // 単一面リターンマッピング
      // =================================================================
      if (tension_violated) {
        yield_type = mpm::mohrcoulomb::FailureState::Tensile;
        (*state_vars).at("yield_state") = 2;
      } else {
        yield_type = mpm::mohrcoulomb::FailureState::Shear;
        (*state_vars).at("yield_state") = 1;
      }
      
      auto [single_correction, single_dpdstrain, single_success] =
          this->compute_single_surface_return(
              yield_type, yield_function, current_stress, de, state_vars);
      stress_correction = single_correction;
      dpdstrain_inc = single_dpdstrain;
    }

    // ---------------------------------------------------------------------
    // D. 応力と塑性ひずみの更新
    // ---------------------------------------------------------------------
    current_stress += stress_correction;
    (*state_vars).at("pdstrain") += dpdstrain_inc;

    // 軟化パラメータの更新
    if (softening_) {
      this->update_softening_parameters(state_vars);
    }
  }

  // =========================================================================
  // 4. 収束判定と警告処理
  // =========================================================================
  if (!converged) {
    // 収束しなかった場合の警告
    console_->warn(
        "MohrCoulomb::compute_stress: Return mapping did not converge "
        "after {} iterations. Final yield functions: f_t = {}, f_s = {}",
        itr_max, yield_function(0), yield_function(1));
    
    // フォールバック処理: 最後の状態をそのまま使用するが、
    // 降伏面から大きく外れている場合は追加処理
    if (yield_function(0) > 1.0 || yield_function(1) > 1.0) {
      // 大きく違反している場合は、近似的なスケーリング補正
      this->apply_fallback_correction(&current_stress, state_vars, de);
    }
  }

  // 最終的な不変量計算
  this->compute_stress_invariants(current_stress, state_vars);

  return current_stress;
}
//! 単一面リターンマッピング
//! 戻り値: (応力補正量, 塑性偏差ひずみ増分, 成功フラグ)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, double, bool>
mpm::MohrCoulomb<Tdim>::compute_single_surface_return(
    mpm::mohrcoulomb::FailureState yield_type,
    const Eigen::Matrix<double, 2, 1>& yield_function,
    const Vector6d& current_stress,
    const Matrix6x6& de,
    mpm::dense_map* state_vars) {
  
  const double Min_Denominator = 1E-15;
  
  // 降伏関数値の取得
  double f_current = (yield_type == mpm::mohrcoulomb::FailureState::Tensile) 
                     ? yield_function(0) : yield_function(1);
  
  // 勾配と硬化係数の計算
  Vector6d df_dsigma = Vector6d::Zero();
  Vector6d dp_dsigma = Vector6d::Zero();
  double dp_dq = 0.0;
  double softening_modulus = 0.0;

  this->compute_df_dp(yield_type, state_vars, current_stress,
                      &df_dsigma, &dp_dsigma, &dp_dq, &softening_modulus);

  // 分母の計算と検証
  double denominator = (df_dsigma.transpose() * de).dot(dp_dsigma) + softening_modulus;
  
  // =========================================================================
  // 分母のゼロ/負チェック（修正ポイント1）
  // =========================================================================
  bool success = true;
  if (denominator < Min_Denominator) {
    if (denominator < -Min_Denominator) {
      // 負の分母: 材料の局所的不安定性（軟化が強すぎる）
      console_->warn(
          "MohrCoulomb: Negative denominator detected ({:.6e}). "
          "This indicates material instability (strong softening). "
          "Using regularization.",
          denominator);
      // 正則化: 小さな正の値を使用
      denominator = Min_Denominator;
      success = false;  // 完全な成功ではないことを示す
    } else {
      // ゼロに近い分母: 数値的特異性
      console_->debug(
          "MohrCoulomb: Near-zero denominator detected ({:.6e}). "
          "Applying regularization.",
          denominator);
      denominator = Min_Denominator;
    }
  }

  // 塑性乗数の計算
  double dlambda = f_current / denominator;
  
  // 塑性乗数の妥当性チェック
  if (dlambda < 0.0) {
    // 負の塑性乗数は物理的に不正
    console_->warn(
        "MohrCoulomb: Negative plastic multiplier ({:.6e}). Setting to zero.",
        dlambda);
    dlambda = 0.0;
    success = false;
  }
  
  // 過大な塑性乗数のクランプ（数値安定性のため）
  const double max_dlambda = 1.0;  // 問題に応じて調整
  if (dlambda > max_dlambda) {
    console_->debug(
        "MohrCoulomb: Large plastic multiplier ({:.6e}). Clamping to {:.6e}.",
        dlambda, max_dlambda);
    dlambda = max_dlambda;
  }

  // 応力補正量と塑性ひずみ増分の計算
  Vector6d stress_correction = -dlambda * de * dp_dsigma;
  double dpdstrain = dlambda * dp_dq;

  return {stress_correction, dpdstrain, success};
}

//! コーナーリターンマッピング（マルチサーフェス法）
//! Tension面とShear面の両方が違反している場合のKoiter's rule適用
//! 戻り値: (応力補正量, 塑性偏差ひずみ増分, 成功フラグ)
template <unsigned Tdim>
std::tuple<Eigen::Matrix<double, 6, 1>, double, bool>
mpm::MohrCoulomb<Tdim>::compute_corner_return(
    const Vector6d& current_stress,
    const Matrix6x6& de,
    mpm::dense_map* state_vars) {
  
  const double Min_Denominator = 1E-15;
  
  // 両方の降伏面に対する勾配を計算
  Vector6d df_dsigma_t = Vector6d::Zero();  // Tension
  Vector6d dp_dsigma_t = Vector6d::Zero();
  double dp_dq_t = 0.0;
  double softening_t = 0.0;
  
  Vector6d df_dsigma_s = Vector6d::Zero();  // Shear
  Vector6d dp_dsigma_s = Vector6d::Zero();
  double dp_dq_s = 0.0;
  double softening_s = 0.0;

  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Tensile, state_vars, 
                      current_stress, &df_dsigma_t, &dp_dsigma_t, &dp_dq_t, &softening_t);
  this->compute_df_dp(mpm::mohrcoulomb::FailureState::Shear, state_vars,
                      current_stress, &df_dsigma_s, &dp_dsigma_s, &dp_dq_s, &softening_s);

  // 降伏関数値を再計算
  Eigen::Matrix<double, 2, 1> yield_function;
  this->compute_yield_state(&yield_function, (*state_vars));
  const double f_t = yield_function(0);  // Tension
  const double f_s = yield_function(1);  // Shear

  // =========================================================================
  // 2x2連立方程式を解いて2つの塑性乗数を求める
  // [A11 A12] [λ_t]   [f_t]
  // [A21 A22] [λ_s] = [f_s]
  // 
  // A_ij = n_i^T * De * m_j + H_i * δ_ij
  // =========================================================================
  
  Eigen::Matrix<double, 2, 2> A;
  Eigen::Vector2d b;
  
  // 係数行列の構築
  Vector6d de_dp_t = de * dp_dsigma_t;
  Vector6d de_dp_s = de * dp_dsigma_s;
  
  A(0, 0) = df_dsigma_t.dot(de_dp_t) + softening_t;
  A(0, 1) = df_dsigma_t.dot(de_dp_s);
  A(1, 0) = df_dsigma_s.dot(de_dp_t);
  A(1, 1) = df_dsigma_s.dot(de_dp_s) + softening_s;
  
  b(0) = f_t;
  b(1) = f_s;

  // 行列式のチェック
  double det_A = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
  
  if (std::abs(det_A) < Min_Denominator) {
    // 特異または準特異行列: コーナーリターン失敗
    console_->debug(
        "MohrCoulomb: Corner return matrix is singular (det = {:.6e}). "
        "Falling back to single surface return.",
        det_A);
    return {Vector6d::Zero(), 0.0, false};
  }

  // 逆行列を使って解く（2x2なので直接計算）
  Eigen::Matrix<double, 2, 2> A_inv;
  A_inv(0, 0) =  A(1, 1) / det_A;
  A_inv(0, 1) = -A(0, 1) / det_A;
  A_inv(1, 0) = -A(1, 0) / det_A;
  A_inv(1, 1) =  A(0, 0) / det_A;
  
  Eigen::Vector2d lambdas = A_inv * b;
  double lambda_t = lambdas(0);
  double lambda_s = lambdas(1);

  // =========================================================================
  // 塑性乗数の妥当性チェック（修正ポイント1の一部）
  // =========================================================================
  bool success = true;
  
  // 負の塑性乗数のチェック
  if (lambda_t < 0.0 || lambda_s < 0.0) {
    if (lambda_t < 0.0 && lambda_s < 0.0) {
      // 両方が負: コーナーリターンは不適切
      console_->debug(
          "MohrCoulomb: Both plastic multipliers negative (λ_t={:.6e}, λ_s={:.6e}). "
          "Corner return failed.",
          lambda_t, lambda_s);
      return {Vector6d::Zero(), 0.0, false};
    } else if (lambda_t < 0.0) {
      // Tension側のみ負: Shear面への単一面リターンが適切
      console_->debug(
          "MohrCoulomb: Tension plastic multiplier negative. "
          "Should use single shear surface return.");
      return {Vector6d::Zero(), 0.0, false};
    } else {
      // Shear側のみ負: Tension面への単一面リターンが適切
      console_->debug(
          "MohrCoulomb: Shear plastic multiplier negative. "
          "Should use single tension surface return.");
      return {Vector6d::Zero(), 0.0, false};
    }
  }

  // 過大な塑性乗数のクランプ
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
  // 応力補正量の計算（Koiter's rule）
  // Δσ = -De * (λ_t * m_t + λ_s * m_s)
  // =========================================================================
  Vector6d stress_correction = -de * (lambda_t * dp_dsigma_t + lambda_s * dp_dsigma_s);
  
  // 塑性偏差ひずみ増分（両方の寄与を合算）
  double dpdstrain = lambda_t * dp_dq_t + lambda_s * dp_dq_s;

  return {stress_correction, dpdstrain, success};
}

//! 収束失敗時のフォールバック補正（修正ポイント3）
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::apply_fallback_correction(
    Vector6d* current_stress,
    mpm::dense_map* state_vars,
    const Matrix6x6& de) {
  
  // 降伏関数を再評価
  this->compute_stress_invariants(*current_stress, state_vars);
  Eigen::Matrix<double, 2, 1> yield_function;
  auto yield_type = this->compute_yield_state(&yield_function, (*state_vars));

  // 最も違反している面を特定
  double max_violation = std::max(yield_function(0), yield_function(1));
  if (max_violation <= 0.0) return;  // 違反なし

  // 違反している場合、応力をスケーリングして降伏面上に近似的に戻す
  // これは厳密なリターンではないが、発散を防ぐための緊急措置
  
  console_->warn(
      "MohrCoulomb: Applying fallback stress scaling. "
      "Max yield violation = {:.6e}",
      max_violation);

  // 簡易的なスケーリング: 偏差応力を縮小
  const double p = mpm::materials::p(*current_stress);
  Vector6d dev_stress = *current_stress;
  dev_stress(0) -= p;
  dev_stress(1) -= p;
  dev_stress(2) -= p;

  // スケーリング係数（経験的な値）
  double scale_factor = 1.0 / (1.0 + max_violation);
  if (scale_factor < 0.1) scale_factor = 0.1;  // 過度な縮小を防止

  dev_stress *= scale_factor;
  
  (*current_stress)(0) = dev_stress(0) + p;
  (*current_stress)(1) = dev_stress(1) + p;
  (*current_stress)(2) = dev_stress(2) + p;
  (*current_stress)(3) = dev_stress(3);
  (*current_stress)(4) = dev_stress(4);
  (*current_stress)(5) = dev_stress(5);

  // Tensionカットオフの処理
  if (yield_function(0) > yield_function(1)) {
    // 引張が支配的: 平均応力も調整
    const double tension_cutoff = (*state_vars).at("tension_cutoff");
    double new_p = mpm::materials::p(*current_stress);
    if (new_p > tension_cutoff * 0.9) {  // 90%に制限
      double dp = new_p - tension_cutoff * 0.9;
      (*current_stress)(0) -= dp;
      (*current_stress)(1) -= dp;
      (*current_stress)(2) -= dp;
    }
  }
}
// //! Compute stress
// template <unsigned Tdim>
// Eigen::Matrix<double, 6, 1> mpm::MohrCoulomb<Tdim>::compute_stress(
//     const Vector6d& stress, const Vector6d& dstrain,
//     const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {

//   // Density and packing parameters
//   const double current_packing_density = ptr->mass_density();
//   const double critical_density = minimum_packing_fraction_ * grain_density_;

//   // Get previous time step state variable
//   const auto prev_state_vars = (*state_vars);
//   const double pdstrain = (*state_vars).at("pdstrain");
//   // Update MC parameters using a linear softening rule
//   if (softening_ && pdstrain > pdstrain_peak_) {
//     if (pdstrain < pdstrain_residual_) {
//       (*state_vars).at("phi") =
//           phi_residual_ +
//           ((phi_peak_ - phi_residual_) * (pdstrain - pdstrain_residual_) /
//            (pdstrain_peak_ - pdstrain_residual_));
//       (*state_vars).at("psi") =
//           psi_residual_ +
//           ((psi_peak_ - psi_residual_) * (pdstrain - pdstrain_residual_) /
//            (pdstrain_peak_ - pdstrain_residual_));
//       (*state_vars).at("cohesion") =
//           cohesion_residual_ + ((cohesion_peak_ - cohesion_residual_) *
//                                 (pdstrain - pdstrain_residual_) /
//                                 (pdstrain_peak_ - pdstrain_residual_));
//     } else {
//       (*state_vars).at("phi") = phi_residual_;
//       (*state_vars).at("psi") = psi_residual_;
//       (*state_vars).at("cohesion") = cohesion_residual_;
//     }
//     // Modify tension cutoff acoording to softening law
//     const double apex =
//         (*state_vars).at("cohesion") / std::tan((*state_vars).at("phi"));
//     if ((*state_vars).at("tension_cutoff") > apex)
//       (*state_vars).at("tension_cutoff") = check_low(apex);
//   }
//   //-------------------------------------------------------------------------
//   // Elastic-predictor stage: compute the trial stress
//   (*state_vars).at("yield_state") = 0;
//   Matrix6x6 de = this->compute_elastic_tensor(state_vars);
//   Vector6d trial_stress =
//       this->compute_trial_stress(stress, dstrain, de, ptr, state_vars);
//   // Separated state: current packing density is less than critical density
//   if (current_packing_density <= critical_density) {
//     (*state_vars).at("pdstrain") +=
//         mpm::materials::q(trial_stress) / 3.0 / shear_modulus_;
//     (*state_vars).at("yield_state") = 3;
//     return Vector6d::Zero();
//   }
//   // Compute stress invariants based on trial stress
//   this->compute_stress_invariants(trial_stress, state_vars);
//   // Compute yield function based on the trial stress
//   Eigen::Matrix<double, 2, 1> yield_function_trial;
//   auto yield_type_trial =
//       this->compute_yield_state(&yield_function_trial, (*state_vars));
//   // Return the updated stress in elastic state
//   if (yield_type_trial == mpm::mohrcoulomb::FailureState::Elastic) {
//     (*state_vars).at("yield_state") = 0;
//     return trial_stress;
//   }
//   //-------------------------------------------------------------------------
//   // Plastic-corrector stage: correct the stress back to the yield surface
//   // Define tolerance of yield function
//   const double Tolerance = 1E-1;
//   // Compute plastic multiplier based on trial stress (Lambda trial)
//   double softening_trial = 0.;
//   double dp_dq_trial = 0.;
//   Vector6d df_dsigma_trial = Vector6d::Zero();
//   Vector6d dp_dsigma_trial = Vector6d::Zero();
//   this->compute_df_dp(yield_type_trial, state_vars, trial_stress,
//                       &df_dsigma_trial, &dp_dsigma_trial, &dp_dq_trial,
//                       &softening_trial);
//   double yield_trial = 0.;
//   if (yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile) {
//     (*state_vars).at("yield_state") = 2;
//     yield_trial = yield_function_trial(0);
//   }
//   if (yield_type_trial == mpm::mohrcoulomb::FailureState::Shear) {
//     (*state_vars).at("yield_state") = 1;
//     yield_trial = yield_function_trial(1);
//   }
//   de = this->compute_elastic_tensor(state_vars);
//   double lambda_trial =
//       yield_trial /
//       ((df_dsigma_trial.transpose() * de).dot(dp_dsigma_trial.transpose()) +
//        softening_trial);
//   // Compute stress invariants based on stress input
//   this->compute_stress_invariants(stress, state_vars);
//   // Compute yield function based on stress input
//   Eigen::Matrix<double, 2, 1> yield_function;
//   auto yield_type = this->compute_yield_state(&yield_function, (*state_vars));
//   // Initialise value of yield function based on stress
//   double yield{std::numeric_limits<double>::max()};
//   if (yield_type == mpm::mohrcoulomb::FailureState::Tensile)
//     yield = yield_function(0);
//   if (yield_type == mpm::mohrcoulomb::FailureState::Shear)
//     yield = yield_function(1);
//   // Compute plastic multiplier based on stress input (Lambda)
//   double softening = 0.;
//   double dp_dq = 0.;
//   Vector6d df_dsigma = Vector6d::Zero();
//   Vector6d dp_dsigma = Vector6d::Zero();
//   this->compute_df_dp(yield_type, state_vars, stress, &df_dsigma, &dp_dsigma,
//                       &dp_dq, &softening);
//   const double lambda =
//       ((df_dsigma.transpose() * de).dot(dstrain)) /
//       (((df_dsigma.transpose() * de).dot(dp_dsigma)) + softening);
//   // Initialise updated stress
//   Vector6d updated_stress = trial_stress;
//   // Initialise incremental of plastic deviatoric strain
//   double dpdstrain = 0.;
//   // Correction stress based on stress
//   if (fabs(yield) < Tolerance) {
//     // Compute updated stress
//     updated_stress += -(lambda * de * dp_dsigma);
//     // Compute incremental of plastic deviatoric strain
//     dpdstrain = lambda * dp_dq;
//   } else {
//     // Compute updated stress
//     updated_stress += -(lambda_trial * de * dp_dsigma_trial);
//     // Compute incremental of plastic deviatoric strain
//     dpdstrain = lambda_trial * dp_dq_trial;
//   }

//   // Define the maximum iteration step
//   const int itr_max = 100;
//   // Correct the stress again
//   for (unsigned itr = 0; itr < itr_max; ++itr) {
//     // Check the update stress
//     // Compute stress invariants based on updated stress
//     this->compute_stress_invariants(updated_stress, state_vars);
//     // Compute yield function based on updated stress
//     yield_type_trial =
//         this->compute_yield_state(&yield_function_trial, (*state_vars));
//     // Check yield function
//     if (yield_function_trial(0) < Tolerance &&
//         yield_function_trial(1) < Tolerance) {
//       break;
//     }
//     // Compute plastic multiplier based on updated stress
//     this->compute_df_dp(yield_type_trial, state_vars, updated_stress,
//                         &df_dsigma_trial, &dp_dsigma_trial, &dp_dq_trial,
//                         &softening_trial);
//     if (yield_type_trial == mpm::mohrcoulomb::FailureState::Tensile)
//       yield_trial = yield_function_trial(0);
//     if (yield_type_trial == mpm::mohrcoulomb::FailureState::Shear)
//       yield_trial = yield_function_trial(1);
//     // Compute plastic multiplier based on updated stress
//     lambda_trial =
//         yield_trial /
//         ((df_dsigma_trial.transpose() * de).dot(dp_dsigma_trial.transpose()) +
//          softening_trial);
//     // Correct stress back to the yield surface
//     updated_stress += -(lambda_trial * de * dp_dsigma_trial);
//     // Update incremental of plastic deviatoric strain
//     dpdstrain += lambda_trial * dp_dq_trial;
//   }
//   // Compute stress invariants based on updated stress
//   this->compute_stress_invariants(updated_stress, state_vars);

//   // Update plastic deviatoric strain
//   (*state_vars).at("pdstrain") += dpdstrain;

//   return updated_stress;
// }
//! 軟化パラメータの更新（既存コードの改良版）
template <unsigned Tdim>
void mpm::MohrCoulomb<Tdim>::update_softening_parameters(mpm::dense_map* state_vars) {
  const double pdstrain = (*state_vars).at("pdstrain");
  
  if (pdstrain <= pdstrain_peak_) {
    // ピーク前: ピーク値を維持
    (*state_vars).at("phi") = phi_peak_;
    (*state_vars).at("psi") = psi_peak_;
    (*state_vars).at("cohesion") = cohesion_peak_;
  } else if (pdstrain >= pdstrain_residual_) {
    // 残留状態: 残留値を維持
    (*state_vars).at("phi") = phi_residual_;
    (*state_vars).at("psi") = psi_residual_;
    (*state_vars).at("cohesion") = cohesion_residual_;
  } else {
    // 軟化領域: 線形補間
    const double ratio = (pdstrain - pdstrain_peak_) / 
                         (pdstrain_residual_ - pdstrain_peak_);
    (*state_vars).at("phi") = phi_peak_ + ratio * (phi_residual_ - phi_peak_);
    (*state_vars).at("psi") = psi_peak_ + ratio * (psi_residual_ - psi_peak_);
    (*state_vars).at("cohesion") = cohesion_peak_ + ratio * (cohesion_residual_ - cohesion_peak_);
  }
  
  // Tension cutoffの更新（Apex制限）
  const double phi = (*state_vars).at("phi");
  const double cohesion = (*state_vars).at("cohesion");
  
  // tan(phi)がゼロに近い場合の保護
  const double tan_phi = std::tan(phi);
  const double min_tan_phi = 1.0e-10;
  
  double apex;
  if (std::abs(tan_phi) < min_tan_phi) {
    // φ ≈ 0 の場合、apex は非常に大きい（事実上無限大）
    apex = std::numeric_limits<double>::max();
  } else {
    apex = cohesion / tan_phi;
  }
  
  // Tension cutoffはapex以下に制限
  if ((*state_vars).at("tension_cutoff") > apex) {
    (*state_vars).at("tension_cutoff") = apex;
  }
  
  // Tension cutoffが負にならないよう保護
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