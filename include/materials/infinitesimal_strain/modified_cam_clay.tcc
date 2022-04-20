//! Constructor with id and material properties
template <unsigned Tdim>
mpm::ModifiedCamClay<Tdim>::ModifiedCamClay(unsigned id,
                                            const Json& material_properties)
    : InfinitesimalElastoPlastic<Tdim>(id, material_properties) {
  try {
    // Elastic parameters
    // Density
    density_ = material_properties.at("density").template get<double>();
    // Initial shear modulus
    shear_modulus_ =
        material_properties.at("shear_modulus").template get<double>();
    // Poisson ratio
    poisson_ratio_ =
        material_properties.at("poisson_ratio").template get<double>();

    // Modified Cam Clay parameters
    // Reference void ratio
    e_ref_ = material_properties.at("e_ref").template get<double>();
    // Reference mean pressure
    p_ref_ = material_properties.at("p_ref").template get<double>();
    // Initial preconsolidation pressure
    pc0_ = material_properties.at("pc0").template get<double>();
    // M (or triaxial compression M (Mtc) in "Three invariants type")
    if (material_properties.contains("m")) {
      m_ = material_properties.at("m").template get<double>();
    } else if (material_properties.contains("friction_cs")) {
      const double friction_cs =
          material_properties.at("friction_cs").template get<double>() * M_PI /
          180.;
      const double sin_friction_cs = sin(friction_cs);
      m_ = (6 * sin_friction_cs) / (3 - sin_friction_cs);
    } else
      throw std::runtime_error("Critical state ratio M is undefined!");
    // Lambda
    lambda_ = material_properties.at("lambda").template get<double>();
    // Kappa
    kappa_ = material_properties.at("kappa").template get<double>();
    // Initial void ratio computed from different sources e0
    // From initial porosity
    if (material_properties.contains("porosity")) {
      const double n =
          material_properties.at("porosity").template get<double>();
      e0_ = n / (1. - n);
    }
    // From initial void ratio
    else if (material_properties.contains("void_ratio_initial")) {
      e0_ = material_properties.at("void_ratio_initial").template get<double>();
    }
    // From OCR
    else if (material_properties.contains("ocr")) {
      ocr_ = material_properties.at("ocr").template get<double>();
      e0_ = e_ref_ - lambda_ * log(pc0_ / ocr_ / p_ref_) - kappa_ * log(ocr_);
    } else
      throw std::runtime_error("Initial void ratio is undefined!");
    // Three invariants option
    if (material_properties.contains("three_invariants"))
      three_invariants_ =
          material_properties.at("three_invariants").template get<bool>();
    // Subloading option
    if (material_properties.contains("subloading"))
      subloading_ = material_properties.at("subloading").template get<bool>();
    // Bonding option
    if (material_properties.contains("bonding"))
      bonding_ = material_properties.at("bonding").template get<bool>();
    // Subloading surface ratio
    if (subloading_) {
      // Material constant controling plastic deformation
      subloading_u_ =
          material_properties.at("subloading_u").template get<double>();
    }
    // Bonding material constants
    if (bonding_) {
      // Hydrate saturation
      s_h_ = material_properties.at("s_h").template get<double>();
      // Material constants a
      mc_a_ = material_properties.at("mc_a").template get<double>();
      // Material constants b
      mc_b_ = material_properties.at("mc_b").template get<double>();
      // Material constants c
      mc_c_ = material_properties.at("mc_c").template get<double>();
      // Material constants d
      mc_d_ = material_properties.at("mc_d").template get<double>();
      // Degradation
      m_degradation_ =
          material_properties.at("m_degradation").template get<double>();
      // Increment in shear modulus
      m_shear_ = material_properties.at("m_shear").template get<double>();
    }

    // Properties
    properties_ = material_properties;
  } catch (std::exception& except) {
    console_->error("Material parameter not set: {}\n", except.what());
  }
}

//! Initialise state variables
template <unsigned Tdim>
mpm::dense_map mpm::ModifiedCamClay<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {
      // Yield state: 0: elastic, 1: yield
      {"yield_state", 0},
      // Bulk modulus
      {"bulk_modulus", 2. * shear_modulus_ * (1. + poisson_ratio_) /
                           (3. * (1. - 2. * poisson_ratio_))},
      // Shear modulus
      {"shear_modulus", shear_modulus_},
      // Stress invariants
      // Volumetric stress
      {"p", 0.},
      // Deviatoric stress
      {"q", 0.},
      // Lode's angle
      {"theta", 0.},
      // Modified Cam clay parameters
      // Preconsolidation pressure
      {"pc", pc0_},
      // void_ratio
      {"void_ratio", e0_},
      // Consistency parameter
      {"delta_phi", 0.},
      // M_theta
      {"m_theta", m_},
      // Yield function
      {"f_function", 0.},
      // Incremental plastic strain
      // Incremental plastic volumetic strain
      {"dpvstrain", 0.},
      // Incremental plastic deviatoric strain
      {"dpdstrain", 0.},
      // Plastic volumetic strain
      {"pvstrain", 0.},
      // Plastic deviatoric strain
      {"pdstrain", 0.},
      // Bonding parameters
      // Chi
      {"chi", 1.},
      // Pcd
      {"pcd", 0.},
      // Pcc
      {"pcc", 0.},
      // Subloading surface ratio
      {"subloading_r", 1.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::ModifiedCamClay<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {
      "yield_state", "bulk_modulus", "shear_modulus", "p",           "q",
      "theta",       "pc",           "void_ratio",    "delta_phi",   "m_theta",
      "f_function",  "dpvstrain",    "dpdstrain",     "pvstrain",    "pdstrain",
      "chi",         "pcd",          "pcc",           "subloading_r"};
  return state_vars;
}

//! Compute elastic tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6> mpm::ModifiedCamClay<Tdim>::compute_elastic_tensor(
    const Vector6d& stress, mpm::dense_map* state_vars) {
  // Compute trial stress invariants
  this->compute_stress_invariants(stress, state_vars);
  // Compute elastic modulus based on stress status
  if ((*state_vars).at("p") > std::numeric_limits<double>::epsilon()) {
    // Bulk modulus
    (*state_vars).at("bulk_modulus") =
        (1 + (*state_vars).at("void_ratio")) / kappa_ * (*state_vars).at("p");
    // Shear modulus
    (*state_vars).at("shear_modulus") = 3 * (*state_vars).at("bulk_modulus") *
                                        (1 - 2 * poisson_ratio_) /
                                        (2 * (1 + poisson_ratio_));
  }
  // Compute bonding part
  if (bonding_) {
    // Bonded shear modulus
    (*state_vars).at("shear_modulus") +=
        m_shear_ * (*state_vars).at("chi") * s_h_;
    // Bonded bulk modulus
    (*state_vars).at("bulk_modulus") = (*state_vars).at("shear_modulus") *
                                       (2 * (1 + poisson_ratio_)) /
                                       (1 - 2 * poisson_ratio_) / 3;
  }
  // Components in stiffness matrix
  const double G = (*state_vars).at("shear_modulus");
  const double a1 = (*state_vars).at("bulk_modulus") + (4.0 / 3.0) * G;
  const double a2 = (*state_vars).at("bulk_modulus") - (2.0 / 3.0) * G;
  // Compute elastic stiffness matrix
  // clang-format off
  Matrix6x6 de = Matrix6x6::Zero();
  de(0,0)=a1;    de(0,1)=a2;    de(0,2)=a2;
  de(1,0)=a2;    de(1,1)=a1;    de(1,2)=a2;
  de(2,0)=a2;    de(2,1)=a2;    de(2,2)=a1;
  de(3,3)=G;     de(4,4)=G;     de(5,5)=G;
  // clang-format on
  return de;
}

//! Compute plastic tensor (used for drained test)
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::ModifiedCamClay<Tdim>::compute_elasto_plastic_tensor(
        const Vector6d& stress, const Vector6d& dstrain,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars,
        bool hardening) {

  mpm::modifiedcamclay::FailureState yield_type =
      yield_type_.at(int((*state_vars).at("yield_state")));
  // Return the updated stress in elastic state
  const Matrix6x6 de = this->compute_elastic_tensor(stress, state_vars);
  if (yield_type == mpm::modifiedcamclay::FailureState::Elastic) {
    return de;
  }

  // Current stress
  const double p = (*state_vars).at("p");
  const double q = (*state_vars).at("q");
  // Update Mtheta
  if (three_invariants_)
    (*state_vars).at("m_theta") =
        m_ - std::pow(m_, 2) / (3 + m_) * cos(1.5 * (*state_vars).at("theta"));
  // Preconsolidation pressure
  const double pc = (*state_vars).at("pc");
  // Bonding parameters
  const double pcc = (*state_vars).at("pcc");
  const double pcd = (*state_vars).at("pcd");
  // Subloading ratio
  const double subloading_r = (*state_vars).at("subloading_r");
  // Compute dF / dp
  double df_dp = 2 * p - pc - pcd;
  // Compute dF / dq
  const double df_dq = 2 * q / std::pow((*state_vars).at("m_theta"), 2);
  // Compute dF / dpc
  double df_dpc = -p - pcc;
  // Compute dF / dpcd
  double df_dpcd = -p - pcc;
  // Compute dF / dpcc
  double df_dpcc = -2 * pcc - pc - pcd;
  // Subloading parameters
  if (subloading_) {
    df_dp = 2 * p + pcc - subloading_r * (pc + pcc + pcd);
    df_dpc *= subloading_r;
    df_dpcd *= subloading_r;
    df_dpcc = p - subloading_r * (p + pc + pcd + 2 * pcc);
  }
  // Upsilon
  const double upsilon =
      (1 + (*state_vars).at("void_ratio")) / (lambda_ - kappa_);
  // Coefficients in plastic stiffness matrix
  const double a1 = std::pow(((*state_vars).at("bulk_modulus") * df_dp), 2);
  const double a2 = -std::sqrt(6) * (*state_vars).at("bulk_modulus") * df_dp *
                    (*state_vars).at("shear_modulus") * df_dq;
  const double a3 =
      6 * std::pow(((*state_vars).at("shear_modulus") * df_dq), 2);
  // Numerator
  const double num = (*state_vars).at("bulk_modulus") * (df_dp * df_dp) +
                     3 * (*state_vars).at("shear_modulus") * (df_dq * df_dq);

  // Hardening parameter
  double hardening_par = upsilon * pc * df_dp * df_dpc;
  // Compute bonding
  if (bonding_) {
    // Compute pcd hardening parameter
    const double hardening_pcd =
        df_dpcd * (-m_degradation_ * mc_b_ * pcd) * df_dq;
    // Compute pcc hardening parameter
    const double hardening_pcc =
        df_dpcc * (-m_degradation_ * mc_d_ * pcc) * df_dq;
    // Update hardening parameter
    hardening_par += (hardening_pcd + hardening_pcc);
  }
  // Compute subloading
  if (subloading_) {
    // Compute dF / dR
    const double df_dr = -(p + pcc) * (pc + pcd + pcc);
    // Compute subloading hardening parameter
    const double hardening_subloading =
        -df_dr * subloading_u_ * (1 + (pcd + pcc) / pc) * log(subloading_r) *
        std::sqrt(std::pow((*state_vars).at("dpvstrain"), 2) +
                  std::pow((*state_vars).at("dpdstrain"), 2));
    // Update hardening parameter
    hardening_par += hardening_subloading;
  }
  // Compute the deviatoric stress
  auto dev_stress = stress;
  for (unsigned i = 0; i < 3; ++i) dev_stress(i) += (*state_vars).at("p");
  // Initialise matrix
  Eigen::Matrix<double, 6, 6> n_l = Matrix6x6::Zero();
  Eigen::Matrix<double, 6, 6> l_n = Matrix6x6::Zero();
  Eigen::Matrix<double, 6, 6> l_l = Matrix6x6::Zero();
  Eigen::Matrix<double, 6, 6> n_n = Matrix6x6::Zero();
  // Norm of deviatoric stress tensor
  const double xi = q / std::sqrt(1.5);
  // lxn
  if (xi > std::numeric_limits<double>::epsilon()) {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 6; ++j) l_n(i, j) = dev_stress(j) / xi;
    }
    // nxl
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 3; ++j) n_l(i, j) = dev_stress(i) / xi;
    }
    // nxn
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        n_n(i, j) = dev_stress(i) * dev_stress(j) / (xi * xi);
      }
    }
    // lxl
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) l_l(i, j) = 1;
    }
  }
  // Compute plastic tensor
  if (!hardening) hardening_par = 0.;
  Matrix6x6 dep =
      (a1 * l_l + a2 * (n_l + l_n) + a3 * (n_n)) / (num - hardening_par);

  return dep;
}

//! Compute stress invariants
template <unsigned Tdim>
bool mpm::ModifiedCamClay<Tdim>::compute_stress_invariants(
    const Vector6d& stress, mpm::dense_map* state_vars) {
  // Compute volumetic stress
  (*state_vars).at("p") = -mpm::materials::p(stress);
  // Compute deviatoric q
  (*state_vars).at("q") = mpm::materials::q(stress);
  // Compute theta (Lode angle)
  if (three_invariants_)
    (*state_vars).at("theta") = mpm::materials::lode_angle(stress);

  return true;
}

//! Compute deviatoric stress tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1>
    mpm::ModifiedCamClay<Tdim>::compute_deviatoric_stress_tensor(
        const Vector6d& stress, mpm::dense_map* state_vars) {
  // Initialise deviatoric stress tensor
  Vector6d n = Vector6d::Zero();
  // Mean stress
  const double p = (*state_vars).at("p");
  // Deviatoric stress
  const double q = (*state_vars).at("q");
  // Compute the deviatoric stress
  Vector6d dev_stress = stress;
  for (unsigned i = 0; i < 3; ++i) dev_stress(i) += p;
  // Compute vector n
  if (q > std::numeric_limits<double>::epsilon()) n = dev_stress / q;

  return n;
}

//! Compute yield function and yield state
template <unsigned Tdim>
typename mpm::modifiedcamclay::FailureState
    mpm::ModifiedCamClay<Tdim>::compute_yield_state(
        mpm::dense_map* state_vars) {
  // Get stress invariants
  const double p = (*state_vars).at("p");
  const double q = (*state_vars).at("q");
  const double m_theta = (*state_vars).at("m_theta");
  // Plastic volumetic strain
  const double pc = (*state_vars).at("pc");
  // Get bonding parameters
  const double pcd = (*state_vars).at("pcd");
  const double pcc = (*state_vars).at("pcc");
  // Subloading surface ratio
  const double subloading_r = (*state_vars).at("subloading_r");
  // Initialise yield status (0: elastic, 1: yield)
  auto yield_type = mpm::modifiedcamclay::FailureState::Elastic;
  // Compute yield functions
  (*state_vars).at("f_function") =
      std::pow(q / m_theta, 2) +
      (p + pcc) * (p - subloading_r * (pc + pcd + pcc));
  // Yielding
  if ((*state_vars).at("f_function") > std::numeric_limits<double>::epsilon())
    yield_type = mpm::modifiedcamclay::FailureState::Yield;

  return yield_type;
}

//! Compute bonding parameters
template <unsigned Tdim>
void mpm::ModifiedCamClay<Tdim>::compute_bonding_parameters(
    const double chi, mpm::dense_map* state_vars) {
  // Compute chi
  (*state_vars).at("chi") =
      chi - m_degradation_ * chi * (*state_vars).at("dpdstrain");
  if ((*state_vars).at("chi") < 0.) (*state_vars).at("chi") = 0.;
  if ((*state_vars).at("chi") > 1.) (*state_vars).at("chi") = 1.;
  // Compute pcd
  (*state_vars).at("pcd") =
      mc_a_ * std::pow((*state_vars).at("chi") * s_h_, mc_b_);
  // Compute pcc
  (*state_vars).at("pcc") =
      mc_c_ * std::pow((*state_vars).at("chi") * s_h_, mc_d_);
}

//! Compute subloading parameters
template <unsigned Tdim>
void mpm::ModifiedCamClay<Tdim>::compute_subloading_parameters(
    const double subloading_r, mpm::dense_map* state_vars) {
  // Mean pressure
  const double p = (*state_vars).at("p");
  // Preconsolidation pressure
  const double pc = (*state_vars).at("pc");
  // Get bonding parameters
  const double pcd = (*state_vars).at("pcd");
  const double pcc = (*state_vars).at("pcc");
  // Plastic strain
  const double dpvstrain = (*state_vars).at("dpvstrain");
  const double dpdstrain = (*state_vars).at("dpdstrain");
  // Initialise subloading surface ratio
  if (std::abs((*state_vars).at("subloading_r") - 1.0) <
      std::numeric_limits<double>::epsilon())
    (*state_vars).at("subloading_r") = p / (pc + pcd + pcc);
  else
    // Update subloading surface ratio
    (*state_vars).at("subloading_r") =
        subloading_r -
        subloading_u_ * (1 + (pcd + pcc) / pc) * log(subloading_r) *
            std::sqrt(dpvstrain * dpvstrain + dpdstrain * dpdstrain);
  // Threshhold
  if ((*state_vars).at("subloading_r") < std::numeric_limits<double>::epsilon())
    (*state_vars).at("subloading_r") = 1.E-5;
  if ((*state_vars).at("subloading_r") > 1.)
    (*state_vars).at("subloading_r") = 1.;
}

//! Compute dF/dmul
template <unsigned Tdim>
void mpm::ModifiedCamClay<Tdim>::compute_df_dmul(
    const mpm::dense_map* state_vars, double* df_dmul) {
  // Stress invariants
  const double p = (*state_vars).at("p");
  const double q = (*state_vars).at("q");
  const double m_theta = (*state_vars).at("m_theta");
  // Preconsolidation pressure
  const double pc = (*state_vars).at("pc");
  // Get bonding parameters
  const double pcd = (*state_vars).at("pcd");
  const double pcc = (*state_vars).at("pcc");
  // Get elastic modulus
  const double e_b = (*state_vars).at("bulk_modulus");
  const double e_s = (*state_vars).at("shear_modulus");
  // Get consistency parameter
  const double mul = (*state_vars).at("delta_phi");
  // Compute dF / dp
  double df_dp = 2 * p - pc - pcd;
  // Compute dF / dq
  double df_dq = 2 * q / std::pow(m_theta, 2);
  // Compute dF / dpc
  double df_dpc = -(p + pcc);
  // Upsilon
  double upsilon = (1 + (*state_vars).at("void_ratio")) / (lambda_ - kappa_);
  // A_den
  double a_den = 1 + (2 * e_b + upsilon * (pc + pcd)) * mul;
  // Compute dp / dmul
  double dp_dmul = -e_b * (2 * p - pc - pcd) / a_den;
  // Compute dpc / dmul
  double dpc_dmul = upsilon * (pc + pcd) * (2 * p - pc - pcd) / a_den;
  // Compute dq / dmul
  double dq_dmul = -q / (mul + std::pow(m_theta, 2) / (6 * e_s));
  // Compute dF / dmul
  if (!bonding_)
    (*df_dmul) = (df_dp * dp_dmul) + (df_dq * dq_dmul) + (df_dpc * dpc_dmul);
  // Compute bonding parameters
  else {
    // Compute dF / dpcd
    double df_dpcd = -p - pcc;
    // Compute dF / dpcc
    double df_dpcc = -2 * pcc - pc - pcd;
    // Compute dpcd / dmul
    double dpcd_dmul = 0;
    if (pcd > std::numeric_limits<double>::epsilon())
      dpcd_dmul = -std::sqrt(6.) * mc_b_ * m_degradation_ * q /
                  (6 * e_s * mul + std::pow(m_theta, 2)) * pcd;
    // Compute dpcc / dmul
    double dpcc_dmul = 0;
    if (pcc > std::numeric_limits<double>::epsilon())
      dpcc_dmul = -std::sqrt(6.) * mc_d_ * m_degradation_ * q /
                  (6 * e_s * mul + std::pow(m_theta, 2)) * pcc;
    // Compute dpc /dmul
    dpc_dmul -= dpcd_dmul;
    // Compute dF / dmul
    (*df_dmul) = (df_dp * dp_dmul) + (df_dq * dq_dmul) + (df_dpc * dpc_dmul) +
                 (df_dpcd * dpcd_dmul) + (df_dpcc * dpcc_dmul);
  }
}

//! Compute dg/dpc
template <unsigned Tdim>
void mpm::ModifiedCamClay<Tdim>::compute_dg_dpc(
    const mpm::dense_map* state_vars, const double pc_n, const double p_trial,
    double* g_function, double* dg_dpc) {
  // Upsilon
  const double upsilon =
      (1 + (*state_vars).at("void_ratio")) / (lambda_ - kappa_);
  // Exponential index
  double e_index =
      upsilon * (*state_vars).at("delta_phi") *
      (2 * p_trial - (*state_vars).at("pc") - (*state_vars).at("pcd")) /
      (1 +
       2 * (*state_vars).at("delta_phi") * (*state_vars).at("bulk_modulus"));
  // Compute consistency parameter function
  (*g_function) = pc_n * exp(e_index) - (*state_vars).at("pc");
  // Compute dG / dpc
  (*dg_dpc) = pc_n * exp(e_index) *
                  (-upsilon * (*state_vars).at("delta_phi") /
                   (1 + 2 * (*state_vars).at("delta_phi") *
                            (*state_vars).at("bulk_modulus"))) -
              1;
}

//! Compute dF/dSigma
template <unsigned Tdim>
void mpm::ModifiedCamClay<Tdim>::compute_df_dsigma(
    const mpm::dense_map* state_vars, const Vector6d& stress,
    Vector6d* df_dsigma) {
  // Get stress invariants
  const double p = (*state_vars).at("p");
  const double q = (*state_vars).at("q");
  const double theta = (*state_vars).at("theta");
  // Get MCC parameters
  const double m_theta = (*state_vars).at("m_theta");
  const double pc = (*state_vars).at("pc");
  const double pcc = (*state_vars).at("pcc");
  const double pcd = (*state_vars).at("pcd");
  // Compute the deviatoric stress
  Vector6d dev_stress = stress;
  for (unsigned i = 0; i < 3; ++i) dev_stress(i) += p;
  // Compute dF / dp
  double df_dp = 2 * p - pc - pcd;
  // Compute dF / dq
  double df_dq = 2 * q / std::pow(m_theta, 2);
  // Compute dF / dpc
  double df_dpc = -(p + pcc);
  // Compute dp / dSigma
  Vector6d dp_dsigma = -mpm::materials::dp_dsigma();
  // Compute dq / dSigma
  Vector6d dq_dsigma = mpm::materials::dq_dsigma(stress);
  // Compute dF/dSigma
  (*df_dsigma) = (df_dp * dp_dsigma) + (df_dq * dq_dsigma);
  // Compute dTheta / dSigma
  if (three_invariants_) {
    // Compute dF / dM
    double df_dm = -2 * q * q / std::pow(m_theta, 3);
    // Compute dM / dtheta
    double dm_dtheta = m_ * m_ / (3 + m_) * sin(1.5 * theta) * 1.5;
    // Compute dtheta / dsigma
    Vector6d dtheta_dsigma = mpm::materials::dtheta_dsigma(
        stress, std::numeric_limits<double>::epsilon());
    (*df_dsigma) += (df_dm * dm_dtheta * dtheta_dsigma);
  }
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::ModifiedCamClay<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {
  // Tolerance for yield function
  const double Ftolerance = 1.E-5;
  // Tolerance for preconsolidation function
  const double Gtolerance = 1.E-5;
  // Maximum iteration step number
  const int itrstep = 100;
  // Maximum subiteration step number
  const int substep = 100;
  // Set elastic tensor
  const Matrix6x6 de = this->compute_elastic_tensor(stress, state_vars);
  //-------------------------------------------------------------------------
  // Elastic step
  // Compute trial stress
  const Vector6d trial_stress = stress + (de * dstrain);
  // Initialise vector n
  Vector6d n_trial = Vector6d::Zero();
  // Compute trial stress invariants
  this->compute_stress_invariants(trial_stress, state_vars);
  // Compute deviatoric stress tensor
  n_trial = this->compute_deviatoric_stress_tensor(trial_stress, state_vars);
  // Bonding parameter of last step
  const double chi_n = (*state_vars).at("chi");
  // Compute bonding parameters
  if (bonding_) this->compute_bonding_parameters(chi_n, state_vars);
  // Subloading parameter of last step
  const double subloading_r = (*state_vars).at("subloading_r");
  // Compute subloading parameters
  if (subloading_)
    this->compute_subloading_parameters(subloading_r, state_vars);
  // Update Mtheta
  if (three_invariants_)
    (*state_vars).at("m_theta") =
        m_ - std::pow(m_, 2) / (3 + m_) * cos(1.5 * (*state_vars).at("theta"));
  // Check yield status
  auto yield_type = this->compute_yield_state(state_vars);
  // Return the updated stress in elastic state
  if (yield_type == mpm::modifiedcamclay::FailureState::Elastic) {
    (*state_vars).at("yield_state") = 0.;
    return trial_stress;
  } else
    (*state_vars).at("yield_state") = 1.;
  //-------------------------------------------------------------------------
  // Plastic step
  // Counters for interations
  int counter_f = 0;
  int counter_g = 0;
  // Initialise consistency parameter
  (*state_vars).at("delta_phi") = 0.;
  // Volumetric trial stress
  const double p_trial = (*state_vars).at("p");
  // Deviatoric trial stress
  const double q_trial = (*state_vars).at("q");
  // M_theta of trial stress
  const double m_theta_trial = (*state_vars).at("m_theta");
  // Preconsolidation pressure of last step
  const double pc_n = (*state_vars).at("pc");
  // Initialise dF / dmul
  double df_dmul = 0;
  // Initialise updated stress
  Vector6d updated_stress = trial_stress;
  // Iteration for consistency parameter
  while (std::fabs((*state_vars).at("f_function")) > Ftolerance &&
         counter_f < itrstep) {
    // Get back the m_theta of trial_stress
    (*state_vars).at("m_theta") = m_theta_trial;
    // Compute dF / dmul
    this->compute_df_dmul(state_vars, &df_dmul);
    // Update consistency parameter
    (*state_vars).at("delta_phi") -= ((*state_vars).at("f_function") / df_dmul);
    // Initialise G and dG / dpc
    double g_function = 0;
    double dg_dpc = 0;
    // Compute G and dG / dpc
    this->compute_dg_dpc(state_vars, pc_n, p_trial, &g_function, &dg_dpc);
    // Subiteraction for preconsolidation pressure
    while (std::fabs(g_function) > Gtolerance && counter_g < substep) {
      // Update preconsolidation pressure
      (*state_vars).at("pc") -= g_function / dg_dpc;
      // Update G and dG / dpc
      this->compute_dg_dpc(state_vars, pc_n, p_trial, &g_function, &dg_dpc);
      // Counter subiteration step
      ++counter_g;
    }
    // Update mean pressure p
    (*state_vars).at("p") =
        (p_trial + (*state_vars).at("bulk_modulus") *
                       (*state_vars).at("delta_phi") * (*state_vars).at("pc")) /
        (1 +
         2 * (*state_vars).at("bulk_modulus") * (*state_vars).at("delta_phi"));
    // Update deviatoric stress q
    // Equation(3.10b)
    (*state_vars).at("q") =
        q_trial / (1 + 6 * (*state_vars).at("shear_modulus") *
                           (*state_vars).at("delta_phi") /
                           std::pow((*state_vars).at("m_theta"), 2));
    // Compute incremental plastic volumetic strain
    // Equation(2.8)
    (*state_vars).at("dpvstrain") =
        (*state_vars).at("delta_phi") *
        (2 * (*state_vars).at("p") - (*state_vars).at("pc") -
         (*state_vars).at("pcd"));
    // Compute plastic deviatoric strain
    (*state_vars).at("dpdstrain") = (*state_vars).at("delta_phi") *
                                    (std::sqrt(6) * (*state_vars).at("q") /
                                     std::pow((*state_vars).at("m_theta"), 2));
    // Update bonding parameters
    if (bonding_) this->compute_bonding_parameters(chi_n, state_vars);
    // Compute subloading parameters
    if (subloading_)
      this->compute_subloading_parameters(subloading_r, state_vars);
    // Compute three invariants parameters
    if (three_invariants_) {
      // Update stress
      // Type-1 Equation(3.16)
      updated_stress = (*state_vars).at("q") * n_trial;
      for (int i = 0; i < 3; ++i) updated_stress(i) -= (*state_vars).at("p");
      (*state_vars).at("theta") = mpm::materials::lode_angle(updated_stress);
      // Update Mtheta
      (*state_vars).at("m_theta") =
          m_ -
          std::pow(m_, 2) / (3 + m_) * cos(1.5 * (*state_vars).at("theta"));
    }
    // Update yield function
    yield_type = this->compute_yield_state(state_vars);
    // Counter iteration step
    ++counter_f;
  }
  // Update plastic strain
  (*state_vars).at("pvstrain") += (*state_vars).at("dpvstrain");
  (*state_vars).at("pdstrain") += (*state_vars).at("dpdstrain");
  // Update stress
  updated_stress = (*state_vars).at("q") * n_trial;
  for (int i = 0; i < 3; ++i) updated_stress(i) -= (*state_vars).at("p");
  // Update void_ratio
  (*state_vars).at("void_ratio") +=
      ((dstrain(0) + dstrain(1) + dstrain(2)) * (1 + e0_));

  return updated_stress;
}
