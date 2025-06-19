// #include "materials/infinitesimal_strain/pm4sand.h" // Include the header

#include <algorithm>  // For std::max, std::min
#include <cmath>
#include <iostream>         // For potential debug output
#include <limits>           // Required for std::numeric_limits
#include <map>              // Assuming mpm::dense_map is std::map or similar
#include <spdlog/spdlog.h>  // For logging
#include <stdexcept>  // Required for std::runtime_error, std::out_of_range
#include <string>
#include <vector>

namespace mpm {
template <unsigned Tdim>
constexpr double PM4Sand<Tdim>::c_tolerance;  // '= 1.0e-10'

template <unsigned Tdim>
constexpr double PM4Sand<Tdim>::c_root12;  // '= M_SQRT1_2'

template <unsigned Tdim>
constexpr double PM4Sand<Tdim>::c_one3;  // '= 1.0 / 3.0'

template <unsigned Tdim>
constexpr double PM4Sand<Tdim>::c_two3;  // '= 2.0 / 3.0'

template <unsigned Tdim>
constexpr double PM4Sand<Tdim>::c_root23;  // '= M_SQRT2 / M_SQRT3'
// --- Helper Functions for State Variable Access ---
namespace {  // Anonymous namespace for internal linkage

// Helper to get a scalar value
inline double get_scalar(const mpm::dense_map* state_vars,
                         const std::string& key) {
  try {
    // Ensure state_vars is not null
    if (!state_vars) {
      throw std::runtime_error("[PM4Sand::get_scalar] state_vars map is null.");
    }
    return state_vars->at(key);
  } catch (const std::out_of_range& oor) {
    throw std::runtime_error(
        "[PM4Sand::get_scalar] State variable not found: " + key);
  } catch (const std::exception& e) {
    throw std::runtime_error("[PM4Sand::get_scalar] debug accessing key '" +
                             key + "': " + e.what());
  }
}

// Helper to set a scalar value
inline void set_scalar(mpm::dense_map* state_vars, const std::string& key,
                       double value) {
  // Ensure state_vars is not null
  if (!state_vars) {
    throw std::runtime_error("[PM4Sand::set_scalar] state_vars map is null.");
  }
  (*state_vars)[key] = value;
}

// Helper to get a 3x3 tensor from state_vars (xx, yy, zz, xy, yz, zx)
inline Eigen::Matrix3d get_tensor(const mpm::dense_map* state_vars,
                                  const std::string& base_key) {
  Eigen::Matrix3d tensor = Eigen::Matrix3d::Zero();
  try {
    // Ensure state_vars is not null
    if (!state_vars) {
      throw std::runtime_error(
          "[PM4Sand::get_tensor] state_vars map is null for key: " + base_key);
    }
    tensor(0, 0) = state_vars->at(base_key + "_xx");
    tensor(1, 1) = state_vars->at(base_key + "_yy");
    tensor(2, 2) = state_vars->at(base_key + "_zz");
    tensor(0, 1) = state_vars->at(base_key + "_xy");
    tensor(1, 0) = tensor(0, 1);  // xy symmetry
    // tensor(1, 2) = state_vars->at(base_key + "_yz");
    // tensor(2, 1) = tensor(1, 2); // yz symmetry
    // tensor(2, 0) = state_vars->at(base_key + "_zx");
    // tensor(0, 2) = tensor(2, 0); // zx symmetry

  } catch (const std::out_of_range& oor) {
    throw std::runtime_error(
        "[PM4Sand::get_tensor] Tensor state variable component not found for "
        "base key: " +
        base_key);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "[PM4Sand::get_tensor] debug accessing tensor components for key '" +
        base_key + "': " + e.what());
  }
  return tensor;
}

// Helper to set a 3x3 tensor into state_vars
inline void set_tensor(mpm::dense_map* state_vars, const std::string& base_key,
                       const Eigen::Matrix3d& tensor) {
  // Ensure state_vars is not null
  if (!state_vars) {
    throw std::runtime_error(
        "[PM4Sand::set_tensor] state_vars map is null for key: " + base_key);
  }
  (*state_vars)[base_key + "_xx"] = tensor(0, 0);
  (*state_vars)[base_key + "_yy"] = tensor(1, 1);
  (*state_vars)[base_key + "_zz"] = tensor(2, 2);
  (*state_vars)[base_key + "_xy"] = tensor(0, 1);
  // (*state_vars)[base_key + "_yz"] = tensor(1, 2);
  // (*state_vars)[base_key + "_zx"] = tensor(2, 0);
}

}  // anonymous namespace

// --- Constructor ---
template <unsigned Tdim>
PM4Sand<Tdim>::PM4Sand(unsigned id, const Json& material_properties)
    : InfinitesimalElastoPlastic<Tdim>(id, material_properties) {
  // Call parameter initialization
  this->initialise_parameters(material_properties);
  // Initialize Identity Tensor (already done in header initializer list)
  // m_I = Eigen::Matrix3d::Identity();
  properties_ = material_properties;
}

template <unsigned Tdim>
void PM4Sand<Tdim>::initialise_parameters(const Json& props) {
  // Use the pattern from Cycliq: Try reading with .at(), use default if
  // missing. We can achieve this using contains() check before accessing with
  // .at() or by catching the exception from .at(). Using contains() is often
  // cleaner.

  try {
    // --- Read parameters from JSON or use defaults ---

    // Dr: Relative Density
    if (props.contains("Dr")) {
      m_Dr = props.at("Dr").template get<double>();
    } else {
      m_Dr = 0.55;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'Dr' not found in JSON, using default: {}",
          this->id_, m_Dr);
    }

    // G0: Shear modulus coefficient
    if (props.contains("G0")) {
      m_G0 = props.at("G0").template get<double>();
    } else {
      m_G0 = 890.0;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'G0' not found in JSON, using default: {}",
          this->id_, m_G0);
    }

    // hpo: Contraction rate parameter
    if (props.contains("hpo")) {
      m_hpo = props.at("hpo").template get<double>();
    } else {
      m_hpo = 0.63;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'hpo' not found in JSON, using default: {}",
          this->id_, m_hpo);
    }

    // P_atm: Atmospheric pressure
    if (props.contains("P_atm")) {
      m_P_atm = props.at("P_atm").template get<double>();
    } else {
      m_P_atm = 101.3;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'P_atm' not found in JSON, using default: {}",
          this->id_, m_P_atm);
    }

    // phi_cv: Critical state friction angle (degrees)
    if (props.contains("phi_cv")) {
      phi_cv_deg = props.at("phi_cv").template get<double>();
    } else {
      phi_cv_deg = 33.0;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'phi_cv' not found in JSON, using default: "
          "{}",
          this->id_, phi_cv_deg);
    }
    if (phi_cv_deg < 0)
      phi_cv_deg =
          33.0;  // Ensure non-negative, use default if input is negative
    m_Mc = 2.0 * std::sin(phi_cv_deg * M_PI / 180.0);  // Calculate Mc

    // h0: Hardening rate parameter (conditional default)
    if (props.contains("h0")) {
      m_h0 = props.at("h0").template get<double>();
      // If user explicitly provides h0 < 0, use the conditional default
      if (m_h0 < 0) {
        m_h0 = std::max(0.3, (0.25 + m_Dr) / 2.0);
        this->console_->debug(
            "Material {}: Parameter 'h0' was < 0, calculating default: {}",
            this->id_, m_h0);
      }
    } else {
      m_h0 = std::max(0.3, (0.25 + m_Dr) / 2.0);  // Default calculation
      this->console_->debug(
          "Material {}: Parameter 'h0' not found in JSON, calculating default: "
          "{}",
          this->id_, m_h0);
    }

    // emax: Maximum void ratio
    if (props.contains("emax")) {
      m_emax = props.at("emax").template get<double>();
    } else {
      m_emax = 0.8;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'emax' not found in JSON, using default: {}",
          this->id_, m_emax);
    }

    // emin: Minimum void ratio
    if (props.contains("emin")) {
      m_emin = props.at("emin").template get<double>();
    } else {
      m_emin = 0.5;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'emin' not found in JSON, using default: {}",
          this->id_, m_emin);
    }
    // Ensure emax > emin
    if (m_emax <= m_emin) {
      this->console_->debug(
          "Material {}: emax ({}) must be greater than emin ({}).", this->id_,
          m_emax, m_emin);
      throw std::runtime_error("Invalid emax/emin configuration for PM4Sand.");
    }

    // nb: Bounding surface parameter
    if (props.contains("nb")) {
      m_nb = props.at("nb").template get<double>();
    } else {
      m_nb = 0.5;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'nb' not found in JSON, using default: {}",
          this->id_, m_nb);
    }

    // nd: Dilatancy surface parameter
    if (props.contains("nd")) {
      m_nd = props.at("nd").template get<double>();
    } else {
      m_nd = 0.1;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'nd' not found in JSON, using default: {}",
          this->id_, m_nd);
    }

    // Ado: Dilatancy parameter (default calculation depends on state)
    if (props.contains("Ado")) {
      m_Ado = props.at("Ado").template get<double>();
    } else {
      m_Ado = -1.0;  // Indicate default calculation needed later
      this->console_->debug(
          "Material {}: Parameter 'Ado' not found in JSON, will calculate "
          "based on initial state.",
          this->id_);
    }

    // z_max: Maximum fabric parameter (default calculation depends on state)
    if (props.contains("z_max")) {
      m_z_max = props.at("z_max").template get<double>();
    } else {
      m_z_max = -1.0;  // Indicate default calculation needed later
      this->console_->debug(
          "Material {}: Parameter 'z_max' not found in JSON, will calculate "
          "based on initial state.",
          this->id_);
    }

    // cz: Fabric evolution parameter
    if (props.contains("cz")) {
      m_cz = props.at("cz").template get<double>();
    } else {
      m_cz = 250.0;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'cz' not found in JSON, using default: {}",
          this->id_, m_cz);
    }

    // ce: Fabric evolution parameter (conditional default)
    if (props.contains("ce")) {
      m_ce = props.at("ce").template get<double>();
      // If user explicitly provides ce <= 0, use the conditional default
      if (m_ce <= 0) {
        if (m_Dr > 0.75)
          m_ce = 0.2;
        else if (m_Dr < 0.55)
          m_ce = 0.5;
        else
          m_ce = 0.5 - (m_Dr - 0.55) * 1.5;
        this->console_->debug(
            "Material {}: Parameter 'ce' was <= 0, calculating default: {}",
            this->id_, m_ce);
      }
    } else {
      if (m_Dr > 0.75)
        m_ce = 0.2;
      else if (m_Dr < 0.55)
        m_ce = 0.5;
      else
        m_ce = 0.5 - (m_Dr - 0.55) * 1.5;  // Default calculation
      this->console_->debug(
          "Material {}: Parameter 'ce' not found in JSON, calculating default: "
          "{}",
          this->id_, m_ce);
    }

    // nu: Poisson's ratio
    if (props.contains("nu")) {
      m_nu = props.at("nu").template get<double>();
    } else {
      m_nu = 0.3;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'nu' not found in JSON, using default: {}",
          this->id_, m_nu);
    }
    if (m_nu >= 0.5) {
      this->console_->debug(
          "Material {}: Poisson's ratio nu >= 0.5 is not allowed. Setting nu = "
          "0.499.",
          this->id_);
      m_nu = 0.499;
    }

    // Cgd: Degradation parameter
    if (props.contains("Cgd")) {
      m_Cgd = props.at("Cgd").template get<double>();
    } else {
      m_Cgd = 2.0;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'Cgd' not found in JSON, using default: {}",
          this->id_, m_Cgd);
    }

    // Cdr: Dilatancy strain parameter (conditional default)
    if (props.contains("Cdr")) {
      m_Cdr = props.at("Cdr").template get<double>();
      // If user explicitly provides Cdr < 0, use the conditional default
      if (m_Cdr < 0.0) {
        m_Cdr = std::min(10.0, (5.0 + 25.0 * (m_Dr - 0.35)));
        this->console_->debug(
            "Material {}: Parameter 'Cdr' was < 0, calculating default: {}",
            this->id_, m_Cdr);
      }
    } else {
      m_Cdr =
          std::min(10.0, (5.0 + 25.0 * (m_Dr - 0.35)));  // Default calculation
      this->console_->debug(
          "Material {}: Parameter 'Cdr' not found in JSON, calculating "
          "default: {}",
          this->id_, m_Cdr);
    }

    // Ckaf: Dilatancy parameter (conditional default)
    if (props.contains("Ckaf")) {
      m_Ckaf = props.at("Ckaf").template get<double>();
      // If user explicitly provides Ckaf < 0, use the conditional default
      if (m_Ckaf < 0) {
        m_Ckaf = std::min(
            35.0,
            std::max(4.0, (5.0 +
                           220.0 * std::pow(std::max(0.0, m_Dr - 0.26), 3.0))));
        this->console_->debug(
            "Material {}: Parameter 'Ckaf' was < 0, calculating default: {}",
            this->id_, m_Ckaf);
      }
    } else {
      m_Ckaf = std::min(
          35.0,
          std::max(4.0, (5.0 + 220.0 * std::pow(std::max(0.0, m_Dr - 0.26),
                                                3.0))));  // Default calculation
      this->console_->debug(
          "Material {}: Parameter 'Ckaf' not found in JSON, calculating "
          "default: {}",
          this->id_, m_Ckaf);
    }

    // Q: Critical state parameter
    if (props.contains("Q")) {
      m_Q = props.at("Q").template get<double>();
    } else {
      m_Q = 10.0;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'Q' not found in JSON, using default: {}",
          this->id_, m_Q);
    }

    // R: Critical state parameter
    if (props.contains("R")) {
      m_R = props.at("R").template get<double>();
    } else {
      m_R = 1.5;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'R' not found in JSON, using default: {}",
          this->id_, m_R);
    }

    // m: Yield surface constant
    if (props.contains("m")) {
      m_m = props.at("m").template get<double>();
    } else {
      m_m = 0.01;  // Default value
      this->console_->debug(
          "Material {}: Parameter 'm' not found in JSON, using default: {}",
          this->id_, m_m);
    }

    // Fsed_min: Post-liquefaction parameter (conditional default)
    if (props.contains("Fsed_min")) {
      m_Fsed_min = props.at("Fsed_min").template get<double>();
      // If user explicitly provides Fsed_min < 0, use the conditional default
      if (m_Fsed_min < 0.0) {
        m_Fsed_min = std::min(0.99, (0.03 * std::exp(2.6 * m_Dr)));
        this->console_->debug(
            "Material {}: Parameter 'Fsed_min' was < 0, calculating default: "
            "{}",
            this->id_, m_Fsed_min);
      }
    } else {
      m_Fsed_min =
          std::min(0.99, (0.03 * std::exp(2.6 * m_Dr)));  // Default calculation
      this->console_->debug(
          "Material {}: Parameter 'Fsed_min' not found in JSON, calculating "
          "default: {}",
          this->id_, m_Fsed_min);
    }

    // p_sedo: Post-liquefaction parameter (conditional default)
    if (props.contains("p_sedo")) {
      m_p_sedo = props.at("p_sedo").template get<double>();
      // If user explicitly provides p_sedo < 0, use the conditional default
      if (m_p_sedo < 0.0) {
        m_p_sedo = m_P_atm / 5.0;
        this->console_->debug(
            "Material {}: Parameter 'p_sedo' was < 0, calculating default: {}",
            this->id_, m_p_sedo);
      }
    } else {
      m_p_sedo = m_P_atm / 5.0;  // Default calculation
      this->console_->debug(
          "Material {}: Parameter 'p_sedo' not found in JSON, calculating "
          "default: {}",
          this->id_, m_p_sedo);
    }

    // integration_scheme: Integration scheme choice
    if (props.contains("integration_scheme")) {
      m_integrationScheme = props.at("integration_scheme").template get<int>();
    } else {
      m_integrationScheme = 1;  // Default: Modified Euler
      this->console_->debug(
          "Material {}: Parameter 'integration_scheme' not found in JSON, "
          "using default: {}",
          this->id_, m_integrationScheme);
    }

    // Initialize p_min, p_min2 with default guesses. They will be refined based
    // on initial stress.
    m_Pmin = m_P_atm / 200.0;
    m_Pmin2 = m_Pmin * 10.0;

  } catch (const nlohmann::json::exception& e) {
    // Catch JSON specific exceptions (like type debugs or missing keys if using
    // .at())
    this->console_->debug(
        "Material {}: debug reading PM4Sand properties from JSON: {}",
        this->id_, e.what());
    throw std::runtime_error(
        "Failed to initialize PM4Sand parameters from JSON.");
  } catch (const std::exception& e) {
    // Catch other potential exceptions
    this->console_->debug(
        "Material {}: Unexpected debug during PM4Sand parameter "
        "initialization: {}",
        this->id_, e.what());
    throw;  // Re-throw exception after logging
  }
}

// --- State Variable Initialisation ---
template <unsigned Tdim>
mpm::dense_map PM4Sand<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars;

  // Initial state assuming zero stress/strain before first compute_stress call.
  // Values will be properly set based on initial conditions during first
  // compute_stress.
  set_scalar(&state_vars, "mean_stress", m_P_atm / 200.0);  // Initial guess
  set_scalar(&state_vars, "total_vol_strain", 0.0);
  set_scalar(&state_vars, "elastic_vol_strain", 0.0);

  // Initial void ratio based on input Dr
  double e_init = m_emax - (m_emax - m_emin) * m_Dr;
  set_scalar(&state_vars, "void_ratio", e_init);

  // Tensors (initialize as zero) - Use full 3D tensor helper
  Eigen::Matrix3d zero_tensor = Eigen::Matrix3d::Zero();
  set_tensor(&state_vars, "alpha", zero_tensor);
  set_tensor(&state_vars, "alpha_in", zero_tensor);
  set_tensor(&state_vars, "alpha_in_true", zero_tensor);
  set_tensor(&state_vars, "alpha_in_p", zero_tensor);
  set_tensor(&state_vars, "alpha_in_max", zero_tensor);
  set_tensor(&state_vars, "alpha_in_min", zero_tensor);
  set_tensor(&state_vars, "fabric", zero_tensor);
  set_tensor(&state_vars, "fabric_in", zero_tensor);

  // Scalars
  set_scalar(&state_vars, "zcum", 0.0);
  set_scalar(&state_vars, "zpeak", 0.0);
  set_scalar(&state_vars, "pzp",
             m_P_atm / 100.0);  // Initial guess based on default p_min
  set_scalar(&state_vars, "zxp", 0.0);
  set_scalar(&state_vars, "pzpFlag", 1.0);  // True initially
  set_scalar(&state_vars, "dGamma", 0.0);
  set_scalar(&state_vars, "yield_state", 0.0);  // Elastic initially

  // Flag to indicate if model parameters (pmin, Mb, Md etc.) have been
  // initialized based on stress
  set_scalar(&state_vars, "model_initialized", 0.0);  // False initially

  return state_vars;
}

template <unsigned Tdim>
std::vector<std::string> PM4Sand<Tdim>::state_variables() const {
  std::vector<std::string> v = {"mean_stress", "total_vol_strain",
                                "elastic_vol_strain", "void_ratio"};

  constexpr std::array<const char*, 8> tensor_names = {
      "alpha",        "alpha_in",     "alpha_in_true", "alpha_in_p",
      "alpha_in_max", "alpha_in_min", "fabric",        "fabric_in"};

  for (auto name : tensor_names) {
    v.emplace_back(std::string(name) + "_xx");
    v.emplace_back(std::string(name) + "_yy");
    v.emplace_back(std::string(name) + "_zz");
    v.emplace_back(std::string(name) + "_xy");
  }
  // scalar tails
  v.insert(v.end(), {"zcum", "zpeak", "pzp", "zxp", "pzpFlag", "dGamma",
                     "yield_state", "model_initialized"});
  return v;
}

// --- Initialise Function ---
template <unsigned Tdim>
void PM4Sand<Tdim>::initialise(mpm::dense_map* state_vars) {
  // This function is called at the beginning of a step or analysis.
  // We can reset step-specific flags or perform checks here if needed.
  // The main initialization happens in the constructor and the first
  // compute_stress call.
  (*state_vars)["yield_state"] = 0.0;  // Reset yield state for the step (will
                                       // be updated in compute_stress)
  (*state_vars)["dGamma"] =
      0.0;  // Reset plastic multiplier increment for the step
  // No other specific initialization seems necessary here based on PM4Sand
  // logic.
}

// --- Voigt <-> Tensor Conversion ---
template <unsigned Tdim>
Eigen::Matrix3d PM4Sand<Tdim>::voigt_to_tensor(const Vector6d& v) const {
  // Input v is [xx, yy, zz, xy, yz, zx] (MPM framework Voigt)
  Eigen::Matrix3d tensor;
  tensor << v(0), v(3), v(5),  // xx, xy, xz (using zx for xz)
      v(3), v(1), v(4),        // yx, yy, yz
      v(5), v(4), v(2);        // zx, zy, zz
  return tensor;
}

template <unsigned Tdim>
typename PM4Sand<Tdim>::Vector6d PM4Sand<Tdim>::tensor_to_voigt(
    const Eigen::Matrix3d& m) const {
  // Output Voigt notation [xx, yy, zz, xy, yz, zx]
  Vector6d v;
  v(0) = m(0, 0);  // xx
  v(1) = m(1, 1);  // yy
  v(2) = m(2, 2);  // zz
  v(3) = m(0, 1);  // xy
  v(4) = m(1, 2);  // yz
  v(5) = m(2, 0);  // zx
  return v;
}

// --- Stress Computation ---
template <unsigned Tdim>
typename PM4Sand<Tdim>::Vector6d PM4Sand<Tdim>::compute_stress(
    const Vector6d& stress_voigt_prev,  // Stress at start of step (Voigt,
                                        // Compressive Negative)
    const Vector6d& dstrain_voigt,      // Strain increment (Voigt, Engineering
                                        // Shear Strain)
    const ParticleBase<Tdim>* /*ptr*/,  // Particle pointer (unused for now)
    dense_map* state_vars, double dt) {

  // ... (Keep the Input Conversion & Initialisation Check section as before)
  // ...
  // -------------------------------------------------
  // 0. Input Conversion & Initialisation Check
  // -------------------------------------------------
  Matrix3d stress_prev = voigt_to_tensor(-stress_voigt_prev) / 1000;
  Vector6d dstrain_tensor_voigt = -dstrain_voigt;
  dstrain_tensor_voigt(3) *= 0.5;
  dstrain_tensor_voigt(4) *= 0.5;
  dstrain_tensor_voigt(5) *= 0.5;
  Matrix3d dEpsilon = voigt_to_tensor(dstrain_tensor_voigt);

  bool model_initialized = (get_scalar(state_vars, "model_initialized") > 0.5);
  if (!model_initialized) {
    // ... (Keep the initialization logic here) ...
    double p0 = stress_prev.trace() * c_one3;
    m_Pmin = std::max(p0 / 200.0, m_P_atm / 200.0);
    m_Pmin2 = m_Pmin * 10.0;
    double e_init = get_scalar(state_vars, "void_ratio");
    double initial_Dr = (m_emax - e_init) / (m_emax - m_emin + c_tolerance);
    double ksi = getKsi(m_Dr, p0);
    if (m_z_max < 0) {
      m_z_max = std::min(20.0, 0.7 * std::exp(-6.1 * ksi));
      if (m_z_max <= 0) m_z_max = 1.0;
    }
    if (ksi < 0.0) {
      this->m_Mb = m_Mc * std::exp(-1.0 * m_nb * ksi);  // this->m_Mb に代入
      this->m_Md = m_Mc * std::exp(m_nd * ksi);         // this->m_Md に代入
      if (m_Ado < 0) {
        if (m_Mb >= 1.999) {
          m_Ado = 1.5;
        } else {
          double term1 = std::asin(m_Mb / 2.0);
          double term2 = std::asin(m_Mc / 2.0);
          double denom = m_Mb - m_Md;
          if (std::abs(denom) < c_tolerance) {
            m_Ado = 1.24;
          } else {
            m_Ado = 2.5 * (term1 - term2) / denom;
          }
        }
      }
    } else {
      this->m_Mb =
          m_Mc * std::exp(-1.0 * m_nb / 4.0 * ksi);    // this->m_Mb に代入
      this->m_Md = m_Mc * std::exp(m_nd * 4.0 * ksi);  // this->m_Md に代入
      if (m_Ado < 0) m_Ado = 1.24;
    }
    m_Ado = std::max(0.1, m_Ado);
    Matrix3d initial_alpha = Matrix3d::Zero();
    p0 = std::max(p0, m_Pmin);
    Matrix3d s0 = getDevPart(stress_prev);
    initial_alpha = s0 / p0;
    double Mfin = std::sqrt(2.0) * tensorNorm(getDevPart(stress_prev)) / p0;
    double Mcut = std::max({m_Mb, m_Md});
    if (Mfin > Mcut) {
      this->console_->debug(
          "Material {}: Initial stress ratio M_fin={} exceeds Mcut={}. "
          "Projecting alpha.",
          this->id_, Mfin, Mcut);
      // initial_alpha *= (Mcut / Mfin); // 元のMPM版
      Matrix3d r =
          getDevPart(stress_prev) / p0 * Mcut / Mfin;  // Standalone版の r 計算
      initial_alpha =
          r * (Mcut - m_m) /
          (Mcut + c_tolerance);  // Standalone版の alpha 計算 (ゼロ割防止追加)
    }
    set_tensor(state_vars, "alpha", initial_alpha);
    set_tensor(state_vars, "alpha_in", initial_alpha);
    set_tensor(state_vars, "alpha_in_true", initial_alpha);
    set_tensor(state_vars, "alpha_in_p", initial_alpha);
    set_tensor(state_vars, "alpha_in_max", initial_alpha);
    set_tensor(state_vars, "alpha_in_min", initial_alpha);
    set_scalar(state_vars, "zpeak", m_z_max / 100000.0);
    set_scalar(state_vars, "pzp", std::max(p0, m_Pmin) / 100.0);
    set_scalar(state_vars, "mean_stress", p0);
    set_scalar(state_vars, "model_initialized", 1.0);
    // this->console_->info("Material {}: PM4Sand model initialized. p0={:.2e},
    // Dr={:.2f}, pmin={:.2e}", this->id_, p0, initial_Dr, m_Pmin);
  }

  // -------------------------------------------------
  // 1. Get current state from state_vars
  // -------------------------------------------------
  // ... (Keep the state variable retrieval section as before) ...
  double p_n = get_scalar(state_vars, "mean_stress");
  double total_vol_strain_n = get_scalar(state_vars, "total_vol_strain");
  double elastic_vol_strain_n = get_scalar(state_vars, "elastic_vol_strain");
  double void_ratio_n = get_scalar(state_vars, "void_ratio");
  Matrix3d alpha_n = get_tensor(state_vars, "alpha");
  Matrix3d alpha_in_n = get_tensor(state_vars, "alpha_in");
  Matrix3d alpha_in_true_n = get_tensor(state_vars, "alpha_in_true");
  Matrix3d alpha_in_p_n = get_tensor(state_vars, "alpha_in_p");
  Matrix3d alpha_in_max_n = get_tensor(state_vars, "alpha_in_max");
  Matrix3d alpha_in_min_n = get_tensor(state_vars, "alpha_in_min");
  Matrix3d fabric_n = get_tensor(state_vars, "fabric");
  Matrix3d fabric_in_n = get_tensor(state_vars, "fabric_in");
  double zcum_n = get_scalar(state_vars, "zcum");
  double zpeak_n = get_scalar(state_vars, "zpeak");
  double pzp_n = get_scalar(state_vars, "pzp");
  double zxp_n = get_scalar(state_vars, "zxp");
  bool pzpFlag_n = (get_scalar(state_vars, "pzpFlag") > 0.5);

  Matrix3d current_stress = stress_prev;
  Matrix3d current_alpha = alpha_n;
  Matrix3d current_fabric = fabric_n;
  Matrix3d current_alpha_in = alpha_in_n;
  Matrix3d current_alpha_in_true = alpha_in_true_n;
  Matrix3d current_alpha_in_p = alpha_in_p_n;
  Matrix3d current_alpha_in_max = alpha_in_max_n;
  Matrix3d current_alpha_in_min = alpha_in_min_n;
  Matrix3d current_fabric_in = fabric_in_n;
  double current_total_vol_strain = total_vol_strain_n;
  double current_elastic_vol_strain = elastic_vol_strain_n;
  double current_void_ratio = void_ratio_n;
  double current_zcum = zcum_n;
  double current_zpeak = zpeak_n;
  double current_pzp = pzp_n;
  double current_zxp = zxp_n;
  bool current_pzpFlag = pzpFlag_n;
  double current_dGamma = 0.0;
  double initial_zcum = current_zcum;
  double initial_zpeak = current_zpeak;
  double accumulated_dFabricNorm = 0.0;
  // ステップ中の最大Fabricノルムを追跡する変数。初期値はステップ開始時のノルム。
  double max_fabric_norm_in_step = tensorNorm(current_fabric) * c_root12;
  // -------------------------------------------------
  // 2. Prepare for integration
  // -------------------------------------------------
  double dEpsilon_vol = dEpsilon.trace();
  Matrix3d dEpsilon_dev = dEpsilon - (dEpsilon_vol * c_one3) * m_I;

  // -------------------------------------------------
  // 3. Check for loading reversal BEFORE integration
  // -------------------------------------------------
  // ... (Keep the reversal check logic as before) ...
  double K_elast, G_elast, Mcur_elast_dummy;
  getElasticModuli(current_stress, K_elast, G_elast, Mcur_elast_dummy,
                   current_zcum, state_vars);
  Matrix3d trialStress = current_stress + 2.0 * G_elast * dEpsilon_dev +
                         K_elast * dEpsilon_vol * m_I;
  bool trial_yield =
      (getF(trialStress, current_alpha, state_vars) > -c_tolerance);
  Matrix3d n_tr = getNormalToYield(trialStress, current_alpha, state_vars);
  Matrix3d alpha_diff_true = current_alpha - current_alpha_in_true;
  if ((tensordot(alpha_diff_true, n_tr) < 0.0)) {
    this->console_->debug("Material {}: Loading reversal detected.", this->id_);
    current_alpha_in_p = current_alpha_in;
    current_alpha_in_true = current_alpha;
    current_fabric_in = current_fabric;
    double p_rev = current_stress.trace() * c_one3;
    p_rev = std::max(p_rev, m_Pmin);
    double zxpTemp = tensorNorm(current_fabric) * p_rev;
    if (((zxpTemp > current_zxp) && (p_rev > current_pzp)) || current_pzpFlag) {
      current_zxp = zxpTemp;
      current_pzp = p_rev;
      current_pzpFlag = false;
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        if (current_alpha_in(i, j) > 0.0) {
          current_alpha_in_min(i, j) =
              std::min(current_alpha_in_min(i, j), current_alpha(i, j));
        } else {
          current_alpha_in_max(i, j) =
              std::max(current_alpha_in_max(i, j), current_alpha(i, j));
        }
      }
    }
    bool opposite_dir = false;
    // if (std::abs(current_alpha(0,1)) > 0.0 &&
    // std::abs(current_alpha_in_p(0,1)) > 0.0) {
    //      if (current_alpha(0,1) * current_alpha_in_p(0,1) <= 0.0)
    //      opposite_dir = true;
    // }
    if (current_alpha(0, 1) * current_alpha_in_p(0, 1) <= 0.0)
      opposite_dir = true;
    if (!opposite_dir) {  // スタンドアローン版の if (m_Alpha(0, 1) *
                          // m_Alpha_in_p(0, 1) > 0) に対応
      Matrix3d alpha_in_temp =
          current_alpha_in;  // Keep previous value temporarily if needed
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (n_tr(i, j) > 0.0) {
            current_alpha_in(i, j) = std::max(0.0, current_alpha_in_min(i, j));
          } else if (n_tr(i, j) < 0.0) {
            current_alpha_in(i, j) = std::min(0.0, current_alpha_in_max(i, j));
          } else {
            current_alpha_in(i, j) = std::min(0.0, current_alpha_in_max(i, j));
          }
          // else { current_alpha_in(i,j) = alpha_in_temp(i,j); } // Neutral
          // loading component, keep previous? Or set to zero? Standalone seems
          // to keep.
        }
      }
    } else {  // スタンドアローン版の else に対応
      current_alpha_in = current_alpha;
    }
  }

  // -------------------------------------------------
  // 4. Perform Integration using Modified Euler with Adaptive Substepping
  // -------------------------------------------------
  Matrix3d nextStress = current_stress;

  Matrix3d nextAlpha = current_alpha;
  Matrix3d nextFabric = current_fabric;
  double next_total_vol_strain = current_total_vol_strain;
  double next_elastic_vol_strain = current_elastic_vol_strain;
  double next_void_ratio = current_void_ratio;

  const double TolE = 1e-5;
  const double dT_min = 1e-4;
  double T = 0.0;
  double dT = 1.0;

  int substep_iter = 0;
  const int max_substeps = 500;

  while (T < 1.0) {
    dT = std::min(dT, 1.0 - T);
    Matrix3d dEpsilon_sub = dT * dEpsilon;
    double dEpsilon_sub_vol = dEpsilon_sub.trace();
    Matrix3d dEpsilon_sub_dev =
        dEpsilon_sub - (dEpsilon_sub_vol * c_one3) * m_I;

    // --- Predictor Step ---
    double K1, G1, Mcur1;
    getElasticModuli(current_stress, K1, G1, Mcur1, current_zcum, state_vars);
    double current_Dr1 =
        (m_emax - current_void_ratio) / (m_emax - m_emin + c_tolerance);
    current_Dr1 = std::max(0.0, std::min(current_Dr1, 1.0));  // Clamp Dr

    Matrix3d n1, R1, b1, alphaD1;
    double D1, Kp1, Cka1, h1, alphaAlphaBDotN1;
    getStateDependent(current_stress, current_alpha, current_alpha_in,
                      current_alpha_in_p, current_fabric, current_fabric_in, G1,
                      current_zcum, current_zpeak, current_pzp, Mcur1,
                      current_Dr1, n1, D1, R1, Kp1, alphaD1, Cka1, h1, b1,
                      alphaAlphaBDotN1, state_vars);

    // Calculate predictor L1
    Matrix3d r1;  // *** USE IF/ELSE ***
    double p1 = current_stress.trace() * c_one3;
    if (p1 < m_Pmin) {
      r1 = Matrix3d::Zero();
    } else {
      double p1_safe = std::max(p1, m_Pmin);  // Use safe p for division
      if (std::abs(p1_safe) < c_tolerance) {
        r1 = Matrix3d::Zero();
      } else {
        r1 = getDevPart(current_stress) / p1_safe;
      }
    }

    double denom1 = Kp1 + 2.0 * G1 - K1 * D1 * tensordot(n1, r1);
    double L1 = 0.0;
    if (std::abs(denom1) > c_tolerance) {
      L1 = (2.0 * G1 * tensordot(n1, dEpsilon_sub_dev) -
            tensordot(n1, r1) * K1 * dEpsilon_sub_vol) /
           denom1;
    }
    L1 = macauley(L1);

    // Calculate predictor increments
    Matrix3d dSigma1_e =
        2.0 * G1 * dEpsilon_sub_dev + K1 * dEpsilon_sub_vol * m_I;

    Matrix3d dSigma1_p;  // *** USE IF/ELSE ***
    if (L1 > c_tolerance) {
      dSigma1_p = L1 * (2.0 * G1 * n1 + K1 * D1 * m_I);
    } else {
      dSigma1_p = Matrix3d::Zero();
    }

    Matrix3d dAlpha1;  // *** USE IF/ELSE ***
    if (L1 > c_tolerance) {
      dAlpha1 = c_two3 * L1 * h1 * b1;
    } else {
      dAlpha1 = Matrix3d::Zero();
    }

    Matrix3d dFabric1 = Matrix3d::Zero();
    if (L1 > c_tolerance && D1 < -c_tolerance) {
      if (tensordot(alphaD1 - current_alpha, n1) < 0.0) {
        double fab_denom = 1.0 + macauley(current_zcum / (2.0 * m_z_max) - 1.0);
        dFabric1 = -m_cz / std::max(fab_denom, c_tolerance) * L1 *
                   (m_z_max * n1 + current_fabric);
      }
    }

    Matrix3d dEpsilon1_p;  // *** USE IF/ELSE ***
    if (L1 > c_tolerance) {
      dEpsilon1_p = L1 * R1;
    } else {
      dEpsilon1_p = Matrix3d::Zero();
    }

    // Predicted mid-step state
    Matrix3d stress_mid = current_stress + dSigma1_e - dSigma1_p;
    Matrix3d alpha_mid = current_alpha + dAlpha1;
    Matrix3d fabric_mid = current_fabric + dFabric1;
    double p_mid = stress_mid.trace() * c_one3;

    // Check for negative pressure in predictor
    if (p_mid < m_Pmin / 5.0) {
      if (dT <= dT_min * (1.0 + c_tolerance)) {
        this->console_->debug(
            "Material {}: p < p_min/5 in predictor step even at dT_min (p={}, "
            "dT={}). Proceeding.",
            this->id_, p_mid, dT);
      } else {
        dT = std::max(dT_min, dT * 0.5);
        this->console_->debug(
            "Material {}: p < p_min/5 in predictor step (p={}). Reducing dT to "
            "{}",
            this->id_, p_mid, dT);
        continue;
      }
    }

    // --- Corrector Step ---
    double K2, G2, Mcur2;
    getElasticModuli(stress_mid, K2, G2, Mcur2, current_zcum, state_vars);
    double current_Dr2 = current_Dr1;

    Matrix3d n2, R2, b2, alphaD2;
    double D2, Kp2, Cka2, h2, alphaAlphaBDotN2;
    getStateDependent(stress_mid, alpha_mid, current_alpha_in,
                      current_alpha_in_p, fabric_mid, current_fabric_in, G2,
                      current_zcum, current_zpeak, current_pzp, Mcur2,
                      current_Dr2, n2, D2, R2, Kp2, alphaD2, Cka2, h2, b2,
                      alphaAlphaBDotN2, state_vars);

    // Calculate corrector L2
    Matrix3d r2;  // *** USE IF/ELSE ***
    if (p_mid < m_Pmin) {
      r2 = Matrix3d::Zero();
    } else {
      double p_mid_safe = std::max(p_mid, m_Pmin);
      if (std::abs(p_mid_safe) < c_tolerance) {
        r2 = Matrix3d::Zero();
      } else {
        r2 = getDevPart(stress_mid) / p_mid_safe;
      }
    }

    double denom2 = Kp2 + 2.0 * G2 - K2 * D2 * tensordot(n2, r2);
    double L2 = 0.0;
    if (std::abs(denom2) > c_tolerance) {
      L2 = (2.0 * G2 * tensordot(n2, dEpsilon_sub_dev) -
            tensordot(n2, r2) * K2 * dEpsilon_sub_vol) /
           denom2;
    }
    L2 = macauley(L2);

    // Calculate corrector increments
    Matrix3d dSigma2_e =
        2.0 * G2 * dEpsilon_sub_dev + K2 * dEpsilon_sub_vol * m_I;

    Matrix3d dSigma2_p;  // *** USE IF/ELSE ***
    if (L2 > c_tolerance) {
      dSigma2_p = L2 * (2.0 * G2 * n2 + K2 * D2 * m_I);
    } else {
      dSigma2_p = Matrix3d::Zero();
    }

    Matrix3d dAlpha2;  // *** USE IF/ELSE ***
    if (L2 > c_tolerance) {
      dAlpha2 = c_two3 * L2 * h2 * b2;
    } else {
      dAlpha2 = Matrix3d::Zero();
    }

    Matrix3d dFabric2 = Matrix3d::Zero();
    if (L2 > c_tolerance && D2 < -c_tolerance) {
      if (tensordot(alphaD2 - alpha_mid, n2) < 0.0) {
        double fab_denom = 1.0 + macauley(current_zcum / (2.0 * m_z_max) - 1.0);
        dFabric2 = -m_cz / std::max(fab_denom, c_tolerance) * L2 *
                   (m_z_max * n2 + fabric_mid);
      }
    }

    Matrix3d dEpsilon2_p;  // *** USE IF/ELSE ***
    if (L2 > c_tolerance) {
      dEpsilon2_p = L2 * R2;
    } else {
      dEpsilon2_p = Matrix3d::Zero();
    }

    // --- debug Estimation and Step Size Control ---
    // ... (Keep this section as before) ...
    double stressNorm = tensorNorm(current_stress);
    Matrix3d dSigma_avg =
        0.5 * ((dSigma1_e - dSigma1_p) + (dSigma2_e - dSigma2_p));
    Matrix3d dSigma_diff = (dSigma2_e - dSigma2_p) - (dSigma1_e - dSigma1_p);
    double debug_estimate;
    if (stressNorm < 1.0) {
      debug_estimate = tensorNorm(dSigma_diff) / 2.0;
    } else {
      debug_estimate = tensorNorm(dSigma_diff) / (2.0 * stressNorm);
    }
    double q = 0.8 * std::pow(TolE / (debug_estimate + c_tolerance), 0.5);
    q = std::max(0.1, std::min(q, 2.0));
    // --- Accept/Reject Step ---
    // ... (Keep this section as before) ...
    if (debug_estimate <= TolE || dT <= dT_min * (1.0 + c_tolerance)) {
      // Accept step
      Matrix3d dAlpha_avg = 0.5 * (dAlpha1 + dAlpha2);
      Matrix3d dFabric_avg = 0.5 * (dFabric1 + dFabric2);
      Matrix3d dEpsilon_p_avg = 0.5 * (dEpsilon1_p + dEpsilon2_p);
      double dEpsilon_p_avg_vol = dEpsilon_p_avg.trace();

      current_stress += dSigma_avg;
      current_alpha += dAlpha_avg;
      current_fabric += dFabric_avg;
      current_total_vol_strain += dEpsilon_sub_vol;
      current_elastic_vol_strain += dEpsilon_sub_vol - dEpsilon_p_avg_vol;
      current_void_ratio += (1.0 + current_void_ratio) * dEpsilon_sub_vol;
      double dFabricNorm_avg = tensorNorm(dFabric_avg);
      accumulated_dFabricNorm += dFabricNorm_avg * c_root12;  // 変化量を累積
      // current_zpeak = std::max(current_zpeak, tensorNorm(current_fabric) *
      // c_root12);
      current_dGamma += 0.5 * (L1 + L2) * dT;  // Sum of average L over substep
                                               // (dT=1 assumed for L)
      max_fabric_norm_in_step =
          std::max(max_fabric_norm_in_step,
                   tensorNorm(current_fabric) *
                       c_root12);  // ステップ中の最大ノルムを更新
      double p_next_sub = current_stress.trace() * c_one3;
      if (p_next_sub < m_Pmin / 5.0) {
        this->console_->debug(
            "Material {}: p < p_min/5 after substep update (p={}). Clamping "
            "stress.",
            this->id_, p_next_sub);
        Matrix3d s_next = getDevPart(current_stress);
        current_stress = s_next + (m_Pmin / 5.0) * m_I;
      }

      T += dT;
      dT *= q;
      dT = std::max(dT_min, dT);
      dT = std::min(dT, 1.0 - T);
    } else {
      // Reject step
      dT *= q;
      dT = std::max(dT_min, dT);
      this->console_->debug(
          "Material {}: Substep rejected (debug={:.2e}). Reducing dT to {}",
          this->id_, debug_estimate, dT);
    }
    substep_iter++;

  }  // End while (T < 1.0)

  // ... (Keep the rest of the function: post-loop check, stress correction
  // call, state update, return value) ...
  if (substep_iter >= max_substeps && T < 1.0 - c_tolerance) {
    this->console_->debug(
        "Material {}: Modified Euler integration failed to converge within max "
        "substeps (T={}). Using last calculated state.",
        this->id_, T);
  }
  nextStress = current_stress;
  nextAlpha = current_alpha;
  nextFabric = current_fabric;
  next_total_vol_strain = current_total_vol_strain;
  next_elastic_vol_strain = current_elastic_vol_strain;
  next_void_ratio = current_void_ratio;

  // -------------------------------------------------
  // 5. Stress Correction
  // -------------------------------------------------
  double F_yield_final = getF(nextStress, nextAlpha, state_vars);
  if (F_yield_final > c_tolerance * 10.0 ||
      nextStress.trace() * c_one3 < m_Pmin / 5.0) {
    if (F_yield_final > c_tolerance * 10.0)
      this->console_->debug(
          "Material {}: Yield condition violated after full step integration "
          "(F={:.2e} > {:.2e}). Applying stress correction.",
          this->id_, F_yield_final, c_tolerance * 10.0);
    else
      this->console_->debug(
          "Material {}: p < p_min/5 after full step integration (p={:.2e}). "
          "Applying stress correction.",
          this->id_, nextStress.trace() * c_one3);

    stressCorrection(nextStress, nextAlpha, stress_prev, alpha_in_n,
                     alpha_in_p_n,  // Use alpha history from start of step
                     nextFabric,    // Use updated fabric
                     next_void_ratio, state_vars);  // Pass updated void ratio

    F_yield_final = getF(nextStress, nextAlpha, state_vars);
    this->console_->debug(
        "Material {}: Stress correction finished (F={:.2e}, p={:.2e}).",
        this->id_, F_yield_final, nextStress.trace() * c_one3);
  }

  // -------------------------------------------------
  // 6. Update state variables for storage (Commit State)
  // -------------------------------------------------
  current_zcum =
      initial_zcum + accumulated_dFabricNorm;  // ステップ終了時に更新
  current_zpeak = std::max(
      initial_zpeak,
      max_fabric_norm_in_step);  // ステップ終了時に更新 (ステップ中の最大
                                 // Fabric ノルムと初期値を比較)

  set_scalar(state_vars, "mean_stress", nextStress.trace() * c_one3);
  set_scalar(state_vars, "total_vol_strain", next_total_vol_strain);
  set_scalar(state_vars, "elastic_vol_strain", next_elastic_vol_strain);
  set_scalar(state_vars, "void_ratio", next_void_ratio);
  set_tensor(state_vars, "alpha", nextAlpha);
  set_tensor(state_vars, "alpha_in", current_alpha_in);
  set_tensor(state_vars, "alpha_in_true", current_alpha_in_true);
  set_tensor(state_vars, "alpha_in_p", current_alpha_in_p);
  set_tensor(state_vars, "alpha_in_max", current_alpha_in_max);
  set_tensor(state_vars, "alpha_in_min", current_alpha_in_min);
  set_tensor(state_vars, "fabric", nextFabric);
  set_tensor(state_vars, "fabric_in", current_fabric_in);
  set_scalar(state_vars, "zcum", current_zcum);
  set_scalar(state_vars, "zpeak", current_zpeak);
  set_scalar(state_vars, "pzp", current_pzp);
  set_scalar(state_vars, "zxp", current_zxp);
  set_scalar(state_vars, "pzpFlag", current_pzpFlag ? 1.0 : 0.0);
  set_scalar(state_vars, "dGamma", current_dGamma);
  bool current_yield_state = (current_dGamma > c_tolerance);
  set_scalar(state_vars, "yield_state", current_yield_state ? 1.0 : 0.0);

  // -------------------------------------------------
  // 7. Return updated stress in Voigt notation (compressive negative)
  // -------------------------------------------------
  return -tensor_to_voigt(nextStress) * 1000;
}

// --- Consistent Tangent Matrix ---
template <unsigned Tdim>
typename PM4Sand<Tdim>::Matrix6x6 PM4Sand<Tdim>::compute_elasto_plastic_tensor(
    const Vector6d& stress_voigt, const Vector6d& /*dstrain*/,
    const ParticleBase<Tdim>* /*ptr*/, dense_map* state_vars, double dt,
    bool /*hardening*/) {

  // Return the elastic tangent matrix based on the *current* state
  // Convert stress back to compression positive for internal calculations
  Matrix3d stress_tensor = voigt_to_tensor(-stress_voigt);
  double K, G, Mcur_dummy;
  double zcum = get_scalar(state_vars, "zcum");
  // Use the helper function to get current K and G
  getElasticModuli(stress_tensor, K, G, Mcur_dummy, zcum, state_vars);

  Matrix6x6 De = Matrix6x6::Zero();
  double a = K + 4.0 / 3.0 * G;
  double b = K - 2.0 / 3.0 * G;

  // Assuming Voigt order [xx, yy, zz, xy, yz, zx]
  // Assuming framework uses engineering shear strain, tangent components are G
  De(0, 0) = a;
  De(1, 1) = a;
  De(2, 2) = a;
  De(0, 1) = b;
  De(0, 2) = b;
  De(1, 0) = b;
  De(1, 2) = b;
  De(2, 0) = b;
  De(2, 1) = b;
  De(3, 3) = G;  // xy (corresponds to gamma_xy)
  De(4, 4) = G;  // yz (corresponds to gamma_yz)
  De(5, 5) = G;  // zx (corresponds to gamma_zx)

  // Note: This is the elastic tangent. The true consistent tangent requires
  // differentiation of the stress update algorithm, which is complex for
  // PM4Sand.
  return De;
}

// --- Helper Function Implementations ---

template <unsigned Tdim>
double PM4Sand<Tdim>::getKsi(double Dr, double p) {
  double pn = std::max(p, m_Pmin);  // Use p_min threshold
  double log_arg = 100.0 * pn / m_P_atm;
  // Prevent log(<=0)
  if (log_arg <= std::numeric_limits<double>::epsilon()) {
    return -100.0;  // Indicate very dense state
  }
  double denom = m_Q - std::log(log_arg);
  // Avoid division by zero
  if (std::abs(denom) < c_tolerance) {
    return (denom > 0 ? 100.0
                      : -100.0);  // Indicate near critical state singularity
  }
  return m_R / denom - Dr;
}

template <unsigned Tdim>
void PM4Sand<Tdim>::getElasticModuli(const Matrix3d& sigma, double& K,
                                     double& G, dense_map* state_vars) {
  // Simplified version calling the full one
  double Mcur_dummy;
  double zcum = get_scalar(state_vars, "zcum");
  getElasticModuli(sigma, K, G, Mcur_dummy, zcum, state_vars);
}

template <unsigned Tdim>
void PM4Sand<Tdim>::getElasticModuli(const Matrix3d& sigma, double& K,
                                     double& G, double& Mcur,
                                     const double& zcum,
                                     dense_map* state_vars) {
  // Full implementation matching PM4Sand_claude_2.cpp logic
  int msr = 4;
  double Csr0 = 0.5;
  double pn = sigma.trace() * c_one3;
  pn = std::max(pn, m_Pmin);  // Use p_min threshold

  Matrix3d deviatoricStress = getDevPart(sigma);
  // Mcur = sqrt(3 J2) / p = sqrt(3/2 * s:s) / p = sqrt(3/2) * ||s|| / p
  Mcur = (pn < c_tolerance)
             ? 0.0
             : (std::sqrt(1.5) * tensorNorm(deviatoricStress) / pn);

  // Recalculate Mb based on current p and Dr
  double current_void_ratio = get_scalar(state_vars, "void_ratio");
  // Clamp void ratio to prevent extreme Dr values
  current_void_ratio =
      std::max(m_emin * 0.9, std::min(current_void_ratio, m_emax * 1.1));
  double current_Dr =
      (m_emax - current_void_ratio) / (m_emax - m_emin + c_tolerance);
  current_Dr =
      std::max(0.0, std::min(current_Dr, 1.0));  // Clamp Dr between 0 and 1

  double ksi = getKsi(current_Dr, pn);
  double Mb_current;
  if (ksi <= 0.0) {  // Dense of critical
    Mb_current = m_Mc * std::exp(-1.0 * m_nb * ksi);
  } else {  // Loose of critical
    Mb_current = m_Mc * std::exp(-1.0 * m_nb / 4.0 * ksi);
  }
  // Clamp Mb if it becomes excessively large (can happen with very dense
  // states/low p)

  double Csr =
      1.0 - Csr0 * std::min(1.0, std::pow(Mcur / (Mb_current + c_tolerance),
                                          msr));  // <<< this->m_Mb を使用
  double temp = (m_z_max < c_tolerance) ? 0.0 : (zcum / m_z_max);

  double G_base = m_G0 * m_P_atm * std::sqrt(std::max(0.1, pn / m_P_atm));
  G = G_base * Csr * (1.0 + temp) / (1.0 + temp * m_Cgd + c_tolerance);

  // Post-shake logic (assuming m_postShake flag exists, currently false)
  bool postShake = false;  // Get from state_vars if needed
  if (postShake) {
    double Md_current;
    if (ksi <= 0.0)
      Md_current = m_Mc * std::exp(m_nd * ksi);
    else
      Md_current = m_Mc * std::exp(m_nd * 4.0 * ksi);

    double p_sed =
        m_p_sedo * (zcum / (zcum + m_z_max + c_tolerance)) *
        std::pow(macauley(1.0 - Mcur / (Md_current + c_tolerance)), 0.25);
    double F_sed = std::min(
        m_Fsed_min + (1.0 - m_Fsed_min) * (pn / (20.0 * (p_sed + c_tolerance))),
        1.0);
    G = G * std::max(0.01, F_sed);
  }

  double nu = (m_nu > 0.499) ? 0.499 : m_nu;  // Clamp nu
  K = c_two3 * (1.0 + nu) / (1.0 - 2.0 * nu) * G;

  // Ensure K and G are positive and reasonable
  G = std::max(G, m_G0 * m_P_atm * 1e-6);  // Minimum shear modulus
  K = std::max(K, G * 1e-3);               // Minimum bulk modulus
}

template <unsigned Tdim>
Eigen::Matrix3d PM4Sand<Tdim>::getNormalToYield(const Matrix3d& stress,
                                                const Matrix3d& alpha,
                                                dense_map* /*state_vars*/) {
  // Implementation matching PM4Sand_claude_2.cpp logic
  double p = stress.trace() * c_one3;
  Matrix3d n;
  Matrix3d s = getDevPart(stress);

  if (p < m_Pmin * 1.1) {  // Use a slightly larger threshold for stability near
                           // p_min
                           //  double s_norm = tensorNorm(s);
                           //  if (s_norm < c_tolerance) {
    // Default direction (e.g., pure shear XY) for zero stress state
    n = Matrix3d::Zero();
    n(0, 1) = c_root12;
    n(1, 0) = c_root12;  // Consistent with PM4Sand code check
                         //  } else {
    //      n = s / s_norm; // Normal based on deviator direction
    //  }
  } else {
    Matrix3d s_eff =
        s - p * alpha;  // Effective deviator relative to yield surface center
    double normN = tensorNorm(s_eff);
    // Check if s_eff is very small relative to yield surface size (m*p)
    if (normN < c_tolerance * m_m * p) {
      normN = 1.0;
    }
    n = s_eff / normN;
  }
  return n;
}

template <unsigned Tdim>
double PM4Sand<Tdim>::getF(const Matrix3d& stress, const Matrix3d& alpha,
                           dense_map* /*state_vars*/) {
  // Implementation matching PM4Sand_claude_2.cpp logic
  Matrix3d s = getDevPart(stress);
  double p = stress.trace() * c_one3;
  p = std::max(p, m_Pmin);  // Use p_min threshold

  Matrix3d s_eff =
      s - p * alpha;  // Effective deviator relative to yield surface center
  // Yield function: f = || s - p*alpha || - sqrt(2/3) * m * p  (Note: Original
  // uses sqrt(1/2)) Let's stick to sqrt(1/2) as per original code logic check
  double f = tensorNorm(s_eff) - c_root12 * m_m * p;
  return f;
}

template <unsigned Tdim>
void PM4Sand<Tdim>::getStateDependent(
    const Matrix3d& stress, const Matrix3d& alpha, const Matrix3d& alphaIn,
    const Matrix3d& alphaInP, const Matrix3d& fabric, const Matrix3d& fabricIn,
    double& G,  // Input G
    const double& zcum, const double& zpeak,
    const double& pzp,   // Input state scalars
    const double& Mcur,  // Input Mcur (const ref OK)
    const double& dr,    // Input current Dr (const ref OK)
    Matrix3d& n, double& D, Matrix3d& R, double& K_p,       // Outputs
    Matrix3d& alphaD, double& Cka, double& h, Matrix3d& b,  // Outputs
    double& alphaAlphaBDotN, dense_map* state_vars) {       // Output

  // --- Implementation based on PM4Sand_claude_2.cpp getStateDependent ---
  // Ensure variables are accessed correctly (inputs vs outputs).

  double p = stress.trace() * c_one3;
  p = std::max(p, m_Pmin);  // Use p_min threshold

  double ksi = getKsi(dr, p);
  n = getNormalToYield(stress, alpha, state_vars);  // Output n

  // Update Mb, Md based on current state
  double Mb_current, Md_current;
  if (ksi <= 0.0) {  // Dense of critical
    Mb_current = m_Mc * std::exp(-1.0 * m_nb * ksi);
    Md_current = m_Mc * std::exp(m_nd * ksi);
  } else {  // Loose of critical
    Mb_current = m_Mc * std::exp(-1.0 * m_nb / 4.0 * ksi);
    Md_current = m_Mc * std::exp(m_nd * 4.0 * ksi);
  }
  // Clamp Mb, Md if they become excessively large/small?

  // Calculate bounding and dilatancy surfaces relative to yield surface center
  Matrix3d alphaB =
      c_root12 * (Mb_current - m_m) * n;  // Use sqrt(1/2) consistent with getF
  alphaD = c_root12 * (Md_current - m_m) * n;  // Output alphaD, use sqrt(1/2)

  // --- Intermediate calculations ---
  double zcum_safe = std::max(zcum, c_tolerance);
  double zmax_safe = std::max(m_z_max, c_tolerance);

  double Czpk1 = zpeak / (zcum_safe + zmax_safe / 5.0 + c_tolerance);
  double Czpk2 = zpeak / (zcum_safe + zmax_safe / 100.0 + c_tolerance);
  Czpk2 = std::min(Czpk2, 1.0 - c_tolerance);  // Clamp Czpk2

  double Cpzp2 = macauley(pzp - p) /
                 (macauley(pzp - p) + m_Pmin + c_tolerance);  // Use m_Pmin2
  double Cg1 = m_h0 / 200.0;
  double Ckp = 2.0;

  // Calculate distance vector to bounding surface
  b = alphaB - alpha;                 // Output b
  alphaAlphaBDotN = tensordot(b, n);  // Output alphaAlphaBDotN

  // Calculate terms related to distance from alpha_in and alpha_in_true
  Matrix3d alphaAlphaIn = alpha - alphaIn;
  double alphaAlphaInDotN = macauley(tensordot(alphaAlphaIn, n));

  Matrix3d alpha_in_true =
      get_tensor(state_vars, "alpha_in_true");  // Get from state_vars
  Matrix3d alphaAlphaInTrue = alpha - alpha_in_true;
  double alphaAlphaInTrueDotN = macauley(tensordot(alphaAlphaInTrue, n));

  // Calculate Cka
  Cka = 1.0 + m_Ckaf / (1.0 + std::pow(2.5 * alphaAlphaInTrueDotN, 2.0)) *
                  Cpzp2 * Czpk1;  // Output Cka

  // Calculate plastic hardening modulus h
  double h_denom_sqrt = std::sqrt(std::abs(alphaAlphaBDotN));
  double h_denom_exp = std::exp(alphaAlphaInDotN) - 1.0 + Cg1;
  double h_denom_czpk =
      (1.0 + Ckp * zpeak / zmax_safe * macauley(alphaAlphaBDotN) *
                 std::sqrt(std::max(0.0, 1.0 - Czpk2)));

  if (std::sqrt(std::abs(alphaAlphaBDotN)) < c_tolerance) {
    h = 1.0e10;  // Use a large number for effectively infinite hardening
  } else {
    h = 1.5 * G * m_h0 / (p * h_denom_exp * h_denom_sqrt * h_denom_czpk) * Cka;

    // Adjust based on alphaInP
    Matrix3d alphaAlphaP = alpha - alphaInP;
    if (tensordot(alphaAlphaP, n) <= 0) {
      h *=
          (alphaAlphaInDotN + Cg1) / (alphaAlphaInTrueDotN + Cg1 + c_tolerance);
    }
  }
  h = std::max(0.0, h);  // Ensure non-negative
  // Output h

  // Calculate plastic modulus K_p
  // K_p = c_two3 * h * p * tensordot(b, n); // Original has sqrt(2/3)
  // Check PM4Sand code: Kp = 2/3 * h * p * tensordot(b, n) - yes, sqrt is not
  // there.
  K_p = c_two3 * h * p * tensordot(b, n);
  K_p = std::max(0.0, K_p);  // Ensure non-negative
  // Output K_p

  // --- Calculate dilatancy D ---
  double Czin1 = macauley(
      1.0 -
      std::exp(-2.0 * std::abs(tensordot(fabricIn, n) - tensordot(fabric, n)) /
               (zmax_safe + c_tolerance)));

  // Rotated dilatancy surface calculation
  Matrix3d minusFabric = -fabric;
  double Crot1_term = macauley(tensordot(minusFabric, n));
  // double Crot1 = 1.0 + 2.0 * Crot1_term / ( sqrt(2.0)* zmax_safe +
  // c_tolerance) * (1.0 - Czin1); // 元のMPM版
  double Crot1 =
      std::max((1.0 + 2.0 * Crot1_term / (sqrt(2.0) * zmax_safe + c_tolerance) *
                          (1.0 - Czin1)),
               1.0);  // Standalone版に合わせる (max と sqrt(2.0) の位置修正)

  double Mdr =
      Md_current / (Crot1 + c_tolerance);  // Rotated dilatancy stress ratio

  // Calculate rotated dilatancy surface center
  Matrix3d alphaDr = c_root12 * (Mdr - m_m) * n;  // sqrt(1/2) here
  Matrix3d alphaDrAlpha = alphaDr - alpha;
  Matrix3d alphaD_alpha = alphaD - alpha;

  if (tensordot(alphaDrAlpha, n) <= 0) {
    // Dilation regime
    double Cpzp =
        (pzp < c_tolerance)
            ? 1.0
            : 1.0 / (1.0 + std::pow(2.5 * p / (pzp + c_tolerance), 5.0));
    double Cpmin = 1.0 / (1.0 + std::pow(m_Pmin2 / (p + c_tolerance), 2.0));
    double Czin2 =
        (1.0 + Czin1 * (zcum - zpeak) / (3.0 * zmax_safe + c_tolerance)) /
        (1.0 + 3.0 * Czin1 * (zcum - zpeak) / (3.0 * zmax_safe + c_tolerance) +
         c_tolerance);

    double zpeak_safe = std::max(zpeak, c_tolerance);
    double temp_fab =
        std::pow(macauley(1.0 - Crot1_term * c_root12 / zpeak_safe),
                 3.0);  // sqrt(1/2) here
    double Ad_denom = (std::pow(zcum_safe, 2.0) / zmax_safe) * temp_fab *
                          std::pow(m_ce, 2.0) * Cpzp * Cpmin * Czin1 +
                      1.0;
    double Ad = m_Ado * Czin2 / (Ad_denom + c_tolerance);

    D = Ad * tensordot(alphaD_alpha, n);

    // Rotation effect on dilatancy
    double Drot = Ad * Crot1_term /
                  (sqrt(2.0) * zmax_safe + c_tolerance) *  // sqrt(1/2) here
                  tensordot(alphaDrAlpha, n) / (m_Cdr + c_tolerance);

    if (D > Drot) {
      D = D + (Drot - D) * macauley(Mb_current - Mcur) /
                  (macauley(Mb_current - Mcur) + 0.01 + c_tolerance);
    }

    // Limit dilatancy at very low pressure
    if (m_Pmin <= p && p <= 2.0 * m_Pmin) {
      D = std::min(D, -3.5 * m_Ado * macauley(Mb_current - Md_current) *
                          (2.0 * m_Pmin - p) / (m_Pmin + c_tolerance));
    }
  } else {
    // Contraction regime
    double hp_exp = std::exp(-0.7 + 7.0 * std::pow(macauley(0.5 - ksi), 2.0));
    double hp = m_hpo * hp_exp;
    double Crot2 = 1.0 - Czpk2;
    // double Cdz_term1 = (1.0 - Crot2 * sqrt(2.0)  * zpeak / zmax_safe); //
    // 元のMPM版の一部 double Cdz_term2 = (zmax_safe / (zmax_safe + Crot2 * zcum
    // + c_tolerance)); double Cdz = std::max(Cdz_term1 * Cdz_term2, 1.0 / (1.0
    // + zmax_safe / 2.0 + c_tolerance)); // 元のMPM版の一部
    double Cdz =
        std::max((1.0 - Crot2 * sqrt(2.0) * zpeak / zmax_safe) *
                     (zmax_safe / (zmax_safe + Crot2 * zcum + c_tolerance)),
                 1.0 / (1.0 + zmax_safe / 2.0 +
                        c_tolerance));  // Standalone版の計算式全体を実装
    Cdz = std::max(c_tolerance, Cdz);  // Ensure positive
    double Adc = m_Ado * (1.0 + macauley(tensordot(fabric, n))) /
                 (hp * Cdz + c_tolerance);

    double Cin = 2.0 * macauley(tensordot(fabric, n)) / sqrt(2.0) /
                 (zmax_safe + c_tolerance);

    double D_term1 = Adc * std::pow(tensordot(alphaAlphaIn, n) + Cin, 2.0);
    double D_denom = tensordot(alphaD_alpha, n) + 0.16 + c_tolerance;
    D = std::min(D_term1, 1.5 * m_Ado) * tensordot(alphaD_alpha, n) / D_denom;

    // Apply factor for small p
    double C_pmin2;
    if (p < m_Pmin * 2.0) {
      C_pmin2 = 0.0;
    } else if (p >= m_Pmin * 18.0) {
      C_pmin2 = 1.0;
    } else {
      C_pmin2 = (p - m_Pmin * 2.0) / (16.0 * m_Pmin + c_tolerance);
    }
    D *= C_pmin2;
  }
  // Output D

  // Calculate flow rule tensor R
  R = n + c_one3 * D * m_I;  // Output R
}

template <unsigned Tdim>
void PM4Sand<Tdim>::stressCorrection(
    Matrix3d& nextStress, Matrix3d& nextAlpha, const Matrix3d& stress_prev,
    const Matrix3d& alphaIn, const Matrix3d& alphaInP,
    const Matrix3d& curFabric, double& nextVoidRatio, dense_map* state_vars) {
  Matrix3d dSigmaP, dfrOverdSigma, dfrOverdAlpha, n_iter, R_iter, alphaD_iter,
      b_iter, r_iter, aBar;
  Matrix3d nAlpha = nextAlpha;  // Use local copies for iteration
  Matrix3d nStress = nextStress;
  double lambda, D_iter, Kp_iter, Cka_iter, h_iter, p_iter, fr,
      alphaAlphaBDotN_iter;
  double K_iter, G_iter, Mcur_iter;  // Elastic moduli

  // Check for small p (Function 2 logic)
  p_iter = nStress.trace() * c_one3;
  fr = getF(nStress, nAlpha,
            state_vars);  // Calculate fr regardless of p initially

  if (p_iter < m_Pmin / 5.0) {
    // this->console_->debug("Material {}: Stress correction initial p < p_min/5
    // ({:.2e})", this->id_, p_iter);
    if (fr < c_tolerance) {
      // Stress state inside yield surface or on it - adjust p only
      nextStress = getDevPart(nStress) +
                   (m_Pmin / 5.0 - p_iter) * m_I;  // Function 2 direct addition
      // Alternative: preserve deviatoric part
      // Matrix3d s_curr = getDevPart(nStress);
      // nextStress = s_curr + (m_Pmin / 5.0) * m_I;
      // this->console_->debug("Material {}: Stress adjusted to p_min/5 while
      // inside yield.", this->id_);
      nextStress = nStress;
      nextAlpha = nAlpha;  // nAlpha はこの分岐では変更されていない
      return;
    } else {
      // Stress state outside yield surface - reset (Function 2 specific reset)
      // this->console_->debug("Material {}: Stress reset to specific state due
      // to p < p_min/5 and outside yield.", this->id_);
      nextStress = (m_Pmin / 5.0) * m_I;
      // Function 2 specific shear component reset (might need review for
      // general cases)
      if (Tdim > 1) {  // Apply only if 2D or 3D
        nextStress(0, 1) = nextStress(1, 0) = 0.8 * m_Mc * m_Pmin / 5.0;
      }
      nextAlpha.setZero();
      if (Tdim > 1) {  // Apply only if 2D or 3D
        nextAlpha(0, 1) = nextAlpha(1, 0) = 0.8 * m_Mc;
      }
      nextStress = nStress;
      nextAlpha = nAlpha;
      return;  // Exit after reset
    }
  } else {
    // p >= p_min / 5.0
    if (fr < c_tolerance) {

      return;
    }
  }

  // --- Correction Loop (Newton-Raphson adapted from Function 2) ---
  double current_Dr =
      (m_emax - nextVoidRatio) / (m_emax - m_emin + c_tolerance);
  current_Dr = std::max(0.0, std::min(current_Dr, 1.0));

  double dGamma_step = get_scalar(state_vars, "dGamma");
  // Function 2 doesn't explicitly handle dGamma <= 0 here, assuming it's
  // positive from the integration step. If dGamma_step could be zero/negative,
  // add handling if necessary.
  if (dGamma_step <= 0) {
    // Option: Use small value like Function 1, or return/throw debug if invalid
    // state this->console_->debug("Material {}: Stress correction called but
    // dGamma<=0. Correction may be ineffective.", this->id_);
    dGamma_step = c_tolerance * 10.0;  // Fallback similar to Function 1
  }

  const int maxIter = 25;
  bool converged = false;
  Matrix3d nStress_newton_start =
      nStress;  // Newton-Raphson開始時の応力（二分法の参照用）
  Matrix3d nAlpha_newton_start = nAlpha;  // Newton-Raphson開始時のアルファ
  for (int i = 1; i <= maxIter; i++) {
    p_iter = nStress.trace() * c_one3;
    p_iter = std::max(p_iter, m_Pmin);  // Ensure p >= p_min

    // Calculate stress ratio tensor r (Function 2 logic)
    r_iter = getDevPart(nStress) /
             p_iter;  // Division by p_iter (already max(p, p_min))
    //  r_iter = (std::abs(p_iter) < c_tolerance) ? Matrix3d::Zero() :
    //  getDevPart(nStress) / p_iter;

    // Get state dependent parameters (using Function 1's helpers)
    double zcum_iter = get_scalar(state_vars, "zcum");
    double zpeak_iter = get_scalar(state_vars, "zpeak");
    double pzp_iter = get_scalar(state_vars, "pzp");
    Matrix3d fabric_in_iter = get_tensor(state_vars, "fabric_in");

    getElasticModuli(nStress, K_iter, G_iter, Mcur_iter, zcum_iter, state_vars);

    getStateDependent(nStress, nAlpha, alphaIn, alphaInP, curFabric,
                      fabric_in_iter, G_iter, zcum_iter, zpeak_iter, pzp_iter,
                      Mcur_iter, current_Dr, n_iter, D_iter, R_iter, Kp_iter,
                      alphaD_iter, Cka_iter, h_iter, b_iter,
                      alphaAlphaBDotN_iter, state_vars);

    // Calculate terms for correction (Function 2 structure)
    // NOTE: Function 2 used m_dGamma implicitly. We use dGamma_step from
    // state_vars.
    if (dGamma_step >
        c_tolerance *
            1e-2) {  // dGammaが意味のある大きさの場合のみdSigmaPを計算
      dSigmaP =
          dGamma_step *
          (2.0 * G_iter * n_iter +
           K_iter * D_iter *
               m_I);  // R_iterではなくn_iterを使用するスタンドアローン版のdSigmaPの計算方法に近づけるか検討が必要。PM4Sand.cpp(OpenSees)ではRを使っている。ここでは元のMPM版のR_iterのままとする。
      // dSigmaP = dGamma_step * (2.0 * G_iter * R_iter + K_iter * D_iter *
      // m_I); // 元のMPM版
    } else {
      dSigmaP = Matrix3d::
          Zero();  // dGammaが小さすぎる場合は塑性的な応力変化は小さいと見なす
    }
    // dSigmaP = dGamma_step * (2.0 * G_iter * R_iter + K_iter * D_iter * m_I);
    // // R_iter used in Func2 dSigmaP calc
    aBar = (c_two3 * h_iter * b_iter);  // Same as Function 1 basically

    dfrOverdSigma =
        n_iter - 0.5 * tensordot(n_iter, r_iter) * m_I;  // Function 2 gradient
    dfrOverdAlpha = -p_iter * n_iter;                    // Function 2 gradient

    // Calculate lambda (Function 2 approach - no stabilization)
    double denom_lambda =
        tensordot(dfrOverdSigma, dSigmaP) - tensordot(dfrOverdAlpha, aBar);

    if (std::abs(denom_lambda) <
        c_tolerance * 1e-5) {  // 分母がゼロに近い場合の処理
      // this->console_->debug("Material {}: Stress correction lambda
      // denominator near zero ({:.2e}). Using alternative lambda. Iter: {}",
      // this->id_, denom_lambda, i);
      double lambda_alt_denom = tensordot(dfrOverdSigma, dfrOverdSigma);
      if (std::abs(lambda_alt_denom) < c_tolerance * 1e-5) {
        // this->console_->warn("Material {}: Stress correction alternative
        // lambda denominator also near zero. Aborting NR. Iter: {}", this->id_,
        // i);
        break;  // Newton-Raphsonループを抜ける -> 二分法へ
      }
      lambda = fr / lambda_alt_denom;
      nStress = nStress - lambda * dfrOverdSigma;  // 応力のみ更新
      // nAlpha はこのケースでは更新しない (スタンドアローン版のロジック参考)
    } else {
      lambda = fr / denom_lambda;
      Matrix3d nStress_trial = nStress - lambda * dSigmaP;
      Matrix3d nAlpha_trial = nAlpha + lambda * aBar;
      double fr_trial = getF(nStress_trial, nAlpha_trial, state_vars);
      if (std::abs(fr_trial) < std::abs(fr)) {
        nStress = nStress_trial;
        nAlpha = nAlpha_trial;
      } else {
        double lambda_alt_denom = tensordot(dfrOverdSigma, dfrOverdSigma);
        if (std::abs(lambda_alt_denom) < c_tolerance * 1e-5) {
          // this->console_->warn("Material {}: Stress correction alternative
          // lambda denominator also near zero. Aborting NR. Iter: {}",
          // this->id_, i);
          break;
        }
        lambda = fr / lambda_alt_denom;
        nStress = nStress - lambda * dfrOverdSigma;
      }
    }

    // Recalculate yield function value for convergence check
    fr = getF(nStress, nAlpha, state_vars);
    if (std::abs(fr) < c_tolerance) {  // Use Function 2 tolerance
      // this->console_->debug("Material {}: Stress correction converged in {}
      // iterations (F={:.2e}).", this->id_, i, fr);
      converged = true;
      nextStress = nStress;
      nextAlpha = nAlpha;
      return;  // Exit loop on convergence
    }

    // Update p_iter for next iteration's r_iter calculation (though loop might
    // end here) p_iter = std::max(c_one3 * nStress.trace(), m_Pmin); //
    // Redundant? Already calculated at loop start.

  }  // End Newton-Raphson correction loop

  // --- Bisection Search if Newton-Raphson didn't converge (Function 2 logic)
  // ---
  if (!converged) {
    Matrix3d dSigma_bisection =
        nStress_newton_start -
        stress_prev;  // Difference between initial state needing correction and
                      // pre-increment state
    double alpha_L = 0.0;  // 補正なし（つまり stress_before_increment + 0.0 *
                           // dSigma_bisection = stress_before_increment）
    double alpha_R = 1.0;  // 完全な増分（つまり stress_before_increment + 1.0 *
                           // dSigma_bisection = nStress_newton_start）
    double fr_L = getF(stress_prev + alpha_L * dSigma_bisection,
                       nAlpha_newton_start, state_vars);
    double fr_R = getF(stress_prev + alpha_R * dSigma_bisection,
                       nAlpha_newton_start, state_vars);
    if (fr_L * fr_R > 0 &&
        std::abs(fr_R) >
            c_tolerance) {  // 解を挟んでいない、かつ右端がすでに降伏面を超えている
      // this->console_->warn("Material {}: Bisection range does not bracket the
      // root initially (fL={:.2e}, fR={:.2e}). Check bisection reference
      // states.", this->id_, fr_L, fr_R);
      // この場合、二分法は効果がない可能性が高い。NRの結果を採用するか、エラーとするか。
      // スタンドアローン版では、この状況でも二分法に進んでいる。
    }

    const int maxBisectionIter = maxIter;
    for (int jj = 0; jj < maxBisectionIter; jj++) {
      double alpha_mid = 0.5 * (alpha_L + alpha_R);
      Matrix3d stress_mid_bisection =
          stress_prev + alpha_mid * dSigma_bisection;
      double fr_mid =
          getF(stress_mid_bisection, nAlpha_newton_start, state_vars);

      if (std::abs(fr_mid) < c_tolerance) {
        nStress = stress_mid_bisection;
        nAlpha = nAlpha_newton_start;  // αは二分法では変更しないのが一般的
        converged = true;
        this->console_->debug(
            "Material {}: Bisection converged in {} iterations (F_mid={:.2e}).",
            this->id_, jj + 1, fr_mid);
        break;
      }
      if (fr_mid < 0.0) {  // 降伏曲面の内側に来すぎた -> 増分を増やす方向に
        alpha_L = alpha_mid;
      } else {  // 降伏曲面の外側 -> 増分を減らす方向に
        alpha_R = alpha_mid;
      }

      if ((alpha_R - alpha_L) < 1e-6) {  // 収束幅が非常に小さい
        this->console_->debug(
            "Material {}: Bisection interval too small at iteration {}. "
            "F_mid={:.2e}",
            this->id_, jj + 1, fr_mid);
        // この時点で stress_mid_bisection を採用することも考えられる
        nStress = stress_mid_bisection;
        nAlpha = nAlpha_newton_start;
        break;
      }
    }
    if (!converged) {
    }
  }  // End if (!converged) for bisection

  // Final assignment
  nextStress = nStress;
  nextAlpha = nAlpha;
  // 最終チェック：平均主応力が負でないか、m_Pminを下回っていないか
  p_iter = nextStress.trace() * c_one3;
  if (p_iter < m_Pmin) {

    Matrix3d s_final = getDevPart(nextStress);
    nextStress = s_final + m_Pmin * m_I;
  }
  if (nextStress.trace() * c_one3 < 0) {
  }
}

// --- Voigt Identity Helper ---
template <unsigned Tdim>
typename PM4Sand<Tdim>::Vector6d PM4Sand<Tdim>::voigtI() const {
  Vector6d I_voigt;
  I_voigt << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
  return I_voigt;
}

// Explicit instantiation for 2D and 3D to avoid linking debugs if used
// elsewhere If the class is only used with Tdim=2, only instantiate for 2. If
// the framework might instantiate with Tdim=3, include 3 as well.

}  // namespace mpm