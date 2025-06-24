//! Constructor with material properties
template <unsigned Tdim>
mpm::Terracotta<Tdim>::Terracotta(unsigned id, const Json& material_properties)
    : Material<Tdim>(id, material_properties) {
  try {
    // General parameters
    // Density
    double density = material_properties.at("density").template get<double>();

    // Initial Packing Fraction
    initial_packing_fraction_ =
        material_properties.at("packing_fraction").template get<double>();

    // Solid grain density
    grain_density_ = density / initial_packing_fraction_;

    // Bulk modulus
    bulk_modulus_ =
        material_properties.at("bulk_modulus").template get<double>();
    // Poisson ratio
    shear_modulus_ =
        material_properties.at("shear_modulus").template get<double>();

    // Parameter lambda
    lambda_ = material_properties.at("lambda").template get<double>();
    // Reference pressure
    p1_ = material_properties.at("reference_pressure").template get<double>();
    // Parameter alpha
    alpha_ = material_properties.at("alpha").template get<double>();
    // Parameter beta
    beta_ = material_properties.at("beta").template get<double>();
    // Parameter gamma
    gamma_ = material_properties.at("gamma").template get<double>();
    // Parameter eta
    eta_ = material_properties.at("eta").template get<double>();
    // Parameter omega
    omega_ = material_properties.at("omega").template get<double>();
    // Critical state ratio
    m_ = material_properties.at("m").template get<double>();
    // Intial meso-scale temperature
    initial_tm_ =
        material_properties.at("meso_temperature").template get<double>();

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

  } catch (Json::exception& except) {
    console_->error("Material parameter not set: {} {}\n", except.what(),
                    except.id);
  }
}

//! Initialise history variables
template <unsigned Tdim>
mpm::dense_map mpm::Terracotta<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {// Terracotta parameters
                               // Pressure
                               {"pressure", 0.0},
                               // Deviatoric stress
                               {"q", 0.0},
                               // Tm
                               {"tm", initial_tm_},
                               // Elastic strain components
                               {"elastic_strain0", 0.},
                               {"elastic_strain1", 0.},
                               {"elastic_strain2", 0.},
                               {"elastic_strain3", 0.},
                               {"elastic_strain4", 0.},
                               {"elastic_strain5", 0.},
                               // Packing fraction
                               {"packing_fraction", initial_packing_fraction_}};
  return state_vars;
}

//! State variables
template <unsigned Tdim>
std::vector<std::string> mpm::Terracotta<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {"pressure",
                                               "q",
                                               "tm",
                                               "elastic_strain0",
                                               "elastic_strain1",
                                               "elastic_strain2",
                                               "elastic_strain3",
                                               "elastic_strain4",
                                               "elastic_strain5",
                                               "packing_fraction"};
  return state_vars;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::Terracotta<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {

  // Density and packing parameters
  const double current_packing_density = ptr->mass_density();
  const double current_packing_fraction =
      current_packing_density / grain_density_;

  // Get strain rate
  auto strain_rate = ptr->strain_rate();
  // Convert strain rate to rate of deformation tensor
  strain_rate.tail(3) *= 0.5;

  // Strain decomposition (vol strain rate is positive in compression)
  Eigen::Matrix<double, 6, 1> strain_rate_dev = strain_rate;
  const double vol_strain_rate = -strain_rate.head(3).sum();
  strain_rate_dev.head(3).noalias() +=
      (1.0 / 3.0) * Eigen::Vector3d::Constant(vol_strain_rate);
  const double dev_strain_rate =
      std::sqrt(2.0 / 3.0 *
                (strain_rate_dev.dot(strain_rate_dev) +
                 strain_rate_dev.tail(3).dot(strain_rate_dev.tail(3))));
  Eigen::Matrix<double, 6, 1> n_d = Eigen::Matrix<double, 6, 1>::Zero();
  if (dev_strain_rate > std::numeric_limits<double>::epsilon())
    n_d = strain_rate_dev / (std::sqrt(3.0 / 2.0) * dev_strain_rate);

  // Compute meso-scale temperature
  const double current_tm = (*state_vars).at("tm");
  double new_tm = current_tm;

  // Start Newton-Raphson iteration for meso-scale temperature
  unsigned iter = 0;
  double initial_rt;
  double rt_m, jac_rt_m;

  while (iter < max_iter_) {
    const double A = alpha_ * std::pow(vol_strain_rate, 2) +
                     beta_ * std::pow(dev_strain_rate, 2);
    const double tm_rate = A - eta_ * std::pow(new_tm, 2);

    rt_m = new_tm - current_tm - dt * (tm_rate);

    // Check convergence based on residual norm
    if (iter == 0)
      initial_rt = rt_m;
    else {
      if (rt_m / initial_rt < rel_tol_) break;
    }

    // Compute Jacobian
    jac_rt_m = 1.0 + dt * (2.0 * eta_ * new_tm);

    // Update meso-scale temperature
    const double delta_tm = -rt_m / jac_rt_m;
    new_tm += delta_tm;

    // Increment iteration counter
    iter++;
  }

  // Current elastic strain tensor
  Eigen::Matrix<double, 6, 1> current_elastic_strain;
  current_elastic_strain << (*state_vars).at("elastic_strain0"),
      (*state_vars).at("elastic_strain1"), (*state_vars).at("elastic_strain2"),
      (*state_vars).at("elastic_strain3"), (*state_vars).at("elastic_strain4"),
      (*state_vars).at("elastic_strain5");
  const double current_vol_elastic_strain =
      -current_elastic_strain.head(3).sum();
  Eigen::Matrix<double, 6, 1> current_elastic_strain_dev =
      current_elastic_strain;
  current_elastic_strain_dev.head(3).noalias() +=
      (1.0 / 3.0) * Eigen::Vector3d::Constant(current_vol_elastic_strain);
  const double current_dev_elastic_strain =
      std::sqrt(2.0 / 3.0 *
                (current_elastic_strain_dev.dot(current_elastic_strain_dev) +
                 current_elastic_strain_dev.tail(3).dot(
                     current_elastic_strain_dev.tail(3))));

  // Initialize new elastic strain tensor and its rate
  double new_vol_elastic_strain = current_vol_elastic_strain;
  double new_dev_elastic_strain = current_dev_elastic_strain;

  // Set constant parameters
  const double xi = new_tm / p1_ / std::pow(current_packing_fraction, lambda_);
  const double a = std::sqrt(eta_ / alpha_);
  const double c1 = std::sqrt(eta_ / beta_) / m_ / omega_;
  const double phi_6 = std::pow(current_packing_fraction, 6);

  // Initialize pe and qe
  double pe_m = phi_6 / 2.0 *
                (bulk_modulus_ * std::pow(new_vol_elastic_strain, 2) +
                 3.0 * shear_modulus_ * std::pow(new_dev_elastic_strain, 2));
  double qe_m = 3.0 * shear_modulus_ * phi_6 * new_vol_elastic_strain *
                new_dev_elastic_strain;
  if (pe_m < tolerance_) pe_m = tolerance_;

  // Start Newton-Raphson iteration for elastic strain
  iter = 0;
  double initial_res_norm;
  Eigen::Matrix<double, 2, 1> res_m;
  Eigen::Matrix<double, 2, 2> jac_m;
  while (iter < max_iter_) {
    // Compute residuals
    res_m(0) =
        new_vol_elastic_strain - current_vol_elastic_strain -
        vol_strain_rate * dt +
        xi * dt * a * (pe_m - std::pow(qe_m, 2) / pe_m / std::pow(m_, 2));
    res_m(1) = new_dev_elastic_strain - current_dev_elastic_strain -
               dev_strain_rate * dt + xi * dt * c1 * qe_m;

    // Check convergence based on residual norm
    if (res_m.norm() < abs_tol_) break;
    if (iter == 0)
      initial_res_norm = res_m.norm();
    else {
      if (res_m.norm() / initial_res_norm < rel_tol_) break;
    }

    // Compute Jacobian
    jac_m(0, 0) =
        1.0 +
        xi * dt * a *
            (phi_6 * bulk_modulus_ * new_vol_elastic_strain -
             phi_6 / std::pow(m_, 2) *
                 (6.0 * qe_m / pe_m * shear_modulus_ * new_dev_elastic_strain -
                  std::pow(qe_m / pe_m, 2) * bulk_modulus_ *
                      new_vol_elastic_strain));
    jac_m(0, 1) =
        3.0 * xi * dt * a * phi_6 * shear_modulus_ * new_dev_elastic_strain -
        xi * dt * a * phi_6 / std::pow(m_, 2) *
            (6.0 * qe_m / pe_m * shear_modulus_ * new_vol_elastic_strain -
             3.0 * std::pow(qe_m / pe_m, 2) * shear_modulus_ *
                 new_dev_elastic_strain);
    jac_m(1, 0) =
        3.0 * xi * dt * c1 * phi_6 * shear_modulus_ * new_dev_elastic_strain;
    jac_m(1, 1) = 1.0 + 3.0 * xi * dt * c1 * phi_6 * shear_modulus_ *
                            new_vol_elastic_strain;

    // Update elastic strain
    const Eigen::Matrix<double, 2, 1> delta_elastic_strain =
        jac_m.inverse() * (-res_m);
    new_vol_elastic_strain += delta_elastic_strain(0);
    new_dev_elastic_strain += delta_elastic_strain(1);

    // Update pe and qe
    pe_m = phi_6 / 2.0 *
           (bulk_modulus_ * std::pow(new_vol_elastic_strain, 2) +
            3.0 * shear_modulus_ * std::pow(new_dev_elastic_strain, 2));
    qe_m = 3.0 * shear_modulus_ * phi_6 * new_vol_elastic_strain *
           new_dev_elastic_strain;

    // Check convergence based on solution
    if (delta_elastic_strain.norm() < abs_tol_) break;

    // Increment iteration counter
    iter++;
  }

  // Identity in voigt
  Vector6d m_voigt;
  m_voigt << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

  // Construct new elastic strain rate tensor
  double new_vol_elastic_strain_rate =
      (new_vol_elastic_strain - current_vol_elastic_strain) / dt;
  double new_dev_elastic_strain_rate =
      (new_dev_elastic_strain - current_dev_elastic_strain) / dt;
  const Eigen::Matrix<double, 6, 1> new_elastic_strain_rate =
      -1.0 / 3.0 * new_vol_elastic_strain_rate * m_voigt +
      std::sqrt(3.0 / 2.0) * new_dev_elastic_strain_rate * n_d;

  // Compute new elastic strain
  Eigen::Matrix<double, 6, 1> new_elastic_strain =
      current_elastic_strain + new_elastic_strain_rate * dt;
  new_vol_elastic_strain = -new_elastic_strain.head(3).sum();
  Eigen::Matrix<double, 6, 1> new_elastic_strain_dev = new_elastic_strain;
  new_elastic_strain_dev.head(3).noalias() +=
      (1.0 / 3.0) * Eigen::Vector3d::Constant(new_vol_elastic_strain);
  new_dev_elastic_strain = std::sqrt(
      2.0 / 3.0 *
      (new_elastic_strain_dev.dot(new_elastic_strain_dev) +
       new_elastic_strain_dev.tail(3).dot(new_elastic_strain_dev.tail(3))));
  Eigen::Matrix<double, 6, 1> n_e = Eigen::Matrix<double, 6, 1>::Zero();
  if (new_dev_elastic_strain > std::numeric_limits<double>::epsilon())
    n_e = new_elastic_strain_dev /
          (std::sqrt(3.0 / 2.0) * new_dev_elastic_strain);

  // Compute new stress invariants
  const double pe =
      phi_6 / 2.0 *
      (bulk_modulus_ * std::pow(new_vol_elastic_strain, 2) +
       3.0 * shear_modulus_ * std::pow(new_dev_elastic_strain, 2));
  const double pd = 2.0 * alpha_ / gamma_ * new_tm * vol_strain_rate;
  const double pt = new_tm * new_tm / gamma_;
  const double p_new = pe + pd + pt;

  const double qe = 3.0 * shear_modulus_ * phi_6 * new_vol_elastic_strain *
                    new_dev_elastic_strain;
  const double qd = 2.0 * beta_ / gamma_ * new_tm * dev_strain_rate;
  const double q_new = qe + qd;

  // Compute new director
  Eigen::Matrix<double, 6, 1> n = Eigen::Matrix<double, 6, 1>::Zero();
  if (q_new > std::numeric_limits<double>::epsilon())
    n = (qe * n_e + qd * n_d) / (std::sqrt(2.0 / 3.0) * q_new);

  // Update stress
  const Eigen::Matrix<double, 6, 1> updated_stress =
      -p_new * m_voigt + std::sqrt(2.0 / 3.0) * q_new * n;

  // Update state variables
  (*state_vars).at("pressure") = p_new;
  (*state_vars).at("q") = q_new;
  (*state_vars).at("tm") = new_tm;
  (*state_vars).at("elastic_strain0") = new_elastic_strain(0);
  (*state_vars).at("elastic_strain1") = new_elastic_strain(1);
  (*state_vars).at("elastic_strain2") = new_elastic_strain(2);
  (*state_vars).at("elastic_strain3") = new_elastic_strain(3);
  (*state_vars).at("elastic_strain4") = new_elastic_strain(4);
  (*state_vars).at("elastic_strain5") = new_elastic_strain(5);
  (*state_vars).at("packing_fraction") = current_packing_fraction;

  return updated_stress;
}