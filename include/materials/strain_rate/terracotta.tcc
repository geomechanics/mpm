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

    // Minimum packing fraction
    minimum_packing_fraction_ =
        material_properties.at("packing_fraction_minimum")
            .template get<double>();

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
  double current_packing_fraction = current_packing_density / grain_density_;

  // TODO: Packing fraction calculation may not be accurate for large
  // deformation, so we set a maximum value to avoid overpacking. It is crucial
  // for this model since phi will be used in computing the stiffness
  // current_packing_fraction = std::min(current_packing_fraction, 1.0);

  // Get strain rate
  auto strain_rate = ptr->strain_rate();
  // Convert strain rate to rate of deformation tensor
  strain_rate.tail(3) *= 0.5;
  // Convert strain rate to Mandel's notation
  strain_rate.tail(3) *= std::sqrt(2.0);

  // Strain decomposition (vol strain rate is positive in compression)
  Vector6d strain_rate_dev = strain_rate;
  const double vol_strain_rate = -strain_rate.head(3).sum();
  strain_rate_dev.head(3).noalias() +=
      (1.0 / 3.0) * Eigen::Vector3d::Constant(vol_strain_rate);
  const double dev_strain_rate =
      std::sqrt(2.0 / 3.0 * strain_rate_dev.dot(strain_rate_dev));
  Vector6d n_d = Vector6d::Zero();
  if (dev_strain_rate > std::numeric_limits<double>::epsilon())
    n_d = strain_rate_dev / (std::sqrt(3.0 / 2.0) * dev_strain_rate);

  // Switch for granular separation
  bool separation_state = false;
  double separation_multiplier = 1.0;
  if (current_packing_fraction < minimum_packing_fraction_) {
    separation_state = true;
    separation_multiplier = 0.0;
  }

  // Compute meso-scale temperature
  const double current_tm = (*state_vars).at("tm");
  double new_tm = current_tm;

  // Start Newton-Raphson iteration for meso-scale temperature
  unsigned iter = 0;
  double initial_rt;
  double rt_m, jac_rt_m;
  const double A = (alpha_ * std::pow(vol_strain_rate, 2) +
                    beta_ * std::pow(dev_strain_rate, 2)) *
                   separation_multiplier;

  while (iter < max_iter_) {
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

  // Update stress and state variables
  Vector6d new_elastic_strain = Vector6d::Zero();
  Vector6d updated_stress = Vector6d::Zero();
  // Second order identity tensor in Mandel's notation
  Vector6d m_mandel;
  m_mandel << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;
  if (separation_state) {
    const double pt = new_tm * new_tm / gamma_;
    const double new_p = pt;

    // Update stress
    updated_stress = -new_p * m_mandel;

    (*state_vars).at("pressure") = new_p;
    (*state_vars).at("q") = 0.0;
    new_elastic_strain.setZero();
  } else {
    // Current elastic strain tensor in tensorial Voigt notation
    Vector6d current_elastic_strain;
    current_elastic_strain << (*state_vars).at("elastic_strain0"),
        (*state_vars).at("elastic_strain1"),
        (*state_vars).at("elastic_strain2"),
        (*state_vars).at("elastic_strain3"),
        (*state_vars).at("elastic_strain4"),
        (*state_vars).at("elastic_strain5");
    // Convert to Mandel's notation
    current_elastic_strain.tail(3) *= std::sqrt(2.0);
    const double current_vol_elastic_strain =
        -current_elastic_strain.head(3).sum();
    Vector6d current_elastic_strain_dev = current_elastic_strain;
    current_elastic_strain_dev.head(3).noalias() +=
        (1.0 / 3.0) * Eigen::Vector3d::Constant(current_vol_elastic_strain);
    const double current_dev_elastic_strain = std::sqrt(
        2.0 / 3.0 * current_elastic_strain_dev.dot(current_elastic_strain_dev));

    // Initialize new elastic strain tensor and its rate
    double new_vol_elastic_strain = current_vol_elastic_strain;
    double new_dev_elastic_strain = current_dev_elastic_strain;
    Vector6d new_elastic_strain_dev = current_elastic_strain_dev;

    // Forth order identity tensor in tensorial Mandel's notation
    Matrix6x6 fourth_order_identity_mandel = Matrix6x6::Zero();
    for (unsigned i = 0; i < 6; ++i) fourth_order_identity_mandel(i, i) = 1.0;

    // Initialize pe and se
    const double phi_6 = std::pow(current_packing_fraction, 6);
    double pe_m =
        phi_6 / 2.0 *
        (bulk_modulus_ * std::pow(this->macaulay(new_vol_elastic_strain), 2) +
         3.0 * shear_modulus_ * this->heaviside(new_vol_elastic_strain) *
             std::pow(new_dev_elastic_strain, 2));
    Vector6d se_m = 2.0 * shear_modulus_ * phi_6 *
                    this->macaulay(new_vol_elastic_strain) *
                    new_elastic_strain_dev;

    // To avoid division by zero in the next step
    if (pe_m < tolerance_) pe_m = tolerance_;

    // Initialize transport parameters (a and c are constants, while b changing
    // over iterations)
    const double a = std::sqrt(eta_ / alpha_) * (1 - current_packing_fraction) /
                     p1_ / std::pow(current_packing_fraction, lambda_);
    Vector6d b_m = -3. / 2. * a / m_ / m_ / pe_m * se_m;
    Matrix6x6 c =
        3. / 2. *
        (std::sqrt(eta_ / beta_) * (1 - current_packing_fraction) / m_ /
             omega_ / p1_ / std::pow(current_packing_fraction, lambda_) +
         a / m_ / m_) *
        fourth_order_identity_mandel;

    // Start Newton-Raphson iteration for elastic strain
    iter = 0;
    double initial_res_norm;
    Eigen::Matrix<double, 7, 1> res_m;
    Eigen::Matrix<double, 7, 7> jac_m;
    while (iter < max_iter_) {
      // Compute residuals
      res_m(0) = new_vol_elastic_strain - current_vol_elastic_strain -
                 vol_strain_rate * dt +
                 new_tm * dt * (a * pe_m + b_m.dot(se_m));
      res_m.tail(6) = new_elastic_strain_dev - current_elastic_strain_dev -
                      dt * (strain_rate_dev - new_tm * (b_m * pe_m + c * se_m));

      // Check convergence based on residual norm
      if (res_m.norm() < abs_tol_) break;
      if (iter == 0)
        initial_res_norm = res_m.norm();
      else {
        if (res_m.norm() / initial_res_norm < rel_tol_) break;
      }

      // Compute necessary derivatives
      const double dp_depsv =
          phi_6 * bulk_modulus_ * this->macaulay(new_vol_elastic_strain);
      const Vector6d dp_dgamma = 2.0 * phi_6 * shear_modulus_ *
                                 this->heaviside(new_vol_elastic_strain) *
                                 new_elastic_strain_dev;
      const Vector6d ds_depsv = dp_dgamma;
      const Matrix6x6 ds_dgamma = 2.0 * phi_6 * shear_modulus_ *
                                  this->macaulay(new_vol_elastic_strain) *
                                  fourth_order_identity_mandel;

      // Compute derivatives of transport variables b with respect to stress
      const Vector6d db_dp = 3. / 2. * a / m_ / m_ / pe_m / pe_m * se_m;
      const Matrix6x6 db_ds =
          -3. / 2. * a / m_ / m_ / pe_m * fourth_order_identity_mandel;

      // Compute derivatives of transport variables b with respect to elastic
      // strain
      const Vector6d db_depsv = db_dp * dp_depsv + db_ds * ds_depsv;
      const Matrix6x6 db_dgamma =
          db_dp * dp_dgamma.transpose() + db_ds * ds_dgamma;

      // Compute residual derivatives
      const double drv_epsv =
          1.0 +
          dt * new_tm * (a * dp_depsv + db_depsv.dot(se_m) + b_m.dot(ds_depsv));
      const Vector6d drv_dgamma = dt * new_tm *
                                  (a * dp_dgamma + db_dgamma * se_m +
                                   (b_m.transpose() * ds_dgamma).transpose());
      const Vector6d drs_depsv =
          dt * new_tm * (pe_m * db_depsv + dp_depsv * b_m + c * ds_depsv);
      const Matrix6x6 drs_dgamma =
          fourth_order_identity_mandel +
          dt * new_tm *
              (pe_m * db_dgamma + b_m * dp_dgamma.transpose() + c * ds_dgamma);

      // Construct Jacobian matrix
      jac_m.setZero();
      jac_m(0, 0) = drv_epsv;
      jac_m.block(0, 1, 1, 6) = drv_dgamma.transpose();
      jac_m.block(1, 0, 6, 1) = drs_depsv;
      jac_m.block(1, 1, 6, 6) = drs_dgamma;

      // Update elastic strain
      const Eigen::Matrix<double, 7, 1> delta_elastic_strain =
          jac_m.inverse() * (-res_m);
      new_vol_elastic_strain += delta_elastic_strain(0);
      new_elastic_strain_dev.noalias() += delta_elastic_strain.tail(6);
      new_dev_elastic_strain = std::sqrt(
          2.0 / 3.0 * new_elastic_strain_dev.dot(new_elastic_strain_dev));

      // Update pe and qe
      pe_m =
          phi_6 / 2.0 *
          (bulk_modulus_ * std::pow(this->macaulay(new_vol_elastic_strain), 2) +
           3.0 * shear_modulus_ * this->heaviside(new_vol_elastic_strain) *
               std::pow(new_dev_elastic_strain, 2));
      se_m = 2.0 * shear_modulus_ * phi_6 *
             this->macaulay(new_vol_elastic_strain) * new_elastic_strain_dev;

      // To avoid division by zero in the next step
      if (pe_m < tolerance_) pe_m = tolerance_;

      // Update transport parameters
      b_m = -3. / 2. * a / m_ / m_ / pe_m * se_m;

      // Check convergence based on solution
      if (delta_elastic_strain.norm() < abs_tol_) break;

      // Increment iteration counter
      iter++;
    }

    // Compute new elastic strain
    new_elastic_strain =
        -new_vol_elastic_strain / 3.0 * m_mandel + new_elastic_strain_dev;

    // Compute new stress invariants
    const double pe = pe_m;
    const double pd = 2.0 * alpha_ / gamma_ * new_tm * vol_strain_rate;
    const double pt = new_tm * new_tm / gamma_;
    const double new_p = pe + pd + pt;

    const Vector6d se = se_m;
    const Vector6d sd = 4. / 3. * beta_ / gamma_ * new_tm * strain_rate_dev;
    const Vector6d new_s = se + sd;
    const double new_q = std::sqrt(3.0 / 2.0 * new_s.dot(new_s));

    // Update stress
    updated_stress = -new_p * m_mandel + new_s;

    // Convert Mandel's notation to tensorial Voigt notation
    updated_stress.tail(3) /= std::sqrt(2.0);
    new_elastic_strain.tail(3) /= std::sqrt(2.0);

    (*state_vars).at("pressure") = new_p;
    (*state_vars).at("q") = new_q;
  }

  // Update state variables
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