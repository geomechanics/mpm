//! Constructor with id and material properties
template <unsigned Tdim>
mpm::MuI<Tdim>::MuI(unsigned id, const Json& material_properties)
    : InfinitesimalElastoPlastic<Tdim>(id, material_properties) {
  try {
    // General parameters
    // Density
    double density = material_properties.at("density").template get<double>();

    // Initial Packing Fraction
    if (material_properties.contains("packing_fraction"))
      initial_packing_fraction_ =
          material_properties.at("packing_fraction").template get<double>();
    else if (material_properties.contains("porosity"))
      initial_packing_fraction_ =
          1. - material_properties.at("porosity").template get<double>();

    // Solid grain density
    bool is_wet = false;
    if (material_properties.contains("is_wet"))
      is_wet = material_properties.at("is_wet").template get<bool>();
    else if (material_properties.contains("porosity"))
      is_wet = true;
    if (!is_wet)
      grain_density_ = density / initial_packing_fraction_;
    else
      grain_density_ = density;

    // Young's modulus
    if (material_properties.contains("youngs_modulus") &&
        material_properties.contains("poisson_ratio")) {
      // Young's modulus
      youngs_modulus_ =
          material_properties.at("youngs_modulus").template get<double>();
      // Poisson ratio
      poisson_ratio_ =
          material_properties.at("poisson_ratio").template get<double>();
      // Bulk modulus
      bulk_modulus_ = youngs_modulus_ / (3.0 * (1. - 2. * poisson_ratio_));
      // Shear modulus
      shear_modulus_ = youngs_modulus_ / (2.0 * (1 + poisson_ratio_));
    } else if (material_properties.contains("bulk_modulus") &&
               material_properties.contains("shear_modulus")) {
      // Bulk modulus
      bulk_modulus_ =
          material_properties.at("bulk_modulus").template get<double>();
      // Shear modulus
      shear_modulus_ =
          material_properties.at("shear_modulus").template get<double>();
      // Young's modulus
      youngs_modulus_ = 9.0 * bulk_modulus_ * shear_modulus_ /
                        (3.0 * bulk_modulus_ + shear_modulus_);
      // Poisson ratio
      poisson_ratio_ = (3.0 * bulk_modulus_ - 2.0 * shear_modulus_) /
                       (2.0 * (3.0 * bulk_modulus_ + shear_modulus_));
    } else {
      throw std::runtime_error(
          "Young's modulus and Poisson's ratio, or bulk modulus and shear "
          "modulus, must be defined.");
    }

    // Friction parameter at zero strain rate
    mu_s_ = material_properties.at("friction_static").template get<double>();
    // Critical packing fraction
    critical_packing_fraction_ =
        material_properties.at("packing_fraction_critical")
            .template get<double>();
    // Minimum packing fraction
    minimum_packing_fraction_ =
        material_properties.at("packing_fraction_minimum")
            .template get<double>();
    // Scaling constant (chi)
    dilation_scaling_ =
        material_properties.at("dilation_scaling").template get<double>();

    // Mean particle size
    d_ = material_properties.at("mean_particle_size").template get<double>();

    // Parameters for rate dependency
    // Rate dependency flag
    if (material_properties.contains("rate_dependent")) {
      rate_dependent_ =
          material_properties.at("rate_dependent").template get<bool>();
    }

    if (rate_dependent_) {
      // Friction parameter at high strain rate
      mu_2_ = material_properties.at("friction_maximum").template get<double>();
      // Friction I0 parameter
      I0_ = material_properties.at("friction_I0").template get<double>();
      // Dilation parameter a
      dilation_a_ = material_properties.at("dilation_a").template get<double>();

      // Bulk viscosity
      if (material_properties.contains("bulk_viscosity")) {
        bulk_viscosity_ =
            material_properties.at("bulk_viscosity").template get<double>();
      } else if (material_properties.contains("restitution")) {
        double rest =
            material_properties.at("restitution").template get<double>();
        if (rest > 1.0)
          rest = 1.0;
        else if (rest < tolerance_)
          rest = tolerance_;
        bulk_viscosity_ =
            0.237 * d_ *
            std::sqrt((bulk_modulus_ + 4. / 3. * shear_modulus_) *
                      initial_packing_fraction_ * grain_density_) *
            std::pow(-std::log(rest), M_PI / 2.);
      }
      // Shear viscosity
      if (material_properties.contains("shear_viscosity")) {
        shear_viscosity_ =
            material_properties.at("shear_viscosity").template get<double>();
      } else if (material_properties.contains("rayleigh_proportionality")) {
        bool rayleigh_proportionality =
            material_properties.at("rayleigh_proportionality")
                .template get<bool>();
        if (rayleigh_proportionality)
          shear_viscosity_ = bulk_viscosity_ / bulk_modulus_ * shear_modulus_;
      }
    }

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
mpm::dense_map mpm::MuI<Tdim>::initialise_state_variables() {
  mpm::dense_map state_vars = {
      // MC parameters
      // Yield state: 0: elastic, 1: shear, 2: apex, 3: separated
      {"yield_state", 0},
      // Number of iteration in return mapping algorithm
      {"rmap_niteration", 0},
      // Elastic stress
      {"elastic_stress_0", 0.0},
      {"elastic_stress_1", 0.0},
      {"elastic_stress_2", 0.0},
      {"elastic_stress_3", 0.0},
      {"elastic_stress_4", 0.0},
      {"elastic_stress_5", 0.0},
      // Packing fraction
      {"packing_fraction", initial_packing_fraction_},
      {"packing_fraction_dilation", initial_packing_fraction_},
      // Plastic deviatoric strain rate
      {"pgamma_dot", 0.},
      // Pressure
      {"pressure", 0.},
      // Deviatoric stress
      {"tau", 0.},
      // Shear stress ratio
      {"shear_stress_ratio", 0.},
      // Plastic deviatoric strain (following p-q stress framework)
      {"pdstrain", 0.}};
  return state_vars;
}

//! Initialise state variables
template <unsigned Tdim>
std::vector<std::string> mpm::MuI<Tdim>::state_variables() const {
  const std::vector<std::string> state_vars = {"yield_state",
                                               "rmap_niteration",
                                               "elastic_stress_0",
                                               "elastic_stress_1",
                                               "elastic_stress_2",
                                               "elastic_stress_3",
                                               "elastic_stress_4",
                                               "elastic_stress_5",
                                               "packing_fraction",
                                               "packing_fraction_dilation",
                                               "pgamma_dot",
                                               "pressure",
                                               "tau",
                                               "shear_stress_ratio",
                                               "pdstrain"};
  return state_vars;
}

//! Compute stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1> mpm::MuI<Tdim>::compute_stress(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt) {

  // Density and packing parameters
  const double current_packing_density = ptr->mass_density();
  const double current_packing_fraction =
      current_packing_density / grain_density_;
  const double current_packing_fraction_dilation =
      (*state_vars).at("packing_fraction_dilation");

  // Elastic stress
  Vector6d elastic_stress = Vector6d::Zero();
  elastic_stress << (*state_vars).at("elastic_stress_0"),
      (*state_vars).at("elastic_stress_1"),
      (*state_vars).at("elastic_stress_2"),
      (*state_vars).at("elastic_stress_3"),
      (*state_vars).at("elastic_stress_4"),
      (*state_vars).at("elastic_stress_5");

  //-------------------------------------------------------------------------
  // Elastic-predictor stage: compute the trial stress
  (*state_vars).at("yield_state") = 0;
  Matrix6x6 de = this->compute_elastic_tensor();
  Vector6d trial_stress =
      this->compute_trial_stress(elastic_stress, dstrain, de, ptr, state_vars);

  //-------------------------------------------------------------------------
  // Identity in voigt
  Vector6d m_voigt;
  m_voigt << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

  // Global boolean to reset elastic stress and packing_fraction_dilation
  bool reset_elastic_stress = false;
  bool reset_phid = false;

  // Add viscoelastic contribution
  Vector6d strain_rate = dstrain / dt;
  strain_rate.tail(3) *= 0.5;
  const double vol_strain_rate = strain_rate.head(Tdim).sum();
  Vector6d dev_strain_rate = strain_rate - vol_strain_rate / 3.0 * m_voigt;
  trial_stress.noalias() += bulk_viscosity_ * vol_strain_rate * m_voigt +
                            2.0 * shear_viscosity_ * dev_strain_rate;

  // Effective moduli
  double eff_bulk_modulus = bulk_modulus_ + bulk_viscosity_ / dt;
  double eff_shear_modulus = shear_modulus_ + shear_viscosity_ / dt;

  // Compute trial invariants
  const double p_tr = -mpm::materials::p(trial_stress);
  const Vector6d& deviatoric_stress_tr =
      mpm::materials::deviatoric_stress(trial_stress);
  const double q_tr = mpm::materials::q(trial_stress);
  const double tau_tr = q_tr / sqrt(3.0);

  // Compute initial friction and dilation parameters (rate-independent)
  const double delta_phi0 =
      current_packing_fraction_dilation - critical_packing_fraction_;
  const double mu_d0 = mu_s_ + delta_phi0 * dilation_scaling_;
  const double dilation0 =
      (rate_dependent_)
          ? (current_packing_fraction_dilation - minimum_packing_fraction_) *
                dilation_scaling_
          : delta_phi0 * dilation_scaling_;
  const double alpha0 = eff_bulk_modulus / eff_shear_modulus * dilation0;

  // Compute parameters for rate dependency
  double xi = 0.0, zeta = 0.0;
  if (rate_dependent_) {
    xi = I0_ / std::sqrt(d_ * d_ * grain_density_);
    zeta = dilation_a_ * std::sqrt(d_ * d_ * grain_density_);
  }

  // Volumetric plastic strain rate - might be used for cohesion
  double vol_plastic_strain_rate = 0.0;
  Vector6d dev_plastic_strain_rate = Vector6d::Zero();

  // Compute new stress
  Vector6d updated_stress = Vector6d::Zero();
  double gamma_dot = 0.0;
  // Separated state: current packing density is less than critical density
  if (current_packing_fraction <= current_packing_fraction_dilation) {
    updated_stress.setZero();
    (*state_vars).at("yield_state") = 3;
    (*state_vars).at("rmap_niteration") = 1;
    (*state_vars).at("pressure") = 0.0;
    (*state_vars).at("tau") = 0.0;
    (*state_vars).at("shear_stress_ratio") = 0.0;
    reset_elastic_stress = true;
    reset_phid =
        (current_packing_fraction < minimum_packing_fraction_) ? true : false;
    // Recompute gamma_dot for visualization (should not affect simulation)
    gamma_dot = tau_tr / (eff_shear_modulus * dt);
  } else {
    // Elastic state: stress point is less than yield surface
    if (tau_tr <= mu_d0 * p_tr) {
      updated_stress = trial_stress;
      (*state_vars).at("yield_state") = 0;
      (*state_vars).at("rmap_niteration") = 1;
      (*state_vars).at("pressure") = p_tr;
      (*state_vars).at("tau") = tau_tr;
      (*state_vars).at("shear_stress_ratio") = 1.0;
    }
    // Plastic state: stress point is outside yield surface
    else {
      // Initialize variables
      double p_new = 0.0, tau_new = 0.0;
      double phi_d_new = current_packing_fraction_dilation;

      // Separated state: pressure is less than zero and stress point cannot
      // be returned to yield surface
      if ((tau_tr * alpha0 + p_tr) <= 0.0 && (alpha0 * mu_d0 + 1.0) >= 0.0) {
        updated_stress.setZero();
        (*state_vars).at("yield_state") = 2;
        (*state_vars).at("rmap_niteration") = 1;
        (*state_vars).at("pressure") = 0.0;
        (*state_vars).at("tau") = 0.0;
        (*state_vars).at("shear_stress_ratio") = 0.0;
        reset_elastic_stress = true;

        // Update packing_fraction_dilation
        if (p_tr > 0) {
          // Compute plastic shear strain rate
          const double den =
              (std::abs(alpha0) < tolerance_) ? tolerance_ : alpha0;
          const double tau0 = tau_tr + p_tr / den;
          if ((tau0 > 0.0) && (tau0 < tau_tr))
            gamma_dot = (tau_tr - tau0) / (eff_shear_modulus * dt);

          // Update packing_fraction_dilation
          if (gamma_dot > 0.0) {
            const double k = dt * dilation_scaling_ * gamma_dot;
            const double temp = (rate_dependent_)
                                    ? (1. - k * minimum_packing_fraction_)
                                    : (1. - k * critical_packing_fraction_);
            phi_d_new =
                -temp + std::sqrt(temp * temp +
                                  4. * k * current_packing_fraction_dilation);
            phi_d_new /= (2. * k);
          }
        }
        // Recompute gamma_dot for visualization (should not affect simulation)
        gamma_dot = tau_tr / (eff_shear_modulus * dt);
      }
      // Shearing state: stress point is outside yield surface but can be
      // returned to yield surface
      else {
        // Rate-independent return map algorithm
        if (!rate_dependent_) {
          double p_m = p_tr;
          double gamma_m = 0.0;
          double phi_d_m = current_packing_fraction_dilation;
          Eigen::Matrix<double, 3, 1> lambda_m;
          lambda_m << p_m, gamma_m, phi_d_m;
          double tau_m = tau_tr - dt * shear_modulus_ * gamma_m;

          // Start Newton-Raphson iteration
          unsigned iter = 0;
          double initial_res_norm;
          Eigen::Matrix<double, 3, 1> res_m;
          Eigen::Matrix<double, 3, 3> jac_m;

          while (iter < max_iter_) {
            // Compute dilation and friction
            const double phi_c = critical_packing_fraction_;
            const double beta = 1. / 3. * (phi_d_m - phi_c) * dilation_scaling_;
            const double mu_c = mu_s_;
            const double mu_d = mu_c + 3.0 * beta;

            // Compute residuals
            res_m(0) = tau_m - mu_d * p_m;
            res_m(1) = p_m - p_tr - 3.0 * dt * bulk_modulus_ * beta * gamma_m;
            res_m(2) =
                bulk_modulus_ * (phi_d_m - current_packing_fraction_dilation +
                                 3.0 * dt * phi_d_m * beta * gamma_m);

            // Check convergence based on residual norm
            if (res_m.norm() < abs_tol_) break;
            if (iter == 0)
              initial_res_norm = res_m.norm();
            else {
              if (res_m.norm() / initial_res_norm < rel_tol_) break;
            }

            // Compute Jacobian components
            const double dr1_dp = -mu_d;
            const double dr1_dgamma = -dt * shear_modulus_;
            const double dr1_dphi_d = -dilation_scaling_ * p_m;
            const double dr2_dp = 1.0;
            const double dr2_dgamma = -3.0 * dt * bulk_modulus_ * beta;
            const double dr2_dphi_d =
                -dt * bulk_modulus_ * dilation_scaling_ * gamma_m;
            const double dr3_dp = 0.0;
            const double dr3_dgamma = bulk_modulus_ * 3.0 * dt * phi_d_m * beta;
            const double dr3_dphi_d =
                bulk_modulus_ *
                (1. + dt * dilation_scaling_ * gamma_m * phi_d_m +
                 3. * dt * beta * gamma_m);

            // Compute Jacobian
            jac_m(0, 0) = dr1_dp;
            jac_m(0, 1) = dr1_dgamma;
            jac_m(0, 2) = dr1_dphi_d;
            jac_m(1, 0) = dr2_dp;
            jac_m(1, 1) = dr2_dgamma;
            jac_m(1, 2) = dr2_dphi_d;
            jac_m(2, 0) = dr3_dp;
            jac_m(2, 1) = dr3_dgamma;
            jac_m(2, 2) = dr3_dphi_d;

            // Solve for increment in transformed space
            Eigen::Matrix<double, 3, 1> delta_lambda_m =
                jac_m.inverse() * (-res_m);

            // Update unknowns in transformed space
            lambda_m += delta_lambda_m;

            // Update unknowns in real space
            p_m = lambda_m(0);
            gamma_m = lambda_m(1);
            phi_d_m = lambda_m(2);
            tau_m = tau_tr - dt * shear_modulus_ * gamma_m;

            // Check convergence based on solution norm
            if (delta_lambda_m.norm() < abs_tol_) {
              break;
            }

            // Increment iteration counter
            iter++;
          }

          // Check bound before finalizing NR iteration, there could be
          // precision error which may lead to long-term instabilities
          if ((p_m < 0.0) || (tau_m < 0.0)) {
            p_m = 0.0;
            tau_m = 0.0;
            phi_d_m = current_packing_fraction_dilation;
          }

          p_new = p_m;
          tau_new = tau_m;
          phi_d_new = phi_d_m;
          (*state_vars).at("rmap_niteration") = iter + 1;
        }
        // Rate-dependent algorithm
        else {
          // Check if dilation is active
          if (std::abs(dilation_scaling_) <
              std::numeric_limits<double>::epsilon()) {
            // Dilation is not active, use non-dilating root finder algorithm
            // (Dunatunga and Kamrin, 2015)
            const double S0 = mu_d0 * p_tr;
            const double S2 = mu_2_ * p_tr;
            const double alpha = xi * eff_shear_modulus * dt * std::sqrt(p_tr);
            const double B = S2 + tau_tr + alpha;
            const double H = S2 * tau_tr + S0 * alpha;
            p_new = p_tr;
            tau_new = 2.0 * H / (B + std::sqrt(B * B - 4.0 * H));
          } else {
            // Precompute guess to assure convergence in the following
            // Newton-Raphson's algorithm
            const double p_guess = 1.0;
            const double phi_c_guess =
                current_packing_fraction_dilation +
                eff_shear_modulus * p_tr /
                    (eff_bulk_modulus * tau_tr * dilation_scaling_);
            const double I_guess = (critical_packing_fraction_ - phi_c_guess) /
                                   dilation_a_ /
                                   (phi_c_guess - minimum_packing_fraction_);
            const double gamma_guess =
                I_guess / d_ * std::sqrt(p_guess / grain_density_);

            // Dilation is active, use Newton-Raphson algorithm with
            // quadratic transformation algorithm to impose non-negative
            // pressure Initialize unknowns in real space and transformed
            // space
            double p_m = p_guess;
            double gamma_m = gamma_guess;
            double phi_d_m = current_packing_fraction_dilation;
            Eigen::Matrix<double, 3, 1> lambda_m;
            lambda_m << std::sqrt(p_m), gamma_m, phi_d_m;
            double tau_m = tau_tr - dt * eff_shear_modulus * gamma_m;

            // Start Newton-Raphson iteration
            unsigned iter = 0;
            double initial_res_norm;
            Eigen::Matrix<double, 3, 1> res_m;
            Eigen::Matrix<double, 3, 3> jac_m;
            while (iter < max_iter_) {
              // Compute dilation and friction
              const double delta_phi =
                  critical_packing_fraction_ - minimum_packing_fraction_;
              double mu_c = mu_s_ + (mu_2_ - mu_s_) * gamma_m /
                                        (gamma_m + xi * std::sqrt(p_m));
              if (xi < tolerance_) mu_c = mu_s_;
              double phi_c = minimum_packing_fraction_ +
                             delta_phi / (1 + zeta * gamma_m / std::sqrt(p_m));
              double psi = -1. / 6. *
                           (delta_phi * zeta * dilation_scaling_ * gamma_m *
                            std::sqrt(p_m)) /
                           (std::pow((std::sqrt(p_m) + zeta * gamma_m), 2));
              if (zeta < tolerance_) {
                phi_c = critical_packing_fraction_;
                psi = 0.0;
              }
              const double beta =
                  1. / 3. * (phi_d_m - phi_c) * dilation_scaling_;
              const double mu_d = mu_c + 3.0 * beta;

              // Compute residuals
              res_m(0) = tau_m - mu_d * p_m;
              res_m(1) =
                  p_m - p_tr - 3.0 * dt * eff_bulk_modulus * beta * gamma_m;
              res_m(2) = eff_bulk_modulus *
                         (phi_d_m - current_packing_fraction_dilation +
                          3.0 * dt * phi_d_m * beta * gamma_m);

              // Check convergence based on residual norm
              if (res_m.norm() < abs_tol_) break;
              if (iter == 0)
                initial_res_norm = res_m.norm();
              else {
                if (res_m.norm() / initial_res_norm < rel_tol_) break;
              }

              // Compute Jacobian components
              double temp = ((mu_2_ - mu_s_) * xi * std::sqrt(p_m)) /
                            (std::pow(gamma_m + xi * std::sqrt(p_m), 2));
              if (xi < tolerance_) temp = 0.0;
              const double dr1_dp = -mu_d - 3.0 * psi + temp * gamma_m / 2.0;
              const double dr1_dgamma =
                  -dt * eff_shear_modulus - (temp - 6.0 * psi / gamma_m) * p_m;
              const double dr1_dphi_d = -dilation_scaling_ * p_m;
              const double dr2_dp =
                  1.0 - 3.0 * dt * eff_bulk_modulus * gamma_m * psi / p_m;
              const double dr2_dgamma =
                  -3.0 * dt * eff_bulk_modulus * (beta - 2.0 * psi);
              const double dr2_dphi_d =
                  -dt * eff_bulk_modulus * dilation_scaling_ * gamma_m;
              const double dr3_dp =
                  eff_bulk_modulus * (3.0 * dt * phi_d_m * gamma_m * psi / p_m);
              const double dr3_dgamma =
                  eff_bulk_modulus * (3.0 * dt * phi_d_m * (beta - 2.0 * psi));
              const double dr3_dphi_d =
                  eff_bulk_modulus *
                  (1. + dt * dilation_scaling_ * gamma_m * phi_d_m +
                   3. * dt * beta * gamma_m);

              // Compute Jacobian
              jac_m(0, 0) = 2 * lambda_m(0) * dr1_dp;
              jac_m(0, 1) = dr1_dgamma;
              jac_m(0, 2) = dr1_dphi_d;
              jac_m(1, 0) = 2 * lambda_m(0) * dr2_dp;
              jac_m(1, 1) = dr2_dgamma;
              jac_m(1, 2) = dr2_dphi_d;
              jac_m(2, 0) = 2 * lambda_m(0) * dr3_dp;
              jac_m(2, 1) = dr3_dgamma;
              jac_m(2, 2) = dr3_dphi_d;

              // Solve for increment in transformed space
              Eigen::Matrix<double, 3, 1> delta_lambda_m =
                  jac_m.inverse() * (-res_m);

              // Update unknowns in transformed space
              lambda_m += delta_lambda_m;

              // Update unknowns in real space
              p_m = lambda_m(0) * lambda_m(0);
              gamma_m = lambda_m(1);
              phi_d_m = lambda_m(2);
              tau_m = tau_tr - dt * eff_shear_modulus * gamma_m;

              // Check convergence based on solution norm
              if (delta_lambda_m.norm() < abs_tol_) {
                break;
              }

              // Increment iteration counter
              iter++;
            }

            // Check bound before finalizing NR iteration, there could be
            // precision error which may lead to long-term instabilities
            if ((p_m < 0.0) || (tau_m < 0.0)) {
              p_m = 0.0;
              tau_m = 0.0;
              phi_d_m = current_packing_fraction_dilation;
            }

            p_new = p_m;
            tau_new = tau_m;
            phi_d_new = phi_d_m;
            (*state_vars).at("rmap_niteration") = iter + 1;
          }
        }

        const double shear_stress_ratio = tau_new / tau_tr;
        updated_stress =
            shear_stress_ratio * deviatoric_stress_tr - p_new * m_voigt;
        gamma_dot = ((tau_tr - tau_new) / eff_shear_modulus) / dt;
        (*state_vars).at("yield_state") = 1;
        (*state_vars).at("pressure") = p_new;
        (*state_vars).at("tau") = tau_new;
        (*state_vars).at("shear_stress_ratio") = shear_stress_ratio;
        vol_plastic_strain_rate = (p_new - p_tr) / (dt * eff_bulk_modulus);
        dev_plastic_strain_rate =
            gamma_dot * deviatoric_stress_tr / 2.0 / tau_tr;
      }

      // Check bounds for packing_fraction_dilation
      if (phi_d_new < minimum_packing_fraction_)
        phi_d_new = minimum_packing_fraction_;
      if (phi_d_new >
          std::max(critical_packing_fraction_, initial_packing_fraction_))
        phi_d_new =
            std::max(critical_packing_fraction_, initial_packing_fraction_);

      (*state_vars).at("packing_fraction_dilation") = phi_d_new;
    }
  }

  // Update elastic stress
  if (!reset_elastic_stress)
    elastic_stress =
        updated_stress -
        bulk_viscosity_ * (vol_strain_rate - vol_plastic_strain_rate) *
            m_voigt -
        2.0 * shear_viscosity_ * (dev_strain_rate - dev_plastic_strain_rate);
  else
    elastic_stress.setZero();

  // Store elastic strain for next step
  (*state_vars).at("elastic_stress_0") = elastic_stress(0);
  (*state_vars).at("elastic_stress_1") = elastic_stress(1);
  (*state_vars).at("elastic_stress_2") = elastic_stress(2);
  (*state_vars).at("elastic_stress_3") = elastic_stress(3);
  (*state_vars).at("elastic_stress_4") = elastic_stress(4);
  (*state_vars).at("elastic_stress_5") = elastic_stress(5);

  // Update state variables
  (*state_vars).at("packing_fraction") = current_packing_fraction;
  (*state_vars).at("pgamma_dot") = gamma_dot;
  (*state_vars).at("pdstrain") += gamma_dot * dt / std::sqrt(3.0);

  // Check if reset packing_fraction_dilation is needed
  if (reset_phid) {
    (*state_vars).at("packing_fraction_dilation") = minimum_packing_fraction_;
  }

  return updated_stress;
}

//! Compute elastic tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6> mpm::MuI<Tdim>::compute_elastic_tensor() {
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

//! Compute visco-elastic tensor
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6> mpm::MuI<Tdim>::compute_visco_tensor(double dt) {
  // Viscosity
  const double mu = shear_viscosity_ / dt;
  const double bulk = bulk_viscosity_ / dt;
  const double a1 = bulk + (4.0 / 3.0) * mu;
  const double a2 = bulk - (2.0 / 3.0) * mu;
  // compute elastic stiffness matrix
  // clang-format off
  Matrix6x6 de = Matrix6x6::Zero();
  de(0,0)=a1;    de(0,1)=a2;    de(0,2)=a2;
  de(1,0)=a2;    de(1,1)=a1;    de(1,2)=a2;
  de(2,0)=a2;    de(2,1)=a2;    de(2,2)=a1;
  de(3,3)=mu;    de(4,4)=mu;    de(5,5)=mu;
  // clang-format on

  return de;
}

//! Compute constitutive relations matrix for elasto-plastic material
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6> mpm::MuI<Tdim>::compute_elasto_plastic_tensor(
    const Vector6d& stress, const Vector6d& dstrain,
    const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars, double dt,
    bool hardening) {

  // Get yield type after return mapping algorithm
  mpm::mu_i::FailureState yield_type =
      yield_type_.at(int((*state_vars).at("yield_state")));

  // Return the elastic consitutive tensor in elastic state
  if (yield_type == mpm::mu_i::FailureState::Elastic) {
    const Matrix6x6 de = this->compute_elastic_tensor();
    const Matrix6x6 dv = this->compute_visco_tensor(dt);
    return de + dv;
  }

  // Return zero tensor matrix in apex and separated state
  if ((yield_type == mpm::mu_i::FailureState::Apex) ||
      (yield_type == mpm::mu_i::FailureState::Separated)) {
    return Matrix6x6::Zero();
  }

  // Prepare necessary stress parameters
  const double p = -mpm::materials::p(stress);
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

  Matrix6x6 tensor_1xN = Matrix6x6::Zero();
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 6; ++j) {
      tensor_1xN(i, j) = director_n[j];
    }
  }

  Matrix6x6 tensor_Nx1 = Matrix6x6::Zero();
  for (unsigned i = 0; i < 6; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      tensor_Nx1(i, j) = director_n[i];
    }
  }

  // Compute the elasto-plastic stiffness matrix in shear state
  // Necessary parameters
  double eff_bulk_modulus = bulk_modulus_ + bulk_viscosity_ / dt;
  double eff_shear_modulus = shear_modulus_ + shear_viscosity_ / dt;
  const double pgamma_dot = (*state_vars).at("pgamma_dot");
  const double phi_d = (*state_vars).at("packing_fraction_dilation");
  double phi_c, mu_c;
  double xi = 0.0, zeta = 0.0, psi = 0.0;
  if (!rate_dependent_) {
    phi_c = critical_packing_fraction_;
    mu_c = mu_s_;
  } else {
    const double delta_phi =
        critical_packing_fraction_ - minimum_packing_fraction_;
    xi = I0_ / std::sqrt(d_ * d_ * grain_density_);
    if (xi < tolerance_) {
      mu_c = mu_s_;
      xi = 0.0;
    } else
      mu_c = mu_s_ +
             (mu_2_ - mu_s_) * pgamma_dot / (pgamma_dot + xi * std::sqrt(p));

    zeta = dilation_a_ * std::sqrt(d_ * d_ * grain_density_);
    if (zeta < tolerance_) {
      phi_c = critical_packing_fraction_;
      psi = 0.0;
      zeta = 0.0;
    } else {
      phi_c = minimum_packing_fraction_ +
              delta_phi / (1 + zeta * pgamma_dot / std::sqrt(p));
      psi = -1. / 6. *
            (delta_phi * zeta * dilation_scaling_ * pgamma_dot * std::sqrt(p)) /
            (std::pow((std::sqrt(p) + zeta * pgamma_dot), 2));
    }
  }
  const double beta = 1. / 3. * (phi_d - phi_c) * dilation_scaling_;
  const double mu_d = mu_c + 3.0 * beta;

  // Compute derivative
  const double denom_p = (p < tolerance_) ? tolerance_ : p;
  const double denom_pgammadot =
      (pgamma_dot < tolerance_) ? tolerance_ : pgamma_dot;
  const double dbeta_dp = psi / denom_p;
  const double dbeta_dgamma = -2.0 * psi / denom_pgammadot;
  const double dbeta_dphid = dilation_scaling_ / 3.0;

  // Compute scalar parameters
  const double a = 1.0 - 3.0 * eff_bulk_modulus * dt * pgamma_dot * dbeta_dp;
  const double a_1 = -eff_bulk_modulus / a;
  const double a_2 =
      3.0 * eff_bulk_modulus * dt * (beta + pgamma_dot * dbeta_dgamma) / a;
  const double a_3 = 3.0 * eff_bulk_modulus * dt * pgamma_dot * dbeta_dphid / a;

  const double d = 1 + 3.0 * dt * pgamma_dot *
                           (beta + phi_d * (dbeta_dphid + a_3 * dbeta_dp));
  const double d_1 = -3.0 * dt * phi_d * pgamma_dot * a_1 * dbeta_dp / d;
  const double d_2 = -3.0 * dt * phi_d *
                     (beta + pgamma_dot * (dbeta_dgamma + a_2 * dbeta_dp)) / d;

  double temp = ((mu_2_ - mu_s_) * xi * std::sqrt(p)) /
                (std::pow(pgamma_dot + xi * std::sqrt(p), 2));
  if (xi < tolerance_) temp = 0.0;
  const double b_1 = 3.0 * psi / denom_p - temp * pgamma_dot / (2.0 * denom_p);
  const double b_2 = temp - 6.0 * psi / denom_pgammadot;
  const double b_3 = dilation_scaling_;

  const double c = eff_shear_modulus * dt + mu_d * (a_2 + a_3 * d_2) +
                   p * (a_2 * b_1 + b_2 + d_2 * (a_3 * b_1 + b_3));
  const double c_1 = std::sqrt(2.0) * eff_shear_modulus / c;
  const double c_2 =
      (-mu_d * (a_1 + a_3 * d_1) - p * (a_1 * b_1 + d_1 * (a_3 * b_1 + b_3))) /
      c;

  const double e_1 = 2.0 * eff_shear_modulus * tau_ratio;
  const double e_2 = -(a_1 + a_3 * d_1 + c_2 * (a_2 + a_3 * d_2) +
                       2.0 / 3.0 * eff_shear_modulus * tau_ratio);
  const double e_3 = -c_1 * (a_2 + a_3 * d_2);
  const double e_4 = -std::sqrt(2.0) * eff_shear_modulus * dt * c_2;
  const double e_5 = 2.0 * eff_shear_modulus * (1.0 - tau_ratio) -
                     std::sqrt(2.0) * eff_shear_modulus * dt * c_1;

  //! Elasto-plastic stiffness matrix
  Matrix6x6 d_ep = e_1 * fourth_order_identity + e_2 * identity_cross +
                   e_3 * tensor_1xN + e_4 * tensor_Nx1 + e_5 * tensor_NxN;

  return d_ep;
}
