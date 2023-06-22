//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticlePML<Tdim>::ParticlePML(Index id, const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  // Logger
  std::string logger =
      "particle_pml" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticlePML<Tdim>::ParticlePML(Index id, const VectorDim& coord,
                                    bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  //! Logger
  std::string logger =
      "particle_pml" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Map particle mass and momentum to nodes
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_mass_momentum_to_nodes(
    mpm::VelocityUpdate velocity_update) noexcept {

  switch (velocity_update) {
    case mpm::VelocityUpdate::APIC:
      this->map_mass_momentum_to_nodes_affine();
      break;
    case mpm::VelocityUpdate::ASFLIP:
      this->map_mass_momentum_to_nodes_affine();
      break;
    case mpm::VelocityUpdate::TPIC:
      this->map_mass_momentum_to_nodes_taylor();
      break;
    default:
      // Check if particle mass is set
      assert(mass_ != std::numeric_limits<double>::max());

      // Damping functions
      const VectorDim& damping_functions = this->mass_damping_functions();

      // Modify Velocity to include damping term
      const VectorDim& velocity_mod = velocity_.cwiseProduct(damping_functions);

      // Map mass and momentum to nodes
      for (unsigned i = 0; i < nodes_.size(); ++i) {
        // Map mass and momentum
        nodes_[i]->update_mass(true, mpm::ParticlePhase::SinglePhase,
                               mass_ * shapefn_[i]);
        nodes_[i]->update_momentum(true, mpm::ParticlePhase::SinglePhase,
                                   mass_ * shapefn_[i] * velocity_mod);
      }
      break;
  }
}

//! Map particle mass and momentum to nodes for affine transformation
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_mass_momentum_to_nodes_affine() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Initialise Mapping matrix if necessary
  if (mapping_matrix_.rows() != Tdim) {
    mapping_matrix_.resize(Tdim, Tdim);
    mapping_matrix_.setZero();
  }

  // Shape tensor computation for APIC
  Eigen::MatrixXd shape_tensor;
  shape_tensor.resize(Tdim, Tdim);
  shape_tensor.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const auto& branch_vector = nodes_[i]->coordinates() - this->coordinates_;
    shape_tensor.noalias() +=
        shapefn_[i] * branch_vector * branch_vector.transpose();
  }

  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Map mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Initialise map velocity
    VectorDim map_velocity = velocity_;
    map_velocity.noalias() += mapping_matrix_ * shape_tensor.inverse() *
                              (nodes_[i]->coordinates() - this->coordinates_);

    // Modify Velocity to include damping term
    const VectorDim& velocity_mod =
        map_velocity.cwiseProduct(damping_functions);

    // Map mass and momentum
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                           mass_ * shapefn_[i]);
    nodes_[i]->update_momentum(true, mpm::ParticlePhase::Solid,
                               mass_ * shapefn_[i] * velocity_mod);
  }
}

//! Map particle mass and momentum to nodes for approximate taylor expansion
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_mass_momentum_to_nodes_taylor() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Initialise Mapping matrix if necessary
  if (mapping_matrix_.rows() != Tdim) {
    mapping_matrix_.resize(Tdim, Tdim);
    mapping_matrix_.setZero();
  }

  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Map mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Initialise map velocity
    VectorDim map_velocity = velocity_;
    map_velocity.noalias() +=
        mapping_matrix_ * (nodes_[i]->coordinates() - this->coordinates_);

    // Modify Velocity to include damping term
    const VectorDim& velocity_mod =
        map_velocity.cwiseProduct(damping_functions);

    // Map mass and momentum
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                           mass_ * shapefn_[i]);
    nodes_[i]->update_momentum(true, mpm::ParticlePhase::Solid,
                               mass_ * shapefn_[i] * velocity_mod);
  }
}

//! Map particle mass, momentum and inertia to nodes
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_mass_momentum_inertia_to_nodes() noexcept {
  // Map mass and momentum to nodes
  this->map_mass_momentum_to_nodes();

  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Modify Acceleration
  const VectorDim& acceleration_mod =
      acceleration_.cwiseProduct(damping_functions);

  // Map inertia to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_inertia(true, mpm::ParticlePhase::SinglePhase,
                              mass_ * shapefn_[i] * acceleration_mod);
  }
}

//! Map damped mass vector to nodes
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_pml_properties_to_nodes() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Recompute damping function at the beginning of time step
  this->compute_damping_functions(
      state_variables_[mpm::ParticlePhase::SinglePhase]);

  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Modify displacements
  const VectorDim& displacement_mod =
      displacement_.cwiseProduct(damping_functions);

  // Check if visco_elastic is needed
  bool viscoelastic =
      (this->material())
          ->template property<bool>(std::string("visco_elasticity"));

  // Read internal displacement variables
  VectorDim displacement_mod_j1;
  VectorDim displacement_mod_j2;
  VectorDim displacement_mod_j3;
  VectorDim displacement_mod_j4;

  if (viscoelastic) {
    unsigned phase = mpm::ParticlePhase::SinglePhase;
    // j = 1
    Eigen::Matrix<double, 3, 1> u_j1;
    u_j1[0] = state_variables_[phase]["prev_disp_x_j1"];
    u_j1[1] = state_variables_[phase]["prev_disp_y_j1"];
    u_j1[2] = state_variables_[phase]["prev_disp_z_j1"];
    for (unsigned i = 0; i < Tdim; i++)
      displacement_mod_j1(i) = u_j1(i) * damping_functions(i);

    // j = 2
    Eigen::Matrix<double, 3, 1> u_j2;
    u_j2[0] = state_variables_[phase]["prev_disp_x_j2"];
    u_j2[1] = state_variables_[phase]["prev_disp_y_j2"];
    u_j2[2] = state_variables_[phase]["prev_disp_z_j2"];
    for (unsigned i = 0; i < Tdim; i++)
      displacement_mod_j2(i) = u_j2(i) * damping_functions(i);

    // j = 3
    Eigen::Matrix<double, 3, 1> u_j3;
    u_j3[0] = state_variables_[phase]["prev_disp_x_j3"];
    u_j3[1] = state_variables_[phase]["prev_disp_y_j3"];
    u_j3[2] = state_variables_[phase]["prev_disp_z_j3"];
    for (unsigned i = 0; i < Tdim; i++)
      displacement_mod_j3(i) = u_j3(i) * damping_functions(i);

    // j = 4
    Eigen::Matrix<double, 3, 1> u_j4;
    u_j4[0] = state_variables_[phase]["prev_disp_x_j4"];
    u_j4[1] = state_variables_[phase]["prev_disp_y_j4"];
    u_j4[2] = state_variables_[phase]["prev_disp_z_j4"];
    for (unsigned i = 0; i < Tdim; i++)
      displacement_mod_j4(i) = u_j4(i) * damping_functions(i);
  }

  // Map damped mass, displacement vector to nodal property
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const auto& damped_mass = mass_ * shapefn_[i] * damping_functions;
    nodes_[i]->update_property(true, "damped_masses", damped_mass, 0, Tdim);
    nodes_[i]->update_property(true, "damped_mass_displacements",
                               mass_ * shapefn_[i] * displacement_mod, 0, Tdim);

    // Historical displacement components
    if (viscoelastic) {
      nodes_[i]->update_property(true, "damped_mass_displacements_j1",
                                 mass_ * shapefn_[i] * displacement_mod_j1, 0,
                                 Tdim);
      nodes_[i]->update_property(true, "damped_mass_displacements_j2",
                                 mass_ * shapefn_[i] * displacement_mod_j2, 0,
                                 Tdim);
      nodes_[i]->update_property(true, "damped_mass_displacements_j3",
                                 mass_ * shapefn_[i] * displacement_mod_j3, 0,
                                 Tdim);
      nodes_[i]->update_property(true, "damped_mass_displacements_j4",
                                 mass_ * shapefn_[i] * displacement_mod_j4, 0,
                                 Tdim);
    }
    nodes_[i]->assign_pml(true);
  }
}

//! Finalise pml properties
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::finalise_pml_properties(double dt) noexcept {
  this->update_pml_displacement_functions(dt);
}

//! Map body force
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_body_force(
    const VectorDim& pgravity) noexcept {
  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Modify gravity
  const VectorDim& pgravity_mod = pgravity.cwiseProduct(damping_functions);

  // Compute nodal body forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::SinglePhase,
                                     (pgravity_mod * mass_ * shapefn_(i)));
}

//! Map internal force
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_internal_force(double dt) noexcept {
  // Compute local stiffness matrix
  Eigen::MatrixXd local_stiffness = this->compute_pml_stiffness_matrix();

  // Displacement vector
  Eigen::VectorXd disp(Tdim * nodes_.size());
  disp.setZero();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const VectorDim& u_n = nodes_[i]->previous_pml_displacement();
    const VectorDim& v_n = nodes_[i]->previous_pml_velocity();
    const VectorDim& a_n = nodes_[i]->previous_pml_acceleration();
    const VectorDim& a_new =
        nodes_[i]->acceleration(mpm::ParticlePhase::SinglePhase);
    disp.segment(i * Tdim, Tdim) +=
        u_n + v_n * dt + dt * dt / 2.0 * (0.5 * a_n + 0.5 * a_new);
  }

  // Compute nodal internal force = K*u
  Eigen::VectorXd nodal_force = local_stiffness * disp;

  // Check if visco_elastic is needed
  bool viscoelastic =
      (this->material())
          ->template property<bool>(std::string("visco_elasticity"));

  if (viscoelastic) {
    // Read parameters
    const double E_inf =
        (this->material())
            ->template property<double>(std::string("youngs_modulus"));
    const double E_0 = (this->material())
                           ->template property<double>(std::string(
                               "visco_elastic_youngs_modulus_relaxed"));
    const double alpha = (this->material())
                             ->template property<double>(
                                 std::string("visco_elastic_fractional_power"));
    const double tau = (this->material())
                           ->template property<double>(
                               std::string("visco_elastic_relaxed_time"));

    // Check parameter
    assert(E_inf > E_0);
    assert((alpha > 0) && (alpha <= 1.0));
    assert(tau > 0);
    double c =
        std::pow(tau, alpha) / (std::pow(tau, alpha) + std::pow(dt, alpha));

    // Add non-historical component
    nodal_force *= (1.0 + c * (E_inf - E_0) / E_0);

    // Read Gruenwald coefficient
    const unsigned nt = 4;
    const double A_2 = -alpha;
    const double A_3 = A_2 * (1. - alpha) / 2.;
    const double A_4 = A_3 * (2. - alpha) / 3.;
    const double A_5 = A_4 * (3. - alpha) / 4.;
    Eigen::Matrix<double, nt, 1> A_j;
    A_j << A_2, A_3, A_4, A_5;

    // Read internal nodal displacement vector
    Eigen::VectorXd old_displacements(Tdim * nodes_.size());
    old_displacements.setZero();
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      VectorDim disp_sum = VectorDim::Zero();
      for (unsigned j = 0; j < nt; j++) {
        const VectorDim& u_j = nodes_[i]->previous_pml_displacement(j + 1);
        disp_sum.noalias() += A_j(j) * u_j;
      }
      old_displacements.segment(i * Tdim, Tdim) += disp_sum;
    }

    // Add historical component
    nodal_force.noalias() +=
        c * E_inf / E_0 * local_stiffness * old_displacements;
  }

  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -force * volume
    Eigen::Matrix<double, Tdim, 1> force = nodal_force.segment(i * Tdim, Tdim);
    force *= -1. * this->volume_;
    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);
  }
}

//! Function to update pml displacement functions
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::update_pml_displacement_functions(
    double dt) noexcept {
  // Check if visco_elastic is needed
  bool viscoelastic =
      (this->material())
          ->template property<bool>(std::string("visco_elasticity"));

  if (viscoelastic) {
    // Read parameters
    const double E_inf =
        (this->material())
            ->template property<double>(std::string("youngs_modulus"));
    const double E_0 = (this->material())
                           ->template property<double>(std::string(
                               "visco_elastic_youngs_modulus_relaxed"));
    const double alpha = (this->material())
                             ->template property<double>(
                                 std::string("visco_elastic_fractional_power"));
    const double tau = (this->material())
                           ->template property<double>(
                               std::string("visco_elastic_relaxed_time"));

    // Check parameter
    assert(E_inf > E_0);
    assert((alpha > 0) && (alpha <= 1.0));
    assert(tau > 0);
    double c =
        std::pow(tau, alpha) / (std::pow(tau, alpha) + std::pow(dt, alpha));

    // Read internal displacement function
    unsigned phase = mpm::ParticlePhase::SinglePhase;
    // j = 0
    Eigen::Matrix<double, 3, 1> current_disp;
    for (unsigned i = 0; i < Tdim; i++)
      current_disp(i) = this->displacement_(i);

    // j = 1
    Eigen::Matrix<double, 3, 1> u_j1;
    u_j1[0] = state_variables_[phase]["prev_disp_x_j1"];
    u_j1[1] = state_variables_[phase]["prev_disp_y_j1"];
    u_j1[2] = state_variables_[phase]["prev_disp_z_j1"];

    // j = 2
    Eigen::Matrix<double, 3, 1> u_j2;
    u_j2[0] = state_variables_[phase]["prev_disp_x_j2"];
    u_j2[1] = state_variables_[phase]["prev_disp_y_j2"];
    u_j2[2] = state_variables_[phase]["prev_disp_z_j2"];

    // j = 3
    Eigen::Matrix<double, 3, 1> u_j3;
    u_j3[0] = state_variables_[phase]["prev_disp_x_j3"];
    u_j3[1] = state_variables_[phase]["prev_disp_y_j3"];
    u_j3[2] = state_variables_[phase]["prev_disp_z_j3"];

    // j = 4
    Eigen::Matrix<double, 3, 1> u_j4;
    u_j4[0] = state_variables_[phase]["prev_disp_x_j4"];
    u_j4[1] = state_variables_[phase]["prev_disp_y_j4"];
    u_j4[2] = state_variables_[phase]["prev_disp_z_j4"];

    // Read Gruenwald coefficient
    const double A_2 = -alpha;
    const double A_3 = A_2 * (1. - alpha) / 2.;
    const double A_4 = A_3 * (2. - alpha) / 3.;
    const double A_5 = A_4 * (3. - alpha) / 4.;

    // Compute new displacement function
    Eigen::Matrix<double, 3, 1> new_disp_funct =
        ((1.0 - c) * (E_inf - E_0) / E_inf) * current_disp -
        c * (A_2 * u_j1 + A_3 * u_j2 + A_4 * u_j3 + A_5 * u_j4);

    // Store internal displacement function
    state_variables_[phase]["prev_disp_x_j1"] = new_disp_funct[0];
    state_variables_[phase]["prev_disp_y_j1"] = new_disp_funct[1];
    state_variables_[phase]["prev_disp_z_j1"] = new_disp_funct[2];

    state_variables_[phase]["prev_disp_x_j2"] = u_j1[0];
    state_variables_[phase]["prev_disp_y_j2"] = u_j1[1];
    state_variables_[phase]["prev_disp_z_j2"] = u_j1[2];

    state_variables_[phase]["prev_disp_x_j3"] = u_j2[0];
    state_variables_[phase]["prev_disp_y_j3"] = u_j2[1];
    state_variables_[phase]["prev_disp_z_j3"] = u_j2[2];

    state_variables_[phase]["prev_disp_x_j4"] = u_j3[0];
    state_variables_[phase]["prev_disp_y_j4"] = u_j3[1];
    state_variables_[phase]["prev_disp_z_j4"] = u_j3[2];
  }
}

//! Map inertial force
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_inertial_force() noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);

  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Compute nodal inertial forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Modify nodal acceleration
    const VectorDim& acceleration_mod =
        nodes_[i]
            ->acceleration(mpm::ParticlePhase::SinglePhase)
            .cwiseProduct(damping_functions);

    nodes_[i]->update_external_force(
        true, mpm::ParticlePhase::SinglePhase,
        (-1. * acceleration_mod * mass_ * shapefn_(i)));
  }
}

//! Map material stiffness matrix to cell (used in equilibrium equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticlePML<Tdim>::map_material_stiffness_matrix_to_cell(
    double dt) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Compute local stiffness matrix
    Eigen::MatrixXd local_stiffness = this->compute_pml_stiffness_matrix();

    // Check if visco_elastic is needed
    bool viscoelastic =
        (this->material())
            ->template property<bool>(std::string("visco_elasticity"));

    if (viscoelastic) {
      // Read parameters
      const double E_inf =
          (this->material())
              ->template property<double>(std::string("youngs_modulus"));
      const double E_0 = (this->material())
                             ->template property<double>(std::string(
                                 "visco_elastic_youngs_modulus_relaxed"));
      const double alpha = (this->material())
                               ->template property<double>(std::string(
                                   "visco_elastic_fractional_power"));
      const double tau = (this->material())
                             ->template property<double>(
                                 std::string("visco_elastic_relaxed_time"));

      // Check parameter
      assert(E_inf > E_0);
      assert((alpha > 0) && (alpha <= 1.0));
      assert(tau > 0);
      double c =
          std::pow(tau, alpha) / (std::pow(tau, alpha) + std::pow(dt, alpha));

      // Add non-historical component
      local_stiffness *= (1.0 + c * (E_inf - E_0) / E_0);
    }

    // Compute local material stiffness matrix
    cell_->compute_local_stiffness_matrix_block(0, 0, local_stiffness, volume_,
                                                1.0);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map mass matrix to cell (used in poisson equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticlePML<Tdim>::map_mass_matrix_to_cell(double newmark_beta,
                                                            double dt) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Damping functions
    const VectorDim& damping_functions = this->mass_damping_functions();

    // Modify mass density
    const VectorDim& mass_density_mod = damping_functions * mass_density_;

    // Construct matrix
    Eigen::MatrixXd mass_density_matrix(Tdim * nodes_.size(),
                                        Tdim * nodes_.size());
    mass_density_matrix.setZero();
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      for (unsigned k = 0; k < Tdim; ++k) {
        mass_density_matrix(Tdim * i + k, Tdim * i + k) +=
            shapefn_(i) * mass_density_mod(k);
      }
    }

    // Compute local mass matrix
    cell_->compute_local_stiffness_matrix_block(
        0, 0, mass_density_matrix, volume_, 1.0 / (newmark_beta * dt * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map PML rayleigh damping force
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::map_rayleigh_damping_force(
    double damping_factor) noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);

  // Scale damping factor
  const auto& state_vars = state_variables_[mpm::ParticlePhase::SinglePhase];
  double L = state_vars.at("boundary_thickness");
  VectorDim dist_function;
  switch (Tdim) {
    case 1:
      dist_function << state_vars.at("distance_function_x") / L;
      break;
    case 2:
      dist_function << state_vars.at("distance_function_x") / L,
          state_vars.at("distance_function_y") / L;
      break;
    case 3:
      dist_function << state_vars.at("distance_function_x") / L,
          state_vars.at("distance_function_y") / L,
          state_vars.at("distance_function_z") / L;
      break;
  }
  double scaling_factor = dist_function.norm();

  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Compute nodal rayleigh damping forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Modify nodal velocity
    const VectorDim& velocity_mod =
        nodes_[i]
            ->velocity(mpm::ParticlePhase::SinglePhase)
            .cwiseProduct(damping_functions);

    nodes_[i]->update_external_force(true, mpm::ParticlePhase::SinglePhase,
                                     (-1. * scaling_factor * damping_factor *
                                      velocity_mod * mass_ * shapefn_(i)));
  }
}

//! Map PML rayleigh damping matrix to cell (used in equilibrium
//! equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticlePML<Tdim>::map_rayleigh_damping_matrix_to_cell(
    double newmark_gamma, double newmark_beta, double dt,
    double damping_factor) {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Scale damping factor
    const auto& state_vars = state_variables_[mpm::ParticlePhase::SinglePhase];
    double L = state_vars.at("boundary_thickness");
    VectorDim dist_function;
    switch (Tdim) {
      case 1:
        dist_function << state_vars.at("distance_function_x") / L;
        break;
      case 2:
        dist_function << state_vars.at("distance_function_x") / L,
            state_vars.at("distance_function_y") / L;
        break;
      case 3:
        dist_function << state_vars.at("distance_function_x") / L,
            state_vars.at("distance_function_y") / L,
            state_vars.at("distance_function_z") / L;
        break;
    }
    double scaling_factor = dist_function.norm();

    // Damping functions
    const VectorDim& damping_functions = this->mass_damping_functions();

    // Modify mass density
    const VectorDim& mass_density_mod =
        scaling_factor * damping_factor * damping_functions * mass_density_;

    // Construct matrix
    Eigen::MatrixXd mass_density_matrix(Tdim * nodes_.size(),
                                        Tdim * nodes_.size());
    mass_density_matrix.setZero();
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      for (unsigned k = 0; k < Tdim; ++k) {
        mass_density_matrix(Tdim * i + k, Tdim * i + k) +=
            shapefn_(i) * mass_density_mod(k);
      }
    }

    // Compute local mass matrix
    cell_->compute_local_stiffness_matrix_block(
        0, 0, mass_density_matrix, volume_,
        newmark_gamma / (newmark_beta * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Initialise damping functions
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::compute_damping_functions(
    mpm::dense_map& state_vars) noexcept {
  // PML material properties
  const double alpha =
      (this->material())
          ->template property<double>(std::string("maximum_damping_ratio"));
  const double dpower =
      (this->material())
          ->template property<double>(std::string("damping_power"));
  const double boundary_thickness = state_vars.at("boundary_thickness");
  const double multiplier = alpha * std::pow(1.0 / boundary_thickness, dpower);

  // Compute damping functions in state variables
  state_vars.at("damping_function_x") =
      multiplier * std::pow(state_vars.at("distance_function_x"), dpower);
  state_vars.at("damping_function_y") =
      multiplier * std::pow(state_vars.at("distance_function_y"), dpower);
  state_vars.at("damping_function_z") =
      multiplier * std::pow(state_vars.at("distance_function_z"), dpower);
}

//! Function to return mass damping functions
template <unsigned Tdim>
Eigen::Matrix<double, Tdim, 1> mpm::ParticlePML<Tdim>::mass_damping_functions()
    const noexcept {
  // Damping functions
  const double c_x = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_x");
  const double c_y = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_y");
  const double c_z = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_z");
  Eigen::Matrix<double, Tdim, 1> damping_functions;
  switch (Tdim) {
    case 1:
      damping_functions << std::pow((1. + c_x), 2);
      break;
    case 2:
      damping_functions << std::pow((1. + c_x), 2), std::pow((1. + c_y), 2);
      break;
    case 3:
      damping_functions << std::pow((1. + c_x), 2), std::pow((1. + c_y), 2),
          std::pow((1. + c_z), 2);
      break;
  }

  return damping_functions;
}

//! Compute PML stiffness matrix
template <>
inline Eigen::MatrixXd
    mpm::ParticlePML<1>::compute_pml_stiffness_matrix() noexcept {
  // Initialise
  Eigen::MatrixXd local_stiffness(1 * nodes_.size(), 1 * nodes_.size());
  local_stiffness.setZero();

  // Material constants
  const double E =
      (this->material())
          ->template property<double>(std::string("youngs_modulus"));

  // Compute K
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    for (unsigned j = 0; j < nodes_.size(); ++j) {
      Eigen::MatrixXd k(1, 1);
      k.setZero();
      k(0, 0) = E * dn_dx_(i, 0) * dn_dx_(j, 0);
      local_stiffness.block(i * 1, j * 1, 1, 1) += k;
    }
  }
  return local_stiffness;
}

//! Compute PML stiffness matrix
template <>
inline Eigen::MatrixXd
    mpm::ParticlePML<2>::compute_pml_stiffness_matrix() noexcept {
  // Initialise
  Eigen::MatrixXd local_stiffness(2 * nodes_.size(), 2 * nodes_.size());
  local_stiffness.setZero();

  // Material constants
  const double E =
      (this->material())
          ->template property<double>(std::string("youngs_modulus"));
  const double nu =
      (this->material())
          ->template property<double>(std::string("poisson_ratio"));
  const double lambda = E * nu / (1. + nu) / (1. - 2. * nu);
  const double shear_modulus = E / (2.0 * (1. + nu));
  const double c_x = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_x");
  const double c_y = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_y");

  // Compute K
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    for (unsigned j = 0; j < nodes_.size(); ++j) {
      Eigen::MatrixXd k(2, 2);
      k.setZero();
      k(0, 0) = (lambda + 2.0 * shear_modulus) * dn_dx_(i, 0) * dn_dx_(j, 0) +
                (1.0 + c_x) * (1.0 + c_x) * shear_modulus * dn_dx_(i, 1) *
                    dn_dx_(j, 1);
      k(0, 1) = (1.0 + c_x) * (lambda * dn_dx_(i, 0) * dn_dx_(j, 1) +
                               shear_modulus * dn_dx_(i, 1) * dn_dx_(j, 0));
      k(1, 0) = (1.0 + c_y) * (lambda * dn_dx_(i, 1) * dn_dx_(j, 0) +
                               shear_modulus * dn_dx_(i, 0) * dn_dx_(j, 1));
      k(1, 1) = (lambda + 2.0 * shear_modulus) * dn_dx_(i, 1) * dn_dx_(j, 1) +
                (1.0 + c_y) * (1.0 + c_y) * shear_modulus * dn_dx_(i, 0) *
                    dn_dx_(j, 0);

      local_stiffness.block(i * 2, j * 2, 2, 2) += k;
    }
  }
  return local_stiffness;
}

//! Compute PML stiffness matrix
template <>
inline Eigen::MatrixXd
    mpm::ParticlePML<3>::compute_pml_stiffness_matrix() noexcept {
  // Initialise
  Eigen::MatrixXd local_stiffness(3 * nodes_.size(), 3 * nodes_.size());
  local_stiffness.setZero();

  // Material constants
  const double E =
      (this->material())
          ->template property<double>(std::string("youngs_modulus"));
  const double nu =
      (this->material())
          ->template property<double>(std::string("poisson_ratio"));
  const double lambda = E * nu / (1. + nu) / (1. - 2. * nu);
  const double shear_modulus = E / (2.0 * (1. + nu));
  const double c_x = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_x");
  const double c_y = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_y");
  const double c_z = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_z");

  // Compute K
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    for (unsigned j = 0; j < nodes_.size(); ++j) {
      Eigen::MatrixXd k(3, 3);
      k.setZero();
      // 0th row
      k(0, 0) = (lambda + 2.0 * shear_modulus) * dn_dx_(i, 0) * dn_dx_(j, 0) +
                (1.0 + c_x) * (1.0 + c_x) * shear_modulus *
                    (dn_dx_(i, 1) * dn_dx_(j, 1) + dn_dx_(i, 2) * dn_dx_(j, 2));
      k(0, 1) = (1.0 + c_x) * (lambda * dn_dx_(i, 0) * dn_dx_(j, 1) +
                               shear_modulus * dn_dx_(i, 1) * dn_dx_(j, 0));
      k(0, 2) = (1.0 + c_x) * (lambda * dn_dx_(i, 0) * dn_dx_(j, 2) +
                               shear_modulus * dn_dx_(i, 2) * dn_dx_(j, 0));

      // 1st row
      k(1, 0) = (1.0 + c_y) * (lambda * dn_dx_(i, 1) * dn_dx_(j, 0) +
                               shear_modulus * dn_dx_(i, 0) * dn_dx_(j, 1));
      k(1, 1) = (lambda + 2.0 * shear_modulus) * dn_dx_(i, 1) * dn_dx_(j, 1) +
                (1.0 + c_y) * (1.0 + c_y) * shear_modulus *
                    (dn_dx_(i, 0) * dn_dx_(j, 0) + dn_dx_(i, 2) * dn_dx_(j, 2));
      k(1, 2) = (1.0 + c_y) * (lambda * dn_dx_(i, 1) * dn_dx_(j, 2) +
                               shear_modulus * dn_dx_(i, 2) * dn_dx_(j, 1));

      // 2nd row
      k(2, 0) = (1.0 + c_z) * (lambda * dn_dx_(i, 2) * dn_dx_(j, 0) +
                               shear_modulus * dn_dx_(i, 0) * dn_dx_(j, 2));
      k(2, 1) = (1.0 + c_z) * (lambda * dn_dx_(i, 2) * dn_dx_(j, 1) +
                               shear_modulus * dn_dx_(i, 1) * dn_dx_(j, 2));
      k(2, 2) = (lambda + 2.0 * shear_modulus) * dn_dx_(i, 2) * dn_dx_(j, 2) +
                (1.0 + c_z) * (1.0 + c_z) * shear_modulus *
                    (dn_dx_(i, 0) * dn_dx_(j, 0) + dn_dx_(i, 1) * dn_dx_(j, 1));

      local_stiffness.block(i * 3, j * 3, 3, 3) += k;
    }
  }
  return local_stiffness;
}