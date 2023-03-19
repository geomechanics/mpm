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

// Compute strain of the particle
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::compute_strain(double dt) noexcept {
  // Assign strain rate
  strain_rate_ =
      this->compute_strain_rate(dn_dx_, mpm::ParticlePhase::SinglePhase);
  // Update dstrain
  dstrain_ = strain_rate_ * dt;
  // Update strain
  strain_.noalias() += dstrain_;

  // Compute at centroid
  // Strain rate for reduced integration
  const Eigen::Matrix<double, 6, 1> strain_rate_centroid =
      mpm::Particle<Tdim>::compute_strain_rate(dn_dx_centroid_,
                                               mpm::ParticlePhase::SinglePhase);

  // Assign volumetric strain at centroid
  dvolumetric_strain_ = dt * strain_rate_centroid.head(Tdim).sum();
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
void mpm::ParticlePML<Tdim>::map_damped_masses_to_nodes() noexcept {
  // Check if particle mass is set
  assert(mass_ != std::numeric_limits<double>::max());

  // Damping functions
  const VectorDim& damping_functions = this->mass_damping_functions();

  // Map damped mass vector to nodal property
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const auto& damped_mass = mass_ * shapefn_[i] * damping_functions;
    nodes_[i]->update_property(true, "damped_masses", damped_mass, 0, Tdim);
    nodes_[i]->assign_pml(true);
  }
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

// Compute strain and volume of the particle using nodal displacement
template <unsigned Tdim>
void mpm::ParticlePML<Tdim>::compute_strain_volume_newmark() noexcept {
  // Compute the volume at the previous time step
  this->volume_ /= (1. + dvolumetric_strain_);
  this->mass_density_ *= (1. + dvolumetric_strain_);

  // Compute strain increment from previous time step
  this->dstrain_ =
      this->compute_strain_increment(dn_dx_, mpm::ParticlePhase::SinglePhase);

  // Updated volumetric strain increment
  const Eigen::Matrix<double, 6, 1>& real_strain =
      mpm::Particle<Tdim>::compute_strain_increment(
          dn_dx_, mpm::ParticlePhase::SinglePhase);
  this->dvolumetric_strain_ = real_strain.head(Tdim).sum();

  // Update volume using volumetric strain increment
  this->volume_ *= (1. + dvolumetric_strain_);
  this->mass_density_ /= (1. + dvolumetric_strain_);
}

// Compute strain rate of the PML particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticlePML<1>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 1, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] += dn_dx(i, 0) * vel[0];
  }

  if (std::fabs(strain_rate(0)) < 1.E-15) strain_rate[0] = 0.;
  return strain_rate;
}

// Compute strain rate of the PML particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticlePML<2>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] += dn_dx(i, 0) * vel[0];
    strain_rate[1] += dn_dx(i, 1) * vel[1];

    // Damping functions
    const double c_x = state_variables_[phase]["damping_function_x"];
    const double c_y = state_variables_[phase]["damping_function_y"];

    strain_rate[3] += ((1 + c_x) / (1 + c_y)) * dn_dx(i, 1) * vel[0] +
                      ((1 + c_y) / (1 + c_x)) * dn_dx(i, 0) * vel[1];
  }

  if (std::fabs(strain_rate[0]) < 1.E-15) strain_rate[0] = 0.;
  if (std::fabs(strain_rate[1]) < 1.E-15) strain_rate[1] = 0.;
  if (std::fabs(strain_rate[3]) < 1.E-15) strain_rate[3] = 0.;
  return strain_rate;
}

// Compute strain rate of the PML particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticlePML<3>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] += dn_dx(i, 0) * vel[0];
    strain_rate[1] += dn_dx(i, 1) * vel[1];
    strain_rate[2] += dn_dx(i, 2) * vel[2];

    // Damping functions
    const double c_x = state_variables_[phase]["damping_function_x"];
    const double c_y = state_variables_[phase]["damping_function_y"];
    const double c_z = state_variables_[phase]["damping_function_z"];

    strain_rate[3] += ((1 + c_x) / (1 + c_y)) * dn_dx(i, 1) * vel[0] +
                      ((1 + c_y) / (1 + c_x)) * dn_dx(i, 0) * vel[1];
    strain_rate[4] += ((1 + c_y) / (1 + c_z)) * dn_dx(i, 2) * vel[1] +
                      ((1 + c_z) / (1 + c_y)) * dn_dx(i, 1) * vel[2];
    strain_rate[5] += ((1 + c_x) / (1 + c_z)) * dn_dx(i, 2) * vel[0] +
                      ((1 + c_z) / (1 + c_x)) * dn_dx(i, 0) * vel[2];
  }

  for (unsigned i = 0; i < strain_rate.size(); ++i)
    if (std::fabs(strain_rate[i]) < 1.E-15) strain_rate[i] = 0.;
  return strain_rate;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticlePML<1>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                  unsigned phase) noexcept {
  // Define strain rincrement
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 1, 1> displacement = nodes_[i]->displacement(phase);
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
  }

  if (std::fabs(strain_increment(0)) < 1.E-15) strain_increment[0] = 0.;
  return strain_increment;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticlePML<2>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                  unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> displacement = nodes_[i]->displacement(phase);
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
    strain_increment[1] += dn_dx(i, 1) * displacement[1];

    // Damping functions
    const double c_x = state_variables_[phase]["damping_function_x"];
    const double c_y = state_variables_[phase]["damping_function_y"];

    strain_increment[3] +=
        ((1 + c_x) / (1 + c_y)) * dn_dx(i, 1) * displacement[0] +
        ((1 + c_y) / (1 + c_x)) * dn_dx(i, 0) * displacement[1];
  }

  if (std::fabs(strain_increment[0]) < 1.E-15) strain_increment[0] = 0.;
  if (std::fabs(strain_increment[1]) < 1.E-15) strain_increment[1] = 0.;
  if (std::fabs(strain_increment[3]) < 1.E-15) strain_increment[3] = 0.;
  return strain_increment;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticlePML<3>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                  unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> displacement = nodes_[i]->displacement(phase);
    strain_increment[0] += dn_dx(i, 0) * displacement[0];
    strain_increment[1] += dn_dx(i, 1) * displacement[1];
    strain_increment[2] += dn_dx(i, 2) * displacement[2];

    // Damping functions
    const double c_x = state_variables_[phase]["damping_function_x"];
    const double c_y = state_variables_[phase]["damping_function_y"];
    const double c_z = state_variables_[phase]["damping_function_z"];

    strain_increment[3] +=
        ((1 + c_x) / (1 + c_y)) * dn_dx(i, 1) * displacement[0] +
        ((1 + c_y) / (1 + c_x)) * dn_dx(i, 0) * displacement[1];
    strain_increment[4] +=
        ((1 + c_y) / (1 + c_z)) * dn_dx(i, 2) * displacement[1] +
        ((1 + c_z) / (1 + c_y)) * dn_dx(i, 1) * displacement[2];
    strain_increment[5] +=
        ((1 + c_x) / (1 + c_z)) * dn_dx(i, 2) * displacement[0] +
        ((1 + c_z) / (1 + c_x)) * dn_dx(i, 0) * displacement[2];
  }

  for (unsigned i = 0; i < strain_increment.size(); ++i)
    if (std::fabs(strain_increment[i]) < 1.E-15) strain_increment[i] = 0.;
  return strain_increment;
}

//! Map material stiffness matrix to cell (used in equilibrium equation LHS)
template <unsigned Tdim>
inline bool mpm::ParticlePML<Tdim>::map_material_stiffness_matrix_to_cell() {
  bool status = true;
  try {
    // Check if material ptr is valid
    assert(this->material() != nullptr);

    // Reduce constitutive relations matrix depending on the dimension
    const Eigen::MatrixXd reduced_dmatrix =
        this->reduce_dmatrix(constitutive_matrix_);

    // Calculate B matrix
    const Eigen::MatrixXd bmatrix = this->compute_bmatrix();

    // Calculate B matrix modified
    const Eigen::MatrixXd bmatrix_mod = this->compute_bmatrix_pml();

    // Compute local material stiffness matrix
    cell_->compute_local_material_stiffness_matrix_pml(
        bmatrix, bmatrix_mod, reduced_dmatrix, volume_);
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

    // Compute local mass matrix
    cell_->compute_local_mass_matrix_pml(
        shapefn_, volume_, mass_density_mod / (newmark_beta * dt * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
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

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::ParticlePML<1>::compute_bmatrix_pml() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(1, this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, i) = dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::ParticlePML<2>::compute_bmatrix_pml() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(3, 2 * this->nodes_.size());
  bmatrix.setZero();

  // Damping functions
  const double c_x = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_x");
  const double c_y = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_y");

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 2 * i) = dn_dx_(i, 0);
    bmatrix(2, 2 * i) = ((1 + c_x) / (1 + c_y)) * dn_dx_(i, 1);

    bmatrix(1, 2 * i + 1) = dn_dx_(i, 1);
    bmatrix(2, 2 * i + 1) = ((1 + c_y) / (1 + c_x)) * dn_dx_(i, 0);
  }
  return bmatrix;
}

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::ParticlePML<3>::compute_bmatrix_pml() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(6, 3 * this->nodes_.size());
  bmatrix.setZero();

  // Damping functions
  const double c_x = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_x");
  const double c_y = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_y");
  const double c_z = state_variables_[mpm::ParticlePhase::SinglePhase].at(
      "damping_function_z");

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 3 * i) = dn_dx_(i, 0);
    bmatrix(3, 3 * i) = ((1 + c_x) / (1 + c_y)) * dn_dx_(i, 1);
    bmatrix(5, 3 * i) = ((1 + c_x) / (1 + c_z)) * dn_dx_(i, 2);

    bmatrix(1, 3 * i + 1) = dn_dx_(i, 1);
    bmatrix(3, 3 * i + 1) = ((1 + c_y) / (1 + c_x)) * dn_dx_(i, 0);
    bmatrix(4, 3 * i + 1) = ((1 + c_y) / (1 + c_z)) * dn_dx_(i, 2);

    bmatrix(2, 3 * i + 2) = dn_dx_(i, 2);
    bmatrix(4, 3 * i + 2) = ((1 + c_z) / (1 + c_y)) * dn_dx_(i, 1);
    bmatrix(5, 3 * i + 2) = ((1 + c_z) / (1 + c_x)) * dn_dx_(i, 0);
  }
  return bmatrix;
}