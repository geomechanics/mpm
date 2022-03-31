//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleFiniteStrain<Tdim>::ParticleFiniteStrain(Index id,
                                                      const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Set material containers
  this->initialise_material(1);
  // Logger
  std::string logger = "particle_finite_strain" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleFiniteStrain<Tdim>::ParticleFiniteStrain(Index id,
                                                      const VectorDim& coord,
                                                      bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  //! Logger
  std::string logger = "particle_finite_strain" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Initialise particle properties
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::initialise() {
  mpm::Particle<Tdim>::initialise();
  deformation_gradient_increment_.setIdentity();
}

// Compute stress
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_stress() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);
  // Calculate stress
  this->stress_ =
      (this->material())
          ->compute_stress(stress_, deformation_gradient_,
                           deformation_gradient_increment_, this,
                           &state_variables_[mpm::ParticlePhase::Solid]);

  // Update deformation gradient
  this->deformation_gradient_ =
      this->deformation_gradient_increment_ * this->deformation_gradient_;
}

// Compute strain of the particle
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_strain(double dt) noexcept {
  // Compute deformation gradient increment
  // Note: Deformation gradient must be updated after compute_stress
  deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment(
          dn_dx_, mpm::ParticlePhase::Solid, dt);

  // Update volume and mass density
  const double deltaJ = this->deformation_gradient_increment_.determinant();
  this->volume_ *= deltaJ;
  this->mass_density_ /= deltaJ;
}

// Compute deformation gradient increment using nodal velocity
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::ParticleFiniteStrain<1>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase,
        const double dt) noexcept {
  // Define deformation gradient rate
  Eigen::Matrix<double, 3, 3> deformation_gradient_rate =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& velocity = nodes_[i]->velocity(phase);
    deformation_gradient_rate(0, 0) += dn_dx(i, 0) * velocity[0] * dt;
  }

  if (std::fabs(deformation_gradient_rate(0, 0) - 1.) < 1.E-15)
    deformation_gradient_rate(0, 0) = 1.;
  return deformation_gradient_rate;
}

// Compute deformation gradient increment using nodal velocity
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::ParticleFiniteStrain<2>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase,
        const double dt) noexcept {
  // Define deformation gradient rate
  Eigen::Matrix<double, 3, 3> deformation_gradient_rate =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& velocity = nodes_[i]->velocity(phase);
    deformation_gradient_rate(0, 0) += dn_dx(i, 0) * velocity[0] * dt;
    deformation_gradient_rate(0, 1) += dn_dx(i, 1) * velocity[0] * dt;
    deformation_gradient_rate(1, 0) += dn_dx(i, 0) * velocity[1] * dt;
    deformation_gradient_rate(1, 1) += dn_dx(i, 1) * velocity[1] * dt;
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; i < 2; ++i) {
      if (i != j && std::fabs(deformation_gradient_rate(i, j)) < 1.E-15)
        deformation_gradient_rate(i, j) = 0.;
      if (i == j && std::fabs(deformation_gradient_rate(i, j) - 1.) < 1.E-15)
        deformation_gradient_rate(i, j) = 1.;
    }
  }
  return deformation_gradient_rate;
}

// Compute deformation gradient increment using nodal velocity
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::ParticleFiniteStrain<3>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase,
        const double dt) noexcept {
  // Define deformation gradient rate
  Eigen::Matrix<double, 3, 3> deformation_gradient_rate =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& velocity = nodes_[i]->velocity(phase);
    deformation_gradient_rate(0, 0) += dn_dx(i, 0) * velocity[0] * dt;
    deformation_gradient_rate(0, 1) += dn_dx(i, 1) * velocity[0] * dt;
    deformation_gradient_rate(0, 2) += dn_dx(i, 2) * velocity[0] * dt;
    deformation_gradient_rate(1, 0) += dn_dx(i, 0) * velocity[1] * dt;
    deformation_gradient_rate(1, 1) += dn_dx(i, 1) * velocity[1] * dt;
    deformation_gradient_rate(1, 2) += dn_dx(i, 2) * velocity[1] * dt;
    deformation_gradient_rate(2, 0) += dn_dx(i, 0) * velocity[2] * dt;
    deformation_gradient_rate(2, 1) += dn_dx(i, 1) * velocity[2] * dt;
    deformation_gradient_rate(2, 2) += dn_dx(i, 2) * velocity[2] * dt;
  }

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; i < 3; ++i) {
      if (i != j && std::fabs(deformation_gradient_rate(i, j)) < 1.E-15)
        deformation_gradient_rate(i, j) = 0.;
      if (i == j && std::fabs(deformation_gradient_rate(i, j) - 1.) < 1.E-15)
        deformation_gradient_rate(i, j) = 1.;
    }
  }
  return deformation_gradient_rate;
}

//! Function to reinitialise material to be run at the beginning of each time
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::initialise_constitutive_law() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);

  // Reset material to be Elastic
  material_[mpm::ParticlePhase::Solid]->initialise(
      &state_variables_[mpm::ParticlePhase::Solid]);

  // Compute initial consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, deformation_gradient_,
          deformation_gradient_increment_, this,
          &state_variables_[mpm::ParticlePhase::Solid]);
}

// Compute stress using implicit updating scheme
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_stress_newmark() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);
  // Clone state variables
  auto temp_state_variables = state_variables_[mpm::ParticlePhase::Solid];
  // Calculate stress
  this->stress_ = (this->material())
                      ->compute_stress(previous_stress_, deformation_gradient_,
                                       deformation_gradient_increment_, this,
                                       &temp_state_variables);

  // Compute current consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]->compute_consistent_tangent_matrix(
          stress_, previous_stress_, deformation_gradient_,
          deformation_gradient_increment_, this, &temp_state_variables);
}

// Compute deformation gradient increment of the particle
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_strain_newmark() noexcept {
  // Compute volume and mass density at the previous time step
  double deltaJ = this->deformation_gradient_increment_.determinant();
  this->volume_ /= deltaJ;
  this->mass_density_ *= deltaJ;

  // Compute deformation gradient increment from previous time step
  this->deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment(this->dn_dx_,
                                                   mpm::ParticlePhase::Solid);

  // Update volume and mass density
  deltaJ = this->deformation_gradient_increment_.determinant();
  this->volume_ *= deltaJ;
  this->mass_density_ /= deltaJ;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::ParticleFiniteStrain<1>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    deformation_gradient_increment(0, 0) += dn_dx(i, 0) * displacement[0];
  }

  if (std::fabs(deformation_gradient_increment(0, 0) - 1.) < 1.E-15)
    deformation_gradient_increment(0, 0) = 1.;
  return deformation_gradient_increment;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::ParticleFiniteStrain<2>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    deformation_gradient_increment(0, 0) += dn_dx(i, 0) * displacement[0];
    deformation_gradient_increment(0, 1) += dn_dx(i, 1) * displacement[0];
    deformation_gradient_increment(1, 0) += dn_dx(i, 0) * displacement[1];
    deformation_gradient_increment(1, 1) += dn_dx(i, 1) * displacement[1];
  }

  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; i < 2; ++i) {
      if (i != j && std::fabs(deformation_gradient_increment(i, j)) < 1.E-15)
        deformation_gradient_increment(i, j) = 0.;
      if (i == j &&
          std::fabs(deformation_gradient_increment(i, j) - 1.) < 1.E-15)
        deformation_gradient_increment(i, j) = 1.;
    }
  }
  return deformation_gradient_increment;
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::ParticleFiniteStrain<3>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Identity();

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    const auto& displacement = nodes_[i]->displacement(phase);
    deformation_gradient_increment(0, 0) += dn_dx(i, 0) * displacement[0];
    deformation_gradient_increment(0, 1) += dn_dx(i, 1) * displacement[0];
    deformation_gradient_increment(0, 2) += dn_dx(i, 2) * displacement[0];
    deformation_gradient_increment(1, 0) += dn_dx(i, 0) * displacement[1];
    deformation_gradient_increment(1, 1) += dn_dx(i, 1) * displacement[1];
    deformation_gradient_increment(1, 2) += dn_dx(i, 2) * displacement[1];
    deformation_gradient_increment(2, 0) += dn_dx(i, 0) * displacement[2];
    deformation_gradient_increment(2, 1) += dn_dx(i, 1) * displacement[2];
    deformation_gradient_increment(2, 2) += dn_dx(i, 2) * displacement[2];
  }

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; i < 3; ++i) {
      if (i != j && std::fabs(deformation_gradient_increment(i, j)) < 1.E-15)
        deformation_gradient_increment(i, j) = 0.;
      if (i == j &&
          std::fabs(deformation_gradient_increment(i, j) - 1.) < 1.E-15)
        deformation_gradient_increment(i, j) = 1.;
    }
  }
  return deformation_gradient_increment;
}

// Compute Hencky strain
template <unsigned Tdim>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleFiniteStrain<Tdim>::compute_hencky_strain() const {

  // Left Cauchy-Green strain
  const Eigen::Matrix<double, 3, 3> left_cauchy_green_tensor =
      deformation_gradient_ * deformation_gradient_.transpose();

  // Principal values of left Cauchy-Green strain
  Eigen::Matrix<double, 3, 3> directors = Eigen::Matrix<double, 3, 3>::Zero();
  const Eigen::Matrix<double, 3, 1> principal_left_cauchy_green_strain =
      mpm::materials::principal_tensor(left_cauchy_green_tensor, directors);

  // Principal value of Hencky (logarithmic) strain
  Eigen::Matrix<double, 3, 3> principal_hencky_strain =
      Eigen::Matrix<double, 3, 3>::Zero();
  principal_hencky_strain(0, 0) =
      0.5 * std::log(principal_left_cauchy_green_strain(0));
  principal_hencky_strain(1, 1) =
      0.5 * std::log(principal_left_cauchy_green_strain(1));
  principal_hencky_strain(2, 2) =
      0.5 * std::log(principal_left_cauchy_green_strain(2));

  // Hencky strain tensor and vector
  const Eigen::Matrix<double, 3, 3> hencky_strain =
      directors * principal_hencky_strain * directors.transpose();
  Eigen::Matrix<double, 6, 1> hencky_strain_vector;
  hencky_strain_vector(0) = hencky_strain(0, 0);
  hencky_strain_vector(1) = hencky_strain(1, 1);
  hencky_strain_vector(2) = hencky_strain(2, 2);
  hencky_strain_vector(3) = 2. * hencky_strain(0, 1);
  hencky_strain_vector(4) = 2. * hencky_strain(1, 2);
  hencky_strain_vector(5) = 2. * hencky_strain(2, 0);

  return hencky_strain_vector;
}

// Update stress and strain after convergence of Newton-Raphson iteration
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::update_stress_strain() noexcept {
  // Update converged stress
  this->stress_ =
      (this->material())
          ->compute_stress(this->previous_stress_, this->deformation_gradient_,
                           this->deformation_gradient_increment_, this,
                           &state_variables_[mpm::ParticlePhase::Solid]);

  // Update initial stress of the time step
  this->previous_stress_ = this->stress_;

  // Update deformation gradient
  this->deformation_gradient_ =
      this->deformation_gradient_increment_ * this->deformation_gradient_;

  // Reset deformation gradient increment
  this->deformation_gradient_increment_.setIdentity();
}