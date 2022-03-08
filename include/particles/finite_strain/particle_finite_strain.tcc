//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleFiniteStrain<Tdim>::ParticleFiniteStrain(Index id,
                                                      const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
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

// Compute stress using implicit updating scheme
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_stress_newmark() noexcept {
  // Check if material ptr is valid
  assert(this->material() != nullptr);
  // Clone state variables
  auto temp_state_variables = state_variables_[mpm::ParticlePhase::Solid];
  // Calculate stress
  this->stress_ =
      (this->material())
          ->compute_stress_finite_strain(
              previous_stress_, deformation_gradient_,
              deformation_gradient_increment_, this, &temp_state_variables);

  // Compute current consititutive matrix
  this->constitutive_matrix_ =
      material_[mpm::ParticlePhase::Solid]
          ->compute_consistent_tangent_matrix_finite_strain(
              stress_, previous_stress_, deformation_gradient_,
              deformation_gradient_increment_, this, &temp_state_variables);
}

// Compute deformation gradient increment of the particle
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_strain_newmark() noexcept {
  // Compute deformation gradient increment from previous time step
  this->deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment(this->dn_dx_,
                                                   mpm::ParticlePhase::Solid);
}

// Compute deformation gradient increment of the particle
template <>
inline Eigen::Matrix<double, 3, 3>
    mpm::ParticleFiniteStrain<1>::compute_deformation_gradient_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define deformation gradient increment
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment =
      Eigen::Matrix<double, 3, 3>::Zero();
  deformation_gradient_increment(0, 0) = 1.;
  deformation_gradient_increment(1, 1) = 1.;
  deformation_gradient_increment(2, 2) = 1.;

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 1, 1> displacement = nodes_[i]->displacement(phase);
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
      Eigen::Matrix<double, 3, 3>::Zero();
  deformation_gradient_increment(0, 0) = 1.;
  deformation_gradient_increment(1, 1) = 1.;
  deformation_gradient_increment(2, 2) = 1.;

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> displacement = nodes_[i]->displacement(phase);
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
      Eigen::Matrix<double, 3, 3>::Zero();
  deformation_gradient_increment(0, 0) = 1.;
  deformation_gradient_increment(1, 1) = 1.;
  deformation_gradient_increment(2, 2) = 1.;

  // Reference configuration is the beginning of the time step
  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> displacement = nodes_[i]->displacement(phase);
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

// Update volume based on the deformation gradient increment
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::update_volume() noexcept {
  // Check if particle has a valid cell ptr and a valid volume
  assert(cell_ != nullptr && volume_ != std::numeric_limits<double>::max());
  // Compute at centroid
  // Strain rate for reduced integration
  this->volume_ *= this->deformation_gradient_increment_.determinant();
  this->mass_density_ =
      this->mass_density_ / this->deformation_gradient_increment_.determinant();
}