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

// Compute strain of the particle using nodal displacement
template <unsigned Tdim>
void mpm::ParticleFiniteStrain<Tdim>::compute_strain_newmark() noexcept {
  // Compute strain increment from previous time step
  this->dstrain_ =
      this->compute_strain_increment(dn_dx_, mpm::ParticlePhase::Solid);
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

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleFiniteStrain<1>::compute_strain_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
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
    mpm::ParticleFiniteStrain<2>::compute_strain_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> displacement = nodes_[i]->displacement(phase);
    // clang-format off
    strain_increment[0] += (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 2.) * displacement[0] +
                           (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 2. * displacement[1];
    strain_increment[1] += (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 2. * displacement[0] +
                           (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 2.) * displacement[1];
    strain_increment[3] += dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
    // clang-format on
  }

  if (std::fabs(strain_increment[0]) < 1.E-15) strain_increment[0] = 0.;
  if (std::fabs(strain_increment[1]) < 1.E-15) strain_increment[1] = 0.;
  if (std::fabs(strain_increment[3]) < 1.E-15) strain_increment[3] = 0.;
  return strain_increment;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleFiniteStrain<3>::compute_strain_increment(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain increment
  Eigen::Matrix<double, 6, 1> strain_increment =
      Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> displacement = nodes_[i]->displacement(phase);
    // clang-format off
    strain_increment[0] += (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3.) * displacement[0] +
                           (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3. * displacement[1] +
                           (dn_dx_centroid_(i, 2) - dn_dx(i, 2)) / 3. * displacement[2];
    strain_increment[1] += (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3. * displacement[0] +
                           (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3.) * displacement[1] +
                           (dn_dx_centroid_(i, 2) - dn_dx(i, 2)) / 3. * displacement[2];
    strain_increment[2] += (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3. * displacement[0] +
                           (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3. * displacement[1] +
                           (dn_dx(i, 2) + (dn_dx_centroid_(i, 2) - dn_dx(i, 2)) / 3.) * displacement[2];
    strain_increment[3] += dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
    strain_increment[4] += dn_dx(i, 2) * displacement[1] + dn_dx(i, 1) * displacement[2];
    strain_increment[5] += dn_dx(i, 2) * displacement[0] + dn_dx(i, 0) * displacement[2];
    // clang-format on
  }

  for (unsigned i = 0; i < strain_increment.size(); ++i)
    if (std::fabs(strain_increment[i]) < 1.E-15) strain_increment[i] = 0.;
  return strain_increment;
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