//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleFiniteStrainFbar<Tdim>::ParticleFiniteStrainFbar(
    Index id, const VectorDim& coord)
    : mpm::ParticleFiniteStrain<Tdim>(id, coord) {
  // Logger
  std::string logger = "particle_finite_strain_fbar" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleFiniteStrainFbar<Tdim>::ParticleFiniteStrainFbar(
    Index id, const VectorDim& coord, bool status)
    : mpm::ParticleFiniteStrain<Tdim>(id, coord, status) {
  //! Logger
  std::string logger = "particle_finite_strain_fbar" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Compute strain of the particle
template <unsigned Tdim>
void mpm::ParticleFiniteStrainFbar<Tdim>::compute_strain(double dt) noexcept {
  // Compute deformation gradient increment
  // Note: Deformation gradient must be updated after compute_stress
  deformation_gradient_increment_ =
      this->compute_deformation_gradient_increment(
          dn_dx_, mpm::ParticlePhase::Solid, dt);

  // Deformation gradient increment at cell center
  Eigen::Matrix<double, 3, 3> deformation_gradient_increment_centroid;
  deformation_gradient_increment_centroid =
      this->compute_deformation_gradient_increment(
          dn_dx_centroid_, mpm::ParticlePhase::Solid, dt);

  double deltaJ = this->deformation_gradient_increment_.determinant();
  const double deltaJ_centroid =
      deformation_gradient_increment_centroid.determinant();

  // incremental F-bar matrix
  deformation_gradient_increment_.block(0, 0, Tdim, Tdim) *=
      std::pow(deltaJ_centroid / deltaJ, 1.0 / Tdim);

  // Update volume and mass density
  deltaJ = this->deformation_gradient_increment_.determinant();
  this->volume_ *= deltaJ;
  this->mass_density_ /= deltaJ;
}