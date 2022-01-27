//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleBbar<Tdim>::ParticleBbar(Index id, const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Set material containers
  this->initialise_material(1);
  // Logger
  std::string logger =
      "particle" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleBbar<Tdim>::ParticleBbar(Index id, const VectorDim& coord,
                                      bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  this->initialise();
  cell_ = nullptr;
  nodes_.clear();
  // Set material containers
  this->initialise_material(1);
  //! Logger
  std::string logger =
      "particle" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticleBbar<1>::compute_strain_rate(
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

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticleBbar<2>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] +=
        (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3.) * vel[0] +
        (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3. * vel[1];
    strain_rate[1] +=
        (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3. * vel[0] +
        (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3.) * vel[1];
    strain_rate[2] += (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3. * vel[0] +
                      (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3. * vel[1];
    strain_rate[3] += dn_dx(i, 1) * vel[0] + dn_dx(i, 0) * vel[1];
  }

  if (std::fabs(strain_rate[0]) < 1.E-15) strain_rate[0] = 0.;
  if (std::fabs(strain_rate[1]) < 1.E-15) strain_rate[1] = 0.;
  if (std::fabs(strain_rate[2]) < 1.E-15) strain_rate[2] = 0.;
  if (std::fabs(strain_rate[3]) < 1.E-15) strain_rate[3] = 0.;
  return strain_rate;
}

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1> mpm::ParticleBbar<3>::compute_strain_rate(
    const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> vel = nodes_[i]->velocity(phase);
    strain_rate[0] +=
        (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.) * vel[0] +
        (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * vel[1] +
        (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * vel[2];
    strain_rate[1] +=
        (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * vel[0] +
        (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.) * vel[1] +
        (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * vel[2];
    strain_rate[2] +=
        (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * vel[0] +
        (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * vel[1] +
        (dn_dx(i, 2) + (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3.) * vel[2];
    strain_rate[3] += dn_dx(i, 1) * vel[0] + dn_dx(i, 0) * vel[1];
    strain_rate[4] += dn_dx(i, 2) * vel[1] + dn_dx(i, 1) * vel[2];
    strain_rate[5] += dn_dx(i, 2) * vel[0] + dn_dx(i, 0) * vel[2];
  }

  for (unsigned i = 0; i < strain_rate.size(); ++i)
    if (std::fabs(strain_rate[i]) < 1.E-15) strain_rate[i] = 0.;
  return strain_rate;
}

// Compute strain of the particle
template <unsigned Tdim>
void mpm::ParticleBbar<Tdim>::compute_strain(double dt) noexcept {
  // Assign strain rate
  strain_rate_ = this->compute_strain_rate(dn_dx_, mpm::ParticlePhase::Solid);
  // Update dstrain
  dstrain_ = strain_rate_ * dt;
  // Update strain
  strain_ += dstrain_;

  // Compute at centroid
  // Strain rate for reduced integration
  const Eigen::Matrix<double, 6, 1> strain_rate_centroid =
      this->compute_strain_rate(dn_dx_centroid_, mpm::ParticlePhase::Solid);

  // Assign volumetric strain at centroid
  dvolumetric_strain_ = dt * strain_rate_centroid.head(Tdim).sum();
  volumetric_strain_centroid_ += dvolumetric_strain_;
}

//! Map internal force
template <>
inline void mpm::ParticleBbar<1>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 1, 1> force;
    force[0] = -1. * dn_dx_(i, 0) * volume_ * stress_[0];

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);
  }
}

//! Map internal force
template <>
inline void mpm::ParticleBbar<2>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 2, 1> force;
    force[0] = (dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.) *
                   stress_[0] +
               (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * stress_[1] +
               (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * stress_[2] +
               dn_dx_(i, 1) * stress_[3];
    force[1] = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * stress_[0] +
               (dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.) *
                   stress_[1] +
               (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * stress_[2] +
               dn_dx_(i, 0) * stress_[3];

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);
  }
}

//! Map internal force
template <>
inline void mpm::ParticleBbar<3>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 3, 1> force;
    force[0] = (dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.) *
                   stress_[0] +
               (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * stress_[1] +
               (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * stress_[2] +
               dn_dx_(i, 1) * stress_[3] + dn_dx_(i, 2) * stress_[5];

    force[1] = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * stress_[0] +
               (dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.) *
                   stress_[1] +
               (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * stress_[2] +
               dn_dx_(i, 0) * stress_[3] + dn_dx_(i, 2) * stress_[4];

    force[2] = (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * stress_[0] +
               (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * stress_[1] +
               (dn_dx_(i, 2) + (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.) *
                   stress_[2] +
               dn_dx_(i, 1) * stress_[4] + dn_dx_(i, 0) * stress_[5];

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);
  }
}