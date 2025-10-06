//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleLevelsetBbar<Tdim>::ParticleLevelsetBbar(Index id,
                                                      const VectorDim& coord)
    : mpm::ParticleLevelset<Tdim>(id, coord) {
  // Logger
  std::string logger =
      "particle_bbar" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleLevelsetBbar<Tdim>::ParticleLevelsetBbar(Index id,
                                                      const VectorDim& coord,
                                                      bool status)
    : mpm::ParticleLevelset<Tdim>(id, coord, status) {
  //! Logger
  std::string logger = "particle_levelset_bbar" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleLevelsetBbar<1>::compute_strain_rate(
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
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleLevelsetBbar<2>::compute_strain_rate(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 2, 1> vel = nodes_[i]->velocity(phase);
    // clang-format off
    strain_rate[0] += (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 2.) * vel[0] +
                      (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 2. * vel[1];
    strain_rate[1] += (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 2. * vel[0] +
                      (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 2.) * vel[1];
    strain_rate[3] += dn_dx(i, 1) * vel[0] + dn_dx(i, 0) * vel[1];
    // clang-format on
  }

  if (std::fabs(strain_rate[0]) < 1.E-15) strain_rate[0] = 0.;
  if (std::fabs(strain_rate[1]) < 1.E-15) strain_rate[1] = 0.;
  if (std::fabs(strain_rate[3]) < 1.E-15) strain_rate[3] = 0.;
  return strain_rate;
}

// Compute strain rate of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleLevelsetBbar<3>::compute_strain_rate(
        const Eigen::MatrixXd& dn_dx, unsigned phase) noexcept {
  // Define strain rate
  Eigen::Matrix<double, 6, 1> strain_rate = Eigen::Matrix<double, 6, 1>::Zero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    Eigen::Matrix<double, 3, 1> vel = nodes_[i]->velocity(phase);
    // clang-format off
    strain_rate[0] += (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3.) * vel[0] +
                      (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3. * vel[1] +
                      (dn_dx_centroid_(i, 2) - dn_dx(i, 2)) / 3. * vel[2];
    strain_rate[1] += (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3. * vel[0] +
                      (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3.) * vel[1] +
                      (dn_dx_centroid_(i, 2) - dn_dx(i, 2)) / 3. * vel[2];
    strain_rate[2] += (dn_dx_centroid_(i, 0) - dn_dx(i, 0)) / 3. * vel[0] +
                      (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3. * vel[1] +
                      (dn_dx(i, 2) + (dn_dx_centroid_(i, 2) - dn_dx(i, 2)) / 3.) * vel[2];
    strain_rate[3] += dn_dx(i, 1) * vel[0] + dn_dx(i, 0) * vel[1];
    strain_rate[4] += dn_dx(i, 2) * vel[1] + dn_dx(i, 1) * vel[2];
    strain_rate[5] += dn_dx(i, 2) * vel[0] + dn_dx(i, 0) * vel[2];
    // clang-format on
  }

  for (unsigned i = 0; i < strain_rate.size(); ++i)
    if (std::fabs(strain_rate[i]) < 1.E-15) strain_rate[i] = 0.;
  return strain_rate;
}

//! Map internal force
template <>
inline void mpm::ParticleLevelsetBbar<1>::map_internal_force() noexcept {
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
inline void mpm::ParticleLevelsetBbar<2>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 2, 1> force;
    // clang-format off
    force[0] = (dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 2.) * stress_[0] +
               (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 2. * stress_[1] +
               dn_dx_(i, 1) * stress_[3];
    force[1] = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 2. * stress_[0] +
               (dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 2.) * stress_[1] +
               dn_dx_(i, 0) * stress_[3];
    // clang-format on

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);
  }
}

//! Map internal force
template <>
inline void mpm::ParticleLevelsetBbar<3>::map_internal_force() noexcept {
  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 3, 1> force;
    // clang-format off
    force[0] = (dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.) * stress_[0] +
               (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * stress_[1] +
               (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * stress_[2] +
               dn_dx_(i, 1) * stress_[3] + dn_dx_(i, 2) * stress_[5];

    force[1] = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * stress_[0] +
               (dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.) * stress_[1] +
               (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * stress_[2] +
               dn_dx_(i, 0) * stress_[3] + dn_dx_(i, 2) * stress_[4];

    force[2] = (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * stress_[0] +
               (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * stress_[1] +
               (dn_dx_(i, 2) + (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.) * stress_[2] +
               dn_dx_(i, 1) * stress_[4] + dn_dx_(i, 0) * stress_[5];
    // clang-format on

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Solid, force);
  }
}