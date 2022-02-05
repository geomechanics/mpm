//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleBbar<Tdim>::ParticleBbar(Index id, const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  // Logger
  std::string logger =
      "particle_bbar" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleBbar<Tdim>::ParticleBbar(Index id, const VectorDim& coord,
                                      bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  //! Logger
  std::string logger =
      "particle_bbar" + std::to_string(Tdim) + "d::" + std::to_string(id);
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
inline Eigen::Matrix<double, 6, 1> mpm::ParticleBbar<3>::compute_strain_rate(
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
inline void mpm::ParticleBbar<3>::map_internal_force() noexcept {
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

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleBbar<1>::compute_bmatrix() noexcept {
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
inline Eigen::MatrixXd mpm::ParticleBbar<2>::compute_bmatrix() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(3, 2 * this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    // clang-format off
    bmatrix(0, 2 * i) = dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 2.;
    bmatrix(1, 2 * i) =                (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 2.;
    bmatrix(2, 2 * i) = dn_dx_(i, 1);

    bmatrix(0, 2 * i + 1) =                (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 2.;
    bmatrix(1, 2 * i + 1) = dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 2.;
    bmatrix(2, 2 * i + 1) = dn_dx_(i, 0);
    // clang-format on
  }
  return bmatrix;
}

// Compute B matrix
template <>
inline Eigen::MatrixXd mpm::ParticleBbar<3>::compute_bmatrix() noexcept {
  Eigen::MatrixXd bmatrix;
  bmatrix.resize(6, 3 * this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    // clang-format off
    bmatrix(0, 3 * i) = dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(1, 3 * i) =                (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(2, 3 * i) =                (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(3, 3 * i) = dn_dx_(i, 1);
    bmatrix(5, 3 * i) = dn_dx_(i, 2);

    bmatrix(0, 3 * i + 1) =                (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(1, 3 * i + 1) = dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(2, 3 * i + 1) =                (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(3, 3 * i + 1) = dn_dx_(i, 0);
    bmatrix(4, 3 * i + 1) = dn_dx_(i, 2);

    bmatrix(0, 3 * i + 2) =                (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.;
    bmatrix(1, 3 * i + 2) =                (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.;
    bmatrix(2, 3 * i + 2) = dn_dx_(i, 2) + (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.;
    bmatrix(4, 3 * i + 2) = dn_dx_(i, 1);
    bmatrix(5, 3 * i + 2) = dn_dx_(i, 0);
    // clang-format on
  }
  return bmatrix;
}

// Compute strain increment of the particle
template <>
inline Eigen::Matrix<double, 6, 1>
    mpm::ParticleBbar<1>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
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
    mpm::ParticleBbar<2>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                   unsigned phase) noexcept {
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
    mpm::ParticleBbar<3>::compute_strain_increment(const Eigen::MatrixXd& dn_dx,
                                                   unsigned phase) noexcept {
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
