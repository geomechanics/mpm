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
  bmatrix.resize(4, 2 * this->nodes_.size());
  bmatrix.setZero();

  for (unsigned i = 0; i < this->nodes_.size(); ++i) {
    bmatrix(0, 2 * i) =
        dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(1, 2 * i) = (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(2, 2 * i) = (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(3, 2 * i) = dn_dx_(i, 1);

    bmatrix(0, 2 * i + 1) = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(1, 2 * i + 1) =
        dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(2, 2 * i + 1) = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(3, 2 * i + 1) = dn_dx_(i, 0);
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
    bmatrix(0, 3 * i) =
        dn_dx_(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(1, 3 * i) = (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(2, 3 * i) = (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.;
    bmatrix(3, 3 * i) = dn_dx_(i, 1);
    bmatrix(5, 3 * i) = dn_dx_(i, 2);

    bmatrix(0, 3 * i + 1) = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(1, 3 * i + 1) =
        dn_dx_(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(2, 3 * i + 1) = (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.;
    bmatrix(3, 3 * i + 1) = dn_dx_(i, 0);
    bmatrix(4, 3 * i + 1) = dn_dx_(i, 2);

    bmatrix(0, 3 * i + 2) = (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.;
    bmatrix(1, 3 * i + 2) = (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.;
    bmatrix(2, 3 * i + 2) =
        dn_dx_(i, 2) + (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3.;
    bmatrix(4, 3 * i + 2) = dn_dx_(i, 1);
    bmatrix(5, 3 * i + 2) = dn_dx_(i, 0);
  }
  return bmatrix;
}

//! Reduce constitutive relations matrix depending on the dimension
template <>
inline Eigen::MatrixXd mpm::ParticleBbar<1>::reduce_dmatrix(
    const Eigen::MatrixXd& dmatrix) noexcept {

  // Convert to 1x1 matrix in 1D
  Eigen::MatrixXd dmatrix1x1;
  dmatrix1x1.resize(1, 1);
  dmatrix1x1(0, 0) = dmatrix(0, 0);

  return dmatrix1x1;
}

//! Reduce constitutive relations matrix depending on the dimension
template <>
inline Eigen::MatrixXd mpm::ParticleBbar<2>::reduce_dmatrix(
    const Eigen::MatrixXd& dmatrix) noexcept {

  // Convert to 3x3 matrix in 2D
  Eigen::MatrixXd dmatrix4x4;
  dmatrix4x4.resize(4, 4);
  dmatrix4x4(0, 0) = dmatrix(0, 0);
  dmatrix4x4(0, 1) = dmatrix(0, 1);
  dmatrix4x4(0, 2) = dmatrix(0, 2);
  dmatrix4x4(0, 3) = dmatrix(0, 3);
  dmatrix4x4(1, 0) = dmatrix(1, 0);
  dmatrix4x4(1, 1) = dmatrix(1, 1);
  dmatrix4x4(1, 2) = dmatrix(1, 2);
  dmatrix4x4(1, 3) = dmatrix(1, 3);
  dmatrix4x4(2, 0) = dmatrix(2, 0);
  dmatrix4x4(2, 1) = dmatrix(2, 1);
  dmatrix4x4(2, 2) = dmatrix(2, 2);
  dmatrix4x4(2, 3) = dmatrix(2, 3);
  dmatrix4x4(3, 0) = dmatrix(3, 0);
  dmatrix4x4(3, 1) = dmatrix(3, 1);
  dmatrix4x4(3, 2) = dmatrix(3, 2);
  dmatrix4x4(3, 3) = dmatrix(3, 3);

  return dmatrix4x4;
}

//! Reduce constitutive relations matrix depending on the dimension
template <>
inline Eigen::MatrixXd mpm::ParticleBbar<3>::reduce_dmatrix(
    const Eigen::MatrixXd& dmatrix) noexcept {
  return dmatrix;
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
    strain_increment[0] +=
        (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.) *
            displacement[0] +
        (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * displacement[1];
    strain_increment[1] +=
        (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * displacement[0] +
        (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.) *
            displacement[1];
    strain_increment[2] +=
        (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * displacement[0] +
        (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * displacement[1];
    strain_increment[3] +=
        dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
  }

  if (std::fabs(strain_increment[0]) < 1.E-15) strain_increment[0] = 0.;
  if (std::fabs(strain_increment[1]) < 1.E-15) strain_increment[1] = 0.;
  if (std::fabs(strain_increment[2]) < 1.E-15) strain_increment[2] = 0.;
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
    strain_increment[0] +=
        (dn_dx(i, 0) + (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3.) *
            displacement[0] +
        (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * displacement[1] +
        (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * displacement[2];
    strain_increment[1] +=
        (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * displacement[0] +
        (dn_dx(i, 1) + (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3.) *
            displacement[1] +
        (dn_dx_centroid_(i, 2) - dn_dx_(i, 2)) / 3. * displacement[2];
    strain_increment[2] +=
        (dn_dx_centroid_(i, 0) - dn_dx_(i, 0)) / 3. * displacement[0] +
        (dn_dx_centroid_(i, 1) - dn_dx_(i, 1)) / 3. * displacement[1];
    +(dn_dx(i, 2) + (dn_dx_centroid_(i, 1) - dn_dx(i, 1)) / 3.) *
        displacement[2];
    strain_increment[3] +=
        dn_dx(i, 1) * displacement[0] + dn_dx(i, 0) * displacement[1];
    strain_increment[4] +=
        dn_dx(i, 2) * displacement[1] + dn_dx(i, 1) * displacement[2];
    strain_increment[5] +=
        dn_dx(i, 2) * displacement[0] + dn_dx(i, 0) * displacement[2];
  }

  for (unsigned i = 0; i < strain_increment.size(); ++i)
    if (std::fabs(strain_increment[i]) < 1.E-15) strain_increment[i] = 0.;
  return strain_increment;
}

// Compute strain of the particle using nodal displacement
template <unsigned Tdim>
void mpm::ParticleBbar<Tdim>::compute_strain_newmark() noexcept {
  // Compute strain increment from previous time step
  this->dstrain_ =
      this->compute_strain_increment(dn_dx_, mpm::ParticlePhase::Solid);
}