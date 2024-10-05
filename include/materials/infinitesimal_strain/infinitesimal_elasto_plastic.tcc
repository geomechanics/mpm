//! Compute trial stress
template <unsigned Tdim>
Eigen::Matrix<double, 6, 1>
    mpm::InfinitesimalElastoPlastic<Tdim>::compute_trial_stress(
        const Vector6d& stress, const Vector6d& dstrain, const Matrix6x6& de,
        const ParticleBase<Tdim>* ptr, mpm::dense_map* state_vars) {

  // Compute trial stress
  Vector6d trial_stress;
  switch (stress_rate_) {
    case mpm::StressRate::None:
      trial_stress = stress + de * dstrain;
      break;
    case mpm::StressRate::Jaumann:
      trial_stress =
          this->compute_jaumann_stress(stress, dstrain, de, ptr, state_vars);
      break;
    case mpm::StressRate::GreenNaghdi:
      trial_stress = this->compute_green_naghdi_stress(stress, dstrain, de, ptr,
                                                       state_vars);
      break;
  }

  return trial_stress;
}

//! Compute consistent tangent matrix
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::InfinitesimalElastoPlastic<Tdim>::compute_consistent_tangent_matrix(
        const Vector6d& stress, const Vector6d& prev_stress,
        const Vector6d& dstrain, const ParticleBase<Tdim>* ptr,
        mpm::dense_map* state_vars) {
  //! Consistent tangent matrix
  Matrix6x6 const_tangent = this->compute_elasto_plastic_tensor(
      stress, dstrain, ptr, state_vars, true);

  return const_tangent;
}