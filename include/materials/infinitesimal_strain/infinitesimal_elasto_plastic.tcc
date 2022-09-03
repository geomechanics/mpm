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