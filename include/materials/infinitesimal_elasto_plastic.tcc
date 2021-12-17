//! Compute consistent tangent matrix
template <unsigned Tdim>
Eigen::Matrix<double, 6, 6>
    mpm::InfinitesimalElastoPlastic<Tdim>::compute_consistent_tangent_matrix(
        const Vector6d& stress, const Vector6d& prev_stress,
        const Vector6d& dstrain, const ParticleBase<Tdim>* ptr,
        mpm::dense_map* state_vars) {
}