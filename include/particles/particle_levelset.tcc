//! Initialise levelset particle properties
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::initialise() {}

//! Map levelset contact force
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::map_particle_contact_force_to_nodes(
    double dt) {
  // Compute levelset values at particle
  double levelset = 0;
  double levelset_mu = 0;
  double barrier_stiffness = 0;
  double slip_threshold = 0;
  double levelset_mp_radius = 0;
  VectorDim levelset_gradient = VectorDim::Zero();
  for (unsigned i = 0; i < nodes_.size(); i++) {
    // Map levelset and compute gradient
    levelset +=
        shapefn_[i] * nodes_[i]->levelset();  // LEDT need to assign to nodes
    levelset_gradient += dn_dx_.row(i).transpose() * nodes_[i]->levelset();
    // Map other input variables
    levelset_mu += shapefn_[i] * nodes_[i]->levelset_mu();
    barrier_stiffness += shapefn_[i] * nodes_[i]->barrier_stiffness();
    slip_threshold += shapefn_[i] * nodes_[i]->slip_threshold();
    levelset_mp_radius += shapefn_[i] * nodes_[i]->levelset_mp_radius();
  }
  // Compute normals
  VectorDim levelset_normal = levelset_gradient.normalized();

  // Compute contact force in particle
  VectorDim force = compute_levelset_contact_force(
      levelset, levelset_normal, levelset_mu, barrier_stiffness, slip_threshold,
      levelset_mp_radius, dt);

  // Compute nodal contact force // LEDT should add rather than replace
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::Solid,
                                     (shapefn_[i] * force));
  }
}

//! Compute levelset contact force
template <unsigned Tdim>
typename mpm::ParticleLevelset<Tdim>::VectorDim
    mpm::ParticleLevelset<Tdim>::compute_levelset_contact_force(
        double levelset, const VectorDim& levelset_normal, double levelset_mu,
        double barrier_stiffness, double slip_threshold,
        double levelset_mp_radius, double dt) noexcept {
  // Temporary computer error minimum value // LEDT check
  if (levelset < std::numeric_limits<double>::epsilon())
    levelset = std::numeric_limits<double>::epsilon();

  // Calculate normal coupling force magnitude
  double couple_force_normal_mag = 0.0;
  if (levelset < levelset_mp_radius && levelset > 0.0)
    double couple_force_normal_mag =
        barrier_stiffness * (levelset - levelset_mp_radius) *
        (2 * log(levelset / levelset_mp_radius -
                 (levelset_mp_radius / levelset) + 1));

  // Calculate normal coupling force
  VectorDim couple_force_normal = couple_force_normal_mag * levelset_normal;

  // Calculate cumulative slip magnitude
  double vel_n = velocity_.dot(levelset_normal);
  VectorDim levelset_tangential =
      (velocity_ - vel_n * levelset_normal) /
      (velocity_ - vel_n * levelset_normal).normalize();
  cumulative_slip_mag += dt * velocity_.dot(levelset_tangential);

  // Calculate friction smoothing function
  double friction_smoothing = 1.0;
  if (abs(cumulative_slip_mag) < slip_threshold)
    friction_smoothing =
        -(std::pow(cumulative_slip_mag, 2) / std::pow(slip_threshold, 2)) +
        2 * abs(cumulative_slip_mag) / slip_threshold;

  // Calculate tangential coupling force
  VectorDim couple_force_tangential = friction_smoothing * levelset_mu *
                                      couple_force_normal_mag *
                                      levelset_tangential;

  // Calculate total coupling force
  VectorDim couple_force_ = couple_force_normal + couple_force_tangential;

  return couple_force_;
}
