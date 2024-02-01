//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleLevelset<Tdim>::ParticleLevelset(Index id, const VectorDim& coord)
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
      "particlelevelset" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleLevelset<Tdim>::ParticleLevelset(Index id, const VectorDim& coord,
                                              bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  this->initialise();
  cell_ = nullptr;
  nodes_.clear();
  // Set material containers
  this->initialise_material(1);
  //! Logger
  std::string logger =
      "particlelevelset" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// LEDT above is repeat from particle.tcc
// //! Initialise levelset particle properties // LEDT REMOVE
// template <unsigned Tdim>
// void mpm::ParticleLevelset<Tdim>::initialise() {}

//! Map levelset contact force
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::map_particle_contact_force_to_nodes(
    double dt, const double levelset_mp_radius) {
  // std::cout << "-->5.1a" << std::endl;  // LEDT REMOVE!
  // Compute levelset values at particle
  double levelset = 0;
  double levelset_mu = 0;
  double barrier_stiffness = 0;
  double slip_threshold = 0;
  VectorDim levelset_gradient = VectorDim::Zero();

  // std::cout << "-->map_particle_contact_force_to_nodes() levelset_mp_radius "
  //           << levelset_mp_radius << std::endl;  // LEDT REMOVE!
  // std::cout << "-->5.1b" << std::endl;  // LEDT REMOVE!
  // std::cout << "-->nodes_.size() " << nodes_.size() << std::endl;
  // LEDT REMOVE!

  // std::cout << "-->id " << this->id_ << std::endl;  // LEDT REMOVE!
  // std::cout << "-->shapefn0 " << shapefn_[0] << std::endl;  // LEDT REMOVE!
  // std::cout << "-->c0 " << nodes_[0]->coordinates()
  //           << std::endl;                                        // LEDT
  //           REMOVE!
  // std::cout << "-->ls0 " << nodes_[0]->levelset() << std::endl;  // LEDT
  // REMOVE! std::cout << "-->c1 " << nodes_[1]->coordinates()
  //           << std::endl;                                        // LEDT
  //           REMOVE!
  // std::cout << "-->ls1 " << nodes_[1]->levelset() << std::endl;  // LEDT
  // REMOVE! std::cout << "-->c2 " << nodes_[2]->coordinates()
  //           << std::endl;                                        // LEDT
  //           REMOVE!
  // std::cout << "-->ls2 " << nodes_[2]->levelset() << std::endl;  // LEDT
  // REMOVE! std::cout << "-->c3 " << nodes_[3]->coordinates()
  //           << std::endl;                                        // LEDT
  //           REMOVE!
  // std::cout << "-->ls3 " << nodes_[3]->levelset() << std::endl;  // LEDT
  // REMOVE!

  for (unsigned i = 0; i < nodes_.size(); i++) {
    // for (unsigned i = 0; i < particles_.size(); i++) {

    // std::cout << "-->5.1b1" << std::endl;    // LEDT REMOVE!
    // Map levelset and compute gradient
    levelset +=
        shapefn_[i] * nodes_[i]->levelset();  // LEDT need to assign to nodes
    // std::cout << "-->5.1b2" << std::endl;     // LEDT REMOVE!
    levelset_gradient += dn_dx_.row(i).transpose() * nodes_[i]->levelset();
    // std::cout << "-->5.1b3" << std::endl;  // LEDT REMOVE!
    // Map other input variables
    levelset_mu += shapefn_[i] * nodes_[i]->levelset_mu();
    barrier_stiffness += shapefn_[i] * nodes_[i]->barrier_stiffness();
    slip_threshold += shapefn_[i] * nodes_[i]->slip_threshold();
    // std::cout << "-->5.1b4" << std::endl;  // LEDT REMOVE!
  }
  // std::cout << "-->5.1c" << std::endl;  // LEDT REMOVE!
  // Compute normals
  VectorDim levelset_normal = levelset_gradient.normalized();

  // std::cout << "-->5.1d" << std::endl;  // LEDT REMOVE!
  // Compute contact force in particle
  VectorDim force = compute_levelset_contact_force(
      levelset, levelset_normal, levelset_mu, barrier_stiffness, slip_threshold,
      levelset_mp_radius, dt);

  // std::cout << "-->node force " << force << std::endl;  // LEDT REMOVE!

  // std::cout << "-->5.1e" << std::endl;  // LEDT REMOVE!
  // Compute nodal contact force // LEDT should add rather than replace
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::Solid,
                                     (shapefn_[i] * force));
  }
  // std::cout << "-->5.1f" << std::endl;  // LEDT REMOVE!
}

//! Compute levelset contact force
template <unsigned Tdim>
typename mpm::ParticleLevelset<Tdim>::VectorDim
    mpm::ParticleLevelset<Tdim>::compute_levelset_contact_force(
        double levelset, const VectorDim& levelset_normal, double levelset_mu,
        double barrier_stiffness, double slip_threshold,
        const double levelset_mp_radius, double dt) noexcept {
  // Coupling force zero by default
  VectorDim couple_force_ = VectorDim::Zero();

  // Temporary computer error minimum value // LEDT check
  if (levelset < std::numeric_limits<double>::epsilon())
    levelset = std::numeric_limits<double>::epsilon();

  // Calculate normal coupling force magnitude
  double couple_force_normal_mag = 0.0;
  if (levelset < levelset_mp_radius && levelset > 0.0) {
    couple_force_normal_mag = barrier_stiffness *
                              (levelset - levelset_mp_radius) *
                              (2 * log(levelset / levelset_mp_radius -
                                       (levelset_mp_radius / levelset) + 1));
    // std::cout << "-->couple_force_normal_mag " << couple_force_normal_mag
    //           << std::endl;  // LEDT REMOVE!

    // Calculate normal coupling force
    VectorDim couple_force_normal = couple_force_normal_mag * levelset_normal;
    // std::cout << "-->couple_force_normal " << couple_force_normal
    //           << std::endl;  // LEDT REMOVE!

    // Calculate levelset_tangential for cumulative slip magnitude
    // VectorDim levelset_tangential = (velocity_ - vel_n * levelset_normal) /
    // (velocity_ - vel_n * levelset_normal).normalize();
    double vel_n = velocity_.dot(levelset_normal);
    VectorDim levelset_tangential_expr = velocity_ - vel_n * levelset_normal;
    VectorDim levelset_tangential_expr_norm =
        levelset_tangential_expr.normalized();  // LEDT check normalize()
    VectorDim levelset_tangential =
        levelset_tangential_expr.cwiseQuotient(levelset_tangential_expr_norm);

    // Calculate cumulative slip magnitude
    cumulative_slip_mag += dt * velocity_.dot(levelset_tangential);
    // std::cout << "-->cumulative_slip_mag " << cumulative_slip_mag
    //           << std::endl;  // LEDT REMOVE!

    // Calculate friction smoothing function
    double friction_smoothing = 1.0;
    if (abs(cumulative_slip_mag) < slip_threshold)
      friction_smoothing =
          -(std::pow(cumulative_slip_mag, 2) / std::pow(slip_threshold, 2)) +
          2 * abs(cumulative_slip_mag) / slip_threshold;
    // std::cout << "-->friction_smoothing " << friction_smoothing
    //           << std::endl;  // LEDT REMOVE!

    // Calculate tangential coupling force
    VectorDim couple_force_tangential = friction_smoothing * levelset_mu *
                                        couple_force_normal_mag *
                                        levelset_tangential;
    // std::cout << "-->couple_force_tangential " << couple_force_tangential
    //           << std::endl;  // LEDT REMOVE!

    // Calculate total coupling force
    couple_force_ = couple_force_normal + couple_force_tangential;
  }
  return couple_force_;
}
