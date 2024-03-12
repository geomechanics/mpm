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
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Set material containers
  this->initialise_material(1);
  //! Logger
  std::string logger =
      "particlelevelset" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Return the approximate particle diameter
template <unsigned Tdim>
double mpm::ParticleLevelset<Tdim>::diameter() const {
  double diameter = 0.;
  if (Tdim == 2) diameter = 2.0 * std::sqrt(volume_ / M_PI);
  if (Tdim == 3) diameter = 2.0 * std::pow(volume_ * 0.75 / M_PI, (1 / 3));
  return diameter;
}

//! Map levelset contact force
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::map_particle_contact_force_to_nodes(
    double dt) {
  // Compute levelset values at particle
  double levelset = 0;
  double levelset_mu = 0;
  double barrier_stiffness = 0;
  double slip_threshold = 0;
  VectorDim levelset_gradient = VectorDim::Zero();

  for (unsigned i = 0; i < nodes_.size(); i++) {
    // Map levelset and compute gradient
    levelset += shapefn_[i] * nodes_[i]->levelset();
    levelset_gradient += dn_dx_.row(i).transpose() * nodes_[i]->levelset();

    // Map other input variables
    levelset_mu += shapefn_[i] * nodes_[i]->levelset_mu();
    barrier_stiffness += shapefn_[i] * nodes_[i]->barrier_stiffness();
    slip_threshold += shapefn_[i] * nodes_[i]->slip_threshold();
  }

  // Compute normals // LEDT check this once separate meshes
  VectorDim levelset_normal = levelset_gradient.normalized();

  // Get radius
  const double mp_radius =
      0.5 * diameter() + std::numeric_limits<double>::epsilon();

  // Compute contact force in particle
  VectorDim force = compute_levelset_contact_force(
      levelset, levelset_normal, levelset_mu, barrier_stiffness, slip_threshold,
      mp_radius, dt);

  // Compute nodal contact force
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
        double barrier_stiffness, double slip_threshold, const double mp_radius,
        double dt) noexcept {
  // Coupling force zero by default
  VectorDim couple_force_ = VectorDim::Zero();

  // Temporary computer error minimum value // LEDT check
  if (levelset < std::numeric_limits<double>::epsilon())
    levelset = std::numeric_limits<double>::epsilon();

  // Calculate coupling force if levelset contacted
  if ((levelset < mp_radius) && (levelset > 0.)) {

    // Calculate normal coupling force magnitude
    double couple_force_normal_mag =
        barrier_stiffness * (levelset - mp_radius) *
        (2 * log(levelset / mp_radius) - (mp_radius / levelset) + 1);

    // Calculate normal coupling force
    VectorDim couple_force_normal = couple_force_normal_mag * levelset_normal;

    // Calculate levelset_tangential for cumulative slip magnitude
    double vel_n = velocity_.dot(levelset_normal);
    VectorDim tangent_calc = velocity_ - (vel_n * levelset_normal);
    VectorDim levelset_tangential = tangent_calc.normalized();

    // Fix tangential direction if applicable // LEDT need to add 3D
    if ((Tdim == 2) && (velocity_[0] * levelset_tangential[0] < 0)) {
      levelset_tangential = -levelset_tangential;
    }

    // Replace levelset_tangential if zero velocity
    if (abs(vel_n) < std::numeric_limits<double>::epsilon())
      levelset_tangential = VectorDim::Zero();

    // Calculate cumulative slip magnitude // LEDT check: abs val? per-particle?
    cumulative_slip_mag += dt * velocity_.dot(levelset_tangential);

    // Calculate friction smoothing function
    double friction_smoothing = 1.0;
    if (abs(cumulative_slip_mag) < slip_threshold)
      friction_smoothing =
          -(std::pow(cumulative_slip_mag, 2) / std::pow(slip_threshold, 2)) +
          2 * abs(cumulative_slip_mag) / slip_threshold;

    // Calculate tangential coupling force
    VectorDim couple_force_tangential = -friction_smoothing * levelset_mu *
                                        couple_force_normal_mag *
                                        levelset_tangential;

    // if (id_ == 0) {
    //   std::cout << "levelset_normal: " << levelset_normal << std::endl;
    //   // std::cout << "levelset_normal_mag: " << couple_force_normal_mag
    //   //           << std::endl;
    //   std::cout << "levelset_tangential: " << levelset_tangential <<
    //   std::endl; std::cout << "couple_force_tangential: " <<
    //   couple_force_tangential
    //             << std::endl;
    //   std::cout << "couple_force_normal: " << couple_force_normal <<
    //   std::endl;
    // }

    // Calculate total coupling force
    couple_force_ = couple_force_normal + couple_force_tangential;
  }
  return couple_force_;
}
