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

// Initialise particle properties
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::initialise() {

  // Call the base class initialise function
  Particle<Tdim>::initialise();

  // Initialise extra properties
  couple_force_.setZero();
  this->vector_properties_["levelset_couple"] = [this]() {
    return this->levelset_couple();
  };
}

//! Return the approximate particle diameter
template <unsigned Tdim>
double mpm::ParticleLevelset<Tdim>::diameter() const {
  double diameter = 0.;
  if (Tdim == 2) diameter = 2.0 * std::sqrt(volume_ / M_PI);         // radial
  if (Tdim == 3) diameter = 2.0 * std::cbrt(volume_ * 0.75 / M_PI);  // radial
  return diameter;
}

//! Map levelset contact force
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::map_particle_contact_force_to_nodes(
    const double levelset_damping, const bool levelset_pic, double dt) {
  // Initialise levelset values at particle, per step
  double levelset = 0.;
  double levelset_mu = 0.;
  double levelset_alpha = 0.;
  double barrier_stiffness = 0.;
  double slip_threshold = 0.;
  VectorDim levelset_gradient = VectorDim::Zero();
  VectorDim contact_vel = VectorDim::Zero();

  // Initialise levelset couple vector data property
  this->vector_properties_["levelset_couple"] = []() {
    return VectorDim::Zero();
  };

  // Approximate radius of influence
  double mp_radius = 0.5 * diameter();

  // Global contact velocity update scheme
  if (!levelset_pic) contact_vel = velocity_;

  // Map levelset to particle
  for (unsigned i = 0; i < nodes_.size(); i++) {
    levelset += shapefn_[i] * nodes_[i]->levelset();
  }

  // Compute error minimum value
  if (levelset < std::numeric_limits<double>::epsilon())
    levelset = std::numeric_limits<double>::epsilon();

  // Contact only if mp within contact zone
  if ((levelset < mp_radius) && (levelset > 0.)) {

    // Map other levelset values to particle
    for (unsigned i = 0; i < nodes_.size(); i++) {
      levelset_gradient += dn_dx_.row(i).transpose() * nodes_[i]->levelset();
      levelset_mu += shapefn_[i] * nodes_[i]->levelset_mu();
      levelset_alpha += shapefn_[i] * nodes_[i]->levelset_alpha();
      barrier_stiffness += shapefn_[i] * nodes_[i]->barrier_stiffness();
      slip_threshold += shapefn_[i] * nodes_[i]->slip_threshold();

      // PIC contact velocity update scheme (map contact velocity from the
      // nodes)
      if (levelset_pic)
        contact_vel +=
            shapefn_[i] * nodes_[i]->velocity(mpm::ParticlePhase::Solid);
    }

    // Compute normals // LEDT check this once separate meshes
    VectorDim levelset_normal = levelset_gradient.normalized();

    // Compute contact force in particle
    VectorDim couple_force = compute_levelset_contact_force(
        levelset, levelset_normal, levelset_mu, levelset_alpha,
        barrier_stiffness, slip_threshold, mp_radius, contact_vel,
        levelset_damping, dt);

    // Compute nodal contact force
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      nodes_[i]->update_external_force(true, mpm::ParticlePhase::Solid,
                                       (shapefn_[i] * couple_force));
    }
  }
}

//! Compute levelset contact force
template <unsigned Tdim>
typename mpm::ParticleLevelset<Tdim>::VectorDim
    mpm::ParticleLevelset<Tdim>::compute_levelset_contact_force(
        double levelset, const VectorDim& levelset_normal, double levelset_mu,
        double levelset_alpha, double barrier_stiffness, double slip_threshold,
        const double mp_radius, const VectorDim& contact_vel,
        const double levelset_damping, double dt) noexcept {
  // Calculate normal coupling force magnitude
  double couple_normal_mag =
      barrier_stiffness * (levelset - mp_radius) *
      (2. * log(levelset / mp_radius) - (mp_radius / levelset) + 1.);

  // Calculate normal coupling force
  VectorDim couple_force_normal = couple_normal_mag * levelset_normal;

  // Calculate levelset tangential unit vector
  VectorDim levelset_tangent = VectorDim::Zero();
  double vel_n = contact_vel.dot(levelset_normal);
  VectorDim tangent_calc = contact_vel - (vel_n * levelset_normal);
  if (tangent_calc.norm() > std::numeric_limits<double>::epsilon())
    levelset_tangent = tangent_calc.normalized();

  // Apply friction smoothing function, if applicable
  double friction_smoothing = 1.;
  if (slip_threshold > 0.) {
    // Calculate cumulative slip magnitude // LEDT check: abs val?
    cumulative_slip_mag += dt * contact_vel.dot(levelset_tangent);
    // Calculate friction smoothing
    if (abs(cumulative_slip_mag) < slip_threshold) {
      friction_smoothing =
          -(std::pow(cumulative_slip_mag, 2) / std::pow(slip_threshold, 2)) +
          2 * abs(cumulative_slip_mag) / slip_threshold;
    }
  }

  // Calculate friction tangential coupling force magnitude
  double tangent_friction =
      friction_smoothing * levelset_mu * couple_normal_mag;

  // Calculate adhesion tangential coupling force magnitude
  double contact_area = volume_ / size_[0];
  double tangent_adhesion = levelset_alpha * contact_area;

  // Calculate tangential coupling force magntiude
  double couple_tangent_mag = tangent_friction + tangent_adhesion;

  // Calculate tangential contact force magnitude
  double contact_tangent_mag = (mass_ * contact_vel / dt).dot(levelset_tangent);

  // Couple must not exceed cancellation of contact tangential force
  bool uphill_prevention = true;  // LEDT temporary bool for testing
  if (uphill_prevention)
    couple_tangent_mag = std::min(couple_tangent_mag, contact_tangent_mag);

  // Calculate tangential coupling force vector
  VectorDim couple_force_tangent = -levelset_tangent * couple_tangent_mag;

  // Calculate total coupling force vector
  VectorDim couple_force_ = couple_force_normal + couple_force_tangent;

  // Damp couple if mp moving away from boundary
  // LEDT check if 0 tangent resistance
  if ((contact_vel.dot(levelset_normal)) >= 0.)
    couple_force_ = (1. - levelset_damping) * couple_force_;

  // Store levelset couple vector data property
  this->vector_properties_["levelset_couple"] = [couple_force_]() {
    return couple_force_;
  };

  return couple_force_;
}
