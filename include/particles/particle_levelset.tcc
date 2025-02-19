//! Define static member variables
template <unsigned Tdim>
double mpm::ParticleLevelset<Tdim>::levelset_damping_ = 0.0;

template <unsigned Tdim>
bool mpm::ParticleLevelset<Tdim>::levelset_pic_ = false;

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

//! Update time-independent static levelset properties
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::update_levelset_static_properties(
    double levelset_damping, bool levelset_pic) {
  levelset_damping_ = levelset_damping;  // LEDT not used for ori
  levelset_pic_ = levelset_pic;          // LEDT not used for ori
}

//! Update time-independent mp levelset properties
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::update_levelset_mp_properties() {
  this->mp_radius_ = 0.5 * this->diameter();  // radial influence
}

//! Update contact force due to levelset
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::levelset_contact_force(double dt) {

  // Map levelset to particle
  map_levelset_to_particle();

  // Check if particle in contact with levelset
  if (is_levelset_contact()) {

    // Compute levelset contact force at particle
    compute_particle_contact_force(dt);

    // Map levelset contact force to nodes
    map_contact_force_to_nodes();
  }

  // Update levelset vtk data properties at particle
  update_levelset_vtk();
}

//! Map levelset to particle
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::map_levelset_to_particle() noexcept {
  // Reset mapped levelset values at particle, per step
  levelset_ = 0.;
  levelset_mu_ = 0.;
  barrier_stiffness_ = 0.;
  slip_threshold_ = 0.;
  levelset_gradient_ = VectorDim::Zero();

  // Reset levelset vtk data properties
  couple_force_ = VectorDim::Zero();

  // Map levelset to particle
  for (unsigned i = 0; i < nodes_.size(); i++) {
    levelset_ += shapefn_[i] * nodes_[i]->levelset();
  }

  // Compute error minimum value
  if (levelset_ < std::numeric_limits<double>::epsilon())
    levelset_ = std::numeric_limits<double>::epsilon();
}

//! Check if particle in contact with levelset
template <unsigned Tdim>
bool mpm::ParticleLevelset<Tdim>::is_levelset_contact() noexcept {
  if ((levelset_ < mp_radius_) && (levelset_ > 0.))
    return true;
  else
    return false;
}

//! Compute levelset contact force at particle
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::compute_particle_contact_force(
    double dt) noexcept {

  // Global contact velocity update scheme
  contact_vel_ = velocity_;

  // Map other levelset values to particle
  for (unsigned i = 0; i < nodes_.size(); i++) {
    levelset_gradient_ += dn_dx_.row(i).transpose() * nodes_[i]->levelset();
    levelset_mu_ += shapefn_[i] * nodes_[i]->levelset_mu();
    barrier_stiffness_ += shapefn_[i] * nodes_[i]->barrier_stiffness();
    slip_threshold_ += shapefn_[i] * nodes_[i]->slip_threshold();
  }

  // Compute normals
  levelset_normal_ = levelset_gradient_.normalized();

  // Calculate normal coupling force magnitude
  double couple_normal_mag =
      barrier_stiffness_ * (levelset_ - mp_radius_) *
      (2. * log(levelset_ / mp_radius_) - (mp_radius_ / levelset_) + 1.);

  // Calculate normal coupling force
  VectorDim couple_force_normal = couple_normal_mag * levelset_normal_;

  // Calculate levelset tangential unit vector
  double vel_n = contact_vel_.dot(levelset_normal_);
  VectorDim tangent_calc = contact_vel_ - (vel_n * levelset_normal_);
  if (tangent_calc.norm() > std::numeric_limits<double>::epsilon())
    levelset_tangent_ = tangent_calc.normalized();

  // Apply friction smoothing function, if applicable
  double friction_smoothing = 1.;
  if (slip_threshold_ > 0.) {
    // Calculate cumulative slip magnitude // LEDT check
    cumulative_slip_mag_ += dt * contact_vel_.dot(levelset_tangent_);
    // Calculate friction smoothing
    if (abs(cumulative_slip_mag_) < slip_threshold_) {
      friction_smoothing =
          -(std::pow(cumulative_slip_mag_, 2) / std::pow(slip_threshold_, 2)) +
          2 * abs(cumulative_slip_mag_) / slip_threshold_;
    }
  }

  // Calculate friction tangential coupling force magnitude
  double couple_tangent_mag =
      friction_smoothing * levelset_mu_ * couple_normal_mag;

  // Calculate tangential coupling force vector
  VectorDim couple_force_tangent = -levelset_tangent_ * couple_tangent_mag;

  // Calculate total coupling force vector
  couple_force_ = couple_force_normal + couple_force_tangent;
}

//! Map levelset contact force to nodes
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::map_contact_force_to_nodes() noexcept {
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_external_force(true, mpm::ParticlePhase::Solid,
                                     (shapefn_[i] * couple_force_));
  }
}

//! Update levelset vtk data properties at particle
template <unsigned Tdim>
void mpm::ParticleLevelset<Tdim>::update_levelset_vtk() noexcept {
  this->vector_properties_["levelset_couple"] = [this]() {
    return this->couple_force_;
  };
  this->scalar_properties_["levelset"] = [this]() { return this->levelset_; };
}