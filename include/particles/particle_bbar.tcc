//! Construct a particle with id and coordinates
template <unsigned Tdim>
mpm::ParticleBbar<Tdim>::ParticleBbar(Index id, const VectorDim& coord)
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
      "particle_bbar" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Construct a particle with id, coordinates and status
template <unsigned Tdim>
mpm::ParticleBbar<Tdim>::ParticleBbar(Index id, const VectorDim& coord,
                                      bool status)
    : mpm::Particle<Tdim>(id, coord, status) {
  this->initialise();
  cell_ = nullptr;
  nodes_.clear();
  // Set material containers
  this->initialise_material(1);
  //! Logger
  std::string logger =
      "particle_bbar" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}