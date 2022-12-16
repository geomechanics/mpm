//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointPenaltyDisplacement<Tdim>::PointPenaltyDisplacement(
    Index id, const VectorDim& coord)
    : mpm::PointBase<Tdim>::PointBase(id, coord) {
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();

  // Logger
  std::string logger = "PointPenaltyDisplacement" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Constructor with id, coordinates and status
template <unsigned Tdim>
mpm::PointPenaltyDisplacement<Tdim>::PointPenaltyDisplacement(
    Index id, const VectorDim& coord, bool status)
    : mpm::PointBase<Tdim>::PointBase(id, coord, status) {
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Logger
  std::string logger = "PointPenaltyDisplacement" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Initialise point properties
template <unsigned Tdim>
void mpm::PointPenaltyDisplacement<Tdim>::initialise() {
  mpm::PointBase<Tdim>::initialise();

  imposed_displacement_.setZero();
  imposed_velocity_.setZero();
  imposed_acceleration_.setZero();
}

// Reinitialise point properties
template <unsigned Tdim>
void mpm::PointPenaltyDisplacement<Tdim>::reinitialise(double dt) {
  // Convert imposition of velocity and acceleration to displacement
  // NOTE: This only consider translational velocity and acceleration: no
  // angular
  imposed_displacement_ =
      (imposed_velocity_ * dt) + (0.5 * imposed_acceleration_ * dt * dt);
}

//! Compute updated position
template <unsigned Tdim>
void mpm::PointPenaltyDisplacement<Tdim>::compute_updated_position(
    double dt, mpm::VelocityUpdate velocity_update,
    double blending_ratio) noexcept {
  // Update position and displacements
  coordinates_.noalias() += imposed_displacement_;
  displacement_.noalias() += imposed_displacement_;
}
