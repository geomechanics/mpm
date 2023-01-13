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
void mpm::PointPenaltyDisplacement<Tdim>::initialise_property(double dt) {
  assert(area_ != std::numeric_limits<double>::max());
  // Convert imposition of velocity and acceleration to displacement
  // NOTE: This only consider translational velocity and acceleration: no
  // angular
  imposed_displacement_ =
      (imposed_velocity_ * dt) + (0.5 * imposed_acceleration_ * dt * dt);

  for (unsigned i = 0; i < Tdim; ++i)
    if (std::abs(imposed_displacement_(i)) < 1.E-15)
      imposed_displacement_(i) = 0.;
}

//! Apply point velocity constraints
template <unsigned Tdim>
void mpm::PointPenaltyDisplacement<Tdim>::apply_point_velocity_constraints(
    unsigned dir, double velocity) {
  // Set particle velocity constraint
  this->imposed_velocity_(dir) = velocity;
}

//! Compute updated position
template <unsigned Tdim>
void mpm::PointPenaltyDisplacement<Tdim>::compute_updated_position(
    double dt) noexcept {
  // Update position and displacements
  coordinates_.noalias() += imposed_displacement_;
  displacement_.noalias() += imposed_displacement_;
}

//! Map penalty stiffness matrix to cell
template <unsigned Tdim>
inline bool
    mpm::PointPenaltyDisplacement<Tdim>::map_stiffness_matrix_to_cell() {
  bool status = true;
  try {
    // Initialise stiffness matrix
    const unsigned matrix_size = nodes_.size() * Tdim;
    Eigen::MatrixXd penalty_stiffness(matrix_size, matrix_size);
    penalty_stiffness.setZero();

    // Arrange shape function
    Eigen::MatrixXd shape_function(Tdim, matrix_size);
    shape_function.setZero();
    for (unsigned i = 0; i < nodes_.size(); i++) {
      if (shapefn_[i] > std::numeric_limits<double>::epsilon()) {
        for (unsigned int j = 0; j < Tdim; j++) {
          shape_function(j, Tdim * i + j) = shapefn_[i];
        }
      }
    }

    // Assign stiffness matrix
    penalty_stiffness.noalias() += shape_function.transpose() * shape_function;

    // Compute local penalty stiffness matrix
    cell_->compute_local_penalty_stiffness_matrix(penalty_stiffness, area_,
                                                  penalty_factor_);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map enforcement force
template <unsigned Tdim>
void mpm::PointPenaltyDisplacement<Tdim>::map_boundary_force(unsigned phase) {
  // Calculate gap_function: nodal_displacement - imposed_displacement
  const unsigned matrix_size = nodes_.size() * Tdim;
  Eigen::VectorXd gap_function(matrix_size);
  gap_function.setZero();
  for (unsigned i = 0; i < nodes_.size(); i++) {
    const auto& n_disp = nodes_[i]->displacement(phase);
    gap_function.segment(i * Tdim, Tdim) = n_disp - imposed_displacement_;
  }

  // Arrange shape function
  Eigen::MatrixXd shape_function(Tdim, matrix_size);
  shape_function.setZero();
  for (unsigned i = 0; i < nodes_.size(); i++) {
    if (shapefn_[i] > std::numeric_limits<double>::epsilon()) {
      for (unsigned int j = 0; j < Tdim; j++) {
        shape_function(j, Tdim * i + j) = shapefn_[i];
      }
    }
  }

  // Penalty force vector
  const auto& penalty_force = shape_function.transpose() * shape_function *
                              gap_function * area_ * penalty_factor_;

  // Compute nodal external forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(
        true, phase, -1.0 * penalty_force.segment(i * Tdim, Tdim));
}