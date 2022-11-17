//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointBase<Tdim>::PointBase(const VectorDim& coord) {
  // Check if the dimension is between 1 & 3
  static_assert((Tdim >= 1 && Tdim <= 3), "Invalid global dimension");
  coordinates_ = coord;
  // Logger
  std::string logger = "point" + std::to_string(Tdim) + "d";
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  initialise();
}

//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointBase<Tdim>::PointBase(Index id, const VectorDim& coord)
    : mpm::PointBase<Tdim>::PointBase(coord) {
  id_ = id;
}

//! Constructor with id, coordinates and status
template <unsigned Tdim>
mpm::PointBase<Tdim>::PointBase(Index id, const VectorDim& coord, bool status)
    : mpm::PointBase<Tdim>::PointBase(id, coord) {
  status_ = status;
}

// Compute reference location cell to point
template <unsigned Tdim>
bool mpm::PointBase<Tdim>::compute_reference_location() noexcept {
  // Set status of compute reference location
  bool status = false;
  // Compute local coordinates
  Eigen::Matrix<double, Tdim, 1> xi;
  // Check if the point is in cell
  if (cell_ != nullptr && cell_->is_point_in_cell(this->coordinates_, &xi)) {
    this->xi_ = xi;
    status = true;
  }

  return status;
}

// Initialise point properties
template <unsigned Tdim>
void mpm::PointBase<Tdim>::initialise() {
  displacement_.setZero();
  // Initialize scalar, vector, and tensor data properties
  this->vector_properties_["displacements"] = [&]() { return displacement(); };
}

// Assign a cell to point
template <unsigned Tdim>
bool mpm::PointBase<Tdim>::assign_cell(
    const std::shared_ptr<Cell<Tdim>>& cellptr) {
  bool status = true;
  try {
    Eigen::Matrix<double, Tdim, 1> xi;
    // Assign cell to the new cell ptr, if point can be found in new cell
    if (cellptr->is_point_in_cell(this->coordinates_, &xi)) {

      cell_ = cellptr;
      cell_id_ = cellptr->id();
      // Copy nodal pointer to cell
      nodes_.clear();
      nodes_ = cell_->nodes();

      // Compute reference location of particle
      bool xi_status = this->compute_reference_location();
      if (!xi_status) return false;
    } else {
      throw std::runtime_error("Point cannot be found in cell!");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assign a cell to point
template <unsigned Tdim>
bool mpm::PointBase<Tdim>::assign_cell_xi(
    const std::shared_ptr<Cell<Tdim>>& cellptr,
    const Eigen::Matrix<double, Tdim, 1>& xi) {
  bool status = true;
  try {
    // Assign cell to the new cell ptr, if point can be found in new cell
    if (cellptr != nullptr) {

      cell_ = cellptr;
      cell_id_ = cellptr->id();

      // Copy nodal pointer to cell
      nodes_.clear();
      nodes_ = cell_->nodes();

      // Assign the reference location of point
      bool xi_nan = false;

      // Check if point is within the cell
      for (unsigned i = 0; i < xi.size(); ++i)
        if (xi(i) < -1. || xi(i) > 1. || std::isnan(xi(i))) xi_nan = true;

      if (xi_nan == false)
        this->xi_ = xi;
      else
        return false;
    } else {
      throw std::runtime_error("Point cannot be found in cell!");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assign a cell id to point
template <unsigned Tdim>
bool mpm::PointBase<Tdim>::assign_cell_id(mpm::Index id) {
  bool status = false;
  try {
    // if a cell ptr is null
    if (cell_ == nullptr && id != std::numeric_limits<Index>::max()) {
      cell_id_ = id;
      status = true;
    } else {
      throw std::runtime_error("Invalid cell id or cell is already assigned!");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Remove cell for the point
template <unsigned Tdim>
void mpm::PointBase<Tdim>::remove_cell() {
  cell_id_ = std::numeric_limits<Index>::max();
  // Clear all the nodes
  nodes_.clear();
}

// Compute shape functions and gradients
template <unsigned Tdim>
void mpm::PointBase<Tdim>::compute_shapefn() noexcept {
  // Check if point has a valid cell ptr
  assert(cell_ != nullptr);
  // Get element ptr of a cell
  const auto element = cell_->element_ptr();

  // Identity matrix
  const auto& identity = Eigen::Matrix<double, Tdim, Tdim>::Identity();

  // Compute shape function of the point
  Eigen::Matrix<double, Tdim, 1> zero_natural_size =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  shapefn_ = element->shapefn(this->xi_, zero_natural_size, identity);
}

// Compute updated position of the point
template <unsigned Tdim>
void mpm::PointBase<Tdim>::compute_updated_position(
    double dt, bool velocity_update) noexcept {
  // Check if point has a valid cell ptr
  assert(cell_ != nullptr);
  // Get interpolated nodal velocity
  Eigen::Matrix<double, Tdim, 1> nodal_velocity =
      Eigen::Matrix<double, Tdim, 1>::Zero();

  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodal_velocity.noalias() +=
        shapefn_[i] * nodes_[i]->velocity(mpm::ParticlePhase::Solid);

  // Acceleration update
  if (!velocity_update) {
    // Get interpolated nodal acceleration
    Eigen::Matrix<double, Tdim, 1> nodal_acceleration =
        Eigen::Matrix<double, Tdim, 1>::Zero();
    for (unsigned i = 0; i < nodes_.size(); ++i)
      nodal_acceleration.noalias() +=
          shapefn_[i] * nodes_[i]->acceleration(mpm::ParticlePhase::Solid);

    // Update point velocity from interpolated nodal acceleration
    this->velocity_.noalias() += nodal_acceleration * dt;
  }
  // Update point velocity using interpolated nodal velocity
  else
    this->velocity_ = nodal_velocity;

  // New position  current position + velocity * dt
  this->coordinates_.noalias() += nodal_velocity * dt;
  // Update displacement (displacement is initialized from zero)
  this->displacement_.noalias() += nodal_velocity * dt;
}

//! Return point scalar data
template <unsigned Tdim>
inline double mpm::PointBase<Tdim>::scalar_data(
    const std::string& property) const {
  return (this->scalar_properties_.find(property) !=
          this->scalar_properties_.end())
             ? this->scalar_properties_.at(property)()
             : std::numeric_limits<double>::quiet_NaN();
}

//! Return point vector data
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, 1> mpm::PointBase<Tdim>::vector_data(
    const std::string& property) const {
  return (this->vector_properties_.find(property) !=
          this->vector_properties_.end())
             ? this->vector_properties_.at(property)()
             : Eigen::Matrix<double, Tdim, 1>::Constant(
                   std::numeric_limits<double>::quiet_NaN());
}

//! Return point tensor data
template <unsigned Tdim>
inline Eigen::VectorXd mpm::PointBase<Tdim>::tensor_data(
    const std::string& property) const {
  return (this->tensor_properties_.find(property) !=
          this->tensor_properties_.end())
             ? this->tensor_properties_.at(property)()
             : Eigen::Matrix<double, 6, 1>::Constant(
                   std::numeric_limits<double>::quiet_NaN());
}