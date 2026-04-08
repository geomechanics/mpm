//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointDirichletDirect<Tdim>::PointDirichletDirect(Index id,
                                                      const VectorDim& coord)
    : mpm::PointBase<Tdim>::PointBase(id, coord) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();

  // Logger
  std::string logger = "PointDirichletDirect" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Constructor with id, coordinates and status
template <unsigned Tdim>
mpm::PointDirichletDirect<Tdim>::PointDirichletDirect(Index id,
                                                      const VectorDim& coord,
                                                      bool status)
    : mpm::PointBase<Tdim>::PointBase(id, coord, status) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Logger
  std::string logger = "PointDirichletDirect" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Initialise point properties
template <unsigned Tdim>
void mpm::PointDirichletDirect<Tdim>::initialise() {
  mpm::PointBase<Tdim>::initialise();

  imposed_displacement_.setZero();
  imposed_velocity_.setZero();
  imposed_acceleration_.setZero();
  constraint_flags_.setZero();
}

//! Assign point properties
template <unsigned Tdim>
void mpm::PointDirichletDirect<Tdim>::assign_properties(
    const std::map<std::string, double>& scalar_properties,
    const std::map<std::string, std::vector<double>>& vector_properties) {
  // Assign constraint flags
  if (vector_properties.count("constraint_flags")) {
    const auto& flags = vector_properties.at("constraint_flags");
    for (unsigned i = 0; i < Tdim; ++i)
      constraint_flags_(i) = static_cast<int>(flags[i]);
  }
}

// Reinitialise point properties
template <unsigned Tdim>
void mpm::PointDirichletDirect<Tdim>::initialise_properties(double dt) {
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

//! Assign point velocity constraints
template <unsigned Tdim>
void mpm::PointDirichletDirect<Tdim>::assign_velocity_constraints(
    unsigned dir, double velocity) {
  // Set particle velocity constraint
  this->imposed_velocity_(dir) = velocity;

  // Iterater over nodes and assign velocity constraint
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    for (unsigned dir = 0; dir < Tdim; ++dir) {
      if (constraint_flags_(dir) != 0)
        nodes_[i]->assign_moving_velocity_constraint(dir,
                                                     imposed_velocity_(dir));
    }
  }
}

//! Compute updated position
template <unsigned Tdim>
void mpm::PointDirichletDirect<Tdim>::compute_updated_position(
    double dt) noexcept {
  // Update position and displacements
  coordinates_.noalias() += imposed_displacement_;
  displacement_.noalias() += imposed_displacement_;
}

//! Compute size of serialized point data
template <unsigned Tdim>
int mpm::PointDirichletDirect<Tdim>::compute_pack_size() const {
  int total_size = mpm::PointBase<Tdim>::compute_pack_size();
  int partial_size;
#ifdef USE_MPI
  // Constraint flags
  MPI_Pack_size(1 * Tdim, MPI_INT, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;
#endif
  return total_size;
}

//! Serialize point data
template <unsigned Tdim>
std::vector<uint8_t> mpm::PointDirichletDirect<Tdim>::serialize() {
  // Compute pack size
  if (pack_size_ == 0) pack_size_ = compute_pack_size();
  // Initialize data buffer
  std::vector<uint8_t> data;
  data.resize(pack_size_);
  uint8_t* data_ptr = &data[0];
  int position = 0;

#ifdef USE_MPI
  // Type
  int type = PointType.at(this->type());
  MPI_Pack(&type, 1, MPI_INT, data_ptr, data.size(), &position, MPI_COMM_WORLD);

  // ID
  MPI_Pack(&id_, 1, MPI_UNSIGNED_LONG_LONG, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);
  // Area
  MPI_Pack(&area_, 1, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

  // Coordinates
  MPI_Pack(coordinates_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Displacement
  MPI_Pack(displacement_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Normal vector
  MPI_Pack(normal_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

  // Cell id
  MPI_Pack(&cell_id_, 1, MPI_UNSIGNED_LONG_LONG, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Status
  MPI_Pack(&status_, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

  // Constraint flags
  MPI_Pack(constraint_flags_.data(), Tdim, MPI_INT, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

#endif
  return data;
}

//! Deserialize point data
template <unsigned Tdim>
void mpm::PointDirichletDirect<Tdim>::deserialize(
    const std::vector<uint8_t>& data) {
  uint8_t* data_ptr = const_cast<uint8_t*>(&data[0]);
  int position = 0;

#ifdef USE_MPI
  // Type
  int type;
  MPI_Unpack(data_ptr, data.size(), &position, &type, 1, MPI_INT,
             MPI_COMM_WORLD);
  assert(type == PointType.at(this->type()));

  // ID
  MPI_Unpack(data_ptr, data.size(), &position, &id_, 1, MPI_UNSIGNED_LONG_LONG,
             MPI_COMM_WORLD);
  // area
  MPI_Unpack(data_ptr, data.size(), &position, &area_, 1, MPI_DOUBLE,
             MPI_COMM_WORLD);

  // Coordinates
  MPI_Unpack(data_ptr, data.size(), &position, coordinates_.data(), Tdim,
             MPI_DOUBLE, MPI_COMM_WORLD);

  // Displacement
  MPI_Unpack(data_ptr, data.size(), &position, displacement_.data(), Tdim,
             MPI_DOUBLE, MPI_COMM_WORLD);

  // Normal vector
  MPI_Unpack(data_ptr, data.size(), &position, normal_.data(), Tdim, MPI_DOUBLE,
             MPI_COMM_WORLD);

  // cell id
  MPI_Unpack(data_ptr, data.size(), &position, &cell_id_, 1,
             MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
  // status
  MPI_Unpack(data_ptr, data.size(), &position, &status_, 1, MPI_C_BOOL,
             MPI_COMM_WORLD);

  // Constraint flags
  MPI_Unpack(data_ptr, data.size(), &position, constraint_flags_.data(), Tdim,
             MPI_INT, MPI_COMM_WORLD);

#endif
}