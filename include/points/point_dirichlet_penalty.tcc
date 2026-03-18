//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointDirichletPenalty<Tdim>::PointDirichletPenalty(Index id,
                                                        const VectorDim& coord)
    : mpm::PointBase<Tdim>::PointBase(id, coord) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();

  // Logger
  std::string logger = "PointDirichletPenalty" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Constructor with id, coordinates and status
template <unsigned Tdim>
mpm::PointDirichletPenalty<Tdim>::PointDirichletPenalty(Index id,
                                                        const VectorDim& coord,
                                                        bool status)
    : mpm::PointBase<Tdim>::PointBase(id, coord, status) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Logger
  std::string logger = "PointDirichletPenalty" + std::to_string(Tdim) +
                       "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Initialise point properties
template <unsigned Tdim>
void mpm::PointDirichletPenalty<Tdim>::initialise() {
  mpm::PointBase<Tdim>::initialise();

  imposed_displacement_.setZero();
  imposed_velocity_.setZero();
  imposed_acceleration_.setZero();
  contact_ = false;
  constraint_flags_.setZero();
}

//! Assign point properties
template <unsigned Tdim>
void mpm::PointDirichletPenalty<Tdim>::assign_properties(
    const std::map<std::string, double>& scalar_properties,
    const std::map<std::string, std::vector<double>>& vector_properties) {
  // Assign scalar properties
  if (scalar_properties.count("penalty_factor"))
    penalty_factor_ = scalar_properties.at("penalty_factor");

  // Assign constraint flags
  if (vector_properties.count("constraint_flags")) {
    const auto& flags = vector_properties.at("constraint_flags");
    for (unsigned i = 0; i < Tdim; ++i)
      constraint_flags_(i) = static_cast<int>(flags[i]);
  }

  // Assign contact conditions
  if (scalar_properties.count("contact"))
    contact_ = static_cast<bool>(scalar_properties.at("contact"));
}

// Reinitialise point properties
template <unsigned Tdim>
void mpm::PointDirichletPenalty<Tdim>::initialise_properties(double dt) {
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
void mpm::PointDirichletPenalty<Tdim>::apply_velocity_constraints(
    unsigned dir, double velocity) {
  // Set particle velocity constraint
  this->imposed_velocity_(dir) = velocity;
}

//! Compute updated position
template <unsigned Tdim>
void mpm::PointDirichletPenalty<Tdim>::compute_updated_position(
    double dt) noexcept {
  // Update position and displacements
  coordinates_.noalias() += imposed_displacement_;
  displacement_.noalias() += imposed_displacement_;
}

//! Map penalty stiffness matrix to cell
template <unsigned Tdim>
inline bool mpm::PointDirichletPenalty<Tdim>::map_stiffness_matrix_to_cell() {
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
        // Directional multiplier
        Eigen::VectorXd dir_multiplier = Eigen::VectorXd::Constant(Tdim, 1.0);

        // Check if direction is constrained
        for (unsigned j = 0; j < Tdim; ++j)
          if (constraint_flags_(j) == 0) dir_multiplier(j) = 0.0;

        // Arrange shape function
        for (unsigned int j = 0; j < Tdim; j++) {
          shape_function(j, Tdim * i + j) = shapefn_[i] * dir_multiplier[j];
        }
      }
    }

    // Assign stiffness matrix
    penalty_stiffness.noalias() += shape_function.transpose() * shape_function;

    // Compute local penalty stiffness matrix
    cell_->compute_local_stiffness_matrix_block(0, 0, penalty_stiffness, area_,
                                                penalty_factor_);

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map enforcement force
template <unsigned Tdim>
void mpm::PointDirichletPenalty<Tdim>::map_boundary_force(unsigned phase) {

  // Check contact: Check contact penetration: if <0 apply constraint,
  // otherwise no
  bool apply_constraints = true;
  if (contact_) {
    // NOTE: the unit_normal_vector is assumed always pointing outside the
    // boundary
    VectorDim field_displacement = VectorDim::Zero();
    for (unsigned int i = 0; i < nodes_.size(); i++)
      if (shapefn_[i] > std::numeric_limits<double>::epsilon())
        field_displacement.noalias() +=
            shapefn_[i] * nodes_[i]->displacement(phase);

    const double penetration =
        (field_displacement - imposed_displacement_).dot(normal_);

    // If penetrates, apply constraint, otherwise no
    if (penetration >= 0.0) apply_constraints = false;
  }

  if (apply_constraints) {
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
        // Directional multiplier
        Eigen::VectorXd dir_multiplier = Eigen::VectorXd::Constant(Tdim, 1.0);

        // Check if direction is constrained
        for (unsigned j = 0; j < Tdim; ++j)
          if (constraint_flags_(j) == 0) dir_multiplier(j) = 0.0;

        // Arrange shape function
        for (unsigned int j = 0; j < Tdim; j++) {
          shape_function(j, Tdim * i + j) = shapefn_[i] * dir_multiplier[j];
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
}

//! Compute size of serialized point data
template <unsigned Tdim>
int mpm::PointDirichletPenalty<Tdim>::compute_pack_size() const {
  int total_size = mpm::PointBase<Tdim>::compute_pack_size();
  int partial_size;
#ifdef USE_MPI
  // Penalty factor
  MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;

  // Constraint flags
  MPI_Pack_size(1 * Tdim, MPI_INT, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;

  // Contact
  MPI_Pack_size(1, MPI_C_BOOL, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;
#endif
  return total_size;
}

//! Serialize point data
template <unsigned Tdim>
std::vector<uint8_t> mpm::PointDirichletPenalty<Tdim>::serialize() {
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

  // Penalty factor
  MPI_Pack(&penalty_factor_, 1, MPI_DOUBLE, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

  // Constraint flags
  MPI_Pack(constraint_flags_.data(), Tdim, MPI_INT, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Contact
  MPI_Pack(&contact_, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

#endif
  return data;
}

//! Deserialize point data
template <unsigned Tdim>
void mpm::PointDirichletPenalty<Tdim>::deserialize(
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

  // Penalty factor
  MPI_Unpack(data_ptr, data.size(), &position, &penalty_factor_, 1, MPI_DOUBLE,
             MPI_COMM_WORLD);

  // Constraint flags
  MPI_Unpack(data_ptr, data.size(), &position, constraint_flags_.data(), Tdim,
             MPI_INT, MPI_COMM_WORLD);

  // Contact
  MPI_Unpack(data_ptr, data.size(), &position, &contact_, 1, MPI_C_BOOL,
             MPI_COMM_WORLD);

#endif
}