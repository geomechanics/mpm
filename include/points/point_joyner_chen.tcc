//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointJoynerChen<Tdim>::PointJoynerChen(Index id, const VectorDim& coord)
    : mpm::PointBase<Tdim>::PointBase(id, coord) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();

  // Logger
  std::string logger =
      "PointJoynerChen" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Constructor with id, coordinates and status
template <unsigned Tdim>
mpm::PointJoynerChen<Tdim>::PointJoynerChen(Index id, const VectorDim& coord,
                                            bool status)
    : mpm::PointBase<Tdim>::PointBase(id, coord, status) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Logger
  std::string logger =
      "PointJoynerChen" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Initialise point properties
template <unsigned Tdim>
void mpm::PointJoynerChen<Tdim>::initialise() {
  mpm::PointBase<Tdim>::initialise();

  imposed_displacement_.setZero();
  imposed_velocity_.setZero();
  imposed_acceleration_.setZero();
  constraint_flags_.setZero();
  normal_.setZero();
}

//! Assign point properties
template <unsigned Tdim>
void mpm::PointJoynerChen<Tdim>::assign_properties(
    const std::map<std::string, double>& scalar_properties,
    const std::map<std::string, std::vector<double>>& vector_properties) {
  assert(area_ != std::numeric_limits<double>::max());
  // Assign scalar properites
  if (scalar_properties.count("youngs_modulus") &&
      scalar_properties.count("density") &&
      scalar_properties.count("poisson_ratio")) {
    youngs_modulus_ = scalar_properties.at("youngs_modulus");
    density_ = scalar_properties.at("density");
    poisson_ratio_ = scalar_properties.at("poisson_ratio");
  } else {
    console_->warn(
        "#{}: Young's modulus, density and poisson ratio are required for "
        "point kelvin voigt. Default values of 0.0 will be assigned.",
        __LINE__);
  }
  
  // Assign constraint flags
  if (vector_properties.count("constraint_flags")) {
    const auto& flags = vector_properties.at("constraint_flags");
    for (unsigned i = 0; i < Tdim; ++i)
      constraint_flags_(i) = static_cast<int>(flags[i]);
  }

  // Assign absorbing boolean
  if (scalar_properties.count("absorbing_factor")) {
    absorbing_factor_ = scalar_properties.at("absorbing_factor");
  }
}

//! Reinitialise point properties
template <unsigned Tdim>
void mpm::PointJoynerChen<Tdim>::initialise_properties(double dt) {
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
void mpm::PointJoynerChen<Tdim>::assign_joyner_chen_constraints(
    unsigned dir, double velocity) {
  // Update imposed velocity
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

// Compute updated position
template <unsigned Tdim>
void mpm::PointJoynerChen<Tdim>::compute_updated_position(
    double dt, unsigned phase, double blending_ratio,
    mpm::VelocityUpdate velocity_update) noexcept {
  // Update position and displacements
  coordinates_.noalias() += imposed_displacement_;
  displacement_.noalias() += imposed_displacement_;
}

template <unsigned Tdim>
inline bool mpm::PointJoynerChen<Tdim>::map_damping_matrix_to_cell(
    double newmark_beta, double newmark_gamma, double dt) {
  bool status = true;
  try {
    // Assumed phase
    unsigned phase = 0;  // mpm::ParticlePhase::SinglePhase;

    // Initialise stiffness matrix
    const unsigned matrix_size = nodes_.size() * Tdim;
    Eigen::MatrixXd point_stiffness(matrix_size, matrix_size);
    point_stiffness.setZero();

    // Initialise material properties
    double vp =
        std::sqrt((youngs_modulus_ * (1 - poisson_ratio_)) /
                  ((1 + poisson_ratio_) * (1 - 2 * poisson_ratio_) * density_));
    double vs =
        std::sqrt(youngs_modulus_ / (2 * (1 + poisson_ratio_) * density_));

    // Normal and Tangent multipliers
    const double normal_mult = density_ * vp;
    const double tangent_mult = density_ * vs;

    // Normal matrix
    normal_.normalize();
    Eigen::Matrix<double, Tdim, Tdim> normal_matrix =
        normal_ * normal_.transpose();

    // Identity matrix
    const Eigen::Matrix<double, Tdim, Tdim> identity =
        Eigen::Matrix<double, Tdim, Tdim>::Identity();

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
          shape_function(j, Tdim * i + j) = shapefn_[i];
        }
      }
    }

    // Assign stiffness matrix
    point_stiffness.noalias() += shape_function.transpose() *
                                 (normal_mult * normal_matrix +
                                  tangent_mult * (identity - normal_matrix)) *
                                 shape_function;

    // Compute local penalty stiffness matrix
    cell_->compute_local_stiffness_matrix_block(
        0, 0, point_stiffness, area_, absorbing_factor_ * newmark_gamma / (newmark_beta * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map enforcement force
template <unsigned Tdim>
void mpm::PointJoynerChen<Tdim>::map_boundary_force(unsigned phase) {
  // Initialise material properties
  double vp =
      std::sqrt((youngs_modulus_ * (1 - poisson_ratio_)) /
                ((1 + poisson_ratio_) * (1 - 2 * poisson_ratio_) * density_));
  double vs =
      std::sqrt(youngs_modulus_ / (2 * (1 + poisson_ratio_) * density_));

  // Normal and Tangent multipliers
  const double normal_dashpot_mult = density_ * vp;
  const double tangent_dashpot_mult = density_ * vs;

  // Normal matrix
  normal_.normalize();
  Eigen::Matrix<double, Tdim, Tdim> normal_matrix =
      normal_ * normal_.transpose();

  // Identity matrix
  const Eigen::Matrix<double, Tdim, Tdim> identity =
      Eigen::Matrix<double, Tdim, Tdim>::Identity();

  // Arrange shape function
  const unsigned matrix_size = nodes_.size() * Tdim;
  Eigen::MatrixXd shape_function(Tdim, matrix_size);
  shape_function.setZero();
  for (unsigned i = 0; i < nodes_.size(); i++) {
    if (shapefn_[i] > std::numeric_limits<double>::epsilon()) {
      // Arrange shape function
      for (unsigned int j = 0; j < Tdim; j++) {
        shape_function(j, Tdim * i + j) = shapefn_[i];
      }
    }
  }

  // Directional multiplier for constrained directions
  Eigen::VectorXd dir_multiplier = Eigen::VectorXd::Constant(Tdim, 1.0);

  // Check if direction is constrained
  for (unsigned j = 0; j < Tdim; ++j)
    if (constraint_flags_(j) == 0) dir_multiplier(j) = 0.0;

  // Get net velocity (nodal velocity - imposed velocity)
  Eigen::VectorXd net_vel(matrix_size);
  net_vel.setZero();

  for (unsigned i = 0; i < nodes_.size(); i++) {
    net_vel.segment(i * Tdim, Tdim) =
        (absorbing_factor_ * nodes_[i]->velocity(phase) - this->imposed_velocity_)
            .cwiseProduct(dir_multiplier);
  }

  // Dashpot force contribution
  const auto& dashpot_force =
      shape_function.transpose() *
      (normal_dashpot_mult * normal_matrix +
       tangent_dashpot_mult * (identity - normal_matrix)) *
      shape_function * net_vel * area_;

  // Compute nodal external forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_external_force(
        true, phase, -1.0 * dashpot_force.segment(i * Tdim, Tdim));
  }
}

// //! Compute size of serialized point data
// template <unsigned Tdim>
// int mpm::PointJoynerChen<Tdim>::compute_pack_size() const {
//   int total_size = mpm::PointBase<Tdim>::compute_pack_size();
//   int partial_size;
// #ifdef USE_MPI
//   // Penalty factor
//   MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &partial_size);
//   total_size += partial_size;
//
//   // Slip, contact
//   MPI_Pack_size(2, MPI_C_BOOL, MPI_COMM_WORLD, &partial_size);
//   total_size += partial_size;
//
//   // Normal type
//   MPI_Pack_size(1, MPI_UNSIGNED, MPI_COMM_WORLD, &partial_size);
//   total_size += partial_size;
//
//   // Normal vector
//   MPI_Pack_size(Tdim, MPI_DOUBLE, MPI_COMM_WORLD, &partial_size);
//   total_size += partial_size;
// #endif
//   return total_size;
// }

// //! Serialize point data
// template <unsigned Tdim>
// std::vector<uint8_t> mpm::PointJoynerChen<Tdim>::serialize() {
//   // Compute pack size
//   if (pack_size_ == 0) pack_size_ = compute_pack_size();
//   // Initialize data buffer
//   std::vector<uint8_t> data;
//   data.resize(pack_size_);
//   uint8_t* data_ptr = &data[0];
//   int position = 0;
//
// #ifdef USE_MPI
//   // Type
//   int type = PointType.at(this->type());
//   MPI_Pack(&type, 1, MPI_INT, data_ptr, data.size(), &position,
//   MPI_COMM_WORLD);
//
//   // ID
//   MPI_Pack(&id_, 1, MPI_UNSIGNED_LONG_LONG, data_ptr, data.size(),
//   &position,
//            MPI_COMM_WORLD);
//   // Area
//   MPI_Pack(&area_, 1, MPI_DOUBLE, data_ptr, data.size(), &position,
//            MPI_COMM_WORLD);
//
//   // Coordinates
//   MPI_Pack(coordinates_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
//            &position, MPI_COMM_WORLD);
//   // Displacement
//   MPI_Pack(displacement_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
//            &position, MPI_COMM_WORLD);
//
//   // Cell id
//   MPI_Pack(&cell_id_, 1, MPI_UNSIGNED_LONG_LONG, data_ptr, data.size(),
//            &position, MPI_COMM_WORLD);
//
//   // Status
//   MPI_Pack(&status_, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
//            MPI_COMM_WORLD);
//
//   // Penalty factor
//   MPI_Pack(&penalty_factor_, 1, MPI_DOUBLE, data_ptr, data.size(),
//   &position,
//            MPI_COMM_WORLD);
//
//   // Slip
//   MPI_Pack(&slip_, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
//            MPI_COMM_WORLD);
//
//   // Contact
//   MPI_Pack(&contact_, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
//            MPI_COMM_WORLD);
//
//   // Normal type
//   MPI_Pack(&normal_type_, 1, MPI_UNSIGNED, data_ptr, data.size(),
//   &position,
//            MPI_COMM_WORLD);
//
//   // Normal vector
//   MPI_Pack(normal_.data(), Tdim, MPI_DOUBLE, data_ptr, data.size(),
//   &position,
//            MPI_COMM_WORLD);
//
// #endif
//   return data;
// }

// //! Deserialize point data
// template <unsigned Tdim>
// void mpm::PointJoynerChen<Tdim>::deserialize(
//     const std::vector<uint8_t>& data) {
//   uint8_t* data_ptr = const_cast<uint8_t*>(&data[0]);
//   int position = 0;
//
// #ifdef USE_MPI
//   // Type
//   int type;
//   MPI_Unpack(data_ptr, data.size(), &position, &type, 1, MPI_INT,
//              MPI_COMM_WORLD);
//   assert(type == PointType.at(this->type()));
//
//   // ID
//   MPI_Unpack(data_ptr, data.size(), &position, &id_, 1,
//   MPI_UNSIGNED_LONG_LONG,
//              MPI_COMM_WORLD);
//   // area
//   MPI_Unpack(data_ptr, data.size(), &position, &area_, 1, MPI_DOUBLE,
//              MPI_COMM_WORLD);
//
//   // Coordinates
//   MPI_Unpack(data_ptr, data.size(), &position, coordinates_.data(), Tdim,
//              MPI_DOUBLE, MPI_COMM_WORLD);
//   // Displacement
//   MPI_Unpack(data_ptr, data.size(), &position, displacement_.data(), Tdim,
//              MPI_DOUBLE, MPI_COMM_WORLD);
//
//   // cell id
//   MPI_Unpack(data_ptr, data.size(), &position, &cell_id_, 1,
//              MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
//   // status
//   MPI_Unpack(data_ptr, data.size(), &position, &status_, 1, MPI_C_BOOL,
//              MPI_COMM_WORLD);
//
//   // Penalty factor
//   MPI_Unpack(data_ptr, data.size(), &position, &penalty_factor_, 1,
//   MPI_DOUBLE,
//              MPI_COMM_WORLD);
//
//   // Slip
//   MPI_Unpack(data_ptr, data.size(), &position, &slip_, 1, MPI_C_BOOL,
//              MPI_COMM_WORLD);
//
//   // Contact
//   MPI_Unpack(data_ptr, data.size(), &position, &contact_, 1, MPI_C_BOOL,
//              MPI_COMM_WORLD);
//
//   // Normal type
//   MPI_Unpack(data_ptr, data.size(), &position, &normal_type_, 1,
//   MPI_UNSIGNED,
//              MPI_COMM_WORLD);
//
//   // Normal vector
//   MPI_Unpack(data_ptr, data.size(), &position, normal_.data(), Tdim,
//   MPI_DOUBLE,
//              MPI_COMM_WORLD);
//
// #endif
// }