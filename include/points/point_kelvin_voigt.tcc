//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointKelvinVoigt<Tdim>::PointKelvinVoigt(Index id, const VectorDim& coord)
    : mpm::PointBase<Tdim>::PointBase(id, coord) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();

  // Logger
  std::string logger =
      "PointKelvinVoigt" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//! Constructor with id, coordinates and status
template <unsigned Tdim>
mpm::PointKelvinVoigt<Tdim>::PointKelvinVoigt(Index id, const VectorDim& coord,
                                              bool status)
    : mpm::PointBase<Tdim>::PointBase(id, coord, status) {
  this->initialise();
  // Clear cell ptr
  cell_ = nullptr;
  // Nodes
  nodes_.clear();
  // Logger
  std::string logger =
      "PointKelvinVoigt" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

// Initialise point properties
template <unsigned Tdim>
void mpm::PointKelvinVoigt<Tdim>::initialise() {
  mpm::PointBase<Tdim>::initialise();
  normal_.setZero();
}

//! Assign point properties
template <unsigned Tdim>
void mpm::PointKelvinVoigt<Tdim>::assign_properties(
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
}

//! Reinitialise point properties
template <unsigned Tdim>
void mpm::PointKelvinVoigt<Tdim>::initialise_properties(double dt) {
    // Nothing to initialise for point Kelvin Voigt constraints
}

//! Apply point velocity constraints
template <unsigned Tdim>
void mpm::PointKelvinVoigt<Tdim>::assign_kelvin_voigt_constraints(
    unsigned dir, double delta, double h_min, double incidence_a,
    double incidence_b) {
  // Assign point properties
  this->delta_ = delta;
  this->h_min_ = h_min;
  this->incidence_a_ = incidence_a;
  this->incidence_b_ = incidence_b;
}

// Compute updated position
template <unsigned Tdim>
void mpm::PointKelvinVoigt<Tdim>::compute_updated_position(
    double dt, unsigned phase, double blending_ratio,
    mpm::VelocityUpdate velocity_update) noexcept {
  // Define default velocity update scheme
  switch (velocity_update) {
    case mpm::VelocityUpdate::FLIP:
      this->compute_updated_position_flip(dt, blending_ratio, phase);
      break;
    default:
      // Default to no position update scheme
      break;
  }
}

// Compute updated position of the point assuming FLIP scheme
template <unsigned Tdim>
void mpm::PointKelvinVoigt<Tdim>::compute_updated_position_flip(
    double dt, double blending_ratio, unsigned phase) noexcept {
  // Check if point has a valid cell ptr
  assert(cell_ != nullptr);

  // Get interpolated nodal velocity and acceleration
  Eigen::Matrix<double, Tdim, 1> nodal_velocity =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> nodal_acceleration =
      Eigen::Matrix<double, Tdim, 1>::Zero();

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodal_velocity.noalias() += shapefn_[i] * nodes_[i]->velocity(phase);
    nodal_acceleration.noalias() +=
        shapefn_[i] * nodes_[i]->acceleration(phase);
  }

  // Update particle velocity from interpolated nodal acceleration
  this->velocity_.noalias() += nodal_acceleration * dt;
  // If intermediate scheme is considered
  this->velocity_ = blending_ratio * this->velocity_ +
                    (1.0 - blending_ratio) * nodal_velocity;

  // New position current position + velocity * dt
  this->coordinates_.noalias() += nodal_velocity * dt;
  // Update displacement (displacement is initialized from zero)
  this->displacement_.noalias() += nodal_velocity * dt;
}

//! Map penalty stiffness matrix to cell
template <unsigned Tdim>
inline bool mpm::PointKelvinVoigt<Tdim>::map_stiffness_matrix_to_cell() {
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
    const double normal_mult = density_ * vp * vp / delta_;
    const double tangent_mult = density_ * vs * vs / delta_;

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
    cell_->compute_local_stiffness_matrix_block(0, 0, point_stiffness, area_,
                                                1.0);
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

template <unsigned Tdim>
inline bool mpm::PointKelvinVoigt<Tdim>::map_damping_matrix_to_cell(
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
    const double normal_mult = incidence_a_ * density_ * vp;
    const double tangent_mult = incidence_b_ * density_ * vs;

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
        0, 0, point_stiffness, area_, newmark_gamma / (newmark_beta * dt));
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Map enforcement force
template <unsigned Tdim>
void mpm::PointKelvinVoigt<Tdim>::map_boundary_force(unsigned phase) {
  // Initialise material properties
  double vp =
      std::sqrt((youngs_modulus_ * (1 - poisson_ratio_)) /
                ((1 + poisson_ratio_) * (1 - 2 * poisson_ratio_) * density_));
  double vs =
      std::sqrt(youngs_modulus_ / (2 * (1 + poisson_ratio_) * density_));

  // Normal and Tangent multipliers
  const double normal_dashpot_mult = incidence_a_ * density_ * vp;
  const double tangent_dashpot_mult = incidence_b_ * density_ * vs;

  // Normal and Tangent multipliers
  const double normal_spring_mult = density_ * vp * vp / delta_;
  const double tangent_spring_mult = density_ * vs * vs / delta_;

  // Normal matrix
  normal_.normalize();
  Eigen::Matrix<double, Tdim, Tdim> normal_matrix =
      normal_ * normal_.transpose();

  // Identity matrix
  const Eigen::Matrix<double, Tdim, Tdim> identity =
      Eigen::Matrix<double, Tdim, Tdim>::Identity();

  // Get nodal displacement and nodal velocity
  const unsigned matrix_size = nodes_.size() * Tdim;
  Eigen::VectorXd nodal_disp(matrix_size);
  nodal_disp.setZero();
  Eigen::VectorXd nodal_vel(matrix_size);
  nodal_vel.setZero();

  for (unsigned i = 0; i < nodes_.size(); i++) {
    nodal_disp.segment(i * Tdim, Tdim) = nodes_[i]->displacement(phase);
    nodal_vel.segment(i * Tdim, Tdim) = nodes_[i]->velocity(phase);
  }

  // Arrange shape function
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

  // Spring force contribution
  const auto& spring_force =
      shape_function.transpose() *
      (normal_spring_mult * normal_matrix +
       tangent_spring_mult * (identity - normal_matrix)) *
      shape_function * nodal_disp * area_;
  // Dashpot force contribution
  const auto& dashpot_force =
      shape_function.transpose() *
      (normal_dashpot_mult * normal_matrix +
       tangent_dashpot_mult * (identity - normal_matrix)) *
      shape_function * nodal_vel * area_;

  // Compute nodal external forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_external_force(
        true, phase, -1.0 * spring_force.segment(i * Tdim, Tdim));
    nodes_[i]->update_external_force(
        true, phase, -1.0 * dashpot_force.segment(i * Tdim, Tdim));
  }
}

// //! Compute size of serialized point data
// template <unsigned Tdim>
// int mpm::PointKelvinVoigt<Tdim>::compute_pack_size() const {
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
// std::vector<uint8_t> mpm::PointKelvinVoigt<Tdim>::serialize() {
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
// void mpm::PointKelvinVoigt<Tdim>::deserialize(
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