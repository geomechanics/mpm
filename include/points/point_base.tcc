//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::PointBase<Tdim>::PointBase(const VectorDim& coord) {
  // Check if the dimension is between 1 & 3
  static_assert((Tdim >= 1 && Tdim <= 3), "Invalid global dimension");
  coordinates_ = coord;
  this->initialise();
  // Logger
  std::string logger = "point" + std::to_string(Tdim) + "d";
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
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
  area_ = std::numeric_limits<double>::max();
  displacement_.setZero();

  // Initialize scalar, vector, and tensor data properties
  this->scalar_properties_["area"] = [&]() { return area(); };
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
      // if a cell already exists remove point from that cell
      if (cell_ != nullptr) cell_->remove_point_id(this->id_);

      cell_ = cellptr;
      cell_id_ = cellptr->id();
      // Copy nodal pointer to cell
      nodes_.clear();
      nodes_ = cell_->nodes();

      // Compute reference location of point
      bool xi_status = this->compute_reference_location();
      if (!xi_status) return false;
      status = cell_->add_point_id(this->id());
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
      // if a cell already exists remove point from that cell
      if (cell_ != nullptr) cell_->remove_point_id(this->id_);

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

      status = cell_->add_point_id(this->id());
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

//! Initialise point data from POD
template <unsigned Tdim>
bool mpm::PointBase<Tdim>::initialise_point(PODPoint& point) {

  // Assign id
  this->id_ = point.id;
  // Area
  this->area_ = point.area;

  // Coordinates
  Eigen::Vector3d coordinates;
  coordinates << point.coord_x, point.coord_y, point.coord_z;
  // Initialise coordinates
  for (unsigned i = 0; i < Tdim; ++i) this->coordinates_(i) = coordinates(i);

  // Displacement
  Eigen::Vector3d displacement;
  displacement << point.displacement_x, point.displacement_y,
      point.displacement_z;
  // Initialise displacement
  for (unsigned i = 0; i < Tdim; ++i) this->displacement_(i) = displacement(i);

  // Status
  this->status_ = point.status;

  // Cell id
  this->cell_id_ = point.cell_id;
  this->cell_ = nullptr;

  // Clear nodes
  this->nodes_.clear();

  return true;
}

//! Return point data as POD
template <unsigned Tdim>
// cppcheck-suppress *
std::shared_ptr<void> mpm::PointBase<Tdim>::pod() const {
  // Initialise point data
  auto point_data = std::make_shared<mpm::PODPoint>();

  Eigen::Vector3d coordinates;
  coordinates.setZero();
  for (unsigned j = 0; j < Tdim; ++j) coordinates[j] = this->coordinates_[j];

  Eigen::Vector3d displacement;
  displacement.setZero();
  for (unsigned j = 0; j < Tdim; ++j) displacement[j] = this->displacement_[j];

  point_data->id = this->id();
  point_data->area = this->area();

  point_data->coord_x = coordinates[0];
  point_data->coord_y = coordinates[1];
  point_data->coord_z = coordinates[2];

  point_data->displacement_x = displacement[0];
  point_data->displacement_y = displacement[1];
  point_data->displacement_z = displacement[2];

  point_data->status = this->status();

  point_data->cell_id = this->cell_id();

  return point_data;
}

// Compute shape functions and gradients
template <unsigned Tdim>
void mpm::PointBase<Tdim>::compute_shapefn() noexcept {
  // Check if point has a valid cell ptr
  assert(cell_ != nullptr);
  // Get element ptr of a cell
  const auto element = cell_->element_ptr();

  // Compute shape function of the point
  Eigen::Matrix<double, Tdim, 1> zero_natural_size =
      Eigen::Matrix<double, Tdim, 1>::Zero();

  // Identity matrix
  const Eigen::Matrix<double, Tdim, Tdim> identity =
      Eigen::Matrix<double, Tdim, Tdim>::Identity();

  shapefn_ = element->shapefn(this->xi_, zero_natural_size, identity);
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

// Assign area to the point
template <unsigned Tdim>
bool mpm::PointBase<Tdim>::assign_area(double area) {
  bool status = true;
  try {
    if (area <= 0.) throw std::runtime_error("Point area cannot be negative");
    this->area_ = area;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Compute size of serialized point data
template <unsigned Tdim>
int mpm::PointBase<Tdim>::compute_pack_size() const {
  int total_size = 0;
  int partial_size;
#ifdef USE_MPI
  // Type
  MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;

  // ID
  MPI_Pack_size(1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;
  // area
  MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;

  // Coordinates, displacement
  MPI_Pack_size(2 * Tdim, MPI_DOUBLE, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;

  // Cell id
  MPI_Pack_size(1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;

  // Status
  MPI_Pack_size(1, MPI_C_BOOL, MPI_COMM_WORLD, &partial_size);
  total_size += partial_size;
#endif
  return total_size;
}

//! Serialize point data
template <unsigned Tdim>
std::vector<uint8_t> mpm::PointBase<Tdim>::serialize() {
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

  // Cell id
  MPI_Pack(&cell_id_, 1, MPI_UNSIGNED_LONG_LONG, data_ptr, data.size(),
           &position, MPI_COMM_WORLD);

  // Status
  MPI_Pack(&status_, 1, MPI_C_BOOL, data_ptr, data.size(), &position,
           MPI_COMM_WORLD);

#endif
  return data;
}

//! Deserialize point data
template <unsigned Tdim>
void mpm::PointBase<Tdim>::deserialize(const std::vector<uint8_t>& data) {
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

  // cell id
  MPI_Unpack(data_ptr, data.size(), &position, &cell_id_, 1,
             MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
  // status
  MPI_Unpack(data_ptr, data.size(), &position, &status_, 1, MPI_C_BOOL,
             MPI_COMM_WORLD);

#endif
}