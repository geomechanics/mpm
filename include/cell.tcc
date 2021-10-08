//! Constructor with cell id, number of nodes and element
template <unsigned Tdim>
mpm::Cell<Tdim>::Cell(Index id, unsigned nnodes,
                      const std::shared_ptr<const Element<Tdim>>& elementptr,
                      bool isoparametric)
    : id_{id}, nnodes_{nnodes}, isoparametric_{isoparametric} {
  // Check if the dimension is between 1 & 3
  static_assert((Tdim >= 1 && Tdim <= 3), "Invalid global dimension");

  //! Logger
  std::string logger =
      "cell" + std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);

  try {
    if (elementptr->nfunctions() == this->nnodes_) {
      element_ = elementptr;
      // Create an empty nodal coordinates
      nodal_coordinates_.resize(this->nnodes_, Tdim);
    } else {
      throw std::runtime_error(
          "Specified number of shape functions and nodes don't match");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
}

//! Return the initialisation status of cells
template <unsigned Tdim>
bool mpm::Cell<Tdim>::initialise() {
  bool status = false;
  try {
    // Check if node pointers are present and are equal to the expected number
    if (this->nnodes_ == this->nodes_.size()) {
      // Initialise cell properties (volume, centroid, length)
      this->compute_centroid();
      this->compute_mean_length();
      this->compute_volume();

      // Get centroid of a cell in natural coordinates which are zeros
      Eigen::Matrix<double, Tdim, 1> xi_centroid;
      xi_centroid.setZero();

      Eigen::Matrix<double, Tdim, 1> zero;
      zero.setZero();

      // dN/dX at the centroid
      dn_dx_centroid_ =
          element_->dn_dx(xi_centroid, this->nodal_coordinates_, zero, zero);

      status = true;
    } else {
      throw std::runtime_error(
          "Specified number of nodes for a cell is not present");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}

//! Return the initialisation status of cells
//! \retval initialisation_status Cell has nodes, shape functions and volumes
template <unsigned Tdim>
bool mpm::Cell<Tdim>::is_initialised() const {
  // Check if node pointers are present and are equal to the # shape_fns
  return ((this->nnodes_ != 0 && this->nfunctions() == this->nodes_.size()) &&
          // Check if shape function is assigned
          this->nfunctions() != 0 &&
          // Check if volume of a cell is initialised
          (std::fabs(this->volume_ - std::numeric_limits<double>::lowest()) >
           1.0E-10) &&
          // Check if mean length of a cell is initialised
          (std::fabs(this->mean_length_ - std::numeric_limits<double>::max()) >
           1.0E-10));
}

//! Assign quadrature
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_quadrature(unsigned nquadratures) {
  this->quadrature_ = element_->quadrature(nquadratures);
}

//! Assign discontinuity element type
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_type_discontinuity(mpm::EnrichType type) {
  if (nparticles() == 0 && type != mpm::EnrichType::NeighbourTip_2)
    type = mpm::EnrichType::Regular;
  if (discontinuity_element_ == nullptr)
    discontinuity_element_ =
        std::make_shared<mpm::DiscontinuityElement<Tdim>>(type);
  else
    discontinuity_element_->assign_element_type(type);
}
// Initialize discontinuity element type
template <unsigned Tdim>
void mpm::Cell<Tdim>::initialise_element_properties_discontinuity() {
  if (discontinuity_element_ == nullptr) return;
  discontinuity_element_->initialize();
}

// Return discontinuity element type
template <unsigned Tdim>
unsigned mpm::Cell<Tdim>::element_type_discontinuity() {
  if (discontinuity_element_ == nullptr) return mpm::EnrichType::Regular;
  return discontinuity_element_->element_type();
}
//! Assign quadrature
template <unsigned Tdim>
std::vector<Eigen::Matrix<double, Tdim, 1>> mpm::Cell<Tdim>::generate_points() {
  // Assign a default quadrature of 1
  if (this->quadrature_ == nullptr) this->assign_quadrature(1);

  const auto quadratures = quadrature_->quadratures();

  // Vector of gauss points
  std::vector<Eigen::Matrix<double, Tdim, 1>> points;

  // Get indices of corner nodes
  Eigen::VectorXi indices = element_->corner_indices();

  // Matrix of nodal coordinates
  const Eigen::MatrixXd nodal_coords = nodal_coordinates_.transpose();

  // Zeros
  const Eigen::Matrix<double, Tdim, 1> zeros =
      Eigen::Matrix<double, Tdim, 1>::Zero();

  // Get local coordinates of gauss points and transform to global
  for (unsigned i = 0; i < quadratures.cols(); ++i) {
    const auto lpoint = quadratures.col(i);
    // Get shape functions
    const auto sf = element_->shapefn(lpoint, zeros, zeros);
    const auto point = nodal_coords * sf;
    const auto xi = this->transform_real_to_unit_cell(point);
    bool status = true;
    // Check if point is within the cell
    for (unsigned i = 0; i < xi.size(); ++i)
      if (xi(i) < -1. || xi(i) > 1. || std::isnan(xi(i))) status = false;

    if (status)
      points.emplace_back(point);
    else
      console_->warn("Cannot generate point: ({}, {}) in cell xi: ({}, {})",
                     point(0), point(1), xi(0), xi(1));
  }

  return points;
}

//! Add a node pointer and return the status of addition of a node
template <unsigned Tdim>
bool mpm::Cell<Tdim>::add_node(
    unsigned local_id, const std::shared_ptr<mpm::NodeBase<Tdim>>& node_ptr) {
  bool insertion_status = false;
  try {
    // If number of node ptrs in a cell is less than the maximum number of nodes
    // per cell
    // The local id should be between 0 and maximum number of nodes
    if (nodes_.size() < this->nnodes_ && local_id < this->nnodes_) {
      nodes_.emplace_back(node_ptr);
      // Assign coordinates
      nodal_coordinates_.row(local_id) =
          nodes_[local_id]->coordinates().transpose();
      insertion_status = true;
    } else {
      throw std::runtime_error(
          "Number nodes in a cell exceeds the maximum allowed per cell");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return insertion_status;
}

//! Activate nodes if particle is present
template <unsigned Tdim>
void mpm::Cell<Tdim>::activate_nodes() {
  // If number of particles are present, set all associated nodes as active
  if (particles_.size() > 0) {
    // std::lock_guard<std::mutex> guard(cell_mutex_);
    for (unsigned i = 0; i < nodes_.size(); ++i) nodes_[i]->assign_status(true);
  }
}

//! Return a vector of side node id pairs
template <unsigned Tdim>
std::vector<std::array<mpm::Index, 2>> mpm::Cell<Tdim>::side_node_pairs()
    const {
  // Create a vector of node_pairs
  std::vector<std::array<mpm::Index, 2>> node_pairs;
  // Get the indices of sides
  Eigen::MatrixXi indices = element_->sides_indices();
  // Iterate over indices and get node ids
  for (unsigned i = 0; i < indices.rows(); ++i)
    node_pairs.emplace_back(std::array<mpm::Index, 2>(
        {nodes_[indices(i, 0)]->id(), nodes_[indices(i, 1)]->id()}));

  return node_pairs;
}

//! Add a neighbour cell and return the status of addition of a node
template <unsigned Tdim>
bool mpm::Cell<Tdim>::add_neighbour(mpm::Index neighbour_id) {
  bool insertion_status = false;
  try {
    // If cell id is not the same as the current cell
    if (neighbour_id != this->id())
      insertion_status = (neighbours_.insert(neighbour_id)).second;
    else
      throw std::runtime_error("Invalid local id of a cell neighbour");

  } catch (std::exception& exception) {
    console_->error("{} {}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return insertion_status;
}

//! Add a particle id and return the status of addition of a particle id
template <unsigned Tdim>
bool mpm::Cell<Tdim>::add_particle_id(Index id) {
  bool status = false;
  std::lock_guard<std::mutex> guard(cell_mutex_);
  // Check if it is found in the container
  auto itr = std::find(particles_.begin(), particles_.end(), id);
  if (itr == particles_.end()) {
    particles_.emplace_back(id);
    status = true;
  }
  return status;
}

//! Remove a particle id
template <unsigned Tdim>
void mpm::Cell<Tdim>::remove_particle_id(Index id) {
  std::lock_guard<std::mutex> guard(cell_mutex_);
  particles_.erase(std::remove(particles_.begin(), particles_.end(), id),
                   particles_.end());
}

//! Compute volume of a 2D cell
//! Computes the volume of a triangle and a quadrilateral
template <unsigned Tdim>
inline void mpm::Cell<Tdim>::compute_volume() {
  try {
    if (this->nnodes_ != this->nodes_.size())
      throw std::runtime_error(
          "Insufficient number of nodes to compute volume");
    else
      volume_ = element_->compute_volume(this->nodal_coordinates_);
    // Check negative volume
    if (this->volume_ <= 0)
      throw std::runtime_error(
          "Negative or zero volume cell, misconfigured cell!");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
}

//! Compute the centroid of the cell
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_centroid() {
  // Get indices of corner nodes
  Eigen::VectorXi indices = element_->corner_indices();

  // Calculate the centroid of the cell
  centroid_.setZero();
  for (unsigned i = 0; i < indices.size(); ++i)
    centroid_ += nodes_[indices(i)]->coordinates();

  centroid_ /= indices.size();
}

//! Compute mean length of cell
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_mean_length() {
  // Get the indices of sub-triangles
  Eigen::MatrixXi indices = element_->sides_indices();
  this->mean_length_ = 0.;
  // Calculate the mean length
  for (unsigned i = 0; i < indices.rows(); ++i)
    this->mean_length_ += (nodes_[indices(i, 0)]->coordinates() -
                           nodes_[indices(i, 1)]->coordinates())
                              .norm();
  this->mean_length_ /= indices.rows();
}

//! Check if a point is in a 1D cell by checking bounding box range
template <>
inline bool mpm::Cell<1>::point_in_cartesian_cell(
    const Eigen::Matrix<double, 1, 1>& point) {
  if (nodes_[0]->coordinates()(0) <= point(0) &&
      nodes_[1]->coordinates()(0) >= point(0))
    return true;
  else
    return false;
}

//! Check if a point is in a 2D cell by checking bounding box range
template <>
inline bool mpm::Cell<2>::point_in_cartesian_cell(
    const Eigen::Matrix<double, 2, 1>& point) {
  // Check if a point is within x and y ranges
  if (nodes_[0]->coordinates()(0) <= point(0) &&
      nodes_[1]->coordinates()(0) >= point(0) &&
      nodes_[1]->coordinates()(1) <= point(1) &&
      nodes_[2]->coordinates()(1) >= point(1))
    return true;
  else
    return false;
}

//! Check if a point is in a 3D cell by checking bounding box range
template <>
inline bool mpm::Cell<3>::point_in_cartesian_cell(
    const Eigen::Matrix<double, 3, 1>& point) {
  // Check if a point is within x, y and z ranges
  if (nodes_[0]->coordinates()(0) <= point(0) &&
      nodes_[1]->coordinates()(0) >= point(0) &&
      nodes_[1]->coordinates()(1) <= point(1) &&
      nodes_[2]->coordinates()(1) >= point(1) &&
      nodes_[0]->coordinates()(2) <= point(2) &&
      nodes_[4]->coordinates()(2) >= point(2))
    return true;
  else
    return false;
}

//! Check approximately if a point is in a cell by using cell length
template <unsigned Tdim>
inline bool mpm::Cell<Tdim>::approx_point_in_cell(
    const Eigen::Matrix<double, Tdim, 1>& point) {
  const double length = (point - this->centroid_).norm();
  if (length < (this->mean_length_ * 2.))
    return true;
  else
    return false;
}

//! Check if a point is in a cell by affine transformation and newton-raphson
template <unsigned Tdim>
inline bool mpm::Cell<Tdim>::is_point_in_cell(
    const Eigen::Matrix<double, Tdim, 1>& point,
    Eigen::Matrix<double, Tdim, 1>* xi) {

  // Set an initial value of Xi
  (*xi).fill(std::numeric_limits<double>::max());

  // Check if point is approximately in the cell
  if (!this->approx_point_in_cell(point)) return false;

  bool status = true;

  // Check if cell is cartesian, if so use cartesian local coordinates
  if (!isoparametric_) (*xi) = this->local_coordinates_point(point);
  // Isoparametric element
  else
    (*xi) = this->transform_real_to_unit_cell(point);

  // Check if the transformed coordinate is within the unit cell:
  // between 0 and 1-xi(1-i) if the element is a triangle, and between
  // -1 and 1 if otherwise. Also, check if the transformed coordinate lies
  // exactly on cell edge.
  const double tolerance = std::numeric_limits<double>::epsilon();
  if (this->element_->corner_indices().size() == 3) {
    for (unsigned i = 0; i < (*xi).size(); ++i) {
      if ((*xi)(i) < 0. || (*xi)(i) > 1. - (*xi)(1 - i) || std::isnan((*xi)(i)))
        status = false;
      else {
        if ((*xi)(i) < tolerance) (*xi)(i) = tolerance;
        if ((*xi)(i) > 1. - (*xi)(1 - i) - tolerance)
          (*xi)(i) = 1. - (*xi)(1 - i) - tolerance;
      }
    }
  } else {
    for (unsigned i = 0; i < (*xi).size(); ++i) {
      if ((*xi)(i) < -1. || (*xi)(i) > 1. || std::isnan((*xi)(i)))
        status = false;
      else {
        if ((*xi)(i) < -1. + tolerance) (*xi)(i) = -1. + tolerance;
        if ((*xi)(i) > 1. - tolerance) (*xi)(i) = 1. - tolerance;
      }
    }
  }
  return status;
}

//! Return the local coordinates of a point in a 1D cell
template <>
inline Eigen::Matrix<double, 1, 1> mpm::Cell<1>::local_coordinates_point(
    const Eigen::Matrix<double, 1, 1>& point) {
  // Local point coordinates
  Eigen::Matrix<double, 1, 1> xi;

  try {
    // Indices of corner nodes
    Eigen::VectorXi indices = element_->corner_indices();

    // Linear
    if (indices.size() == 2) {
      //
      // 0 0---------0 1
      //        l
      const double length = (nodes_[indices(0)]->coordinates() -
                             nodes_[indices(1)]->coordinates())
                                .norm();

      const Eigen::Matrix<double, 1, 1> centre =
          (nodes_[indices(0)]->coordinates() +
           nodes_[indices(1)]->coordinates()) /
          2.0;

      xi(0) = 2. * (point(0) - centre(0)) / length;
    } else {
      throw std::runtime_error("Unable to compute local coordinates");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return xi;
}

//! Return the local coordinates of a point in a 2D cell
template <>
inline Eigen::Matrix<double, 2, 1> mpm::Cell<2>::local_coordinates_point(
    const Eigen::Matrix<double, 2, 1>& point) {
  // Local point coordinates
  Eigen::Matrix<double, 2, 1> xi;

  try {
    // Indices of corner nodes
    Eigen::VectorXi indices = element_->corner_indices();

    // Triangle
    if (indices.size() == 3) {
      //   2 0
      //     |\
      //     | \  
      //   c |  \ b
      //     |   \
      //     |    \
      //   0 0-----0 1
      //        a
      //

      const auto node0 = nodal_coordinates_.row(0);
      const auto node1 = nodal_coordinates_.row(1);
      const auto node2 = nodal_coordinates_.row(2);

      const double area = ((node1(0) - node0(0)) * (node2(1) - node0(1)) -
                           (node2(0) - node0(0)) * (node1(1) - node0(1))) /
                          2.0;

      xi(0) = 1. / (2. * area) *
              ((point(0) - node0(0)) * (node2(1) - node0(1)) -
               (node2(0) - node0(0)) * (point(1) - node0(1)));

      xi(1) = -1. / (2. * area) *
              ((point(0) - node0(0)) * (node1(1) - node0(1)) -
               (node1(0) - node0(0)) * (point(1) - node0(1)));
      // Quadrilateral
    } else if (indices.size() == 4) {
      //        b
      // 3 0--------0 2
      //   | \   / |
      // a |  \ /  | c
      //   |  / \  |
      //   | /   \ |
      // 0 0---------0 1
      //         d
      const double xlength =
          (nodal_coordinates_.row(0) - nodal_coordinates_.row(1)).norm();
      const double ylength =
          (nodal_coordinates_.row(1) - nodal_coordinates_.row(2)).norm();

      const Eigen::Matrix<double, 2, 1> centre =
          (nodal_coordinates_.row(0) + nodal_coordinates_.row(1) +
           nodal_coordinates_.row(2) + nodal_coordinates_.row(3)) /
          4.0;

      xi(0) = 2. * (point(0) - centre(0)) / xlength;
      xi(1) = 2. * (point(1) - centre(1)) / ylength;
    } else {
      throw std::runtime_error("Unable to compute local coordinates");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return xi;
}

//! Return the local coordinates of a point in a 3D cell
template <>
inline Eigen::Matrix<double, 3, 1> mpm::Cell<3>::local_coordinates_point(
    const Eigen::Matrix<double, 3, 1>& point) {

  // Local point coordinates
  Eigen::Matrix<double, 3, 1> xi;

  try {
    // Indices of corner nodes
    Eigen::VectorXi indices = element_->corner_indices();

    // Hexahedron
    if (indices.size() == 8) {
      // Node numbering as read in by mesh file
      //        3               2
      //          *_ _ _ _ _ _*
      //         /|           /|
      //        / |          / |
      //     7 *_ |_ _ _ _ _* 6|
      //       |  |         |  |
      //       |  |         |  |
      //       |  *_ _ _ _ _|_ *
      //       | / 0        | / 1
      //       |/           |/
      //       *_ _ _ _ _ _ *
      //     4               5
      //

      const double xlength =
          (nodal_coordinates_.row(0) - nodal_coordinates_.row(1)).norm();
      const double ylength =
          (nodal_coordinates_.row(1) - nodal_coordinates_.row(2)).norm();
      const double zlength =
          (nodal_coordinates_.row(1) - nodal_coordinates_.row(5)).norm();

      // Compute centre
      Eigen::Matrix<double, 3, 1> centre;
      centre.setZero();
      for (unsigned i = 0; i < indices.size(); ++i)
        centre += nodal_coordinates_.row(i);
      centre /= indices.size();

      xi(0) = 2. * (point(0) - centre(0)) / xlength;
      xi(1) = 2. * (point(1) - centre(1)) / ylength;
      xi(2) = 2. * (point(2) - centre(2)) / zlength;

    } else {
      throw std::runtime_error("Unable to compute local coordinates");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return xi;
}

//! Return the local coordinates of a point in a 1D cell
template <>
inline Eigen::Matrix<double, 1, 1> mpm::Cell<1>::transform_real_to_unit_cell(
    const Eigen::Matrix<double, 1, 1>& point) {
  return this->local_coordinates_point(point);
}

//! Return the local coordinates of a point in a cell
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, 1>
    mpm::Cell<Tdim>::transform_real_to_unit_cell(
        const Eigen::Matrix<double, Tdim, 1>& point) {

  // If regular cartesian grid use cartesian transformation
  if (!this->isoparametric_) return this->local_coordinates_point(point);

  // Get indices of corner nodes
  Eigen::VectorXi indices = element_->corner_indices();

  // Analytical solution for 2D linear triangle element
  if (Tdim == 2 && indices.size() == 3) {
    if (element_->isvalid_natural_coordinates_analytical())
      return element_->natural_coordinates_analytical(point,
                                                      this->nodal_coordinates_);
  }

  // Local coordinates of a point in an unit cell
  Eigen::Matrix<double, Tdim, 1> xi;
  xi.fill(std::numeric_limits<double>::max());

  // Zeros
  const Eigen::Matrix<double, Tdim, 1> zero =
      Eigen::Matrix<double, Tdim, 1>::Zero();

  // Matrix of nodal coordinates
  const Eigen::MatrixXd nodal_coords = nodal_coordinates_.transpose();

  // Analytical xi
  Eigen::Matrix<double, Tdim, 1> analytical_xi;
  analytical_xi.fill(std::numeric_limits<double>::max());

  Eigen::Matrix<double, Tdim, 1> analytical_residual;
  analytical_residual.fill(std::numeric_limits<double>::max());

  if (Tdim == 2) {
    if (element_->isvalid_natural_coordinates_analytical())
      analytical_xi = element_->natural_coordinates_analytical(
          point, this->nodal_coordinates_);

    // Analytical tolerance
    const double analytical_tolerance = 1.0E-16 * mean_length_ * mean_length_;

    bool analytical_xi_in_cell = true;
    // Check if analytical_xi is within the cell and is not nan
    for (unsigned i = 0; i < analytical_xi.size(); ++i)
      if (analytical_xi(i) < -1. || analytical_xi(i) > 1. ||
          std::isnan(analytical_xi(i)))
        analytical_xi_in_cell = false;

    // Local shape function
    const auto sf = element_->shapefn_local(analytical_xi, zero, zero);
    // f(x) = p(x) - p, where p is the real point
    analytical_residual = (nodal_coords * sf) - point;
    // Early exit
    if ((analytical_residual.squaredNorm() < analytical_tolerance) &&
        analytical_xi_in_cell)
      return analytical_xi;
  }

  // Affine guess of xi
  Eigen::Matrix<double, Tdim, 1> affine_xi;
  // Boolean to check if affine is nan
  bool affine_nan = false;

  // Affine tolerance
  const double affine_tolerance = 1.0E-16 * mean_length_ * mean_length_;

  // Coordinates of a unit cell
  const auto unit_cell = element_->unit_cell_coordinates();

  // Affine residual
  Eigen::Matrix<double, Tdim, 1> affine_residual;

  // Set the size of KA and Kb matrices
  const unsigned KA = (Tdim == 2 ? 4 : 8);

  // Affine transformation, using linear interpolation for the initial guess
  if (element_->degree() == mpm::ElementDegree::Linear) {
    // A = vertex * KA
    const Eigen::Matrix<double, Tdim, Tdim> A =
        nodal_coords * mpm::TransformR2UAffine<Tdim, KA>::KA;

    // b = vertex * Kb
    const Eigen::Matrix<double, Tdim, 1> b =
        point - (nodal_coords * mpm::TransformR2UAffine<Tdim, KA>::Kb);

    // Affine transform: A^-1 * b
    // const Eigen::Matrix<double, Tdim, 1>
    affine_xi = A.inverse() * b;

    // Check for nan
    for (unsigned i = 0; i < affine_xi.size(); ++i)
      if (std::isnan(affine_xi(i))) affine_nan = true;

    // Set xi to affine guess
    if (!affine_nan) {
      // Local shape function
      const auto sf = element_->shapefn_local(affine_xi, zero, zero);

      // f(x) = p(x) - p, where p is the real point
      affine_residual = (nodal_coords * sf) - point;

      // Early exit
      if ((affine_residual.squaredNorm() < affine_tolerance)) return affine_xi;
    }
  }
  // Use the best of analytical and affine transformation
  Eigen::Matrix<double, Tdim, 1> geometry_xi;
  Eigen::Matrix<double, Tdim, 1> geometry_residual;
  geometry_residual.fill(std::numeric_limits<double>::max());

  // Analytical solution is a better initial guess
  if (analytical_residual.norm() < affine_residual.norm()) {
    geometry_residual = analytical_residual;
    geometry_xi = analytical_xi;
  } else if (!affine_nan) {
    // Affine is a better initial guess
    geometry_residual = affine_residual;
    geometry_xi = affine_xi;
  } else {
    // Use zero, we don't have a good guess
    geometry_xi.setZero();
  }

  // Trial guess for NR
  Eigen::Matrix<double, Tdim, 1> nr_xi = geometry_xi;
  // Check if the first trial xi is just outside the box
  const double nudge_tolerance = 1.E-12;
  for (unsigned i = 0; i < nr_xi.size(); ++i) {
    if (nr_xi(i) < -1. && nr_xi(i) > -1.001)
      nr_xi(i) = -1. + nudge_tolerance;
    else if (nr_xi(i) > 1. && nr_xi(i) < 1.001)
      nr_xi(i) = 1. - nudge_tolerance;
  }

  // Maximum iterations of newton raphson
  const unsigned max_iterations = 10000;

  // Tolerance for newton raphson
  const double Tolerance = 1.0E-10;

  // Newton Raphson iteration to solve for x
  // x_{n+1} = x_n - f(x)/f'(x)
  // f(x) = p(x) - p, where p is the real point
  // p(x) is the computed point.
  Eigen::Matrix<double, Tdim, 1> nr_residual;
  unsigned iter = 0;
  for (; iter < max_iterations; ++iter) {
    // Calculate local Jacobian
    const Eigen::Matrix<double, Tdim, Tdim> jacobian =
        element_->jacobian_local(nr_xi, unit_cell, zero, zero);

    // Set guess nr_xi to zero
    if (std::abs(jacobian.determinant()) < 1.0E-10) nr_xi.setZero();

    // Local shape function
    const auto sf = element_->shapefn_local(nr_xi, zero, zero);

    // Residual (f(x))
    // f(x) = p(x) - p, where p is the real point
    nr_residual = (nodal_coords * sf) - point;

    // f(x)/f'(x)
    const Eigen::Matrix<double, Tdim, 1> delta =
        jacobian.inverse() * nr_residual;

    // Line search
    double step_length = 1.;
    for (unsigned line_trials = 0; line_trials < 10; ++line_trials) {
      // Trial nr_xi
      // x_{n+1} = x_n - f(x)/f'(x)
      const Eigen::Matrix<double, Tdim, 1> xi_trial =
          nr_xi - (step_length * delta);

      // Trial shape function
      const auto sf_trial = element_->shapefn_local(xi_trial, zero, zero);

      // Trial residual: f(x) = p(x) - p, where p is the real point
      const Eigen::Matrix<double, Tdim, 1> nr_residual_trial =
          (nodal_coords * sf_trial) - point;

      if (nr_residual_trial.norm() < nr_residual.norm()) {
        nr_xi = xi_trial;
        nr_residual = nr_residual_trial;
        break;
      } else if (step_length > 0.05)
        step_length /= 2.;
      else {
        // Line search failed
        break;
      }
    }
    // Convergence criteria
    if ((step_length * delta).norm() < Tolerance) break;

    // Check for nan and set to a trial xi
    if (std::isnan(nr_xi(0)) || std::isnan(nr_xi(1))) nr_xi.setZero();
  }

  // At end of iteration return affine or xi based on lowest norm
  xi = geometry_residual.norm() < nr_residual.norm() ? geometry_xi : nr_xi;

  if (std::isnan(xi(0)) || std::isnan(xi(1)))
    throw std::runtime_error("Local coordinates of xi is NAN");

  return xi;
}

//! Assign MPI rank to nodes
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_mpi_rank_to_nodes() {
  for (unsigned i = 0; i < nodes_.size(); ++i) nodes_[i]->mpi_rank(this->rank_);
}

//! Compute all face normals 2d
template <>
inline void mpm::Cell<2>::compute_normals() {

  //! Set number of faces from element
  for (unsigned face_id = 0; face_id < element_->nfaces(); ++face_id) {
    // Get the nodes of the face
    const Eigen::VectorXi indices = element_->face_indices(face_id);

    // Compute the vector to calculate normal (perpendicular)
    // a = node(0) - node(1)
    Eigen::Matrix<double, 2, 1> a = (this->nodes_[indices(0)])->coordinates() -
                                    (this->nodes_[indices(1)])->coordinates();

    // Compute normal and make unit vector
    // The normal vector n to vector a is defined such that the dot product
    // between a and n is always 0 In 2D, n(0) = -a(1), n(1) = a(0) Note that
    // the reverse does not work to produce normal that is positive pointing out
    // of the element
    Eigen::Matrix<double, 2, 1> normal_vector;
    normal_vector << -a(1), a(0);
    normal_vector = normal_vector.normalized();

    face_normals_.insert(std::make_pair<unsigned, Eigen::VectorXd>(
        static_cast<unsigned>(face_id), normal_vector));
  }
}

//! Compute all face normals 3d
template <>
inline void mpm::Cell<3>::compute_normals() {

  //! Set number of faces from element
  for (unsigned face_id = 0; face_id < element_->nfaces(); ++face_id) {
    // Get the nodes of the face
    const Eigen::VectorXi indices = element_->face_indices(face_id);

    // Compute two vectors to calculate normal
    // a = node(1) - node(0)
    // b = node(3) - node(0)
    Eigen::Matrix<double, 3, 1> a =
        nodal_coordinates_.row(1) - nodal_coordinates_.row(0);
    Eigen::Matrix<double, 3, 1> b =
        nodal_coordinates_.row(3) - nodal_coordinates_.row(0);

    // Compute normal and make unit vector
    // normal = a x b
    // Note that definition of a and b are such that normal is always out of
    // page
    Eigen::Matrix<double, 3, 1> normal_vector = a.cross(b);
    normal_vector = normal_vector.normalized();

    face_normals_.insert(std::make_pair<unsigned, Eigen::VectorXd>(
        static_cast<unsigned>(face_id), normal_vector));
  }
}

//! Return a sorted list of face node ids
template <unsigned Tdim>
inline std::vector<std::vector<mpm::Index>>
    mpm::Cell<Tdim>::sorted_face_node_ids() {
  std::vector<std::vector<mpm::Index>> set_face_nodes;
  //! Set number of faces from element
  for (unsigned face_id = 0; face_id < element_->nfaces(); ++face_id) {
    std::vector<mpm::Index> face_nodes;

    // Get the nodes of the face
    const Eigen::VectorXi indices = element_->face_indices(face_id);
    for (int id = 0; id < indices.size(); ++id)
      face_nodes.emplace_back(nodes_[indices(id)]->id());

    // Sort in ascending order
    std::sort(face_nodes.begin(), face_nodes.end());
    set_face_nodes.emplace_back(face_nodes);
  }
  return set_face_nodes;
}

//! Assign MPI rank to cell
template <unsigned Tdim>
inline void mpm::Cell<Tdim>::rank(unsigned rank) {
  if (rank_ != rank) {
    this->previous_mpirank_ = this->rank_;
    this->rank_ = rank;
  }
}

//! Return MPI rank of the cell
template <unsigned Tdim>
inline unsigned mpm::Cell<Tdim>::rank() const {
  return this->rank_;
}

//! Return MPI rank of the cell
template <unsigned Tdim>
inline unsigned mpm::Cell<Tdim>::previous_mpirank() const {
  return this->previous_mpirank_;
}

//! potential tip element
template <unsigned Tdim>
void mpm::Cell<Tdim>::potential_tip_element() {
  if (this->discontinuity_element_ == nullptr) return;
  if (this->discontinuity_element_->element_type() !=
      mpm::EnrichType::NeighbourTip_1)
    return;
  if (this->nparticles() == 0) return;

  if (product_levelset() < 0)
    this->discontinuity_element_->assign_element_type(
        mpm::EnrichType::PotentialTip);

  // for (unsigned i = 0; i < nodes_.size(); ++i) {
  //   if (nodes_[i]->discontinuity_enrich())
  //     nodes_[i]->assign_discontinuity_enrich(false);
  // }
}

//! determine tip element
template <unsigned Tdim>
void mpm::Cell<Tdim>::tip_element() {
  if (this->discontinuity_element_ == nullptr) return;
  if (this->discontinuity_element_->element_type() != mpm::EnrichType::Crossed)
    return;

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i]->discontinuity_enrich()) continue;
    this->discontinuity_element_->assign_element_type(mpm::EnrichType::Tip);
  }
}


//! potential tip element
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_discontinuity_point(
    std::vector<VectorDim>& coordinates) {

  // if (this->discontinuity_element_->area() == 0)
  // compute_area_discontinuity();
  std::vector<Eigen::Matrix<double, Tdim, 1>> intersections_list;

  Eigen::Matrix<double, Tdim, 1> center =
      this->discontinuity_element_->cohesion_cor();
  int index_area[6][5] = {{0, 1, 2, 3, 0}, {0, 1, 5, 4, 0}, {1, 2, 6, 5, 1},
                          {3, 2, 6, 7, 3}, {0, 3, 7, 4, 0}, {4, 5, 6, 7, 4}};
  for (int i = 0; i < 6; i++) {
    std::vector<Eigen::Matrix<double, Tdim, 1>> intersections;
    for (int j = 0; j < 4; j++) {
      double phi[2];
      phi[0] = nodes_[index_area[i][j]]->discontinuity_property("levelset_phi",
                                                                1)(0, 0);
      phi[1] = nodes_[index_area[i][j + 1]]->discontinuity_property(
          "levelset_phi", 1)(0, 0);
      if (phi[0] * phi[1] >= 0) continue;
      Eigen::Matrix<double, Tdim, 1> intersection;
      Eigen::Matrix<double, Tdim, 1> cor0 =
          nodes_[index_area[i][j]]->coordinates();
      Eigen::Matrix<double, Tdim, 1> cor1 =
          nodes_[index_area[i][j + 1]]->coordinates();
      intersection = cor0 * std::abs(phi[1] / ((phi[1] - phi[0]))) +
                     cor1 * std::abs(phi[0] / ((phi[1] - phi[0])));

      intersections.push_back(intersection);
      intersections_list.push_back(intersection);
    }
    // if (intersections.size() != 2) continue;
    // if (this->discontinuity_element_->area() == 0) continue;
    // Eigen::Matrix<double, Tdim, 1> cor =
    //     1.0 / 3 * (intersections[0] + intersections[1] + center);
    // double length = (cor - center).norm();
    // if (length < 0.25 * mean_length_) continue;
    // coordinates.push_back(cor);
  }

  //   if (this->discontinuity_element_->area() != 0) {
  //     coordinates.push_back(this->discontinuity_element_->cohesion_cor());
  //   } else {
  //     if (intersections_list.size() < 3) return;

  //     Eigen::Matrix<double, Tdim, 1> cor;
  //     cor.setZero();
  //     for (int i = 0; i < intersections_list.size(); i++)
  //       cor += 1.0 / intersections_list.size() * intersections_list[i];
  //     coordinates.push_back(cor);
  //   }
  if (intersections_list.size() < 3) return;

  Eigen::Matrix<double, Tdim, 1> cor;
  cor.setZero();
  for (int i = 0; i < intersections_list.size(); i++)
    cor += 1.0 / intersections_list.size() * intersections_list[i];
  coordinates.push_back(cor);
}

//! assign the normal direction of the discontinuity in the cell
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_normal_discontinuity(VectorDim normal) {
  this->discontinuity_element_->assign_normal_discontinuity(normal);
}

//! assign the normal direction of the discontinuity in the cell
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_normal_discontinuity(VectorDim normal, double d) {
  this->discontinuity_element_->assign_normal_discontinuity(normal);
  this->discontinuity_element_->assign_d(d);
}

//! compute normal vector of discontinuity by the nodal level set values
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_normal_vector_discontinuity() {
  VectorDim normal;
  normal.setZero();
  // determine the discontinuity plane by the enriched nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    double phi = nodes_[i]->discontinuity_property("levelset_phi", 1)(0, 0);
    for (unsigned int j = 0; j < Tdim; j++) {
      normal[j] += phi * dn_dx_centroid_(i, j);
    }
  }
  normal.normalize();
  this->discontinuity_element_->assign_normal_discontinuity(normal);
}

//! compute gradient of the nodal level set values
template <unsigned Tdim>
Eigen::Matrix<double, Tdim, 1> mpm::Cell<Tdim>::compute_gradient_levelset() {
  VectorDim gradient;
  gradient.setZero();
  // determine the discontinuity plane by the enriched nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    double phi = nodes_[i]->discontinuity_property("levelset_phi", 1)(0, 0);
    for (unsigned int j = 0; j < Tdim; j++) {
      gradient[j] += phi * dn_dx_centroid_(i, j);
    }
  }

  return gradient;
}

//! assign nodal normal vector
template <unsigned Tdim>
void mpm::Cell<Tdim>::add_cell_in_node() {
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->add_cell_id(id_);
  }
}

template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_plane_discontinuity(bool enrich) {
  int enriched_node = 0;
  auto normal = discontinuity_element_->normal_discontinuity();
  double dis = 0;
  // determine the discontinuity plane by the enriched nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    if (enrich) {
      if (!nodes_[i]->discontinuity_enrich()) continue;
    }
    enriched_node++;
    auto node_coordinate = nodes_[i]->coordinates();
    for (unsigned int j = 0; j < Tdim; j++)
      dis -= node_coordinate[j] * normal[j];
    dis = nodes_[i]->discontinuity_property("levelset_phi", 1)(0, 0) + dis;
  }
  // update the level set values of the unenriched nodes
  dis = dis / enriched_node;
  this->discontinuity_element_->assign_d(dis);
}

// product of the nodal level set value
template <unsigned Tdim>
double mpm::Cell<Tdim>::product_levelset() {
  double levelset_max = -std::numeric_limits<double>::max();
  double levelset_min = std::numeric_limits<double>::max();
  for (unsigned i = 0; i < nodes_.size(); ++i) {

    double levelset =
        nodes_[i]->discontinuity_property("levelset_phi", 1)(0, 0);
    levelset_max = levelset > levelset_max ? levelset : levelset_max;
    levelset_min = levelset < levelset_min ? levelset : levelset_min;
  }
  return levelset_max * levelset_min;
}

template <unsigned Tdim>
void mpm::Cell<Tdim>::determine_crossed() {

  //if (this->nparticles() == 0) return;

  double max_phi = -1e15, min_phi = 1e15;

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    double phi = nodes_[i]->discontinuity_property("levelset_phi", 1)(0, 0);
    if (phi > max_phi) max_phi = phi;
    if (phi < min_phi) min_phi = phi;
  }

  this->assign_type_discontinuity(mpm::EnrichType::Regular);
  if (max_phi * min_phi >= 0) return;

  this->assign_type_discontinuity(mpm::EnrichType::Crossed);
}

template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_nodal_levelset_equation() {
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    auto coor = nodes_[i]->coordinates();
    double phi = 0;
    for (unsigned int j = 0; j < Tdim; j++)
      phi += coor[j] * this->discontinuity_element_->normal_discontinuity()[j];
    phi += this->discontinuity_element_->d_discontinuity();
    Eigen::Matrix<double, 1, 1> phi_matrix;
    phi_matrix(0, 0) = phi;
    nodes_[i]->assign_discontinuity_property(true, "levelset_phi", phi_matrix,
                                             0, 1);
  }
}

template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_area_discontinuity() {

  //   if(this->id() == 1671)
  //     int a = 0;

  if (this->discontinuity_element_ == nullptr) return;
  if (this->discontinuity_element_->element_type() != mpm::EnrichType::Crossed)
    return;
  // compute the level set values
  Eigen::VectorXd phi_list(nnodes());
  phi_list.setZero();

  auto normal = this->discontinuity_element_->normal_discontinuity();
  auto d = this->discontinuity_element_->d_discontinuity();
  for (int i = 0; i < nodes_.size(); ++i) {
    phi_list[i] = normal.dot(nodes_[i]->coordinates()) + d;
  }
  // determine the intersections
  std::vector<Eigen::Matrix<double, Tdim, 1>> intersections;

  // node id of the 12 edges of one cell
  int index_line[13][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5},
                           {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 4}};
  for (int i = 0; i < 12; ++i) {
    if (phi_list[index_line[i][0]] * phi_list[index_line[i][1]] >= 0) continue;

    Eigen::Matrix<double, Tdim, 1> intersection;
    Eigen::Matrix<double, Tdim, 1> cor0 =
        nodes_[index_line[i][0]]->coordinates();
    Eigen::Matrix<double, Tdim, 1> cor1 =
        nodes_[index_line[i][1]]->coordinates();
    intersection = cor0 * std::abs(phi_list[index_line[i][1]] /
                                   ((phi_list[index_line[i][1]] -
                                     phi_list[index_line[i][0]]))) +
                   cor1 * std::abs(phi_list[index_line[i][0]] /
                                   ((phi_list[index_line[i][1]] -
                                     phi_list[index_line[i][0]])));

    intersections.push_back(intersection);
  }
  if (intersections.size() < 3) return;
  Eigen::Matrix<double, Tdim, 1> average_cor;
  average_cor.setZero();
  for (int i = 0; i < intersections.size(); ++i)
    average_cor += intersections[i];

  average_cor /= intersections.size();

  // compute angle
  // obtain e1 e2 of the local coordinate system
  Eigen::Matrix<double, Tdim, 1> e1 =
      (intersections[0] - average_cor).normalized();
  Eigen::Matrix<double, Tdim, 1> e2 = normal.cross(e1).normalized();
  // the angle and the order of the intersections
  Eigen::VectorXd angles(intersections.size());
  angles.setZero();
  Eigen::VectorXd orders(intersections.size());
  orders.setZero();

  for (int i = 1; i < intersections.size(); ++i) {
    double costh = (intersections[i] - average_cor).normalized().dot(e1);
    double sinth = (intersections[i] - average_cor).normalized().dot(e2);

    costh = costh > 1 ? 1 : costh;
    costh = costh < -1 ? -1 : costh;

    double theta = std::acos(costh);

    if (sinth < 0) theta = 2 * M_PI - theta;

    angles[i] = theta;
  }
  // compute orders
  for (int i = 1; i < intersections.size(); ++i) {
    for (int j = 0; j < intersections.size(); j++) {
      if (angles[i] > angles[j]) orders[i] += 1;
    }
  }

  // exchange intersections
  auto intersections_copy = intersections;
  for (int i = 1; i < intersections.size(); ++i)
    intersections[orders[i]] = intersections_copy[i];

  // compute area
  double area = 0.0;
  Eigen::Matrix<double, Tdim, 1> subcenters;
  subcenters.setZero();
  for (int i = 0; i < intersections.size() - 2; ++i) {
    // the coordinates of the triangle
    Eigen::Matrix<double, Tdim, 1> cor0 = intersections[0];
    Eigen::Matrix<double, Tdim, 1> cor1 = intersections[i + 1];
    Eigen::Matrix<double, Tdim, 1> cor2 = intersections[i + 2];
    double subarea =
        std::abs(0.5 * (cor1 - cor0).cross(cor2 - cor0).dot(normal));
    area += subarea;
    subcenters += subarea * 1 / 3 * (cor0 + cor1 + cor2);
  }
  subcenters = subcenters / area;

  this->discontinuity_element_->assign_area(area);
  this->discontinuity_element_->assign_cohesion_cor(subcenters);
}

template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_cohesion_area() {

  auto centers = this->discontinuity_element_->cohesion_cor();
  auto area = this->discontinuity_element_->area();

  const Eigen::Matrix<double, Tdim, 1> zeros =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> xi;

  if (!this->is_point_in_cell(centers, &xi)) return;

  auto shapefn = element_->shapefn(xi, zeros, zeros);
  Eigen::Matrix<double, 1, 1> node_area;
  for (int i = 0; i < nodes_.size(); i++) {

    node_area(0, 0) = shapefn[i] * area;
    nodes_[i]->update_discontinuity_property(true, "cohesion_area", node_area,
                                             0, 1);
  }

}