// Constructor with id and coordinates
//! \param[in] id Global cell id
//! \param[in] nnodes Number of nodes per cell
//! \tparam Tdim Dimension
template <unsigned Tdim>
mpm::Cell<Tdim>::Cell(Index id, unsigned nnodes) : id_{id}, nnodes_{nnodes} {
  // Check if the dimension is between 1 & 3
  static_assert((Tdim >= 1 && Tdim <= 3), "Invalid global dimension");
  try {
    // Number of nodes should be greater than dimensions
    if (!(nnodes > Tdim)) {
      throw std::runtime_error(
          "Specified number of nodes for a cell is too low");
    }
  } catch (std::exception& exception) {
    std::cerr << exception.what() << '\n';
    std::abort();
  }
}

// Constructor with id, coordinates and shapefn
//! \param[in] id Global cell id
//! \param[in] nnodes Number of nodes per cell
//! \param[in] shapefnptr Pointer to a shape function
//! \tparam Tdim Dimension
template <unsigned Tdim>
mpm::Cell<Tdim>::Cell(Index id, unsigned nnodes,
                      const std::shared_ptr<ShapeFn<Tdim>>& shapefnptr)
    : mpm::Cell<Tdim>::Cell(id, nnodes) {

  try {
    if (shapefnptr->nfunctions() >= this->nnodes_) {
      shapefn_ = shapefnptr;
    } else {
      throw std::runtime_error(
          "Specified number of shape functions is not defined");
    }
  } catch (std::exception& exception) {
    std::cerr << exception.what() << '\n';
    std::abort();
  }
}

// Assign a shape function to cell
//! \param[in] shapefnptr Pointer to a shape function
//! \tparam Tdim Dimension
template <unsigned Tdim>
bool mpm::Cell<Tdim>::shapefn(
    const std::shared_ptr<ShapeFn<Tdim>>& shapefnptr) {
  bool status = false;
  try {
    if (shapefnptr->nfunctions() >= this->nnodes_) {
      shapefn_ = shapefnptr;
      status = true;
    } else {
      throw std::runtime_error(
          "Specified number of shape functions is not defined");
    }
  } catch (std::exception& exception) {
    std::cerr << exception.what() << '\n';
  }
  return status;
}

//! Add a node pointer
//! \param[in] local_id local id of the node
//! \param[in] ptr A shared pointer
//! \retval insertion_status Return the successful addition of a node
//! \tparam Tdim Dimension
template <unsigned Tdim>
bool mpm::Cell<Tdim>::add_node(
    unsigned local_id, const std::shared_ptr<mpm::NodeBase<Tdim>>& node_ptr) {
  bool insertion_status = false;
  try {
    // If number of node ptrs in a cell is less than the maximum number of nodes
    // per cell
    // The local id should be between 0 and maximum number of nodes
    if (nodes_.size() < this->nnodes_ &&
        (local_id >= 0 && local_id < this->nnodes_)) {
      insertion_status = nodes_.insert(local_id, node_ptr);
    } else {
      throw std::runtime_error(
          "Number nodes in a cell exceeds the maximum allowed per cell");
    }
  } catch (std::exception& exception) {
    std::cerr << exception.what() << "\n";
  }
  return insertion_status;
}

//! Add a neighbour cell
//! \param[in] local_id local id of the cell
//! \param[in] ptr A shared pointer to cell
//! \retval insertion_status Return the successful addition of a node
//! \tparam Tdim Dimension
template <unsigned Tdim>
bool mpm::Cell<Tdim>::add_neighbour(
    unsigned local_id, const std::shared_ptr<mpm::Cell<Tdim>>& cell_ptr) {
  bool insertion_status = false;
  try {
    // If number of cell ptrs id is not the current cell id
    if (cell_ptr->id() != this->id()) {
      insertion_status = neighbour_cells_.insert(local_id, cell_ptr);
    } else {
      throw std::runtime_error("Invalid local id of a cell neighbour");
    }
  } catch (std::exception& exception) {
    std::cerr << exception.what() << "\n";
  }
  return insertion_status;
}

//! Add a particle id
//! \param[in] id Global id of a particle
//! \retval status Return the successful addition of a particle id
//! \tparam Tdim Dimension
template <unsigned Tdim>
bool mpm::Cell<Tdim>::add_particle_id(Index id) {
  bool status = false;
  // Check if it is found in the container
  auto itr = std::find(particles_.begin(), particles_.end(), id);

  if (itr == particles_.end()) {
    particles_.emplace_back(id);
    status = true;
  }

  return status;
}

//! Remove a particle id
//! \param[in] id Global id of a particle
//! \retval status Return the successful removal of a particle id
//! \tparam Tdim Dimension
template <unsigned Tdim>
void mpm::Cell<Tdim>::remove_particle_id(Index id) {
  particles_.erase(std::remove(particles_.begin(), particles_.end(), id),
                   particles_.end());
}

//! Compute volume of cell
//! \retval volume Cell volume / area
//! \tparam Tdim Dimension
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_volume() {
  Eigen::VectorXi indices = shapefn_->volume_indices();
  try {
    // Hexahedron
    if (indices.size() == 8) {
      // Node numbering as read in by mesh file
      //        d               c
      //          *_ _ _ _ _ _*
      //         /|           /|
      //        / |          / |
      //     a *_ |_ _ _ _ _* b|
      //       |  |         |  |
      //       |  |         |  |
      //       |  *_ _ _ _ _|_ *
      //       | / h        | / g
      //       |/           |/
      //       *_ _ _ _ _ _ *
      //     e               f
      //
      // Calculation of hexahedron volume from
      // https://arc.aiaa.org/doi/pdf/10.2514/3.9013

      const Eigen::Vector3d a = nodes_[indices[7]]->coordinates();
      const Eigen::Vector3d b = nodes_[indices[6]]->coordinates();
      const Eigen::Vector3d c = nodes_[indices[2]]->coordinates();
      const Eigen::Vector3d d = nodes_[indices[3]]->coordinates();
      const Eigen::Vector3d e = nodes_[indices[4]]->coordinates();
      const Eigen::Vector3d f = nodes_[indices[5]]->coordinates();
      const Eigen::Vector3d g = nodes_[indices[1]]->coordinates();
      const Eigen::Vector3d h = nodes_[indices[0]]->coordinates();

      volume_ =
          (1.0 / 12) *
              (a - g).dot(((b - d).cross(c - a)) + ((e - b).cross(f - a)) +
                          ((d - e).cross(h - a))) +
          (1.0 / 12) *
              (b - g).dot(((b - d).cross(c - a)) + ((c - g).cross(c - f))) +
          (1.0 / 12) *
              (e - g).dot(((e - b).cross(f - a)) + ((f - g).cross(h - f))) +
          (1.0 / 12) *
              (d - g).dot(((d - e).cross(h - a)) + ((h - g).cross(h - c)));

    } else {
      throw std::runtime_error(
          "Unable to compute volume, number of vertices is incorrect");
    }
  } catch (std::exception& except) {
    std::cout << "Compute volume of a cell: " << except.what() << '\n';
  }
}
