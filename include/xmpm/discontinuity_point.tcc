// Assign a cell to particle
template <unsigned Tdim>
bool mpm::discontinuity_point<Tdim>::assign_cell_xi(
    const std::shared_ptr<Cell<Tdim>>& cellptr,
    const Eigen::Matrix<double, Tdim, 1>& xi) {
  bool status = true;
  try {
    // Assign cell to the new cell ptr, if point can be found in new cell
    if (cellptr != nullptr) {

      cell_ = cellptr;
      cell_id_ = cellptr->id();
      nodes_ = cell_->nodes();
      // Assign the reference location of particle
      bool xi_nan = false;

      // Check if point is within the cell
      for (unsigned i = 0; i < xi.size(); ++i)
        if (xi(i) < -1. || xi(i) > 1. || std::isnan(xi(i))) xi_nan = true;

      if (xi_nan == false)
        this->xi_ = xi;
      else
        return false;
    } else {
      console_->warn("Points of discontinuity cannot be found in cell!");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assign a cell to point
template <unsigned Tdim>
bool mpm::discontinuity_point<Tdim>::assign_cell(
    const std::shared_ptr<Cell<Tdim>>& cellptr) {
  bool status = true;
  try {
    Eigen::Matrix<double, Tdim, 1> xi;
    // Assign cell to the new cell ptr, if point can be found in new cell
    if (cellptr->is_point_in_cell(this->coordinates_, &xi)) {

      cell_ = cellptr;
      cell_id_ = cellptr->id();
      nodes_ = cell_->nodes();
    } else {
      console_->warn("Points of discontinuity cannot be found in cell!");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Compute reference location cell to particle
template <unsigned Tdim>
bool mpm::discontinuity_point<Tdim>::compute_reference_location() noexcept {
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

//! Locate points in a cell
template <unsigned Tdim>
void mpm::discontinuity_point<Tdim>::locate_discontinuity_mesh(
    const Vector<Cell<Tdim>>& cells, const Map<Cell<Tdim>>& map_cells,
    unsigned dis_id) noexcept {
  // Check the current cell if it is not invalid
  if (cell_id() != std::numeric_limits<mpm::Index>::max()) {
    // If a cell id is present, but not a cell locate the cell from map
    if (!cell_ptr()) assign_cell(map_cells[cell_id()]);

    if (compute_reference_location()) {
      assign_cell_enrich(map_cells, dis_id);
      return;
    }

    // Check if discontinuity point is in any of its nearest neighbours
    const auto neighbours = map_cells[cell_id()]->neighbours();
    Eigen::Matrix<double, Tdim, 1> xi;
    for (auto neighbour : neighbours) {
      if (map_cells[neighbour]->is_point_in_cell(coordinates_, &xi)) {
        assign_cell_xi(map_cells[neighbour], xi);
        assign_cell_enrich(map_cells, dis_id);
        return;
      }
    }
  }
#pragma omp parallel for schedule(runtime)
  for (auto citr = cells.cbegin(); citr != cells.cend(); ++citr) {
    // Check if particle is already found, if so don't run for other cells
    // Check if co-ordinates is within the cell, if true
    // add particle to cell
    Eigen::Matrix<double, Tdim, 1> xi;
    if ((*citr)->is_point_in_cell(coordinates(), &xi)) {
      assign_cell_xi(*citr, xi);
      assign_cell_enrich(map_cells, dis_id);
    }
  }
}

// Compute updated position of the particle
template <unsigned Tdim>
void mpm::discontinuity_point<Tdim>::compute_updated_position(
    const double dt, int move_direction) noexcept {
  // Check if point has a valid cell ptr
  if (cell_ == nullptr) return;

  // Get interpolated nodal velocity
  Eigen::Matrix<double, Tdim, 1> nodal_velocity =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  const double tolerance = 1.E-16;
  unsigned int phase = 0;

  // TODO: need to do, points move with which side
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // nodal mass and momentum
    double nodal_mass = nodes_[i]->mass(phase);
    auto nodal_momentum = nodes_[i]->momentum(phase);
    const auto& nodal_mass_enrich = nodes_[i]->mass_enrich();
    const auto& nodal_momentum_enrich = nodes_[i]->momentum_enrich();

    // TODO: to be checked
    const auto discontinuity_id = nodes_[i]->discontinuity_id();

    if (nodes_[i]->enrich_type() == mpm::NodeEnrichType::single_enriched) {
      nodal_mass += nodal_mass_enrich[0] * move_direction;
      nodal_momentum.col(0) += nodal_momentum_enrich.col(0) * move_direction;
    } else if (nodes_[i]->enrich_type() ==
               mpm::NodeEnrichType::double_enriched) {
      nodal_mass += nodal_mass_enrich[0] * move_direction +
                    nodal_mass_enrich[1] * move_direction +
                    nodal_mass_enrich[2] * move_direction * move_direction;

      nodal_momentum.col(0) +=
          nodal_momentum_enrich.col(0) * move_direction +
          nodal_momentum_enrich.col(1) * move_direction +
          nodal_momentum_enrich.col(2) * move_direction * move_direction;
    }
    if (nodal_mass < tolerance) continue;

    nodal_velocity += shapefn_[i] * nodal_momentum / nodal_mass;
  }

  // New position current position += velocity * dt
  this->coordinates_ += nodal_velocity * dt;
}

// Compute updated position of the particle
template <unsigned Tdim>
void mpm::discontinuity_point<Tdim>::compute_shapefn() noexcept {
  // Check if point has a valid cell ptr
  if (cell_ == nullptr) return;
  // Get element ptr of a cell
  const auto element = cell_->element_ptr();

  // Zero matrix
  Eigen::Matrix<double, Tdim, 1> zero = Eigen::Matrix<double, Tdim, 1>::Zero();

  // Compute shape function of the point
  //! Size of particle in natural coordinates
  Eigen::Matrix<double, 1, Tdim> natural_size_;
  natural_size_.setZero();
  shapefn_ = element->shapefn(this->xi_, natural_size_, zero);
}

//! Assign the discontinuity enrich type to cell
template <unsigned Tdim>
void mpm::discontinuity_point<Tdim>::assign_cell_enrich(
    const Map<Cell<Tdim>>& map_cells, unsigned dis_id) {
  if (cell_->nparticles() == 0) return;
  cell_->assign_discontinuity_type(mpm::EnrichType::Crossed, dis_id);
  const auto neighbours_1 = cell_->neighbours();
  for (auto neighbour_1 : neighbours_1) {
    if (map_cells[neighbour_1]->element_discontinuity_type(dis_id) ==
        mpm::EnrichType::Crossed)
      continue;
    map_cells[neighbour_1]->assign_discontinuity_type(
        mpm::EnrichType::NeighbourTip_1, dis_id);
    const auto neighbours_2 = map_cells[neighbour_1]->neighbours();
    for (auto neighbour_2 : neighbours_2) {
      if (map_cells[neighbour_2]->element_discontinuity_type(dis_id) ==
              mpm::EnrichType::Regular ||
          map_cells[neighbour_2]->element_discontinuity_type(dis_id) ==
              mpm::EnrichType::NeighbourTip_3)
        map_cells[neighbour_2]->assign_discontinuity_type(
            mpm::EnrichType::NeighbourTip_2, dis_id);

      const auto neighbours_3 = map_cells[neighbour_2]->neighbours();
      for (auto neighbour_3 : neighbours_3) {
        if (map_cells[neighbour_3]->element_discontinuity_type(dis_id) ==
            mpm::EnrichType::Regular)
          map_cells[neighbour_3]->assign_discontinuity_type(
              mpm::EnrichType::NeighbourTip_3, dis_id);
      }
    }
  }
}