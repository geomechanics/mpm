//! Constructor with id and coordinates
template <unsigned Tdim>
mpm::DiscontinuityPoint<Tdim>::DiscontinuityPoint(const VectorDim& coord,
                                                  mpm::Index dis_id)
    : mpm::PointBase<Tdim>(coord) {
  // dis_id
  dis_id_ = dis_id;
  // Logger
  std::string logger = "discontinuity point" + std::to_string(Tdim) + "d";
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  // initialise
  initialise();
}
//! Initialise properties
template <unsigned Tdim>
void mpm::DiscontinuityPoint<Tdim>::initialise() {
  this->scalar_properties_["discontinuity_id"] = [&]() { return dis_id_; };
  this->scalar_properties_["terminal_point"] = [&]() {
    return terminal_point_;
  };
}

//! Locate points in a cell
template <unsigned Tdim>
void mpm::DiscontinuityPoint<Tdim>::locate_discontinuity_mesh(
    const Vector<Cell<Tdim>>& cells, const Map<Cell<Tdim>>& map_cells,
    unsigned dis_id, bool update) {
  // Check the current cell if it is not invalid
  if (this->cell_id() != std::numeric_limits<mpm::Index>::max()) {
    // If a cell id is present, but not a cell locate the cell from map
    if (!this->cell_ptr()) this->assign_cell(map_cells[this->cell_id()]);

    if (this->compute_reference_location()) {
      if (update) assign_cell_enrich(map_cells, dis_id);
      return;
    }

    // Check if discontinuity point is in any of its nearest neighbours
    const auto neighbours = map_cells[this->cell_id()]->neighbours();
    Eigen::Matrix<double, Tdim, 1> xi;
    for (auto neighbour : neighbours) {
      if (map_cells[neighbour]->is_point_in_cell(coordinates_, &xi)) {
        this->assign_cell_xi(map_cells[neighbour], xi);
        if (update) assign_cell_enrich(map_cells, dis_id);
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
    if ((*citr)->is_point_in_cell(coordinates_, &xi)) {
      this->assign_cell_xi(*citr, xi);
      if (update) assign_cell_enrich(map_cells, dis_id);
    }
  }
}

// Compute updated position of the particle
template <unsigned Tdim>
void mpm::DiscontinuityPoint<
    Tdim>::compute_updated_position_discontinuity_point(const double dt,
                                                        int move_direction) {
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

//! Assign the discontinuity enrich type to cell
template <unsigned Tdim>
void mpm::DiscontinuityPoint<Tdim>::assign_cell_enrich(
    const Map<Cell<Tdim>>& map_cells, unsigned dis_id) {
  if (cell_->nparticles() == 0) return;
  cell_->assign_discontinuity_type(mpm::EnrichType::Crossed, dis_id);
  if (terminal_point_) return;
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