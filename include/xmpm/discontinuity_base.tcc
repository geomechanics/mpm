//! Constructor
template <unsigned Tdim>
mpm::DiscontinuityBase<Tdim>::DiscontinuityBase(const Json& json_generator,
                                                unsigned id) {
  id_ = id;
  friction_coef_ = 0;
  std::string logger = "discontinuity";
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);

  try {

    id_ = json_generator.at("id").template get<int>();

    description_type_ =
        json_generator.at("description_type").template get<std::string>();

    if (json_generator.contains("friction_coefficient_average"))
      friction_coef_average_ = json_generator.at("friction_coefficient_average")
                                   .template get<bool>();

    if (json_generator.contains("propagation"))
      propagation_ = json_generator.at("propagation").template get<bool>();
    // assign friction_coef_ if it's given in input file
    if (json_generator.contains("friction_coefficient"))
      friction_coef_ =
          json_generator.at("friction_coefficient").template get<double>();
    // assign cohesion_ if it's given in input file
    if (json_generator.contains("cohesion"))
      cohesion_ = json_generator.at("cohesion").template get<double>();
    // assign contact_distance_ if it's given in input file
    if (json_generator.contains("contact_distance"))
      contact_distance_ =
          json_generator.at("contact_distance").template get<double>();

    // assign move direction if it's given in input file
    if (json_generator.contains("move_direction"))
      move_direction_ = json_generator.at("move_direction").template get<int>();
    // assign width if it's given in input file
    if (json_generator.contains("width"))
      width_ = json_generator.at("width").template get<double>();

    // assign maximum_pdstrain if it's given in input file
    if (json_generator.contains("maximum_pdstrain"))
      maximum_pdstrain_ =
          json_generator.at("maximum_pdstrain").template get<double>();
    // assign mls if it's given in input file
    if (json_generator.contains("mls"))
      mls_ = json_generator.at("mls").template get<bool>();

  } catch (Json::exception& except) {
    console_->error("discontinuity parameter not set: {} {}\n", except.what(),
                    except.id);
  }
}

//! Constructor for the initiation
template <unsigned Tdim>
mpm::DiscontinuityBase<Tdim>::DiscontinuityBase(
    unsigned id,
    std::tuple<double, double, double, double, double, int, bool, bool>&
        initiation_property) {

  id_ = id;
  std::string logger = "discontinuity";
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  description_type_ == "mark_points";
  propagation_ = true;

  cohesion_ = std::get<0>(initiation_property);
  friction_coef_ = std::get<1>(initiation_property);
  contact_distance_ = std::get<2>(initiation_property);
  width_ = std::get<3>(initiation_property);
  maximum_pdstrain_ = std::get<4>(initiation_property);
  move_direction_ = std::get<5>(initiation_property);
  friction_coef_average_ = std::get<6>(initiation_property);
  mls_ = std::get<7>(initiation_property);
}

//! Create points from file
template <unsigned Tdim>
bool mpm::DiscontinuityBase<Tdim>::create_points(
    const std::vector<VectorDim>& coordinates) {
  bool status = true;
  try {
    // Check if point coordinates is empty
    if (coordinates.empty())
      throw std::runtime_error("List of coordinates is empty");
    // Iterate over all coordinates
    for (const auto& point_coordinates : coordinates) {

      // Create discontinuity point
      const std::string& point_type = "DiscontinuityPoint3D";
      auto point =
          Factory<mpm::PointBase<Tdim>, const Eigen::Matrix<double, Tdim, 1>&,
                  mpm::Index>::instance()
              ->create(point_type, point_coordinates,
                       static_cast<mpm::Index>(id_));

      bool insert_status = this->add_point(point);

      // When addition of point fails
      if (!insert_status)
        throw std::runtime_error(
            "Addition of discontinuity point to mesh failed!");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}
//! Locate points in a cell
template <unsigned Tdim>
void mpm::DiscontinuityBase<Tdim>::locate_discontinuity_mesh(
    const Vector<Cell<Tdim>>& cells,
    const Map<Cell<Tdim>>& map_cells) noexcept {
  for (auto pitr = points_.cbegin(); pitr != points_.cend(); pitr++)
    (*pitr)->locate_discontinuity_mesh(cells, map_cells, id_, true);
}

// Compute updated position of the particle
template <unsigned Tdim>
void mpm::DiscontinuityBase<Tdim>::compute_updated_position(
    double dt) noexcept {
  for (auto pitr = points_.cbegin(); pitr != points_.cend(); pitr++)
    (*pitr)->compute_updated_position_discontinuity_point(dt, move_direction_);
}

// Compute updated position of the particle
template <unsigned Tdim>
void mpm::DiscontinuityBase<Tdim>::compute_shapefn() noexcept {
  for (auto pitr = points_.cbegin(); pitr != points_.cend(); pitr++)
    (*pitr)->compute_shapefn();
}

//! Insert new point
template <unsigned Tdim>
void mpm::DiscontinuityBase<Tdim>::insert_points(
    VectorDim& coordinates, const Vector<Cell<Tdim>>& cells,
    const Map<Cell<Tdim>>& map_cells) {
  const std::string& point_type = "DiscontinuityPoint3D";
  auto point =
      Factory<mpm::PointBase<Tdim>, const Eigen::Matrix<double, Tdim, 1>&,
              mpm::Index>::instance()
          ->create(point_type, coordinates, static_cast<mpm::Index>(id_));

  point->locate_discontinuity_mesh(cells, map_cells, id_, false);
  point->compute_shapefn();
  // assign terminal type
  unsigned cell_interaction_type =
      map_cells[point->cell_id()]->interaction_type(id_);
  if (cell_interaction_type == mpm::InteractionType::Terminated_1 ||
      cell_interaction_type == mpm::InteractionType::Terminated_2)
    point->assign_terminal_point(true);
  bool insert_status = this->add_point(point);

  // When addition of point fails
  if (!insert_status)
    throw std::runtime_error("Addition of discontinuity point to mesh failed!");
}