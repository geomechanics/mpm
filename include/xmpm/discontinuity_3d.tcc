template <unsigned Tdim>
mpm::Discontinuity3D<Tdim>::Discontinuity3D(const Json& json_generator,
                                            unsigned id)
    : DiscontinuityBase<Tdim>(json_generator, id) {}

template <unsigned Tdim>
mpm::Discontinuity3D<Tdim>::Discontinuity3D(
    unsigned id,
    std::tuple<double, double, double, double, double, int, bool, bool>&
        initiation_property)
    : DiscontinuityBase<Tdim>(id, initiation_property) {}

// initialization
template <unsigned Tdim>
bool mpm::Discontinuity3D<Tdim>::initialise(
    const std::vector<VectorDim>& points,
    const std::vector<std::vector<mpm::Index>>& surfs) {
  bool status = true;
  // Create points from file
  bool point_status = this->create_points(points);
  if (!point_status) {
    status = false;
    throw std::runtime_error(
        "Addition of points in discontinuity to mesh failed");
  }
  // Create surfaces from file
  bool surf_status = create_surfaces(surfs);
  if (!surf_status) {
    status = false;
    throw std::runtime_error(
        "Addition of surfaces in discontinuity to mesh failed");
  }

  bool normal_status = initialise_center_normal();
  if (!normal_status) {
    status = false;
    throw std::runtime_error(
        "initialization of the center and the normal vector of the "
        "discontinuity failed");
  }

  this->assign_point_friction_coef();
  return status;
};

//! Create surfaces from file
template <unsigned Tdim>
bool mpm::Discontinuity3D<Tdim>::create_surfaces(
    const std::vector<std::vector<mpm::Index>>& surfs) {

  bool status = true;
  try {
    // Check if surfs is empty
    if (surfs.empty()) throw std::runtime_error("List of surfaces is empty");
    // Iterate over all surfaces
    for (const auto& points : surfs) {

      mpm::discontinuity_surface<Tdim> surf(points);

      surfaces_.emplace_back(surf);
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Initialize the center and normal of the surfaces
template <>
bool mpm::Discontinuity3D<3>::initialise_center_normal() {
  bool status = true;
  try {
    VectorDim center;
    VectorDim normal;
    Eigen::Matrix<mpm::Index, 3, 1> points;

    for (auto& surf : surfaces_) {
      points = surf.points();

      // the center of the surfaces
      // TODO: refactor with map_points_
      for (int i = 0; i < 3; i++)
        center[i] = 1.0 / 3 *
                    (points_[points[0]]->coordinates()[i] +
                     points_[points[1]]->coordinates()[i] +
                     points_[points[2]]->coordinates()[i]);

      surf.assign_center(center);

      // the normal of the surfaces
      normal = three_cross_product(points_[points[0]]->coordinates(),
                                   points_[points[1]]->coordinates(),
                                   points_[points[2]]->coordinates());

      if (normal.norm() > std::numeric_limits<double>::epsilon())
        normal.normalize();
      else
        normal = VectorDim::Zero();
      surf.assign_normal(normal);
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//! Return the cross product of ab X bc
template <unsigned Tdim>
Eigen::Matrix<double, Tdim, 1> mpm::Discontinuity3D<Tdim>::three_cross_product(
    const VectorDim& a, const VectorDim& b, const VectorDim& c) {
  VectorDim threecross;
  threecross[0] = (b[1] - a[1]) * (c[2] - b[2]) - (b[2] - a[2]) * (c[1] - b[1]);
  threecross[1] = (b[2] - a[2]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[2] - b[2]);
  threecross[2] = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
  return threecross;
}

//! Return the levelset values of each coordinates
template <unsigned Tdim>
void mpm::Discontinuity3D<Tdim>::compute_levelset(const VectorDim& coordinates,
                                                  double& phi_particle) {
  // searching and local searching
  double min_distance = std::numeric_limits<double>::max();
  double vertical_distance = std::numeric_limits<double>::max();
  for (const auto& surf : surfaces_) {
    double distance = surf.ptocenter_distance(coordinates);
    if (std::abs(distance) < std::abs(min_distance)) {
      min_distance = distance;
      vertical_distance = surf.vertical_distance(coordinates);
    }
  }
  if (!vertical_distance)
    vertical_distance = std::numeric_limits<double>::min();
  if (abs(min_distance) < 3 * width_)
    phi_particle = vertical_distance;
  else
    phi_particle = 0;
}

//! Return the normal vectors of given coordinates
template <unsigned Tdim>
void mpm::Discontinuity3D<Tdim>::compute_normal(const VectorDim& coordinates,
                                                VectorDim& normal_vector) {
  double min_distance = std::numeric_limits<double>::max();
  for (const auto& surf : surfaces_) {
    double distance = surf.ptocenter_distance(coordinates);
    if (std::abs(distance) < std::abs(min_distance)) {
      min_distance = distance;
      normal_vector = surf.normal();
    }
  }
}

//! Assign point friction coefficient
template <unsigned Tdim>
void mpm::Discontinuity3D<Tdim>::assign_point_friction_coef() noexcept {
  for (auto pitr = points_.cbegin(); pitr != points_.cend(); pitr++)
    (*pitr)->assign_friction_coef(friction_coef_);
}

//! Compute updated position of the particle
template <unsigned Tdim>
void mpm::Discontinuity3D<Tdim>::compute_updated_position(double dt) noexcept {
  for (auto pitr = points_.cbegin(); pitr != points_.cend(); pitr++)
    (*pitr)->compute_updated_position_discontinuity_point(dt, move_direction_);

  initialise_center_normal();
}