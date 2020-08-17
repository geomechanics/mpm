template <unsigned Tdim>
mpm::Discontinuity3D<Tdim>::Discontinuity3D(unsigned id,
                                            const Json& discontinuity_props)
    : DiscontinuityBase<Tdim>(id, discontinuity_props) {

  try {
    // assign friction_coef_ if it's given in input file
    if (discontinuity_props.contains("friction_coefficient"))
      friction_coef_ =
          discontinuity_props.at("friction_coefficient").template get<double>();
    else
      friction_coef_ = 0;
  } catch (Json::exception& except) {
    console_->error("discontinuity parameter not set: {} {}\n", except.what(),
                    except.id);
  }
}

// initialization
template <unsigned Tdim>
bool mpm::Discontinuity3D<Tdim>::initialize(
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
  // Create elements from file
  bool element_status = create_surfaces(surfs);
  if (!element_status) {
    status = false;
    throw std::runtime_error(
        "Addition of elements in discontinuity to mesh failed");
  }

  bool normal_status = initialize_center_normal();
  if (!normal_status) {
    status = false;
    throw std::runtime_error(
        "initialized the center and normal of the discontunity failed");
  }
  return status;
};

//! create elements from file
template <unsigned Tdim>
bool mpm::Discontinuity3D<Tdim>::create_surfaces(
    const std::vector<std::vector<mpm::Index>>& surfs) {

  bool status = true;
  try {
    // Check if elements is empty
    if (surfs.empty()) throw std::runtime_error("List of elements is empty");
    // Iterate over all elements
    for (const auto& points : surfs) {

      mpm::discontinuity_surface<Tdim> element(points);

      surfaces_.emplace_back(element);  //
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// initialize the center and normal of the elements
template <>
bool mpm::Discontinuity3D<3>::initialize_center_normal() {
  bool status = true;
  try {
    VectorDim center;
    VectorDim normal;
    Eigen::Matrix<mpm::Index, 3, 1> points;

    for (auto& element : surfaces_) {
      points = element.points();

      // the center of the element
      for (int i = 0; i < 3; i++)
        center[i] = 1.0 / 3 *
                    (points_[points[0]].coordinates()[i] +
                     points_[points[1]].coordinates()[i] +
                     points_[points[2]].coordinates()[i]);

      element.assign_center(center);

      // the normal of the element
      normal = three_cross_product(points_[points[0]].coordinates(),
                                   points_[points[1]].coordinates(),
                                   points_[points[2]].coordinates());
      double det = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] +
                             normal[2] * normal[2]);
      normal = 1.0 / det * normal;

      element.assign_normal(normal);
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// return the cross product of ab X bc
template <unsigned Tdim>
Eigen::Matrix<double, Tdim, 1> mpm::Discontinuity3D<Tdim>::three_cross_product(
    const VectorDim& a, const VectorDim& b, const VectorDim& c) {

  VectorDim threecross;
  threecross[0] = (b[1] - a[1]) * (c[2] - b[2]) - (b[2] - a[2]) * (c[1] - b[1]);
  threecross[1] = (b[2] - a[2]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[2] - b[2]);
  threecross[2] = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
  return threecross;
}

// return the levelset values of each coordinates
//! \param[in] the vector of the coordinates
template <unsigned Tdim>
void mpm::Discontinuity3D<Tdim>::compute_levelset(
    const std::vector<VectorDim>& coordinates, std::vector<double>& phi_list) {

  mpm::Index i = 0;
  for (const auto& coor : coordinates) {
    // find the nearest distance from particle to cell: need to do by global
    // searching and local searching
    double distance = std::numeric_limits<double>::max();
    for (const auto& element : surfaces_) {
      double vertical_distance =
          element.vertical_distance(coor);  // vertical_distance(coor);
      distance = std::abs(distance) < std::abs(vertical_distance)
                     ? distance
                     : vertical_distance;
      if (!distance) distance = 1e-16;
    }

    phi_list[i] = distance;
    ++i;
  }
}

// return the normal vectors of given coordinates
//! \param[in] the coordinates
template <unsigned Tdim>
void mpm::Discontinuity3D<Tdim>::compute_normal(const VectorDim& coordinates,
                                                VectorDim& normal_vector) {
  // find the nearest distance from particle to cell: need to do by global
  // searching and local searching
  double distance = std::numeric_limits<double>::max();
  for (const auto& element : surfaces_) {
    double vertical_distance =
        element.vertical_distance(coordinates);  // vertical_distance(coor);
    if (std::abs(distance) > std::abs(vertical_distance)) {
      distance = vertical_distance;
      normal_vector = element.normal();
    }
  }
}