//! Assign discontinuity element type
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_type_discontinuity(mpm::EnrichType type,
                                                unsigned dis_id) {
  if (nparticles() == 0 && type != mpm::EnrichType::NeighbourTip_2)
    type = mpm::EnrichType::Regular;
  if (discontinuity_element_[dis_id] == nullptr)
    discontinuity_element_[dis_id] =
        std::make_shared<mpm::DiscontinuityElement<Tdim>>(type);
  else
    discontinuity_element_[dis_id]->assign_element_type(type);
}
// Initialize discontinuity element type
template <unsigned Tdim>
void mpm::Cell<Tdim>::initialise_element_properties_discontinuity() {
  for (int i = 0; i < discontinuity_element_.size(); i++) {
    if (discontinuity_element_[i] == nullptr) continue;
    discontinuity_element_[i]->initialise();
  }
}

// Return discontinuity element type
template <unsigned Tdim>
unsigned mpm::Cell<Tdim>::element_type_discontinuity(unsigned dis_id) {
  if (discontinuity_element_[dis_id] == nullptr)
    return mpm::EnrichType::Regular;
  return discontinuity_element_[dis_id]->element_type();
}

//! Find the potential tip element
template <unsigned Tdim>
void mpm::Cell<Tdim>::find_potential_tip_cell(unsigned dis_id) {
  // Perform necessary checks
  if (this->discontinuity_element_[dis_id] == nullptr) return;
  if (this->discontinuity_element_[dis_id]->element_type() !=
      mpm::EnrichType::NeighbourTip_1)
    return;
  if (this->nparticles() == 0) return;

  if (product_levelset(dis_id) < 0)
    this->discontinuity_element_[dis_id]->assign_element_type(
        mpm::EnrichType::PotentialTip);
}

//! Determine the tip cells
template <unsigned Tdim>
void mpm::Cell<Tdim>::find_tip_cell(unsigned dis_id) {
  if (this->discontinuity_element_[dis_id] == nullptr) return;
  if (this->discontinuity_element_[dis_id]->element_type() !=
      mpm::EnrichType::Crossed)
    return;

  for (unsigned i = 0; i < nodes_.size(); ++i) {
    if (nodes_[i]->discontinuity_enrich(dis_id)) continue;
    this->discontinuity_element_[dis_id]->assign_element_type(
        mpm::EnrichType::Tip);
  }
}

//! Compute the discontinuity point: the average coordinates of the intersection
//! points
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_discontinuity_point(
    std::vector<VectorDim>& coordinates, unsigned dis_id) {

  std::vector<Eigen::Matrix<double, Tdim, 1>> intersections_list;

  Eigen::Matrix<double, Tdim, 1> center =
      this->discontinuity_element_[dis_id]->cohesion_cor();

  const int index_line[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                 {0, 4}, {1, 5}, {2, 6}, {3, 7},
                                 {4, 5}, {5, 6}, {6, 7}, {7, 4}};
  for (int i = 0; i < 12; i++) {

    double phi[2];
    phi[0] = nodes_[index_line[i][0]]->levelset_phi(dis_id);
    phi[1] = nodes_[index_line[i][1]]->levelset_phi(dis_id);
    if (phi[0] * phi[1] >= 0) continue;
    Eigen::Matrix<double, Tdim, 1> intersection;
    Eigen::Matrix<double, Tdim, 1> cor0 =
        nodes_[index_line[i][0]]->coordinates();
    Eigen::Matrix<double, Tdim, 1> cor1 =
        nodes_[index_line[i][1]]->coordinates();
    intersection = cor0 * std::abs(phi[1] / ((phi[1] - phi[0]))) +
                   cor1 * std::abs(phi[0] / ((phi[1] - phi[0])));

    intersections_list.push_back(intersection);
  }

  if (intersections_list.size() < 3) return;

  Eigen::Matrix<double, Tdim, 1> cor = Eigen::Matrix<double, Tdim, 1>::Zero();
  for (int i = 0; i < intersections_list.size(); i++)
    cor.noalias() += 1.0 / intersections_list.size() * intersections_list[i];
  coordinates.push_back(cor);
}

//! Assign the normal direction of the discontinuity in the cell
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_normal_discontinuity(VectorDim normal,
                                                  unsigned dis_id) {
  this->discontinuity_element_[dis_id]->assign_normal_discontinuity(normal);
}

//! Assign the normal direction of the discontinuity in the cell
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_normal_discontinuity(VectorDim normal, double d,
                                                  unsigned dis_id) {
  this->discontinuity_element_[dis_id]->assign_normal_discontinuity(normal);
  this->discontinuity_element_[dis_id]->assign_d(d);
}

//! Compute normal vector of discontinuity by the nodal level set values
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_normal_vector_discontinuity(unsigned dis_id) {

  VectorDim normal = VectorDim::Zero();

  // Determine the discontinuity plane by the enriched nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const double phi = nodes_[i]->levelset_phi(dis_id);
    for (unsigned int j = 0; j < Tdim; j++) {
      normal[j] += phi * dn_dx_centroid_(i, j);
    }
  }

  normal.normalize();
  this->discontinuity_element_[dis_id]->assign_normal_discontinuity(normal);
}

//! Compute the discontinuity plane by the nodal level set values
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_plane_discontinuity(bool enrich,
                                                  unsigned dis_id) {
  int enriched_node = 0;
  const auto& normal = discontinuity_element_[dis_id]->normal_discontinuity();
  double dis = 0;

  // Determine the discontinuity plane by the enriched nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Option to only use enriched nodes
    if (enrich)
      if (!nodes_[i]->discontinuity_enrich(dis_id)) continue;

    enriched_node++;
    const auto& node_coordinate = nodes_[i]->coordinates();
    const double r = node_coordinate.dot(normal);
    dis += nodes_[i]->levelset_phi(dis_id) - r;
  }

  // Update the plane equation constant
  dis = dis / enriched_node;
  this->discontinuity_element_[dis_id]->assign_d(dis);
}

// product of the maximum and minimum nodal level set value
template <unsigned Tdim>
double mpm::Cell<Tdim>::product_levelset(unsigned dis_id) {
  double levelset_max = -std::numeric_limits<double>::max();
  double levelset_min = std::numeric_limits<double>::max();
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    double levelset = nodes_[i]->levelset_phi(dis_id);
    levelset_max = levelset > levelset_max ? levelset : levelset_max;
    levelset_min = levelset < levelset_min ? levelset : levelset_min;
  }
  return levelset_max * levelset_min;
}

//! Determine the celltype by the nodal level set
template <unsigned Tdim>
void mpm::Cell<Tdim>::determine_crossed_cell(unsigned dis_id) {
  if (this->product_levelset(dis_id) >= 0) return;
  this->assign_type_discontinuity(mpm::EnrichType::Crossed, dis_id);
}

//! Compute the nodal level set values by plane equations
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_nodal_levelset_equation(unsigned dis_id) {
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    const auto& coor = nodes_[i]->coordinates();
    double phi = 0;
    for (unsigned int j = 0; j < Tdim; j++)
      phi += coor[j] *
             this->discontinuity_element_[dis_id]->normal_discontinuity()[j];
    phi += this->discontinuity_element_[dis_id]->d_discontinuity();
    nodes_[i]->assign_levelset_phi(phi, dis_id);
  }
}

//! Compute the area of the discontinuity
template <unsigned Tdim>
void mpm::Cell<Tdim>::compute_area_discontinuity(unsigned dis_id) {

  if (this->discontinuity_element_[dis_id] == nullptr) return;
  if (this->discontinuity_element_[dis_id]->element_type() !=
      mpm::EnrichType::Crossed)
    return;

  // Compute the level set values in all nodes
  Eigen::VectorXd phi_list(nnodes());
  phi_list.setZero();

  const auto& normal =
      this->discontinuity_element_[dis_id]->normal_discontinuity();
  const auto d = this->discontinuity_element_[dis_id]->d_discontinuity();
  for (int i = 0; i < nodes_.size(); ++i) {
    phi_list[i] = normal.dot(nodes_[i]->coordinates()) + d;
  }

  // Determine the intersections
  std::vector<Eigen::Matrix<double, Tdim, 1>> intersections;

  // Node id of the 12 edges of one cell
  const int index_line[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                 {0, 4}, {1, 5}, {2, 6}, {3, 7},
                                 {4, 5}, {5, 6}, {6, 7}, {7, 4}};
  for (int i = 0; i < 12; ++i) {
    if (phi_list[index_line[i][0]] * phi_list[index_line[i][1]] >= 0) continue;

    const auto& cor0 = nodes_[index_line[i][0]]->coordinates();
    const auto& cor1 = nodes_[index_line[i][1]]->coordinates();
    const Eigen::Matrix<double, Tdim, 1> intersection =
        cor0 * std::abs(phi_list[index_line[i][1]] /
                        ((phi_list[index_line[i][1]] -
                          phi_list[index_line[i][0]]))) +
        cor1 * std::abs(
                   phi_list[index_line[i][0]] /
                   ((phi_list[index_line[i][1]] - phi_list[index_line[i][0]])));

    intersections.push_back(intersection);
  }

  // Exit if intersection size is less than 3
  if (intersections.size() < 3) return;

  // Compute average coordinates
  Eigen::Matrix<double, Tdim, 1> average_cor =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  for (int i = 0; i < intersections.size(); ++i)
    average_cor += intersections[i];

  average_cor /= intersections.size();

  // Compute angle
  // Obtain bases vectors e1 e2 of the local coordinate system
  const Eigen::Matrix<double, Tdim, 1> e1 =
      (intersections[0] - average_cor).normalized();
  const Eigen::Matrix<double, Tdim, 1> e2 = normal.cross(e1).normalized();

  // The angle and the order of the intersections
  Eigen::VectorXd angles(intersections.size());
  angles.setZero();
  Eigen::VectorXd orders(intersections.size());
  orders.setZero();

  // Loop over the intersections
  for (int i = 1; i < intersections.size(); ++i) {
    double costh = (intersections[i] - average_cor).normalized().dot(e1);
    double sinth = (intersections[i] - average_cor).normalized().dot(e2);

    costh = costh > 1 ? 1 : costh;
    costh = costh < -1 ? -1 : costh;

    double theta = std::acos(costh);
    if (sinth < 0) theta = 2 * M_PI - theta;

    angles[i] = theta;
  }

  // Compute orders
  for (int i = 1; i < intersections.size(); ++i) {
    for (int j = 0; j < intersections.size(); j++) {
      if (angles[i] > angles[j]) orders[i] += 1;
    }
  }

  // Exchange intersections
  auto intersections_copy = intersections;
  for (int i = 1; i < intersections.size(); ++i)
    intersections[orders[i]] = intersections_copy[i];

  // Compute area and weighted center
  double area = 0.0;
  Eigen::Matrix<double, Tdim, 1> weighted_center =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  for (int i = 0; i < intersections.size() - 2; ++i) {
    // the coordinates of the triangle
    const Eigen::Matrix<double, Tdim, 1> cor0 = intersections[0];
    const Eigen::Matrix<double, Tdim, 1> cor1 = intersections[i + 1];
    const Eigen::Matrix<double, Tdim, 1> cor2 = intersections[i + 2];
    const double subarea =
        std::abs(0.5 * (cor1 - cor0).cross(cor2 - cor0).dot(normal));
    area += subarea;
    weighted_center += subarea * 1 / 3 * (cor0 + cor1 + cor2);
  }
  weighted_center = weighted_center / area;

  // Assign area and weighted center to discontinuity plane
  this->discontinuity_element_[dis_id]->assign_area(area);
  this->discontinuity_element_[dis_id]->assign_cohesion_cor(weighted_center);
}

//! Assign the area of the discontinuity to nodes
template <unsigned Tdim>
void mpm::Cell<Tdim>::assign_cohesion_area(unsigned dis_id) {

  auto centers = this->discontinuity_element_[dis_id]->cohesion_cor();
  auto area = this->discontinuity_element_[dis_id]->area();

  const Eigen::Matrix<double, Tdim, 1> zeros =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  Eigen::Matrix<double, Tdim, 1> xi;

  if (!this->is_point_in_cell(centers, &xi)) return;

  auto shapefn = element_->shapefn(xi, zeros, zeros);
  double node_area = 0;
  for (int i = 0; i < nodes_.size(); i++) {

    node_area = shapefn[i] * area;
    nodes_[i]->update_cohesion_area(node_area, dis_id);
  }
}