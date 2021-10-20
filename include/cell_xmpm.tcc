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

  // if (this->nparticles() == 0) return;

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