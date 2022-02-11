//! Create the nodal properties' map for
//! discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::create_nodal_properties_discontinuity() {
  // Initialise the shared pointer to nodal properties
  if (nodal_properties_ == nullptr)
    nodal_properties_ = std::make_shared<mpm::NodalProperties>();

  // Check if nodes_ is empty and throw runtime error if they are
  assert(nodes_.size());
  // Compute number of rows in nodal properties for vector entities
  const unsigned nrows = nodes_.size() * Tdim;
  // Create pool data for each property in the nodal properties struct
  // object. Properties must be named in the plural form
  // nodal_properties_->create_property("mass_enrich", nodes_.size(), 1);
  // nodal_properties_->create_property("levelset_phi", nodes_.size(), 1);
  // nodal_properties_->create_property("momenta_enrich", nrows, 1);
  // nodal_properties_->create_property("internal_force_enrich", nrows, 1);
  // nodal_properties_->create_property("external_force_enrich", nrows, 1);
  nodal_properties_->create_property("normal_unit_vectors_discontinuity", nrows,
                                     1);
  // nodal_properties_->create_property("friction_coef", nodes_.size(), 1);
  // nodal_properties_->create_property("cohesion", nodes_.size(), 1);
  // nodal_properties_->create_property("cohesion_area", nodes_.size(), 1);
  // nodal_properties_->create_property("contact_distance", nodes_.size(), 1);
  // Iterate over all nodes to initialise the property handle in each node
  // and assign its node id as the prop id in the nodal property data pool
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr)
    (*nitr)->initialise_discontinuity_property_handle((*nitr)->id(),
                                                      nodal_properties_);
}

//! Locate points in a cell
template <unsigned Tdim>
void mpm::Mesh<Tdim>::locate_discontinuity() {
  for (int i = 0; i < discontinuity_.size(); i++)
    discontinuity_[i]->locate_discontinuity_mesh(cells_, map_cells_);
}
//! Updated_position of discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_updated_position_discontinuity(double dt) {
  for (int i = 0; i < discontinuity_.size(); i++)
    discontinuity_[i]->compute_updated_position(dt);
}

//! Compute shape function
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_shapefn_discontinuity() {
  for (int i = 0; i < discontinuity_.size(); i++)
    discontinuity_[i]->compute_shapefn();
}

//! Compute the normal vector of cells
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_cell_normal_vector_discontinuity(
    unsigned dis_id) {
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {

    if ((*citr)->element_type_discontinuity(dis_id) == mpm::EnrichType::Regular)
      continue;
    (*citr)->compute_normal_vector_discontinuity(dis_id);
    (*citr)->compute_plane_discontinuity(false, dis_id);
  }
}

//! Compute the normal vector of enriched nodes at the discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_nodal_normal_vector_discontinuity(
    unsigned dis_id) {
  VectorDim normal_cell;
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    normal_cell.setZero();
    int crossed_cell = 0;
    for (auto cell : (*nitr)->cells()) {
      if (map_cells_[cell]->element_type_discontinuity(dis_id) ==
          mpm::EnrichType::Regular)
        continue;
      normal_cell += map_cells_[cell]->normal_discontinuity(dis_id);
      crossed_cell += 1;
    }
    if (crossed_cell == 0) continue;
    normal_cell = normal_cell / crossed_cell;
    normal_cell.normalize();
    (*nitr)->assign_normal(normal_cell, dis_id);
  }
}

//! Initialise level set values at particles
template <unsigned Tdim>
void mpm::Mesh<Tdim>::initialise_levelset_discontinuity() {
  for (int i = 0; i < discontinuity_.size(); i++) {
    auto discontinuity = discontinuity_[i];
    if (discontinuity->description_type() != "mark_points") continue;
    double phi_particle;

    for (mpm::Index j = 0; j < nparticles(); ++j) {
      discontinuity->compute_levelset(particles_[j]->coordinates(),
                                      phi_particle);
      particles_[j]->assign_levelsetphi(phi_particle);
    }
  }
}

//! The evolution of the discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::update_discontinuity(unsigned dis_id) {
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) ==
        mpm::EnrichType::PotentialTip)
      (*citr)->assign_type_discontinuity(mpm::EnrichType::NeighbourTip_1,
                                         dis_id);
  }

  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) != mpm::EnrichType::NextTip)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed)
          continue;
        virtual_enrich = true;
        break;
      }
      if (virtual_enrich) continue;

      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
            mpm::EnrichType::NextTip)
          map_cells_[cell]->assign_type_discontinuity(
              mpm::EnrichType::NeighbourNextTip_1, dis_id);
      }

      VectorDim normal_cell;
      normal_cell.setZero();
      int crossed_cell = 0;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
            mpm::EnrichType::NextTip)
          continue;
        normal_cell += map_cells_[cell]->normal_discontinuity(dis_id);
        crossed_cell += 1;
      }

      normal_cell = normal_cell / crossed_cell;
      normal_cell.normalize();
    }
  }

  // modify normal vector of NextTip cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) != mpm::EnrichType::NextTip)
      continue;
    VectorDim normal_cell;
    normal_cell.setZero();

    for (auto node : (*citr)->nodes()) {
      // normal_cell += node->discontinuity_property(
      //     "normal_unit_vectors_discontinuity", Tdim);
      normal_cell += node->normal(dis_id);
    }
    normal_cell = normal_cell / (*citr)->nodes().size();
    normal_cell.normalize();
    (*citr)->assign_normal_discontinuity(normal_cell, dis_id);

    int enriched_node = 0;
    double dis = 0;
    // determine the discontinuity plane by the virtual enriched nodes

    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed)
          continue;
        virtual_enrich = true;
        break;
      }
      if (!virtual_enrich) continue;
      enriched_node++;
      auto node_coordinate = node->coordinates();
      for (unsigned int j = 0; j < Tdim; j++)
        dis -= node_coordinate[j] * normal_cell[j];
      dis = node->levelset_phi(dis_id) + dis;
    }

    // update the level set values of the unenriched nodes
    dis = dis / enriched_node;
    (*citr)->assign_d_discontinuity(dis, dis_id);
  }

  // compute nodal level set values
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) != mpm::EnrichType::NextTip)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed)
          continue;
        virtual_enrich = true;
        break;
      }
      if (virtual_enrich) continue;

      VectorDim normal_cell;
      normal_cell.setZero();
      int nexttip_cell = 0;
      Eigen::Matrix<double, 1, 1> phi;
      phi.setZero();
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
            mpm::EnrichType::NextTip)
          continue;
        double d = map_cells_[cell]->d_discontinuity(dis_id);
        normal_cell = map_cells_[cell]->normal_discontinuity(dis_id);
        for (unsigned int i = 0; i < Tdim; i++)
          phi(0, 0) += node->coordinates()[i] * normal_cell[i];
        phi(0, 0) += d;
        nexttip_cell += 1;
      }

      if (nexttip_cell == 0) continue;

      node->assign_levelset_phi(phi(0, 0) / nexttip_cell, dis_id);
    }
  }

  // modify normal vector of NeighbourNextTip_1 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) !=
        mpm::EnrichType::NeighbourNextTip_1)
      continue;
    VectorDim normal_cell;
    normal_cell.setZero();
    int enriched_node = 0;
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NextTip)
          continue;
        virtual_enrich = true;
        break;
      }
      if (!virtual_enrich) continue;

      normal_cell += node->normal(dis_id);
      enriched_node += 1;
    }
    normal_cell = normal_cell / enriched_node;
    normal_cell.normalize();
    (*citr)->assign_normal_discontinuity(normal_cell, dis_id);

    enriched_node = 0;
    double dis = 0;
    // determine the discontinuity plane by the virtual enriched nodes

    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NextTip)
          continue;
        virtual_enrich = true;
        break;
      }
      if (!virtual_enrich) continue;
      enriched_node++;
      auto node_coordinate = node->coordinates();
      for (unsigned int j = 0; j < Tdim; j++)
        dis -= node_coordinate[j] * normal_cell[j];
      dis = node->levelset_phi(dis_id) + dis;
    }

    // update the level set values of the unenriched nodes
    dis = dis / enriched_node;
    (*citr)->assign_d_discontinuity(dis, dis_id);
  }

  // update nodal level set values of the NeighbourNextTip_1 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) !=
        mpm::EnrichType::NeighbourNextTip_1)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NextTip)
          continue;
        virtual_enrich = true;
        break;
      }
      if (virtual_enrich) continue;

      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourNextTip_1)
          map_cells_[cell]->assign_type_discontinuity(
              mpm::EnrichType::NeighbourNextTip_2, dis_id);
      }

      VectorDim normal_cell;
      normal_cell.setZero();
      VectorDim normal_cell_sum;
      normal_cell_sum.setZero();
      int cell_num = 0;
      Eigen::Matrix<double, 1, 1> phi;
      phi.setZero();
      for (auto cell : node->cells()) {

        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourNextTip_1 &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourTip_1)
          continue;
        double d = map_cells_[cell]->d_discontinuity(dis_id);
        normal_cell = map_cells_[cell]->normal_discontinuity(dis_id);
        normal_cell_sum += normal_cell;
        for (unsigned int i = 0; i < Tdim; i++)
          phi(0, 0) += node->coordinates()[i] * normal_cell[i];
        phi(0, 0) += d;
        cell_num++;
      }

      if (cell_num == 0) continue;
      normal_cell_sum = normal_cell_sum / cell_num;
      normal_cell_sum.normalize();
      // node->assign_discontinuity_property(
      //     true, "normal_unit_vectors_discontinuity", normal_cell_sum, 0,
      //     Tdim);

      node->assign_levelset_phi(phi(0, 0) / cell_num, dis_id);
    }
  }
  // modify normal vector of NeighbourNextTip_2 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) !=
        mpm::EnrichType::NeighbourNextTip_2)
      continue;

    VectorDim normal_cell;
    normal_cell.setZero();
    int enriched_node = 0;
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NextTip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourNextTip_1)
          continue;
        virtual_enrich = true;
        break;
      }
      if (!virtual_enrich) continue;

      normal_cell += node->normal(dis_id);
      enriched_node += 1;
    }
    normal_cell = normal_cell / enriched_node;
    normal_cell.normalize();
    (*citr)->assign_normal_discontinuity(normal_cell, dis_id);

    enriched_node = 0;
    double dis = 0;
    // determine the discontinuity plane by the virtual enriched nodes

    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NextTip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourNextTip_1)
          continue;
        virtual_enrich = true;
        break;
      }
      if (!virtual_enrich) continue;
      enriched_node++;
      auto node_coordinate = node->coordinates();
      for (unsigned int j = 0; j < Tdim; j++)
        dis -= node_coordinate[j] * normal_cell[j];
      dis = node->levelset_phi(dis_id) + dis;
    }

    // update the level set values of the unenriched nodes

    dis = dis / enriched_node;
    (*citr)->assign_d_discontinuity(dis, dis_id);
  }

  // update nodal level set values of the NeighbourNextTip_2 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) !=
        mpm::EnrichType::NeighbourNextTip_2)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NextTip &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity(dis_id) !=
                mpm::EnrichType::NeighbourNextTip_1)
          continue;
        virtual_enrich = true;
        break;
      }

      if (virtual_enrich) continue;

      VectorDim normal_cell;
      normal_cell.setZero();
      VectorDim normal_cell_sum;
      normal_cell_sum.setZero();
      int cell_num = 0;
      double phi = 0;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity(dis_id) !=
            mpm::EnrichType::NeighbourNextTip_2)
          continue;
        double d = map_cells_[cell]->d_discontinuity(dis_id);
        normal_cell = map_cells_[cell]->normal_discontinuity(dis_id);
        normal_cell_sum += normal_cell;
        for (unsigned int i = 0; i < Tdim; i++)
          phi += node->coordinates()[i] * normal_cell[i];
        phi += d;
        cell_num++;
      }

      if (cell_num == 0) continue;

      normal_cell_sum = normal_cell_sum / cell_num;
      normal_cell_sum.normalize();
      // node->assign_discontinuity_property(
      //     true, "normal_unit_vectors_discontinuity", , 0,
      //     Tdim);
      node->assign_normal(normal_cell_sum, dis_id);
      node->assign_levelset_phi(phi / cell_num, dis_id);
    }
  }
  // update particle level set values
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) ==
            mpm::EnrichType::NextTip ||
        (*citr)->element_type_discontinuity(dis_id) ==
            mpm::EnrichType::NeighbourNextTip_1 ||
        (*citr)->element_type_discontinuity(dis_id) ==
            mpm::EnrichType::NeighbourNextTip_2) {
      for (auto particle_id : (*citr)->particles()) {
        map_particles_[particle_id]->map_levelset_to_particle(dis_id);
      }
    }
  }

  // update discontinuity points
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) != mpm::EnrichType::NextTip)
      continue;
    std::vector<VectorDim> coordinates;
    (*citr)->compute_discontinuity_point(coordinates, dis_id);
    for (int i = 0; i < coordinates.size(); i++) {
      discontinuity_[dis_id]->insert_particles(coordinates[i], cells_,
                                               map_cells_);
    }
  }
}

//! Find next tip element
template <unsigned Tdim>
void mpm::Mesh<Tdim>::next_tip_element_discontinuity(unsigned dis_id) {
  std::string shear;
#pragma omp parallel for schedule(runtime)
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) !=
        mpm::EnrichType::PotentialTip)
      continue;
    mpm::Index pid;
    double max_pdstrain = 0;
    for (auto particle_id : (*citr)->particles()) {
      double pdstrain = map_particles_[particle_id]->state_variable("pdstrain");
      if (pdstrain > max_pdstrain) {
        max_pdstrain = pdstrain;
        pid = particle_id;
      }
    }

    if (max_pdstrain <= discontinuity_[dis_id]->maximum_pdstrain()) continue;
    VectorDim normal;
    bool propagation =
        map_particles_[pid]->minimum_acoustic_tensor(normal, false, dis_id);
    if (propagation) {
      (*citr)->assign_type_discontinuity(mpm::EnrichType::NextTip, dis_id);
      (*citr)->assign_normal_discontinuity(normal, dis_id);
    }
  }
  return;
}

//! Remove spurious potential tip element
template <unsigned Tdim>
void mpm::Mesh<Tdim>::spurious_potential_tip_element(unsigned dis_id) {

  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(0) != mpm::EnrichType::PotentialTip)
      continue;

    bool boundary = false;
    bool potential_tip = false;
    for (auto neighbour : (*citr)->neighbours()) {
      if (cells_[neighbour]->element_type_discontinuity(0) !=
          mpm::EnrichType::NeighbourTip_2)
        continue;

      if (cells_[neighbour]->nparticles() == 0) {
        boundary = true;
      }
      if (cells_[neighbour]->product_levelset(dis_id) < 0) potential_tip = true;
    }

    if (potential_tip) continue;
    (*citr)->assign_type_discontinuity(mpm::EnrichType::Crossed, dis_id);

    continue;
    // to do
    if (!boundary) continue;

    // avoid the node located near the discontinuity
    if ((*citr)->discontinuity_area(dis_id) == 0) continue;

    std::vector<VectorDim> coordinates;
    (*citr)->compute_discontinuity_point(coordinates, dis_id);

    for (int i = 0; i < coordinates.size(); i++)
      discontinuity_[dis_id]->insert_particles(coordinates[i], cells_,
                                               map_cells_);
  }
}

//! Assign node type as enrich
template <unsigned Tdim>
void mpm::Mesh<Tdim>::assign_node_enrich(bool friction_coef_average,
                                         unsigned dis_id) {

  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {

    if ((*citr)->element_type_discontinuity(dis_id) != mpm::EnrichType::Crossed)
      continue;
    double friction_coef;
    friction_coef = discontinuity_[dis_id]->friction_coef();

    double cohesion;
    cohesion = discontinuity_[dis_id]->cohesion();
    for (auto node : (*citr)->nodes()) {
      if (node->discontinuity_enrich(dis_id)) continue;
      node->assign_discontinuity_enrich(true, dis_id);

      // if (!friction_coef_average)
      //   node->assign_discontinuity_property(true, "friction_coef",
      //                                       friction_coef, 0, 1);
      // node->assign_discontinuity_property(true, "cohesion", cohesion, 0,
      // 1);
    }

    // (*citr)->assign_cohesion_area();
  }
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity(dis_id) !=
        mpm::EnrichType::PotentialTip)
      continue;
    for (auto node : (*citr)->nodes()) {
      if (!node->discontinuity_enrich(dis_id)) continue;
      node->assign_discontinuity_enrich(false, dis_id);
    }
  }
}

//! Find all the nodes need to enriched
template <unsigned Tdim>
void mpm::Mesh<Tdim>::update_node_enrich() {

  double tolerance = 1e-16;
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    double positive_mass =
        (*nitr)->mass(mpm::ParticlePhase::Solid) +
        (*nitr)->discontinuity_property("mass_enrich", 1)(0, 0);
    double negative_mass =
        (*nitr)->mass(mpm::ParticlePhase::Solid) -
        (*nitr)->discontinuity_property("mass_enrich", 1)(0, 0);
    if (positive_mass < tolerance || negative_mass < tolerance)
      (*nitr)->assign_discontinuity_enrich(false);
  }
}

//! The initiation of discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::initiation_discontinuity(
    double maximum_pdstrain, double shield_width, int maximum_num,
    std::tuple<double, double, double, double, double, int, bool>&
        initiation_property) {

  while (discontinuity_.size() < maximum_num) {
    mpm::Index pid;
    double particle_max_pdstrain = 0;
    for (int i = 0; i < nparticles(); ++i) {
      // search the region outside of the other discontinuity
      // to do
      bool near_dis = false;
      for (int j = 0; j < discontinuity_num(); ++j) {
        if (map_particles_[i]->levelset_phi(j) != 0 &&
            std::abs(map_particles_[i]->levelset_phi(j)) < shield_width)
          near_dis = true;
      }
      if (near_dis) continue;
      double pdstrain = map_particles_[i]->state_variable("pdstrain");

      if (pdstrain > particle_max_pdstrain) {
        particle_max_pdstrain = pdstrain;
        pid = i;
      }
    }

    if (particle_max_pdstrain <= maximum_pdstrain) return;
    VectorDim normal;
    bool initiation =
        map_particles_[pid]->minimum_acoustic_tensor(normal, true);

    if (initiation) {

      // Create a new discontinuity surface from JSON object
      const Json json_generator;
      std::string type = "3d_initiation";
      int dis_id = discontinuity_num();

      auto discontinuity = Factory<mpm::DiscontinuityBase<Tdim>, unsigned,
                                   std::tuple<double, double, double, double,
                                              double, int, bool>&>::instance()
                               ->create(type, dis_id, initiation_property);

      insert_discontinuity(discontinuity);
      auto cell_id = map_particles_[pid]->cell_id();
      map_cells_[cell_id]->assign_type_discontinuity(
          mpm::EnrichType::InitialTip, dis_id);
      map_cells_[cell_id]->assign_normal_discontinuity(normal, dis_id);
      auto center = map_cells_[cell_id]->centroid();

      double d = 0;

      for (unsigned int i = 0; i < Tdim; i++) d -= center[i] * normal[i];

      map_cells_[cell_id]->assign_normal_discontinuity(normal, d, dis_id);

      map_cells_[cell_id]->compute_nodal_levelset_equation(dis_id);

      std::vector<VectorDim> coordinates_dis;
      map_cells_[cell_id]->compute_discontinuity_point(coordinates_dis, dis_id);

      for (int i = 0; i < coordinates_dis.size(); i++)
        discontinuity_[dis_id]->insert_particles(coordinates_dis[i], cells_,
                                                 map_cells_);
      // initialise neighbour cells

      auto neighbours = map_cells_[cell_id]->neighbours();
      for (auto neighbour : neighbours) {
        if (map_cells_[neighbour]->nparticles() == 0) continue;
        map_cells_[neighbour]->assign_type_discontinuity(
            mpm::EnrichType::NeighbourTip_1, dis_id);
        map_cells_[neighbour]->assign_normal_discontinuity(normal, d, dis_id);
        map_cells_[neighbour]->compute_nodal_levelset_equation(dis_id);
        if (map_cells_[neighbour]->product_levelset(dis_id) >= 0) continue;
        map_cells_[neighbour]->assign_type_discontinuity(
            mpm::EnrichType::InitialTip, dis_id);

        std::vector<VectorDim> coordinates_dis_neigh;
        map_cells_[neighbour]->compute_discontinuity_point(
            coordinates_dis_neigh, dis_id);

        for (int i = 0; i < coordinates_dis_neigh.size(); i++) {
          discontinuity_[dis_id]->insert_particles(coordinates_dis_neigh[i],
                                                   cells_, map_cells_);
        }
      }
      // initialise level set values

      for (int i = 0; i < nparticles(); ++i) {
        bool neighbour = true;
        for (int j = 0; j < Tdim; j++) {

          if (std::abs(center[j] - particles_[i]->coordinates()[j]) >
              shield_width)
            neighbour = false;
        }
        if (!neighbour) continue;
        double phi = particles_[i]->coordinates().dot(normal) + d;
        particles_[i]->assign_levelsetphi(phi, dis_id);
      }
    }
  }
}

//! Adjust the nodal levelset_phi by mls
template <unsigned Tdim>
void mpm::Mesh<Tdim>::modify_nodal_levelset_mls(unsigned dis_id) {
  Eigen::Matrix<double, 4, 4> au;
  Eigen::Matrix<double, 4, 1> bu;
  // double error_max = 0;
  const double tolerance = std::numeric_limits<double>::epsilon();

  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    if ((*nitr)->discontinuity_property("levelset_phi", 1)(0, 0) == 0) continue;
    double phi = 0;

    au.setZero();
    bu.setZero();

    double particle_volume = 0;
    double cell_volume = 0;
    std::vector<Index> cell_list;
    for (auto cell : (*nitr)->cells()) cell_list.push_back(cell);

    for (auto cell : cell_list) {
      double length = discontinuity_[dis_id]->width();
      cell_volume += map_cells_[cell]->volume();
      for (auto particle : map_cells_[cell]->particles()) {
        auto corp = map_particles_[particle]->coordinates();
        phi = map_particles_[particle]->levelset_phi();
        if (phi == 0) continue;
        particle_volume += map_particles_[particle]->volume();
        // compute weight
        double w[3];
        for (int i = 0; i < 3; i++) {
          w[i] = 1 - std::abs(corp[i] - (*nitr)->coordinates()[i]) / length;
          if (w[i] < 0) w[i] = 0;
        }

        double weight = w[0] * w[1] * w[2];
        au(0, 0) += weight;
        au(0, 1) += weight * corp[0];
        au(0, 2) += weight * corp[1];
        au(0, 3) += weight * corp[2];
        au(1, 0) += weight * corp[0];
        au(1, 1) += weight * corp[0] * corp[0];
        au(1, 2) += weight * corp[0] * corp[1];
        au(1, 3) += weight * corp[0] * corp[2];
        au(2, 0) += weight * corp[1];
        au(2, 1) += weight * corp[1] * corp[0];
        au(2, 2) += weight * corp[1] * corp[1];
        au(2, 3) += weight * corp[1] * corp[2];
        au(3, 0) += weight * corp[2];
        au(3, 1) += weight * corp[2] * corp[0];
        au(3, 2) += weight * corp[2] * corp[1];
        au(3, 3) += weight * corp[2] * corp[2];

        bu(0, 0) += weight * phi;
        bu(1, 0) += weight * phi * corp[0];
        bu(2, 0) += weight * phi * corp[1];
        bu(3, 0) += weight * phi * corp[2];
      }
    }

    // find particles from neighbour cells
    if (particle_volume < 0.5 * cell_volume ||
        std::abs(au.determinant()) < tolerance) {
      au.setZero();
      bu.setZero();
      for (auto cells : (*nitr)->cells()) {
        for (auto cell : map_cells_[cells]->neighbours()) {
          std::vector<Index>::iterator ret;
          ret = std::find(cell_list.begin(), cell_list.end(), cell);
          if (ret != cell_list.end()) continue;
          cell_list.push_back(cell);
        }
      }

      for (auto cell : cell_list) {
        for (auto particle : map_cells_[cell]->particles()) {
          auto corp = map_particles_[particle]->coordinates();
          phi = map_particles_[particle]->levelset_phi();
          if (phi == 0) continue;
          // compute weight
          double length = 2 * discontinuity_[dis_id]->width();
          double w[3];
          for (int i = 0; i < 3; i++) {
            w[i] = 1 - std::abs(corp[i] - (*nitr)->coordinates()[i]) / length;
            if (w[i] < 0) w[i] = 0;
          }
          double weight = w[0] * w[1] * w[2];

          au(0, 0) += weight;
          au(0, 1) += weight * corp[0];
          au(0, 2) += weight * corp[1];
          au(0, 3) += weight * corp[2];
          au(1, 0) += weight * corp[0];
          au(1, 1) += weight * corp[0] * corp[0];
          au(1, 2) += weight * corp[0] * corp[1];
          au(1, 3) += weight * corp[0] * corp[2];
          au(2, 0) += weight * corp[1];
          au(2, 1) += weight * corp[1] * corp[0];
          au(2, 2) += weight * corp[1] * corp[1];
          au(2, 3) += weight * corp[1] * corp[2];
          au(3, 0) += weight * corp[2];
          au(3, 1) += weight * corp[2] * corp[0];
          au(3, 2) += weight * corp[2] * corp[1];
          au(3, 3) += weight * corp[2] * corp[2];

          bu(0, 0) += weight * phi;
          bu(1, 0) += weight * phi * corp[0];
          bu(2, 0) += weight * phi * corp[1];
          bu(3, 0) += weight * phi * corp[2];
        }
      }
    }

    if (std::abs(au.determinant()) < tolerance) continue;

    Eigen::Vector4d coef;
    coef.setZero();
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) coef[i] += au.inverse()(i, j) * bu(j, 0);

    // compute the error
    double error = 0;
    int error_p = 0;
    for (auto cell : cell_list) {
      for (auto particle : map_cells_[cell]->particles()) {
        auto corp = map_particles_[particle]->coordinates();
        phi = map_particles_[particle]->levelset_phi();
        if (phi == 0) continue;
        double phi_mls = 1 * coef[0] + corp[0] * coef[1] + corp[1] * coef[2] +
                         corp[2] * coef[3];
        error += std::pow(phi_mls - phi, 2);
        error_p += 1;
      }
    }
    error = std::sqrt(error / error_p) / discontinuity_[dis_id]->width();

    if (error > 1e-3) continue;

    Eigen::Matrix<double, 1, 4> cor;
    Eigen::Matrix<double, 1, 1> phi_mls;

    cor << 1, (*nitr)->coordinates()[0], (*nitr)->coordinates()[1],
        (*nitr)->coordinates()[2];
    phi_mls(0, 0) = cor.dot(coef);

    (*nitr)->assign_discontinuity_property(true, "levelset_phi", phi_mls, 0, 1);
  }
}

//! Compute the distance between two sides of discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::selfcontact_detection() {

  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); nitr++) {
    if ((*nitr)->enrich_type() == mpm::NodeEnrichType::regular) continue;
    if ((*nitr)->enrich_type() == mpm::NodeEnrichType::single_enriched) {
      auto cor = (*nitr)->coordinates();
      auto dis_id = (*nitr)->discontinuity_id();
      auto normal = (*nitr)->normal(dis_id[0]);
      double contact_distance = discontinuity_[dis_id[0]]->contact_distance();
      double dis_negative = -10 * contact_distance;
      double dis_positive = 10 * contact_distance;
      for (auto cell : (*nitr)->cells()) {

        for (auto particle : map_cells_[cell]->particles()) {
          auto corp = map_particles_[particle]->coordinates();
          double phi = map_particles_[particle]->levelset_phi();

          double dis = 0;
          dis = (corp - cor).dot(normal);

          if (phi > 0) dis_positive = dis < dis_positive ? dis : dis_positive;
          if (phi < 0) dis_negative = dis > dis_negative ? dis : dis_negative;
        }
      }
      bool status = true;
      if (dis_positive - dis_negative > contact_distance) status = false;
      (*nitr)->assign_contact(0, status);

    } else if ((*nitr)->enrich_type() == mpm::NodeEnrichType::double_enriched) {

      Eigen::Matrix<int, 4, 2> flag;

      flag << -1, -1, 1, -1, -1, 1, 1, 1;

      int k = -1;
      auto cor = (*nitr)->coordinates();
      auto dis_id = (*nitr)->discontinuity_id();
      for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 4; j++) {
          auto dis_id = (*nitr)->discontinuity_id();
          // loop for 2 normal directions
          bool status = true;
          k++;
          for (int n = 0; n < 2; n++) {
            if (flag(i, n) * flag(j, n) > 0) continue;

            auto normal = (*nitr)->normal(dis_id[n]);
            double contact_distance =
                discontinuity_[dis_id[n]]->contact_distance();
            double dis_negative = -10 * contact_distance;
            double dis_positive = 10 * contact_distance;
            for (auto cell : (*nitr)->cells()) {

              for (auto particle : map_cells_[cell]->particles()) {
                auto corp = map_particles_[particle]->coordinates();
                double phi = map_particles_[particle]->levelset_phi(dis_id[n]);

                double dis = 0;
                dis = (corp - cor).dot(normal);

                if (phi > 0)
                  dis_positive = dis < dis_positive ? dis : dis_positive;
                if (phi < 0)
                  dis_negative = dis > dis_negative ? dis : dis_negative;
              }
            }

            if (dis_positive - dis_negative > contact_distance) status = false;
          }
          (*nitr)->assign_contact(k, status);
        }
    }
  }
}

//! Assign the level set values to the particles which just enter the crossed
//! cell
// template <unsigned Tdim>
// void mpm::Mesh<Tdim>::check_particle_levelset(bool particle_levelset,
//                                               unsigned dis_id) {

//   for (auto pitr = particles_.cbegin(); pitr != particles_.cend(); ++pitr) {
//     if ((*pitr)->levelset_phi(dis_id) != 0) continue;

//     auto cell_id = (*pitr)->cell_id();

//     for (auto node : cells_[cell_id]->nodes()) {
//       // if (!node->discontinuity_enrich()) continue;

//       Eigen::Matrix<double, 4, 4> au;
//       Eigen::Matrix<double, 4, 1> bu;
//       au.setZero();
//       bu.setZero();
//       auto cell_list = cells_[cell_id]->neighbours();
//       cell_list.insert(cell_id);

//       for (auto cell : cell_list) {
//         for (auto particle : cells_[cell]->particles()) {
//           auto corp = map_particles_[particle]->coordinates();
//           double phi = map_particles_[particle]->levelset_phi();
//           if (phi == 0) continue;
//           // compute weight
//           double length = 2.0 * discontinuity_[dis_id]->width();
//           double w[3];
//           for (int i = 0; i < 3; i++) {
//             w[i] = 1 - std::abs(corp[i] - (*pitr)->coordinates()[i]) /
//             length; if (w[i] < 0) w[i] = 0;
//           }
//           double weight = w[0] * w[1] * w[2];

//           au(0, 0) += weight;
//           au(0, 1) += weight * corp[0];
//           au(0, 2) += weight * corp[1];
//           au(0, 3) += weight * corp[2];
//           au(1, 0) += weight * corp[0];
//           au(1, 1) += weight * corp[0] * corp[0];
//           au(1, 2) += weight * corp[0] * corp[1];
//           au(1, 3) += weight * corp[0] * corp[2];
//           au(2, 0) += weight * corp[1];
//           au(2, 1) += weight * corp[1] * corp[0];
//           au(2, 2) += weight * corp[1] * corp[1];
//           au(2, 3) += weight * corp[1] * corp[2];
//           au(3, 0) += weight * corp[2];
//           au(3, 1) += weight * corp[2] * corp[0];
//           au(3, 2) += weight * corp[2] * corp[1];
//           au(3, 3) += weight * corp[2] * corp[2];

//           bu(0, 0) += weight * phi;
//           bu(1, 0) += weight * phi * corp[0];
//           bu(2, 0) += weight * phi * corp[1];
//           bu(3, 0) += weight * phi * corp[2];
//         }
//       }

//       const double tolerance = std::numeric_limits<double>::epsilon();

//       if (std::abs(au.determinant()) < tolerance) {
//         (*pitr)->map_levelset_to_particle(dis_id);
//         continue;
//       }

//       Eigen::Vector4d coef;
//       coef.setZero();
//       for (int i = 0; i < 4; i++)
//         for (int j = 0; j < 4; j++) coef[i] += au.inverse()(i, j) * bu(j, 0);

//       Eigen::Vector4d cor;
//       cor << 1, (*pitr)->coordinates()[0], (*pitr)->coordinates()[1],
//           (*pitr)->coordinates()[2];
//       double phi = cor.dot(coef);

//       (*pitr)->assign_levelsetphi(phi, dis_id);

//       break;
//     }
//   }
//   if (particle_levelset) return;
//   for (auto pitr = particles_.cbegin(); pitr != particles_.cend(); ++pitr) {
//     if ((*pitr)->levelset_phi() == 0) continue;
//     auto cell_id = (*pitr)->cell_id();

//     if (cells_[cell_id]->element_type_discontinuity(0) ==
//             mpm::EnrichType::Regular ||
//         cells_[cell_id]->element_type_discontinuity(0) ==
//             mpm::EnrichType::NeighbourTip_3)
//       (*pitr)->assign_levelsetphi(0);
//   }
// }

//! Read HDF5 particles for xmpm particle
template <unsigned Tdim>
bool mpm::Mesh<Tdim>::read_particles_hdf5_xmpm(
    const std::string& filename, const std::string& particle_type) {

  // Create a new file using default properties.
  hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  // Throw an error if file can't be found
  if (file_id < 0) throw std::runtime_error("HDF5 particle file is not found");

  // Calculate the size and the offsets of our struct members in memory
  hsize_t nrecords = 0;
  hsize_t nfields = 0;
  H5TBget_table_info(file_id, "table", &nfields, &nrecords);

  if (nfields != mpm::pod::particlexmpm::NFIELDS)
    throw std::runtime_error("HDF5 table has incorrect number of fields");

  std::vector<PODParticleXMPM> dst_buf;
  dst_buf.reserve(nrecords);
  // Read the table
  H5TBread_table(file_id, "table", mpm::pod::particlexmpm::dst_size,
                 mpm::pod::particlexmpm::dst_offset,
                 mpm::pod::particlexmpm::dst_sizes, dst_buf.data());

  // Iterate over all HDF5 particles
  for (unsigned i = 0; i < nrecords; ++i) {
    PODParticleXMPM pod_particle = dst_buf[i];
    // Get particle's material from list of materials
    // Get particle's material from list of materials
    std::vector<std::shared_ptr<mpm::Material<Tdim>>> materials;
    materials.emplace_back(materials_.at(pod_particle.material_id));
    // Particle id
    mpm::Index pid = pod_particle.id;
    // Initialise coordinates
    Eigen::Matrix<double, Tdim, 1> coords;
    coords.setZero();

    // Create particle
    auto particle =
        Factory<mpm::ParticleBase<Tdim>, mpm::Index,
                const Eigen::Matrix<double, Tdim, 1>&>::instance()
            ->create(particle_type, static_cast<mpm::Index>(pid), coords);

    // Initialise particle with HDF5 data
    particle->initialise_particle(pod_particle, materials);

    // Add particle to mesh and check
    bool insert_status = this->add_particle(particle, false);

    // If insertion is successful
    if (!insert_status)
      throw std::runtime_error("Addition of particle to mesh failed!");
  }
  // close the file
  H5Fclose(file_id);
  return true;
}

//! Write particles to HDF5 for xmpm particle
template <unsigned Tdim>
bool mpm::Mesh<Tdim>::write_particles_hdf5_xmpm(const std::string& filename) {
  const unsigned nparticles = this->nparticles();

  std::vector<PODParticleXMPM> particle_data;
  particle_data.reserve(nparticles);

  for (auto pitr = particles_.cbegin(); pitr != particles_.cend(); ++pitr) {
    auto pod = std::static_pointer_cast<mpm::PODParticleXMPM>((*pitr)->pod());
    particle_data.emplace_back(*pod);
  }

  // Calculate the size and the offsets of our struct members in memory
  const hsize_t NRECORDS = nparticles;
  const hsize_t NFIELDS = mpm::pod::particlexmpm::NFIELDS;

  hid_t file_id;
  hsize_t chunk_size = 10000;
  int* fill_data = NULL;
  int compress = 0;

  // Create a new file using default properties.
  file_id =
      H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // make a table
  H5TBmake_table(
      "Table Title", file_id, "table", NFIELDS, NRECORDS,
      mpm::pod::particlexmpm::dst_size, mpm::pod::particlexmpm::field_names,
      mpm::pod::particlexmpm::dst_offset, mpm::pod::particlexmpm::field_type,
      chunk_size, fill_data, compress, particle_data.data());

  H5Fclose(file_id);
  return true;
}

//! code for debugging added by yliang start-------------------------------
// FIXME: Remove this before merging to master
template <unsigned Tdim>
void mpm::Mesh<Tdim>::output_celltype(int step) {
  std::ofstream test("cell_type.txt", std::ios::app);
  int dis_id = step;

  test << step << ":" << std::endl;
  for (int i = 0; i < cells_.size(); i++) {
    auto type = cells_[i]->element_type_discontinuity(dis_id);
    if (type == 0)
      test << "0  ";
    else if (type == 1)
      test << "1  ";
    else if (type == 2)
      test << "2  ";
    else if (type == 3)
      test << "3  ";
    else if (type == 4)
      test << "4  ";
    else if (type == 5)
      test << "5  ";
    else if (type == 6)
      test << "6  ";
    else if (type == 7)
      test << "7  ";
    else if (type == 8)
      test << "8  ";
    else
      test << type << " ";
    if (((i + 1) % 26) == 0) test << std::endl;
  }
  test << std::endl;

  return;

  // test << step << ":" << std::endl;
  // for (int i = 0; i < nodes_.size() * 0.5; i++) {
  //   auto type = nodes_[i]->enrich_type();
  //   if (type == 0)
  //     test << "0  ";
  //   else if (type == 1)
  //     test << "1  ";
  //   else if (type == 2)
  //     test << "2  ";
  //   else
  //     test << type << " ";
  //   if (((i + 1) % 27) == 0) test << std::endl;
  // }
  // test << std::endl;
  std::ofstream testm("mass.txt", std::ios::app);
  std::ofstream testme("masse.txt", std::ios::app);
  std::ofstream testp("momentum.txt", std::ios::app);
  std::ofstream testpe("momentume.txt", std::ios::app);
  unsigned phase = 0;
  testm << step << ":" << std::endl;
  testme << step << ":" << std::endl;
  testp << step << ":" << std::endl;
  testpe << step << ":" << std::endl;
  for (int i = 0; i < nodes_.size() * 0.5; i++) {
    // double nodal_mass = nodes_[i]->mass(phase);
    // auto nodal_mass_enrich = nodes_[i]->mass_enrich();

    // auto nodal_momentum = nodes_[i]->momentum(phase);
    // auto nodal_momentum_enrich = nodes_[i]->momentum_enrich();

    auto internal_force = nodes_[i]->internal_force(phase);
    // testm << std::setw(20) << nodal_mass;
    // testme << std::setw(20) << nodal_mass_enrich[0];
    testp << std::setw(20) << internal_force[0];
    // testp << std::setw(20) << nodal_momentum[0];
    // testpe << std::setw(20) << nodal_momentum_enrich(0, 0);

    if (((i + 1) % 27) == 0) testm << std::endl;
    if (((i + 1) % 27) == 0) testme << std::endl;
    if (((i + 1) % 27) == 0) testp << std::endl;
    if (((i + 1) % 27) == 0) testpe << std::endl;
  }
  testm << std::endl;
  testme << std::endl;
  testp << std::endl;
  testpe << std::endl;
  // std::ofstream testnormal("node_normal.txt", std::ios::app);
  // testnormal << step << ":" << std::endl;
  // for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
  //   if (!(*nitr)->discontinuity_enrich()) continue;

  //   if ((*nitr)->coordinates()[0] < 35) continue;

  //   Eigen::Matrix<double, Tdim, 1> normal = (*nitr)->discontinuity_property(
  //       "normal_unit_vectors_discontinuity", Tdim);
  //   testnormal << (*nitr)->coordinates()[0] << "\t" <<
  //   (*nitr)->coordinates()[1]
  //              << "\t" << normal[0] << "\t" << normal[1] << "\t" << normal[2]
  //              << std::endl;
  // }
}

// FIXME: Remove this before merging to master
template <unsigned Tdim>
void mpm::Mesh<Tdim>::define_levelset() {
  // for oso
  std::ifstream in("stage1.txt");
  double stage[63126];
  for (int i = 0; i < 63126; ++i) {
    in >> stage[i];
    if (stage[i] == 0) stage[i] = std::numeric_limits<double>::min();
  }
  int i = 0;
  Eigen::Matrix<double, 1, 1> phi_mls;
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    phi_mls(0, 0) = stage[i];
    i += 1;

    (*nitr)->assign_discontinuity_property(true, "levelset_phi", phi_mls, 0, 1);
  }
  for (auto pitr = particles_.cbegin(); pitr != particles_.cend(); ++pitr) {

    (*pitr)->map_levelset_to_particle();
  }

  return;
  for (auto pitr = particles_.cbegin(); pitr != particles_.cend(); ++pitr) {

    auto cor = (*pitr)->coordinates();
    double phi;
    // phi = 1 / std::sqrt(3) * cor[0] + 1 / std::sqrt(3) * cor[1] +
    // 1 / std::sqrt(3) * cor[2] - 0.5 * std::sqrt(3);

    //   phi = std::sqrt(std::pow(cor[0] - 35, 2) + std::pow(cor[1] - 30, 2) +
    //                   std::pow(cor[2] - 0, 2)) -
    //         25;
    // slide body
    phi = (0.5 * cor[0] + cor[2] - 0.5) / std::sqrt(0.25 + 1);
    phi = cor[1] - 1.5;
    (*pitr)->assign_levelsetphi(phi);
    // case 4-2d
    // if (cor[0] > 35 && cor[1] > 4 && cor[1] < 5) {
    //   phi = 5 - cor[1];
    //   (*pitr)->assign_levelsetphi(phi);
    // }
    // if (cor[0] > 34 && cor[0] < 35 && cor[1] > 4) {
    //   if ((*pitr)->levelset_phi() < 0) continue;
    //   Eigen::Matrix<double, 2, 1> e1, e2, p2n;
    //   e1 << 1 / std::sqrt(5), 2 / std::sqrt(5);
    //   e2 << 2 / std::sqrt(5), -1 / std::sqrt(5);

    //   p2n << cor[0] - 35, cor[1] - 5;
    //   if (e2.dot(p2n) >= 0)
    //     phi = p2n.norm();
    //   else if (e2.dot(p2n) < 0) {
    //     double dis1 = 5.25 - cor[1];  // 5.27118
    //     double dis2 = std::abs(e1.dot(p2n));
    //     if (dis1 < dis2)
    //       dis1 = (*pitr)->levelset_phi();
    //     else
    //       dis1 = dis2;
    //     phi = dis1;
    //   }
    //   (*pitr)->assign_levelsetphi(phi);
    // }
    // case 5
    // if (cor[0] > 35 && cor[1] > 4 && cor[1] < 5) {
    //   phi = 5 - cor[1];
    //   (*pitr)->assign_levelsetphi(phi);
    // }
    // if (cor[0] > 34 && cor[0] < 35 && cor[1] > 4) {
    //   if ((*pitr)->levelset_phi() != 0) continue;
    //   Eigen::Matrix<double, 2, 1> e1, e2, p2n;
    //   e1 << 1 / std::sqrt(5), 2 / std::sqrt(5);
    //   e2 << 2 / std::sqrt(5), -1 / std::sqrt(5);

    //   p2n << cor[0] - 35, cor[1] - 5;
    //   if (e2.dot(p2n) >= 0)
    //     phi = p2n.norm();
    //   else if (e2.dot(p2n) < 0) {
    //     double dis1 = 5.25 - cor[1];  // 5.27118
    //     double dis2 = std::abs(e1.dot(p2n));
    //     phi = dis2;
    //   }
    //   (*pitr)->assign_levelsetphi(phi);
    // }
    // if ((*pitr)->material_id(mpm::ParticlePhase::Solid) == 4)
    //   (*pitr)->assign_levelsetphi(1.0);
    // else if ((*pitr)->material_id(mpm::ParticlePhase::Solid) == 5)
    //   (*pitr)->assign_levelsetphi(-1);
    // else if ((*pitr)->material_id(mpm::ParticlePhase::Solid) == 6)
    //   (*pitr)->assign_levelsetphi(-1);
  }
}
//! code for debugging added by yliang start-------------------------------

//! Assign particle levelset
template <unsigned Tdim>
void mpm::Mesh<Tdim>::assign_particles_levelset(
    const std::vector<std::tuple<mpm::Index, double>>& particles_levelset,
    unsigned dis_id) {

  for (const auto& particle_levelset : particles_levelset) {
    // Particle id
    mpm::Index pid = std::get<0>(particle_levelset);
    // levelset values phi
    double phi = std::get<1>(particle_levelset);

    map_particles_[pid]->assign_levelsetphi(phi, dis_id);
  }
}

//! Assign nodal levelset
template <unsigned Tdim>
void mpm::Mesh<Tdim>::assign_nodes_levelset(
    const std::vector<std::tuple<mpm::Index, double>>& nodes_levelset,
    unsigned dis_id) {

  for (const auto& node_levelset : nodes_levelset) {
    // node id
    mpm::Index nid = std::get<0>(node_levelset);
    // levelset values phi
    double phi = std::get<1>(node_levelset);
    map_nodes_[nid]->assign_levelset_phi(phi, dis_id);
  }
  // initialise the particle level set values
  iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_levelset_to_particle,
                std::placeholders::_1, dis_id));
}

//! Initialise discontinuity
// template <unsigned Tdim>
// void mpm::Mesh<Tdim>::initialise_discontinuity(
//     const std::vector<std::shared_ptr<mpm::DiscontinuityBase<Tdim>>>&
//         discontinuity) {
//   this->discontinuity_ = discontinuity;
//   iterate_over_particles(
//       std::bind(&mpm::ParticleBase<Tdim>::reset_discontinuity_size,
//                 std::placeholders::_1, discontinuity.size()));
// }

//! Insert a new discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::insert_discontinuity(
    const std::shared_ptr<mpm::DiscontinuityBase<Tdim>>& discontinuity) {
  this->discontinuity_.push_back(discontinuity);
  iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::reset_discontinuity_size,
                std::placeholders::_1, discontinuity_.size()));
  iterate_over_cells(std::bind(&mpm::Cell<Tdim>::reset_discontinuity_size,
                               std::placeholders::_1, discontinuity_.size()));
  iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::reset_discontinuity_size,
                               std::placeholders::_1, discontinuity_.size()));
}

template <unsigned Tdim>
void mpm::Mesh<Tdim>::determine_enriched_node_by_mass_h() {

  // Assign mass to nodes
  iterate_over_particles(std::bind(&mpm::ParticleBase<Tdim>::map_mass_to_nodes,
                                   std::placeholders::_1));

  for (int i = 0; i < discontinuity_.size(); i++) {

    // Initialise mass*h at nodes
    iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::initialise_mass_h,
                                 std::placeholders::_1));

    // Assign mass*h to nodes at discontinuity i
    iterate_over_particles(
        std::bind(&mpm::ParticleBase<Tdim>::map_mass_h_to_nodes,
                  std::placeholders::_1, i));
    // determine node type by mass*h at discontinuity i
    iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::determine_node_type,
                                 std::placeholders::_1, i));
  }

  // Initialise mass at nodes
  unsigned phase = 0;
  iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::initialise_mass,
                               std::placeholders::_1, phase));
  // Initialise mass*h at nodes
  iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::initialise_mass_h,
                               std::placeholders::_1));
}

template <unsigned Tdim>
void mpm::Mesh<Tdim>::propagation_discontinuity() {

  if (discontinuity_num() == 0) return;
  // Initialise element properties
  iterate_over_cells(
      std::bind(&mpm::Cell<Tdim>::initialise_element_properties_discontinuity,
                std::placeholders::_1));
  // locate points of discontinuity
  locate_discontinuity();

  // Iterate over each points to compute shapefn
  compute_shapefn_discontinuity();

  iterate_over_particles(std::bind(
      &mpm::ParticleBase<Tdim>::map_volume_to_nodes, std::placeholders::_1));

  for (unsigned dis_id = 0; dis_id < discontinuity_num(); dis_id++) {

    bool propagation = discontinuity_[dis_id]->propagation();
    auto type = discontinuity_[dis_id]->description_type();
    if (propagation) {

      iterate_over_particles(
          std::bind(&mpm::ParticleBase<Tdim>::map_levelset_to_nodes,
                    std::placeholders::_1, dis_id));
      // to do
      // modify_nodal_levelset_mls();

      iterate_over_cells(std::bind(&mpm::Cell<Tdim>::potential_tip_element,
                                   std::placeholders::_1, dis_id));

      // remove the spurious potential tip element
      spurious_potential_tip_element(dis_id);
      // to do
      bool friction_coef_average_ = false;
      assign_node_enrich(friction_coef_average_, dis_id);

      iterate_over_cells(std::bind(&mpm::Cell<Tdim>::tip_element,
                                   std::placeholders::_1, dis_id));
    } else {
      // Determine the node enriched type by mass*h
      if (type == "particle_levelset" || type == "node_levelset") {
        determine_enriched_node_by_mass_h();
      }
      // determine the celltype by the nodal level set
      iterate_over_cells(std::bind(&mpm::Cell<Tdim>::determine_crossed,
                                   std::placeholders::_1, dis_id));
    }
    // obtain the normal direction of each cell
    compute_cell_normal_vector_discontinuity(dis_id);

    // obtain the normal direction of enrich nodes
    compute_nodal_normal_vector_discontinuity(dis_id);

    if (propagation) next_tip_element_discontinuity(dis_id);

    // discontinuity growth
    if (propagation) update_discontinuity(dis_id);

    iterate_over_cells(std::bind(&mpm::Cell<Tdim>::compute_area_discontinuity,
                                 std::placeholders::_1, dis_id));

    // to do: option 1
    iterate_over_particles(std::bind(&mpm::ParticleBase<Tdim>::check_levelset,
                                     std::placeholders::_1, dis_id));
    // to do: option 2
    // check_particle_levelset(false, dis_id);
  }

  selfcontact_detection();
}