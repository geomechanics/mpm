// Create the nodal properties' map for discontinuity
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
  nodal_properties_->create_property("mass_enrich", nodes_.size(), 1);
  nodal_properties_->create_property("levelset_phi", nodes_.size(), 1);
  nodal_properties_->create_property("momenta_enrich", nrows, 1);
  nodal_properties_->create_property("internal_force_enrich", nrows, 1);
  nodal_properties_->create_property("external_force_enrich", nrows, 1);
  nodal_properties_->create_property("normal_unit_vectors_discontinuity", nrows,
                                     1);
  nodal_properties_->create_property("friction_coef", nodes_.size(), 1);
  nodal_properties_->create_property("cohesion", nodes_.size(), 1);
  nodal_properties_->create_property("cohesion_area", nodes_.size(), 1);
  nodal_properties_->create_property("contact_distance", nodes_.size(), 1);
  // Iterate over all nodes to initialise the property handle in each node
  // and assign its node id as the prop id in the nodal property data pool
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr)
    (*nitr)->initialise_discontinuity_property_handle((*nitr)->id(),
                                                      nodal_properties_);
}

// Initialise the nodal properties' map
template <unsigned Tdim>
void mpm::Mesh<Tdim>::initialise_nodal_properties() {
  // Call initialise_properties function from the nodal properties
  nodal_properties_->initialise_nodal_properties();
}

//! Locate points in a cell
template <unsigned Tdim>
void mpm::Mesh<Tdim>::locate_discontinuity() {
  discontinuity_->locate_discontinuity_mesh(cells_, map_cells_);
}
//! updated_position of discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_updated_position_discontinuity(double dt) {
  discontinuity_->compute_updated_position(dt);
}
//! compute shape function
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_shapefn_discontinuity() {
  discontinuity_->compute_shapefn();
}

// compute the normal vector of cells
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_cell_normal_vector_discontinuity() {
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    (*citr)->compute_normal_vector_discontinuity();
    (*citr)->compute_plane_discontinuity(false);
  }
}

// compute the normal vector of enriched nodes at the discontinuity
template <unsigned Tdim>
void mpm::Mesh<Tdim>::compute_nodal_normal_vector_discontinuity() {

  VectorDim normal_cell;
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    if (!(*nitr)->discontinuity_enrich()) continue;
    normal_cell.setZero();
    int crossed_cell = 0;
    for (auto cell : (*nitr)->cells()) {
      normal_cell += map_cells_[cell]->normal_discontinuity();
      crossed_cell += 1;
    }
    if (crossed_cell == 0) continue;
    normal_cell = normal_cell / crossed_cell;

    // normal_cell << 0.5,0,1;
    normal_cell.normalize();
    (*nitr)->assign_discontinuity_property(
        true, "normal_unit_vectors_discontinuity", normal_cell, 0, Tdim);
  }
}

// Initialise level set values particles
template <unsigned Tdim>
void mpm::Mesh<Tdim>::initialise_levelset_discontinuity() {

  double phi_particle;

  for (mpm::Index j = 0; j < nparticles(); ++j) {
    discontinuity_->compute_levelset(particles_[j]->coordinates(),
                                     phi_particle);
    particles_[j]->assign_levelsetphi(phi_particle);
  }
}

// Initialise nodal level set values particles
template <unsigned Tdim>
void mpm::Mesh<Tdim>::initialise_nodal_levelset_discontinuity() {

  Eigen::Matrix<double, 1, 1> phi;
  phi.setZero();
  double phi_node;
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    discontinuity_->compute_levelset((*nitr)->coordinates(), phi_node);
    phi(0, 0) = phi_node;
    (*nitr)->assign_discontinuity_property(true, "levelset_phi", phi, 0, 1);
  }
}

// code for debugging added by yliang
//! solve nodal levelset values
template <unsigned Tdim>
void mpm::Mesh<Tdim>::update_node_levelset() {
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr)
    (*nitr)->update_levelset();
}

// discontinuity growth
template <unsigned Tdim>
void mpm::Mesh<Tdim>::update_discontinuity() {

  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() == mpm::EnrichType::PotentialTip)
      (*citr)->assign_type_discontinuity(mpm::EnrichType::NeighbourTip_1);
  }

  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::NextTip)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Crossed)
          continue;
        virtual_enrich = true;
        break;
      }
      if (virtual_enrich) {
        // node->assign_discontinuity_enrich(true);
        continue;
      }

      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
            mpm::EnrichType::NextTip)
          map_cells_[cell]->assign_type_discontinuity(
              mpm::EnrichType::NeighbourNextTip_1);
      }

      VectorDim normal_cell;
      normal_cell.setZero();
      int crossed_cell = 0;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
            mpm::EnrichType::NextTip)
          continue;
        normal_cell += map_cells_[cell]->normal_discontinuity();
        crossed_cell += 1;
      }

      normal_cell = normal_cell / crossed_cell;
      normal_cell.normalize();
      node->assign_discontinuity_property(
          true, "normal_unit_vectors_discontinuity", normal_cell, 0, Tdim);
    }
  }

  // modify normal vector of NextTip cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::NextTip)
      continue;
    VectorDim normal_cell;
    normal_cell.setZero();

    for (auto node : (*citr)->nodes()) {
      normal_cell += node->discontinuity_property(
          "normal_unit_vectors_discontinuity", Tdim);
    }
    normal_cell = normal_cell / (*citr)->nodes().size();
    normal_cell.normalize();
    (*citr)->assign_normal_discontinuity(normal_cell);

    int enriched_node = 0;
    double dis = 0;
    // determine the discontinuity plane by the virtual enriched nodes

    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
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
      dis = node->discontinuity_property("levelset_phi", 1)(0, 0) + dis;
    }

    // update the level set values of the unenriched nodes
    dis = dis / enriched_node;
    (*citr)->assign_d_discontinuity(dis);
  }

  // compute nodal level set values
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::NextTip)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
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
        if (map_cells_[cell]->element_type_discontinuity() !=
            mpm::EnrichType::NextTip)
          continue;
        double d = map_cells_[cell]->d_discontinuity();
        normal_cell = map_cells_[cell]->normal_discontinuity();
        for (unsigned int i = 0; i < Tdim; i++)
          phi(0, 0) += node->coordinates()[i] * normal_cell[i];
        phi(0, 0) += d;
        nexttip_cell += 1;
      }

      if (nexttip_cell == 0) continue;

      node->assign_discontinuity_property(true, "levelset_phi",
                                          phi / nexttip_cell, 0, 1);
    }
  }

  // modify normal vector of NeighbourNextTip_1 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() !=
        mpm::EnrichType::NeighbourNextTip_1)
      continue;
    VectorDim normal_cell;
    normal_cell.setZero();
    int enriched_node = 0;
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NextTip)
          continue;
        virtual_enrich = true;
        break;
      }
      if (!virtual_enrich) continue;

      normal_cell += node->discontinuity_property(
          "normal_unit_vectors_discontinuity", Tdim);
      enriched_node += 1;
    }
    normal_cell = normal_cell / enriched_node;
    normal_cell.normalize();
    (*citr)->assign_normal_discontinuity(normal_cell);

    enriched_node = 0;
    double dis = 0;
    // determine the discontinuity plane by the virtual enriched nodes

    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity() !=
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
      dis = node->discontinuity_property("levelset_phi", 1)(0, 0) + dis;
    }

    // update the level set values of the unenriched nodes
    dis = dis / enriched_node;
    (*citr)->assign_d_discontinuity(dis);
  }

  // update nodal level set values of the NeighbourNextTip_1 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() !=
        mpm::EnrichType::NeighbourNextTip_1)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NextTip)
          continue;
        virtual_enrich = true;
        break;
      }
      if (virtual_enrich) continue;

      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourNextTip_1)
          map_cells_[cell]->assign_type_discontinuity(
              mpm::EnrichType::NeighbourNextTip_2);
      }

      VectorDim normal_cell;
      normal_cell.setZero();
      VectorDim normal_cell_sum;
      normal_cell_sum.setZero();
      int cell_num = 0;
      Eigen::Matrix<double, 1, 1> phi;
      phi.setZero();
      for (auto cell : node->cells()) {

        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourNextTip_1 &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourTip_1)
          continue;
        double d = map_cells_[cell]->d_discontinuity();
        normal_cell = map_cells_[cell]->normal_discontinuity();
        normal_cell_sum += normal_cell;
        for (unsigned int i = 0; i < Tdim; i++)
          phi(0, 0) += node->coordinates()[i] * normal_cell[i];
        phi(0, 0) += d;
        cell_num++;
      }

      if (cell_num == 0) continue;
      normal_cell_sum = normal_cell_sum / cell_num;
      normal_cell_sum.normalize();
      node->assign_discontinuity_property(
          true, "normal_unit_vectors_discontinuity", normal_cell_sum, 0, Tdim);

      node->assign_discontinuity_property(true, "levelset_phi", phi / cell_num,
                                          0, 1);
    }
  }
  // modify normal vector of NeighbourNextTip_2 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() !=
        mpm::EnrichType::NeighbourNextTip_2)
      continue;

    VectorDim normal_cell;
    normal_cell.setZero();
    int enriched_node = 0;
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NextTip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourNextTip_1)
          continue;
        virtual_enrich = true;
        break;
      }
      if (!virtual_enrich) continue;

      normal_cell += node->discontinuity_property(
          "normal_unit_vectors_discontinuity", Tdim);
      enriched_node += 1;
    }
    normal_cell = normal_cell / enriched_node;
    normal_cell.normalize();
    (*citr)->assign_normal_discontinuity(normal_cell);

    enriched_node = 0;
    double dis = 0;
    // determine the discontinuity plane by the virtual enriched nodes

    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NextTip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity() !=
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
      dis = node->discontinuity_property("levelset_phi", 1)(0, 0) + dis;
    }

    // update the level set values of the unenriched nodes

    dis = dis / enriched_node;
    (*citr)->assign_d_discontinuity(dis);
  }

  // update nodal level set values of the NeighbourNextTip_2 cell
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() !=
        mpm::EnrichType::NeighbourNextTip_2)
      continue;
    // compute nodal normal direction and find neighbour cells
    for (auto node : (*citr)->nodes()) {

      bool virtual_enrich = false;
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Tip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::Crossed &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NextTip &&
            map_cells_[cell]->element_type_discontinuity() !=
                mpm::EnrichType::NeighbourTip_1 &&
            map_cells_[cell]->element_type_discontinuity() !=
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
      Eigen::Matrix<double, 1, 1> phi;
      phi.setZero();
      for (auto cell : node->cells()) {
        if (map_cells_[cell]->element_type_discontinuity() !=
            mpm::EnrichType::NeighbourNextTip_2)
          continue;
        double d = map_cells_[cell]->d_discontinuity();
        normal_cell = map_cells_[cell]->normal_discontinuity();
        normal_cell_sum += normal_cell;
        for (unsigned int i = 0; i < Tdim; i++)
          phi(0, 0) += node->coordinates()[i] * normal_cell[i];
        phi(0, 0) += d;
        cell_num++;
      }

      if (cell_num == 0) continue;

      normal_cell_sum = normal_cell_sum / cell_num;
      normal_cell_sum.normalize();
      node->assign_discontinuity_property(
          true, "normal_unit_vectors_discontinuity", normal_cell_sum, 0, Tdim);
      node->assign_discontinuity_property(true, "levelset_phi", phi / cell_num,
                                          0, 1);
    }
  }
  // update particle level set values
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() == mpm::EnrichType::NextTip ||
        (*citr)->element_type_discontinuity() ==
            mpm::EnrichType::NeighbourNextTip_1 ||
        (*citr)->element_type_discontinuity() ==
            mpm::EnrichType::NeighbourNextTip_2) {
      for (auto particle_id : (*citr)->particles()) {
        map_particles_[particle_id]->map_levelset_to_particle();
      }
    }
  }

  // update discontinuity points
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::NextTip)
      continue;
    std::vector<VectorDim> coordinates;
    (*citr)->compute_discontinuity_point(coordinates);
    for (int i = 0; i < coordinates.size(); i++) {
      discontinuity_->insert_particles(coordinates[i], cells_, map_cells_);

      double d = (*citr)->d_discontinuity();
      auto normal_cell = (*citr)->normal_discontinuity();
    }
  }
}

//! find next tip element
template <unsigned Tdim>
void mpm::Mesh<Tdim>::next_tip_element_discontinuity() {
  std::string shear;
#pragma omp parallel for schedule(runtime)
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::PotentialTip)
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

    if (max_pdstrain <= discontinuity_->maximum_pdstrain()) continue;
    VectorDim normal;
    bool propagation =
        map_particles_[pid]->minimum_acoustic_tensor(normal, false);
    if (propagation) {
      (*citr)->assign_type_discontinuity(mpm::EnrichType::NextTip);
      (*citr)->assign_normal_discontinuity(normal);
    }
  }
  return;
}

//! remove spurious potential tip element
template <unsigned Tdim>
void mpm::Mesh<Tdim>::spurious_potential_tip_element() {

  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::PotentialTip)
      continue;

    bool boundary = false;
    bool potential_tip = false;
    for (auto neighbour : (*citr)->neighbours()) {
      if (cells_[neighbour]->element_type_discontinuity() !=
          mpm::EnrichType::NeighbourTip_2)
        continue;

      if (cells_[neighbour]->nparticles() == 0) {
        boundary = true;
      }
      if (cells_[neighbour]->product_levelset() < 0) potential_tip = true;
    }

    if (potential_tip) continue;
    (*citr)->assign_type_discontinuity(mpm::EnrichType::Crossed);

    continue;
    if (!boundary) continue;

    // avoid the node located near the discontinuity
    if ((*citr)->discontinuity_area() == 0) continue;

    std::vector<VectorDim> coordinates;
    (*citr)->compute_discontinuity_point(coordinates);

    for (int i = 0; i < coordinates.size(); i++)
      discontinuity_->insert_particles(coordinates[i], cells_, map_cells_);
  }
}

// assign_node_enrich
template <unsigned Tdim>
void mpm::Mesh<Tdim>::assign_node_enrich(bool friction_coef_average,
                                         bool enrich_all) {
  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::Crossed)
      continue;
    Eigen::Matrix<double, 1, 1> friction_coef;
    friction_coef(0, 0) = discontinuity_->friction_coef();

    Eigen::Matrix<double, 1, 1> cohesion;
    cohesion(0, 0) = discontinuity_->cohesion();
    for (auto node : (*citr)->nodes()) {
      if (node->discontinuity_enrich()) continue;
      node->assign_discontinuity_enrich(true);

      if (!friction_coef_average)
        node->assign_discontinuity_property(true, "friction_coef",
                                            friction_coef, 0, 1);
      node->assign_discontinuity_property(true, "cohesion", cohesion, 0, 1);
    }

    (*citr)->assign_cohesion_area();
  }

  for (auto citr = cells_.cbegin(); citr != cells_.cend(); ++citr) {
    if ((*citr)->element_type_discontinuity() != mpm::EnrichType::PotentialTip)
      continue;
    for (auto node : (*citr)->nodes()) {
      if (node->discontinuity_enrich())
        node->assign_discontinuity_enrich(false);
    }
  }

  if (!enrich_all) return;

  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    (*nitr)->assign_discontinuity_enrich(true);
  }
}

// modify_node_enrich
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

template <unsigned Tdim>
bool mpm::Mesh<Tdim>::initiation_discontinuity() {
  bool status = false;

  mpm::Index pid;
  double max_pdstrain = 0;
  for (int i = 0; i < nparticles(); ++i) {
    double pdstrain = map_particles_[i]->state_variable("pdstrain");

    if (pdstrain > max_pdstrain) {
      max_pdstrain = pdstrain;
      pid = i;
    }
  }

  if (max_pdstrain <= discontinuity_->maximum_pdstrain()) return status;
  VectorDim normal;
  bool initiation = map_particles_[pid]->minimum_acoustic_tensor(normal, true);

  if (initiation) {
    status = true;
    auto cell_id = map_particles_[pid]->cell_id();
    map_cells_[cell_id]->assign_type_discontinuity(mpm::EnrichType::InitialTip);
    map_cells_[cell_id]->assign_normal_discontinuity(normal);
    auto center = map_cells_[cell_id]->centroid();

    double d = 0;

    for (unsigned int i = 0; i < Tdim; i++) d -= center[i] * normal[i];

    map_cells_[cell_id]->assign_normal_discontinuity(normal, d);

    map_cells_[cell_id]->compute_nodal_levelset_equation();

    std::vector<VectorDim> coordinates_dis;
    map_cells_[cell_id]->compute_discontinuity_point(coordinates_dis);

    for (int i = 0; i < coordinates_dis.size(); i++)
      discontinuity_->insert_particles(coordinates_dis[i], cells_, map_cells_);
    // initialise neighbour cells

    auto neighbours = map_cells_[cell_id]->neighbours();
    for (auto neighbour : neighbours) {
      if (map_cells_[neighbour]->nparticles() == 0) continue;
      map_cells_[neighbour]->assign_type_discontinuity(
          mpm::EnrichType::NeighbourTip_1);
      map_cells_[neighbour]->assign_normal_discontinuity(normal, d);
      map_cells_[neighbour]->compute_nodal_levelset_equation();
      if (map_cells_[neighbour]->product_levelset() >= 0) continue;
      map_cells_[neighbour]->assign_type_discontinuity(
          mpm::EnrichType::InitialTip);

      std::vector<VectorDim> coordinates_dis_neigh;
      map_cells_[neighbour]->compute_discontinuity_point(coordinates_dis_neigh);

      for (int i = 0; i < coordinates_dis_neigh.size(); i++) {
        discontinuity_->insert_particles(coordinates_dis_neigh[i], cells_,
                                         map_cells_);
      }
    }
    // initialise level set values

    for (int i = 0; i < nparticles(); ++i) {
      bool neighbour = true;
      for (int j = 0; j < Tdim; j++) {
        if (std::abs(center[j] - particles_[i]->coordinates()[j]) >
            3.5 * discontinuity_->width())
          neighbour = false;
      }
      if (!neighbour) continue;
      double phi = particles_[i]->coordinates().dot(normal) + d;
      particles_[i]->assign_levelsetphi(phi);
    }
  }
  return status;
}

template <unsigned Tdim>
void mpm::Mesh<Tdim>::modify_nodal_levelset_mls() {
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
      double length = discontinuity_->width();
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
          double length = 2 * discontinuity_->width();
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
    error = std::sqrt(error / error_p) / discontinuity_->width();

    if (error > 1e-3) continue;

    Eigen::Matrix<double, 1, 4> cor;
    Eigen::Matrix<double, 1, 1> phi_mls;

    cor << 1, (*nitr)->coordinates()[0], (*nitr)->coordinates()[1],
        (*nitr)->coordinates()[2];
    phi_mls(0, 0) = cor.dot(coef);

    (*nitr)->assign_discontinuity_property(true, "levelset_phi", phi_mls, 0, 1);
  }
}

template <unsigned Tdim>
void mpm::Mesh<Tdim>::selfcontact_detection() {

  double contact_distance = discontinuity_->contact_distance();

  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    if (!(*nitr)->discontinuity_enrich()) continue;

    auto cor = (*nitr)->coordinates();
    auto normal = (*nitr)->discontinuity_property(
        "normal_unit_vectors_discontinuity", Tdim);
    double dis_negative = -10 * contact_distance;
    double dis_positive = 10 * contact_distance;
    for (auto cell : (*nitr)->cells()) {

      for (auto particle : map_cells_[cell]->particles()) {
        auto corp = map_particles_[particle]->coordinates();
        double phi = map_particles_[particle]->levelset_phi();

        double dis = 0;
        for (unsigned int i = 0; i < Tdim; i++)
          dis += (corp[i] - cor[i]) * normal(i);

        if (phi > 0) dis_positive = dis < dis_positive ? dis : dis_positive;
        if (phi < 0) dis_negative = dis > dis_negative ? dis : dis_negative;
      }
    }
    Eigen::Matrix<double, 1, 1> dis;
    dis(0, 0) = dis_positive - dis_negative - contact_distance;
    (*nitr)->assign_discontinuity_property(true, "contact_distance", dis, 0, 1);
  }
}

template <unsigned Tdim>
void mpm::Mesh<Tdim>::check_particle_levelset(bool particle_levelset) {

  for (auto pitr = particles_.cbegin(); pitr != particles_.cend(); ++pitr) {
    if ((*pitr)->levelset_phi() != 0) continue;

    auto cell_id = (*pitr)->cell_id();

    for (auto node : cells_[cell_id]->nodes()) {
      if (!node->discontinuity_enrich()) continue;

      Eigen::Matrix<double, 4, 4> au;
      Eigen::Matrix<double, 4, 1> bu;
      au.setZero();
      bu.setZero();
      auto cell_list = cells_[cell_id]->neighbours();
      cell_list.insert(cell_id);

      for (auto cell : cell_list) {
        for (auto particle : cells_[cell]->particles()) {
          auto corp = map_particles_[particle]->coordinates();
          double phi = map_particles_[particle]->levelset_phi();
          if (phi == 0) continue;
          // compute weight
          double length = 2.0 * discontinuity_->width();
          double w[3];
          for (int i = 0; i < 3; i++) {
            w[i] = 1 - std::abs(corp[i] - (*pitr)->coordinates()[i]) / length;
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

      const double tolerance = std::numeric_limits<double>::epsilon();

      if (std::abs(au.determinant()) < tolerance) {
        (*pitr)->map_levelset_to_particle();
        continue;
      }

      Eigen::Vector4d coef;
      coef.setZero();
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) coef[i] += au.inverse()(i, j) * bu(j, 0);

      Eigen::Vector4d cor;
      cor << 1, (*pitr)->coordinates()[0], (*pitr)->coordinates()[1],
          (*pitr)->coordinates()[2];
      double phi = cor.dot(coef);

      (*pitr)->assign_levelsetphi(phi);

      break;
    }
  }
  if (particle_levelset) return;
  for (auto pitr = particles_.cbegin(); pitr != particles_.cend(); ++pitr) {
    if ((*pitr)->levelset_phi() == 0) continue;
    auto cell_id = (*pitr)->cell_id();

    if (cells_[cell_id]->element_type_discontinuity() ==
            mpm::EnrichType::Regular ||
        cells_[cell_id]->element_type_discontinuity() ==
            mpm::EnrichType::NeighbourTip_3)
      (*pitr)->assign_levelsetphi(0);
  }
}

// code for debugging added by yliang
template <unsigned Tdim>
void mpm::Mesh<Tdim>::output_celltype(int step) {
  std::ofstream test("cell_type.txt", std::ios::app);

  test << step << ":" << std::endl;
  for (int i = 0; i < cells_.size(); i++) {
    auto type = cells_[i]->element_type_discontinuity();
    if (type == 1)
      test << "o ";
    else if (type == 2)
      test << "\\ ";
    else if (type == 3)
      test << "^ ";
    else if (type == 4)
      test << "1 ";
    else if (type == 5)
      test << "2 ";
    else if (type == 6)
      test << "* ";
    else
      test << type << " ";
    if (((i + 1) % 90) == 0) test << std::endl;
  }
  test << std::endl;

  std::ofstream testnormal("node_normal.txt", std::ios::app);
  testnormal << step << ":" << std::endl;
  for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr) {
    if (!(*nitr)->discontinuity_enrich()) continue;

    if ((*nitr)->coordinates()[0] < 35) continue;

    Eigen::Matrix<double, Tdim, 1> normal = (*nitr)->discontinuity_property(
        "normal_unit_vectors_discontinuity", Tdim);
    testnormal << (*nitr)->coordinates()[0] << "\t" << (*nitr)->coordinates()[1]
               << "\t" << normal[0] << "\t" << normal[1] << "\t" << normal[2]
               << std::endl;
  }
}

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