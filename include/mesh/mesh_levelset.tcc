//! Assign mesh levelset values to nodes
template <unsigned Tdim>
bool mpm::MeshLevelset<Tdim>::assign_nodal_levelset_values(
    const std::vector<std::tuple<mpm::Index, double, double, double, double>>&
        levelset_input_file) {
  bool status = true;
  try {
    if (!nodes_.size())
      throw std::runtime_error(
          "No nodes have been assigned in mesh, cannot assign levelset values");

    for (const auto& levelset_info : levelset_input_file) {
      // Node id
      mpm::Index nid = std::get<0>(levelset_info);
      // Levelset
      double levelset = std::get<1>(levelset_info);
      // Levelset friction
      double levelset_mu = std::get<2>(levelset_info);
      // Levelset adhesion coefficient
      double levelset_alpha = std::get<3>(levelset_info);
      // Barrier stiffness
      double barrier_stiffness = std::get<4>(levelset_info);

      if (map_nodes_.find(nid) != map_nodes_.end())
        status = map_nodes_[nid]->assign_levelset(
            levelset, levelset_mu, levelset_alpha, barrier_stiffness);

      if (!status)
        throw std::runtime_error("Cannot assign invalid nodal levelset values");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Create the nodal properties' map
template <unsigned Tdim>
void mpm::MeshLevelset<Tdim>::create_nodal_properties() {
  // Initialise the shared pointer to nodal properties
  nodal_properties_ = std::make_shared<mpm::NodalProperties>();

  // Check if nodes_ and materials_is empty (if empty throw runtime error)
  if (nodes_.size() != 0 && materials_.size() != 0) {
    // Compute number of rows in nodal properties for vector entities
    const unsigned nrows = nodes_.size() * Tdim;
    // Create pool data for each property in the nodal properties struct
    // object. Properties must be named in the plural form
    nodal_properties_->create_property("masses", nodes_.size(),
                                       materials_.size());
    nodal_properties_->create_property("momenta", nrows, materials_.size());
    nodal_properties_->create_property("change_in_momenta", nrows,
                                       materials_.size());
    nodal_properties_->create_property("displacements", nrows,
                                       materials_.size());
    nodal_properties_->create_property("separation_vectors", nrows,
                                       materials_.size());
    nodal_properties_->create_property("domain_gradients", nrows,
                                       materials_.size());
    nodal_properties_->create_property("normal_unit_vectors", nrows,
                                       materials_.size());
    nodal_properties_->create_property("wave_velocities", nrows,
                                       materials_.size());
    nodal_properties_->create_property("density", nodes_.size(),
                                       materials_.size());
    // levelset properties
    nodal_properties_->create_property("levelsets", nodes_.size(),
                                       materials_.size());
    nodal_properties_->create_property("levelset_mus", nodes_.size(),
                                       materials_.size());
    nodal_properties_->create_property("levelset_alphas", nodes_.size(),
                                       materials_.size());
    nodal_properties_->create_property("barrier_stiffnesses", nodes_.size(),
                                       materials_.size());
    nodal_properties_->create_property("levelset_mp_radii", nodes_.size(),
                                       materials_.size());

    // Iterate over all nodes to initialise the property handle in each node
    // and assign its node id as the prop id in the nodal property data pool
    for (auto nitr = nodes_.cbegin(); nitr != nodes_.cend(); ++nitr)
      (*nitr)->initialise_property_handle((*nitr)->id(), nodal_properties_);
  } else {
    throw std::runtime_error(
        "Number of nodes or number of materials is zero (levelset)");
  }
}