//! Assign mesh levelset values to nodes
template <unsigned Tdim>
bool mpm::MeshLevelset<Tdim>::assign_nodal_levelset_values(
    const std::vector<std::tuple<mpm::Index, double, double, double, double,
                                 double>>& levelset_input_file) {
  bool status = true;
  try {
    if (!nodes_.size())
      throw std::runtime_error(
          "No nodes have been assigned in mesh, cannot assign levelset values");

    for (const auto& levelset_info : levelset_input_file) {
      // Node id
      mpm::Index pid = std::get<0>(levelset_info);
      // Levelset
      double levelset = std::get<1>(levelset_info);
      // Levelset mu
      double levelset_mu = std::get<2>(levelset_info);
      // Barrier stiffness
      double barrier_stiffness = std::get<3>(levelset_info);
      // Slip threshold
      double slip_threshold = std::get<4>(levelset_info);
      // Levelset mp radius
      double levelset_mp_radius = std::get<5>(levelset_info);

      if (map_nodes_.find(pid) != map_nodes_.end())
        status = true;  // LEDT FIX (see assign_particles_volumes in mesh.tcc)

      if (!status)
        throw std::runtime_error("Cannot assign invalid nodal levelset values");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}
