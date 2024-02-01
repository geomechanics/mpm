//! Constructor of contact with mesh
template <unsigned Tdim>
mpm::ContactLevelset<Tdim>::ContactLevelset(
    const std::shared_ptr<mpm::Mesh<Tdim>>& mesh)
    : mpm::Contact<Tdim>(mesh) {
  // Assign mesh
  mesh_ = mesh;
}

//! Initialize nodal properties
template <unsigned Tdim>
void mpm::ContactLevelset<Tdim>::initialise() {
  // Initialise nodal properties
  mesh_->initialise_nodal_properties();
}

//! Compute contact forces
template <unsigned Tdim>
void mpm::ContactLevelset<Tdim>::compute_contact_forces(
    double dt, const double levelset_mp_radius) {

  // std::cout << "-->compute_contact_force() levelset_mp_radius "
  //           << levelset_mp_radius << std::endl;
  // std::cout << "-->5.1" << std::endl;  // LEDT REMOVE!

  // Compute and map contact forces to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_particle_contact_force_to_nodes,
                std::placeholders::_1, dt, levelset_mp_radius));
  // std::cout << "-->5.2" << std::endl;  // LEDT REMOVE!
}
