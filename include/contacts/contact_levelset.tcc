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
inline void mpm::ContactLevelset<Tdim>::initialise() {
  // Initialise nodal properties
  mesh_->initialise_nodal_properties();
}

//! Compute contact forces
template <unsigned Tdim>
inline void mpm::ContactLevelset<Tdim>::compute_contact_forces(double dt) {

  // Compute and map contact forces to nodes
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::map_particle_contact_force_to_nodes,
                std::placeholders::_1, dt));
}
