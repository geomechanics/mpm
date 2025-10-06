//! Constructor of contact with mesh
template <unsigned Tdim>
mpm::ContactLevelset<Tdim>::ContactLevelset(
    const std::shared_ptr<mpm::Mesh<Tdim>>& mesh)
    : mpm::Contact<Tdim>(mesh) {
  // Assign mesh
  mesh_ = mesh;
}

//! Initialise levelset properties
template <unsigned Tdim>
void mpm::ContactLevelset<Tdim>::initialise_levelset_properties(
    double levelset_damping, bool levelset_pic,
    double levelset_violation_corrector) {
  mpm::ParticleLevelset<Tdim>::set_levelset_properties(
      levelset_damping, levelset_pic, levelset_violation_corrector);
}

//! Compute contact reaction forces
template <unsigned Tdim>
void mpm::ContactLevelset<Tdim>::compute_contact_forces(double dt) {
  mesh_->iterate_over_particles(
      std::bind(&mpm::ParticleBase<Tdim>::levelset_contact_force,
                std::placeholders::_1, dt));
}