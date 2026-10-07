//! Constructor
template <unsigned Tdim>
mpm::MPMSchemeMUSL<Tdim>::MPMSchemeMUSL(
    const std::shared_ptr<mpm::Mesh<Tdim>>& mesh, double dt)
    : mpm::MPMScheme<Tdim>(mesh, dt) {}

//! Precompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeMUSL<Tdim>::precompute_stress_strain(
    unsigned phase, bool pressure_smoothing) {}

//! Postcompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeMUSL<Tdim>::postcompute_stress_strain(
    unsigned phase, bool pressure_smoothing) {
  mpm::MPMScheme<Tdim>::compute_stress_strain(phase, pressure_smoothing);
}

//! Postcompute nodal kinematics - map mass and momentum to nodes
template <unsigned Tdim>
inline void mpm::MPMSchemeMUSL<Tdim>::postcompute_nodal_kinematics(
    mpm::VelocityUpdate velocity_update, unsigned phase) {
  // Assign mass and momentum to nodes zero
  // NOTE (MPI): nodes shared between ranks also hold the exchanged mass and
  // momentum of the first mapping on ranks where they are inactive. They must
  // be reset as well, otherwise that stale total is added again in the halo
  // exchange below (wrong nodal velocity, typically after load balancing).
  if (mpi_size_ > 1 && mesh_->active_region()) {
    // Active nodes and nodes shared between ranks are the only nodes that can
    // hold mass / momentum
    auto zero = [phase](const std::shared_ptr<mpm::NodeBase<Tdim>>& node) {
      node->update_mass(false, phase, 0.0);
      node->update_momentum(false, phase, VectorDim::Zero());
    };
    mesh_->iterate_over_status_nodes(zero);
    mesh_->iterate_over_domain_shared_nodes(zero);
  } else if (mpi_size_ > 1) {
    mesh_->iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::update_mass,
                                        std::placeholders::_1, false, phase,
                                        0.0));
    mesh_->iterate_over_nodes(std::bind(&mpm::NodeBase<Tdim>::update_momentum,
                                        std::placeholders::_1, false, phase,
                                        VectorDim::Zero()));
  } else {
    mesh_->iterate_over_status_nodes(
      std::bind(&mpm::NodeBase<Tdim>::update_mass, std::placeholders::_1,
                  false, phase, 0.0));

    mesh_->iterate_over_status_nodes(
      std::bind(&mpm::NodeBase<Tdim>::update_momentum, std::placeholders::_1,
                  false, phase, VectorDim::Zero()));
  }

  this->compute_nodal_kinematics(velocity_update, phase);
}

//! Stress update scheme
template <unsigned Tdim>
inline std::string mpm::MPMSchemeMUSL<Tdim>::scheme() const {
  return "MUSL";
}
