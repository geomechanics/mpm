//! Constructor
template <unsigned Tdim>
mpm::MPMSchemeMUSL<Tdim>::MPMSchemeMUSL(
    const std::shared_ptr<mpm::Mesh<Tdim>>& mesh, double dt)
    : mpm::MPMScheme<Tdim>(mesh, dt) {}

//! Precompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeMUSL<Tdim>::precompute_stress_strain(
    unsigned phase, bool pressure_smoothing, mpm::StressRate stress_rate) {}

//! Postcompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeMUSL<Tdim>::postcompute_stress_strain(
    unsigned phase, bool pressure_smoothing, mpm::StressRate stress_rate) {
  mpm::MPMScheme<Tdim>::compute_stress_strain(phase, pressure_smoothing,
                                              stress_rate);
}

//! Postcompute nodal kinematics - map mass and momentum to nodes
template <unsigned Tdim>
inline void mpm::MPMSchemeMUSL<Tdim>::postcompute_nodal_kinematics(
    mpm::VelocityUpdate velocity_update, unsigned phase) {
  // Assign mass and momentum to nodes zero
  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::update_mass, std::placeholders::_1, false,
                phase, 0.0),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

  mesh_->iterate_over_nodes_predicate(
      std::bind(&mpm::NodeBase<Tdim>::update_momentum, std::placeholders::_1,
                false, phase, VectorDim::Zero()),
      std::bind(&mpm::NodeBase<Tdim>::status, std::placeholders::_1));

  this->compute_nodal_kinematics(velocity_update, phase);
}

//! Stress update scheme
template <unsigned Tdim>
inline std::string mpm::MPMSchemeMUSL<Tdim>::scheme() const {
  return "MUSL";
}
