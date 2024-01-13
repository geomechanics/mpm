//! Constructor
template <unsigned Tdim>
mpm::MPMSchemeUSL<Tdim>::MPMSchemeUSL(
    const std::shared_ptr<mpm::Mesh<Tdim>>& mesh, double dt)
    : mpm::MPMScheme<Tdim>(mesh, dt) {}

//! Precompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeUSL<Tdim>::precompute_stress_strain(
    unsigned phase, bool pressure_smoothing, mpm::StressRate stress_rate) {}

//! Postcompute stresses and strains
template <unsigned Tdim>
inline void mpm::MPMSchemeUSL<Tdim>::postcompute_stress_strain(
    unsigned phase, bool pressure_smoothing, mpm::StressRate stress_rate) {
  mpm::MPMScheme<Tdim>::compute_stress_strain(phase, pressure_smoothing,
                                              stress_rate);
}

//! Postcompute nodal kinematics - map mass and momentum to nodes
template <unsigned Tdim>
inline void mpm::MPMSchemeUSL<Tdim>::postcompute_nodal_kinematics(
    mpm::VelocityUpdate velocity_update, unsigned phase) {}

//! Stress update scheme
template <unsigned Tdim>
inline std::string mpm::MPMSchemeUSL<Tdim>::scheme() const {
  return "USL";
}
