//! Initialise xmpm nodal variables
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
mpm::NodeXMPM<Tdim, Tdof, Tnphases>::NodeXMPM(Index id, const VectorDim& coord)
    : mpm::Node<Tdim, Tdof, Tnphases>(id, coord) {
  this->initialise();
  // Specific variables for xmpm
}

//! Initialise nodal properties
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
void mpm::NodeXMPM<Tdim, Tdof, Tnphases>::initialise() noexcept {

  mpm::Node<Tdim, Tdof, Tnphases>::initialise();
  for (int i = 0; i < 2; i++) {
    // to do
    dis_enrich_[i] = true;
    levelset_phi_[i] = 0;
  }

  for (int i = 0; i < 3; i++) mass_enrich_[i] = 0;

  momentum_enrich_.setZero();
  internal_force_enrich_.setZero();
  external_force_enrich_.setZero();
}

//! Compute momentum for discontinuity
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
bool mpm::NodeXMPM<Tdim, Tdof, Tnphases>::compute_momentum_discontinuity(
    unsigned phase, double dt) noexcept {
  momentum_enrich_.col(phase) =
      momentum_.col(phase) + (this->internal_force(phase).col(phase) +
                              this->external_force(phase).col(phase)) *
                                 dt;

  for (unsigned int i = 0; i < 3; i++)
    momentum_enrich_.col(i) =
        (internal_force_enrich_.col(i) + external_force_enrich_.col(i)) * dt;

  // Apply velocity constraints, which also sets acceleration to 0,
  // when velocity is set.
  //   this->apply_velocity_constraints_discontinuity();

  //   this->self_contact_discontinuity(dt);

  //   this->apply_velocity_constraints_discontinuity();

  return true;
}