#ifndef MPM_CONTACT_FRICTION_H_
#define MPM_CONTACT_FRICTION_H_

#include "contact.h"

namespace mpm {

//! ContactFriction class
//! \brief ContactFriction base class
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ContactFriction : public Contact<Tdim> {
 public:
  //! Default constructor with mesh class
  ContactFriction(const std::shared_ptr<mpm::Mesh<Tdim>>& mesh);

  //! Intialize
  void initialise() override;

  //! Compute contact forces
  void compute_contact_forces() override;

  //! Compute contact forces
  //! \param[in] dt Analysis time step
  //! \param[in] levelset_damping Levelset damping factor
  //! \param[in] levelset_pic Particle in cell method bool for contact velocity
  void compute_contact_forces(const double levelset_damping,
                              const bool levelset_pic, double dt) override;

 protected:
  //! Mesh object
  using mpm::Contact<Tdim>::mesh_;
};  // Contactfriction class
}  // namespace mpm

#include "contact_friction.tcc"

#endif  // MPM_CONTACT_FRICTION_H_
