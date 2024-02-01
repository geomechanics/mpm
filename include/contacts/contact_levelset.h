#ifndef MPM_CONTACT_LEVELSET_H_
#define MPM_CONTACT_LEVELSET_H_

#include "contact.h"
#include "mesh_levelset.h"
#include "particle_levelset.h"

namespace mpm {

//! ContactLevelset class
//! \brief ContactLevelset base class
//! \tparam Tdim Dimension
template <unsigned Tdim>
class ContactLevelset : public Contact<Tdim> {
 public:
  //! Default constructor with mesh class
  ContactLevelset(const std::shared_ptr<mpm::Mesh<Tdim>>& mesh);

  //! Intialize
  void initialise() override;

  //! Compute contact forces
  //! \param[in] dt Analysis time step
  void compute_contact_forces(double dt) override;

  //! Mesh levelset object
  std::shared_ptr<mpm::MeshLevelset<Tdim>> mesh_levelset_;

 protected:
  //! Mesh object
  using mpm::Contact<Tdim>::mesh_;

};  // ContactLevelset class
}  // namespace mpm

#include "contact_levelset.tcc"

#endif  // MPM_CONTACT_LEVELSET_H_
