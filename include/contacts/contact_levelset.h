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

  //! Initialise levelset properties
  //! \param[in] levelset_damping Levelset damping factor
  //! \param[in] levelset_pic Particle in cell method bool for contact velocity
  //! \param[in] levelset_violation_corrector Violation correction factor
  void initialise_levelset_properties(
      double levelset_damping, bool levelset_pic,
      double levelset_violation_corrector) override;

  //! Compute contact forces
  //! \param[in] dt Analysis time step
  void compute_contact_forces(double dt) override;

  //! Mesh levelset object
  std::shared_ptr<mpm::MeshLevelset<Tdim>> mesh_levelset_;

 protected:
  //! levelset_damping_
  double levelset_damping_;
  //! levelset_pic_
  bool levelset_pic_;
  //! levelset_violation_corrector_
  double levelset_violation_corrector_;
  //! Mesh object
  using mpm::Contact<Tdim>::mesh_;
};  // ContactLevelset class
}  // namespace mpm

#include "contact_levelset.tcc"

#endif  // MPM_CONTACT_LEVELSET_H_
