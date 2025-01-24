#ifndef MPM_CONTACT_H_
#define MPM_CONTACT_H_

#include "mesh.h"

namespace mpm {

//! Contact class
//! \brief Contact base class
//! \tparam Tdim Dimension
template <unsigned Tdim>
class Contact {
 public:
  //! Default constructor with mesh class
  Contact(const std::shared_ptr<mpm::Mesh<Tdim>>& mesh);

  //! Intialise
  virtual inline void initialise(){};

  //! Initialise levelset properties
  //! \param[in] levelset_damping Levelset damping factor
  //! \param[in] levelset_pic Particle in cell method bool for contact velocity
  virtual inline void initialise_levelset_properties(
      const double levelset_damping, const bool levelset_pic){};

  //! Compute contact forces
  virtual inline void compute_contact_forces(){};

  //! Compute contact forces
  //! \param[in] dt Analysis time step
  virtual inline void compute_contact_forces(double dt){};

 protected:
  //! Mesh object
  std::shared_ptr<mpm::Mesh<Tdim>> mesh_;
};  // Contact class
}  // namespace mpm

#include "contact.tcc"

#endif  // MPM_CONTACT_H_
