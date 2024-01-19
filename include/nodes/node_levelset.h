#ifndef MPM_NODE_LEVELSET_H_
#define MPM_NODE_LEVELSET_H_

#include "node.h"
#include "node_base.h"

namespace mpm {

//! Levelset subclass
//! \brief subclass that stores the information about levelset node
//! \tparam Tdim Dimension
template <unsigned Tdim, unsigned Tdof, unsigned Tnphases>
class NodeLevelset : public Node<Tdim, Tdof, Tnphases> {

 public:
  // Assign levelset values to nodes
  //! \param[in] levelset Levelset value at the particle
  //! \param[in] levelset_mu Levelset friction
  //! \param[in] barrier_stiffness Barrier stiffness
  //! \param[in] slip_threshold Slip threshold
  //! \param[in] levelset_mp_radius mp radius of influence for contact
  bool assign_levelset(double levelset, double levelset_mu,
                       double barrier_stiffness, double slip_threshold,
                       double levelset_mp_radius) override;

 private:
  //! Levelset value
  double levelset_;
  //! Levelset friction
  double levelset_mu_;
  //! Barrier stiffness
  double barrier_stiffness_;
  //! Slip threshold
  double slip_threshold_;
  //! Levelset mp radius
  double levelset_mp_radius_;

};  // NodeLevelset class

}  // namespace mpm

#include "node_levelset.tcc"

#endif  // MPM_NODE_LEVELSET_H_