#ifndef MPM_MESH_LEVELSET_H_
#define MPM_MESH_LEVELSET_H_

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "logger.h"
#include "mesh.h"

namespace mpm {

//! Levelset subclass
//! \brief subclass that stores the information about levelset node
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MeshLevelset : public Mesh<Tdim> {

 public:
  //! Constructor with mesh as input argument
  MeshLevelset(std::shared_ptr<mpm::Mesh<Tdim>> mesh) {
    mesh_ = mesh;
    console_ = std::make_unique<spdlog::logger>("Levelset", mpm::stdout_sink);
  }

  //! Assign mesh levelset values to nodes
  //! \param[in] levelset Levelset value at the particle
  //! \param[in] levelset_mu Levelset friction
  //! \param[in] barrier_stiffness Barrier stiffness
  //! \param[in] slip_threshold Slip threshold
  //! \param[in] levelset_mp_radius mp radius of influence for contact
  bool assign_nodal_levelset_values(
      const std::vector<std::tuple<mpm::Index, double, double, double, double,
                                   double>>& levelset_input_file);

  //! Return status of the nodes
  bool status() const { return nodes_.size(); }

  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

 private:
  //! Mesh object
  std::shared_ptr<mpm::Mesh<Tdim>> mesh_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Vector of nodes
  Vector<NodeBase<Tdim>> nodes_;
  //! Map of nodes for fast retrieval
  Map<NodeBase<Tdim>> map_nodes_;

};  // Mesh_Levelset class

}  // namespace mpm

#include "mesh_levelset.tcc"

#endif  // MPM_MESH_LEVELSET_H_