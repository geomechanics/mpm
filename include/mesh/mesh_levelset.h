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
//! \brief subclass that stores the information about levelset mesh
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MeshLevelset : public Mesh<Tdim> {

 public:
  // Construct a mesh with a global unique id
  //! \param[in] id Global mesh id
  //! \param[in] isoparametric Mesh is isoparametric
  MeshLevelset(unsigned id, bool isoparametric = true)
      : mpm::Mesh<Tdim>(id, isoparametric) {
    console_ = std::make_unique<spdlog::logger>("Levelset", mpm::stdout_sink);
  };

  //! Assign mesh levelset values to nodes
  //! \param[in] levelset Levelset value at the particle
  //! \param[in] levelset_mu Levelset friction
  //! \param[in] barrier_stiffness Barrier stiffness
  //! \param[in] slip_threshold Slip threshold
  //! \param[in] levelset_mp_radius mp radius of influence for contact
  bool assign_nodal_levelset_values(
      const std::vector<std::tuple<mpm::Index, double, double, double, double,
                                   double>>& levelset_input_file) override;

  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

 private:
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Vector of nodes // LEDT check if necessary
  using mpm::Mesh<Tdim>::nodes_;
  //! Map of nodes for fast retrieval
  using mpm::Mesh<Tdim>::map_nodes_;

};  // MeshLevelset class

}  // namespace mpm

#include "mesh_levelset.tcc"

#endif  // MPM_MESH_LEVELSET_H_