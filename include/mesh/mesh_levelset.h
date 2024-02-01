#ifndef MPM_MESH_LEVELSET_H_
#define MPM_MESH_LEVELSET_H_

#include <string>

#include "logger.h"
#include "mesh.h"

namespace mpm {

//! Levelset subclass
//! \brief subclass that stores the information about levelset mesh
//! \tparam Tdim Dimension
template <unsigned Tdim>
class MeshLevelset : public Mesh<Tdim> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  // Construct a mesh with a global unique id
  //! \param[in] id Global mesh id
  //! \param[in] isoparametric Mesh is isoparametric
  MeshLevelset(unsigned id, bool isoparametric)
      : mpm::Mesh<Tdim>(id, isoparametric),
        id_{id},
        isoparametric_{isoparametric} {
    // Check if the dimension is between 1 & 3
    static_assert((Tdim >= 1 && Tdim <= 3), "Invalid global dimension");
    //! Logger
    std::string logger = "meshlevelset::" + std::to_string(id);
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Default destructor
  ~MeshLevelset() = default;

  //! Delete copy constructor
  MeshLevelset(const MeshLevelset<Tdim>&) = delete;

  //! Delete assignement operator
  MeshLevelset& operator=(const MeshLevelset<Tdim>&) = delete;

  //! Assign mesh levelset values to nodes
  //! \param[in] levelset Levelset value at the particle
  //! \param[in] levelset_mu Levelset friction
  //! \param[in] barrier_stiffness Barrier stiffness
  //! \param[in] slip_threshold Slip threshold
  bool assign_nodal_levelset_values(
      const std::vector<std::tuple<mpm::Index, double, double, double, double>>&
          levelset_input_file) override;

  // Create the nodal properties' map
  void create_nodal_properties() override;

 private:
  //   //! mesh id
  //   using mpm::Mesh<Tdim>::id_;
  //   //! Isoparametric mesh
  //   using mpm::Mesh<Tdim>::isoparametric_;
  //! mesh id
  unsigned id_{std::numeric_limits<unsigned>::max()};
  //! Isoparametric mesh
  bool isoparametric_{true};
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Vector of nodes
  using mpm::Mesh<Tdim>::nodes_;
  //! Map of nodes for fast retrieval
  using mpm::Mesh<Tdim>::map_nodes_;
  //! Materials
  using mpm::Mesh<Tdim>::materials_;
  //! Nodal property pool
  using mpm::Mesh<Tdim>::nodal_properties_;

};  // MeshLevelset class

}  // namespace mpm

#include "mesh_levelset.tcc"

#endif  // MPM_MESH_LEVELSET_H_