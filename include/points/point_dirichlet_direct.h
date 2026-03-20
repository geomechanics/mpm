#ifndef MPM_POINT_DIRICHLET_H_
#define MPM_POINT_DIRICHLET_H_

// MPI
#ifdef USE_MPI
#include "mpi.h"
#endif

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "point_base.h"

namespace mpm {

//! Point class to impose nonconforming displacement BC with direct imposition
//! method
//! \tparam Tdim Dimension
template <unsigned Tdim>
class PointDirichletDirect : public PointBase<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id and coordinates
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  PointDirichletDirect(Index id, const VectorDim& coord);

  //! Constructor with id, coordinates and status
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  //! \param[in] status Point status (active / inactive)
  PointDirichletDirect(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~PointDirichletDirect() override {};

  //! Delete copy constructor
  PointDirichletDirect(const PointDirichletDirect<Tdim>&) = delete;

  //! Delete assignement operator
  PointDirichletDirect& operator=(const PointDirichletDirect<Tdim>&) = delete;

  //! Initialise properties
  void initialise() override;

  //! Compute updated position
  //! \param[in] dt Analysis time step
  void compute_updated_position(double dt) noexcept override;

  //! Assign point velocity constraints
  //! \param[in] dir Direction of point velocity constraint
  //! \param[in] velocity Applied point velocity constraint
  void assign_velocity_constraints(unsigned dir, double velocity) override;

  //! Apply point velocity constraints
  //! \param[in] phase Index corresponding to the phase
  void apply_velocity_constraints(unsigned phase) override;

  //! Serialize
  //! \retval buffer Serialized buffer data
  std::vector<uint8_t> serialize() override;

  //! Deserialize
  //! \param[in] buffer Serialized buffer data
  void deserialize(const std::vector<uint8_t>& buffer) override;

  //! Assign point properties
  //! \param[in] scalar_properties Map of scalar properties
  //! \param[in] vector_properties Map of vector properties
  void assign_properties(const std::map<std::string, double>& scalar_properties,
                         const std::map<std::string, std::vector<double>>&
                             vector_properties) override;

  //! Reinitialise point property
  //! \param[in] dt Time step size
  void initialise_properties(double dt) override;

  //! Type of point
  std::string type() const override {
    return (Tdim == 2) ? "POINT2DDIRDIRECT" : "POINT3DDIRDIRECT";
  }

 protected:
  //! Compute pack size
  //! \retval pack size of serialized object
  int compute_pack_size() const override;

 protected:
  //! point id
  using PointBase<Tdim>::id_;
  //! coordinates
  using PointBase<Tdim>::coordinates_;
  //! Status
  using PointBase<Tdim>::status_;
  //! Cell
  using PointBase<Tdim>::cell_;
  //! Cell id
  using PointBase<Tdim>::cell_id_;
  //! Nodes
  using PointBase<Tdim>::nodes_;
  //! Shape functions
  using PointBase<Tdim>::shapefn_;
  //! Displacement
  using PointBase<Tdim>::displacement_;
  //! Area
  using PointBase<Tdim>::area_;
  //! Normal vector
  using PointBase<Tdim>::normal_;
  //! Pack size
  using PointBase<Tdim>::pack_size_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Imposed displacement
  VectorDim imposed_displacement_;
  //! Imposed velocity
  VectorDim imposed_velocity_;
  //! Imposed acceleration
  VectorDim imposed_acceleration_;
  //! Constraint flags: 1 = constrained, 0 = unconstrained, per direction
  Eigen::Matrix<int, Tdim, 1> constraint_flags_;

};  // PointDirichletDirect class
}  // namespace mpm

#include "point_dirichlet_direct.tcc"

#endif  // MPM_POINT_DIRICHLET_H_