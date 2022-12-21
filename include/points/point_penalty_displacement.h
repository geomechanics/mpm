#ifndef MPM_POINT_PENALTY_DISPLACEMENT_H_
#define MPM_POINT_PENALTY_DISPLACEMENT_H_

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

// Forward declaration of Material
template <unsigned Tdim>
class Material;

//! Point class to impose nonconforming displacement BC with penalty method
//! \tparam Tdim Dimension
template <unsigned Tdim>
class PointPenaltyDisplacement : public PointBase<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id and coordinates
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  PointPenaltyDisplacement(Index id, const VectorDim& coord);

  //! Constructor with id, coordinates and status
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  //! \param[in] status Point status (active / inactive)
  PointPenaltyDisplacement(Index id, const VectorDim& coord, bool status);

  //! Destructor
  ~PointPenaltyDisplacement() override{};

  //! Delete copy constructor
  PointPenaltyDisplacement(const PointPenaltyDisplacement<Tdim>&) = delete;

  //! Delete assignement operator
  PointPenaltyDisplacement& operator=(const PointPenaltyDisplacement<Tdim>&) =
      delete;

  //! Initialise properties
  void initialise() override;

  //! Reinitialise point property
  void reinitialise(double dt) override;

  //! Compute updated position
  //! \param[in] dt Analysis time step
  //! \param[in] velocity_update Method to update particle velocity
  //! \param[in] blending_ratio FLIP-PIC Blending ratio
  void compute_updated_position(
      double dt,
      mpm::VelocityUpdate velocity_update = mpm::VelocityUpdate::FLIP,
      double blending_ratio = 1.0) noexcept override;

  //! Map point stiffness matrix to cell
  inline bool map_stiffness_matrix_to_cell() override;

  //! Map enforcement boundary force to node
  //! \param[in] phase Index corresponding to the phase
  void map_boundary_force(unsigned phase) override;

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
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Imposed displacement
  VectorDim imposed_displacement_;
  //! Imposed velocity
  VectorDim imposed_velocity_;
  //! Imposed acceleration
  VectorDim imposed_acceleration_;
  //! Area
  double area_{0.};
  //! Penalty factor
  double penalty_factor_{0.};

};  // PointPenaltyDisplacement class
}  // namespace mpm

#include "point_penalty_displacement.tcc"

#endif  // MPM_POINT_PENALTY_DISPLACEMENT_H__