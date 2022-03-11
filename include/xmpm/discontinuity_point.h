#ifndef MPM_DISCONTINUITYPOINT_H_
#define MPM_DISCONTINUITYPOINT_H_

#include "point_base.h"

namespace mpm {
//! Discontinuity Point class
//! \brief Discontinuity Point that stores the information of mark points
//! \details Discontinuity class: id_ and coordinates.
//! \tparam Tdim Dimension
template <unsigned Tdim>
class DiscontinuityPoint : public PointBase<Tdim> {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id and coordinates
  //! \param[in] coord coordinates of the point
  //! \param[in] dis_id the discontinuity id
  DiscontinuityPoint(const VectorDim& coord, mpm::Index dis_id);

  //! Destructor
  virtual ~DiscontinuityPoint(){};

  //! Delete copy constructor
  DiscontinuityPoint(const DiscontinuityPoint<Tdim>&) = delete;

  //! Delete assignement operator
  DiscontinuityPoint& operator=(const DiscontinuityPoint<Tdim>&) = delete;

  //! Initialise properties
  void initialise() override;

  //! Assign the discontinuity type to cell
  //! \param[in] map_cells map of cells
  //! \param[in] dis_id the discontinuity id
  void assign_cell_enrich(const Map<Cell<Tdim>>& map_cells, unsigned dis_id);

  //! Locate particles in a cell
  //! \param[in] cells vector of cells
  //! \param[in] map_cells map of cells
  //! \param[in] dis_id the discontinuity id
  //! \param[in] update update cell type or not
  void locate_discontinuity_mesh(const Vector<Cell<Tdim>>& cells,
                                 const Map<Cell<Tdim>>& map_cells,
                                 unsigned dis_id, bool update) override;

  //! Compute updated position
  //! \param[in] dt Analysis time step
  //! \param[in] move_direction the discontinuity point move with move_direction
  //! side
  void compute_updated_position_discontinuity_point(
      double dt, int move_direction) override;

  //! Assign point friction coefficient
  //! \param[in] friction_coef
  void assign_friction_coef(double friction_coef) {
    friction_coef_ = friction_coef;
  }

  //! Assign cohesion
  //! \param[in] friction_coef
  void assign_cohesion(double cohesion) { cohesion_ = cohesion; }

  //! Assign the status of terminal point
  //! \param[in] terminal_point
  void assign_terminal_point(bool terminal_point) {
    terminal_point_ = terminal_point;
  }

 protected:
  //! coordinates
  using PointBase<Tdim>::coordinates_;
  //! Velocity
  using PointBase<Tdim>::velocity_;
  //! displacement
  using PointBase<Tdim>::displacement_;
  //! Cell id
  using PointBase<Tdim>::cell_id_;
  //! Status
  using PointBase<Tdim>::status_;
  //! Reference coordinates (in a cell)
  using PointBase<Tdim>::xi_;
  //! Cell
  using PointBase<Tdim>::cell_;
  //! Vector of nodal pointers
  using PointBase<Tdim>::nodes_;
  //! Shape functions
  using PointBase<Tdim>::shapefn_;
  //! Logger
  using PointBase<Tdim>::console_;
  //! Map of scalar properties
  using PointBase<Tdim>::scalar_properties_;
  //! Map of vector properties
  using PointBase<Tdim>::vector_properties_;
  //! Map of tensor properties
  using PointBase<Tdim>::tensor_properties_;

 private:
  //! discontinuity_id_
  mpm::Index dis_id_;
  //! friction coefficient
  double friction_coef_{0.};
  //! cohesion
  double cohesion_{0.};
  //! tip
  bool tip_{false};
  //! end for search next tip
  bool terminal_point_{false};
};  // DiscontinuityPoint class
}  // namespace mpm

#include "discontinuity_point.tcc"

#endif  // MPM_DISCONTINUITYPOINT_H__
