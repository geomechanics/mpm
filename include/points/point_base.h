#ifndef MPM_POINTBASE_H_
#define MPM_POINTBASE_H_

// MPI
#ifdef USE_MPI
#include "mpi.h"
#endif

#include <array>
#include <limits>
#include <memory>
#include <vector>

#include "cell.h"
#include "data_types.h"
#include "function_base.h"
#include "particle.h"
#include "pod_particle.h"
#include "pod_particle_twophase.h"
#include "vector.h"

namespace mpm {
//! PointBase class
//! \brief Base class that stores the information about PointBases
//! \details PointBase class: id_ and coordinates.
//! \tparam Tdim Dimension
template <unsigned Tdim>
class PointBase {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Constructor with id and coordinates
  //! \param[in] coord coordinates of the point
  PointBase(const VectorDim& coord);

  //! Constructor with id and coordinates
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  PointBase(Index id, const VectorDim& coord);

  //! Constructor with id, coordinates and status
  //! \param[in] id Point id
  //! \param[in] coord coordinates of the point
  //! \param[in] status Point status (active / inactive)
  PointBase(Index id, const VectorDim& coord, bool status);

  //! Destructor
  virtual ~PointBase(){};

  //! Delete copy constructor
  PointBase(const PointBase<Tdim>&) = delete;

  //! Delete assignement operator
  PointBase& operator=(const PointBase<Tdim>&) = delete;

  //! Initialise properties
  virtual void initialise();

  //! Return id of the pointBase
  Index id() const { return id_; }

  //! Assign coordinates
  //! \param[in] coord Assign coord as coordinates of the pointBase
  void assign_coordinates(const VectorDim& coord) { coordinates_ = coord; }

  //! Return coordinates
  //! \retval coordinates_ return coordinates of the pointBase
  VectorDim coordinates() const { return coordinates_; }

  //! Compute reference coordinates in a cell
  virtual bool compute_reference_location() noexcept;

  //! Return reference location
  virtual VectorDim reference_location() const { return xi_; };

  //! Assign cell
  virtual bool assign_cell(const std::shared_ptr<Cell<Tdim>>& cellptr);

  //! Assign cell and xi
  virtual bool assign_cell_xi(const std::shared_ptr<Cell<Tdim>>& cellptr,
                              const Eigen::Matrix<double, Tdim, 1>& xi);

  //! Assign cell id
  virtual bool assign_cell_id(Index id);

  //! Return cell id
  virtual Index cell_id() const { return cell_id_; }

  //! Return cell ptr status
  virtual bool cell_ptr() const { return cell_ != nullptr; }

  //! Remove cell
  virtual void remove_cell();

  //! Compute shape functions
  virtual void compute_shapefn() noexcept;

  //! Return displacement of the point
  virtual VectorDim displacement() const { return displacement_; }

  //! Compute updated position
  //! \param[in] dt Analysis time step
  //! \param[in] velocity_update Update particle velocity from nodal vel
  virtual void compute_updated_position(double dt,
                                        bool velocity_update = false) noexcept;

  //! Return scalar data of points
  //! \param[in] property Property string
  //! \retval data Scalar data of point property
  virtual double scalar_data(const std::string& property) const;

  //! Return vector data of points
  //! \param[in] property Property string
  //! \retval data Vector data of point property
  virtual VectorDim vector_data(const std::string& property) const;

  //! Return tensor data of points
  //! \param[in] property Property string
  //! \retval data Tensor data of point property
  virtual Eigen::VectorXd tensor_data(const std::string& property) const;
  /**
   * \defgroup discontinuity point dealing with XMPM
   */
  /**@{*/

  //! Constructor with id and coordinates
  //! \param[in] coord coordinates of the point
  //! \param[in] dis_id the discontinuity id
  PointBase(const VectorDim& coord, mpm::Index dis_id){};

  //! Locate particles in a cell
  //! \param[in] cells vector of cells
  //! \param[in] map_cells map of cells
  //! \param[in] dis_id the discontinuity id
  //! \param[in] update update cell type or not
  virtual void locate_discontinuity_mesh(const Vector<Cell<Tdim>>& cells,
                                         const Map<Cell<Tdim>>& map_cells,
                                         unsigned dis_id, bool update) {
    throw std::runtime_error(
        "Calling the base class function (locate_discontinuity_mesh) in "
        "PointBase:: "
        "illegal operation!");
  }

  //! Assign the discontinuity type to cell
  //! \param[in] map_cells map of cells
  //! \param[in] dis_id the discontinuity id
  virtual void assign_cell_enrich(const Map<Cell<Tdim>>& map_cells,
                                  unsigned dis_id) {
    throw std::runtime_error(
        "Calling the base class function (assign_cell_enrich) in "
        "PointBase:: "
        "illegal operation!");
  }

  //! Compute updated position
  //! \param[in] dt Analysis time step
  //! \param[in] move_direction the discontinuity point move with move_direction
  //! side
  virtual void compute_updated_position_discontinuity_point(
      double dt, int move_direction) {
    throw std::runtime_error(
        "Calling the base class function "
        "(compute_updated_position_discontinuity_point) in "
        "PointBase:: "
        "illegal operation!");
  }

  //! Assign point friction coefficient
  //! \param[in] friction_coef
  virtual void assign_friction_coef(double friction_coef) {
    throw std::runtime_error(
        "Calling the base class function "
        "(assign_friction_coef) in "
        "PointBase:: "
        "illegal operation!");
  }

  //! Assign cohesion
  //! \param[in] friction_coef
  virtual void assign_cohesion(double cohesion) {
    throw std::runtime_error(
        "Calling the base class function "
        "(assign_cohesion) in "
        "PointBase:: "
        "illegal operation!");
  }

  //! Assign the status of terminal point
  //! \param[in] terminal_point
  virtual void assign_terminal_point(bool terminal_point) {
    throw std::runtime_error(
        "Calling the base class function "
        "(assign_terminal_point) in "
        "PointBase:: "
        "illegal operation!");
  }

  /**@}*/

 protected:
  //! pointBase id
  Index id_{std::numeric_limits<Index>::max()};
  //! coordinates
  VectorDim coordinates_;
  //! Velocity
  Eigen::Matrix<double, Tdim, 1> velocity_;
  //! displacement
  VectorDim displacement_;
  //! Cell id
  Index cell_id_{std::numeric_limits<Index>::max()};
  //! Status
  bool status_{true};
  //! Reference coordinates (in a cell)
  Eigen::Matrix<double, Tdim, 1> xi_;
  //! Cell
  std::shared_ptr<Cell<Tdim>> cell_;
  //! Vector of nodal pointers
  std::vector<std::shared_ptr<NodeBase<Tdim>>> nodes_;
  //! Shape functions
  Eigen::VectorXd shapefn_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Map of scalar properties
  tsl::robin_map<std::string, std::function<double()>> scalar_properties_;
  //! Map of vector properties
  tsl::robin_map<std::string, std::function<VectorDim()>> vector_properties_;
  //! Map of tensor properties
  tsl::robin_map<std::string, std::function<Eigen::VectorXd()>>
      tensor_properties_;
};  // PointBase class
}  // namespace mpm

#include "point_base.tcc"

#endif  // MPM_POINTBASE_H__
