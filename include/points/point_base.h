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
#include "pod_point.h"

namespace mpm {

//! Point type
extern std::map<std::string, int> PointType;
extern std::map<int, std::string> PointTypeName;
extern std::map<std::string, std::string> PointPODTypeName;

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

  //! Return id of the pointBase
  Index id() const { return id_; }

  //! Status
  virtual bool status() const { return status_; }

  //! Initialise point from POD data
  //! \param[in] point POD data of point
  //! \retval status Status of reading POD point
  virtual bool initialise_point(PODPoint& point);

  //! Return point data as POD
  //! \retval point POD of the point
  virtual std::shared_ptr<void> pod() const;

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

  //! Initialise properties
  virtual void initialise();

  //! Return displacement of the point
  virtual VectorDim displacement() const { return displacement_; }

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

  //! Assign area
  //! \param[in] area Point area
  virtual bool assign_area(double area);

  //! Return area
  virtual double area() const { return area_; }

  //! Reinitialise point property
  //! \param[in] dt Time step size
  virtual void initialise_property(double dt) = 0;

  //! Compute updated position
  virtual void compute_updated_position(double dt) noexcept = 0;

  //! Type of point
  virtual std::string type() const = 0;

  //! Serialize
  //! \retval buffer Serialized buffer data
  virtual std::vector<uint8_t> serialize();

  //! Deserialize
  //! \param[in] buffer Serialized buffer data
  virtual void deserialize(const std::vector<uint8_t>& buffer);

  //! Assign penalty factor
  //! \param[in] constraint_type Constraint type, e.g. "fixed", "slip"
  //! \param[in] penalty_factor Penalty factor
  //! \param[in] normal_type Normal type, e.g. "cartesian", "assign", "auto"
  //! \param[in] normal_vector Normal vector
  virtual void assign_penalty_parameter(const std::string& constraint_type,
                                        double penalty_factor,
                                        const std::string& normal_type,
                                        const VectorDim& normal_vector) {
  };

  //! Apply point velocity constraints
  //! \param[in] dir Direction of point velocity constraint
  //! \param[in] velocity Applied point velocity constraint
  virtual void apply_point_velocity_constraints(unsigned dir, double velocity) {};

  //! Apply point kelvin voigt constraints
  //! \param[in] dir Direction of kelvin voigt constraint
  //! \param[in] delta Spring vs. Dashpot Weighting Parameter
  //! \param[in] h_min Characteristic length
  //! \param[in] incidence_a Incidence parameter a
  //! \param[in] incidence_b Incidence parameter b
  virtual void apply_point_kelvin_voigt_constraints(
      unsigned dir, double delta, double h_min, double incidence_a,
      double incidence_b) {};
  
  //! Map point stiffness matrix to cell
  virtual inline bool map_stiffness_matrix_to_cell(double newmark_beta,
  double newmark_gamma, double dt) {
    throw std::runtime_error(
        "Calling the base class function (map_stiffness_matrix_to_cell) in "
        "PointBase:: illegal operation!");
    return false;
  };

  //! Map enforcement boundary force to node
  //! \param[in] phase Index corresponding to the phase
  virtual void map_boundary_force(unsigned phase) {
    throw std::runtime_error(
        "Calling the base class function (map_boundary_force) in "
        "PointBase:: illegal operation!");
  };

 protected:
  //! Compute pack size
  //! \retval pack size of serialized object
  virtual int compute_pack_size() const;

 protected:
  //! pointBase id
  Index id_{std::numeric_limits<Index>::max()};
  //! coordinates
  VectorDim coordinates_;
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
  //! Area
  double area_{0.};
  //! Pack size
  unsigned pack_size_{0};
};  // PointBase class
}  // namespace mpm

#include "point_base.tcc"

#endif  // MPM_POINTBASE_H__