#ifndef MPM_DISCONTINUITY_H_
#define MPM_DISCONTINUITY_H_

#include "cell.h"
#include "data_types.h"
#include "discontinuity_element.h"
#include "io_mesh.h"
#include "logger.h"
#include "memory.h"
#include "node_base.h"
#include "vector.h"
#include <iostream>

namespace mpm {

template <unsigned Tdim>
struct discontinuity_point;

//! Class for describe the discontinuous surface
//! \brief
//! \tparam Tdim Dimension
template <unsigned Tdim>
class DiscontinuityBase {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;
  //! Constructor with id
  //! \param[in] discontinuity id
  //! \param[in] discontinuity properties json
  DiscontinuityBase(const Json& discontinuity_props, unsigned id);

  //! Constructor with id
  //! \param[in] discontinuity id
  //! \param[in] initiation properties: store the properties fot each newly
  //! generated discontinuity: cohesion, friction_coef, contact_distance, width,
  //! maximum_pdstrain move_direction, friction_coef_average
  DiscontinuityBase(unsigned id,
                    std::tuple<double, double, double, double, double, int,
                               bool>& initiation_property);

  //! Destructor
  virtual ~DiscontinuityBase(){};

  //! Delete copy constructor
  DiscontinuityBase(const DiscontinuityBase<Tdim>&) = delete;

  //! Delete assignement operator
  DiscontinuityBase& operator=(const DiscontinuityBase<Tdim>&) = delete;

  //！ Initialization
  //! \param[in] the coordinates of all points
  //! \param[in] the point index of each surface
  virtual bool initialise(
      const std::vector<VectorDim>& points,
      const std::vector<std::vector<mpm::Index>>& surfs) = 0;

  //! Create points from file
  //! \param[in] points the coordinates list of points
  bool create_points(const std::vector<VectorDim>& points);

  //! Create elements from file
  //! \param[in] surfs the point index list of each surface
  virtual bool create_surfaces(
      const std::vector<std::vector<mpm::Index>>& surfs) {
    return true;
  };

  // Return the levelset values of each coordinates
  //! \param[in] coordinates coordinates
  //! \param[in] phi_list the reference of phi for all coordinates
  virtual void compute_levelset(const VectorDim& coordinates,
                                double& phi_particle) = 0;

  //! Compute the normal vectors of coordinates
  //! \param[in] coordinates The coordinates
  //! \param[in] normal vector the normal vector of the given coordinates
  virtual void compute_normal(const VectorDim& coordinates,
                              VectorDim& normal_vector) = 0;

  //! Return self_contact
  //! \retval compute the self-contact force
  bool self_contact() const { return self_contact_; };

  //! Return the friction coefficient
  //! \retval the friction coeffcient of this discontinuity
  double friction_coef() const { return friction_coef_; };

  //! Return the cohesion
  //! \retval the surface cohesion
  double cohesion() const { return cohesion_; };

  //! Return the width
  //! \retval the width of the dsicontinuity
  double width() const { return width_; }

  //! Return the contact_distance
  //! \retval the contact distance for the contact detection
  double contact_distance() const { return contact_distance_; }

  //! Return the maximum_pdstrain
  //! \retval the critical value of the pdstrain
  double maximum_pdstrain() const { return maximum_pdstrain_; }

  //! Return the number of the points
  //! \retval the numberof the mark points
  mpm::Index npoints() const { return points_.size(); };

  //! Locate points in a cell
  //! \param[in] cells vector of cells
  //! \param[in] map_cells map of cells
  void locate_discontinuity_mesh(const Vector<Cell<Tdim>>& cells,
                                 const Map<Cell<Tdim>>& map_cells) noexcept;

  //! Compute updated position
  //! \param[in] dt Time-step
  virtual void compute_updated_position(const double dt) noexcept;

  //! Compute shape function
  void compute_shapefn() noexcept;

  //! Assign point friction coefficient
  virtual void assign_point_friction_coef() noexcept = 0;

  //! Insert new point
  void insert_particles(VectorDim& point, const Vector<Cell<Tdim>>& cells,
                        const Map<Cell<Tdim>>& map_cells);

  //! TODO: to do:output mark points
  void output_markpoints(int step);

  //! Return propagation
  bool propagation() { return propagation_; }

  //! Return description type
  std::string description_type() { return description_type_; }

 protected:
  //! Id
  int id_;
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Vector of points
  std::vector<mpm::discontinuity_point<Tdim>> points_;
  //! Self-contact
  bool self_contact_{true};
  //! Friction coefficient
  double friction_coef_{0};
  //! Cohesion
  double cohesion_{0};
  //! The width of the discontinuity
  double width_{std::numeric_limits<double>::max()};
  //! Move_direction
  int move_direction_{1};
  //! Contact distance, consider the particle size
  double contact_distance_{std::numeric_limits<double>::max()};
  //! Maximum pdstrain
  double maximum_pdstrain_{0};
  //! DescriptionType at the begining
  std::string description_type_;
  //! Compute the average friction coefficient from the neighbour particles
  bool friction_coef_average_{false};
  //! Proparate or not
  bool propagation_{false};
};  // DiscontinuityBase class

//! Struct of discontinuity point
template <unsigned Tdim>
struct discontinuity_point {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct with coordinate
  discontinuity_point(const VectorDim& coordinate) {
    friction_coef_ = 0;
    cohesion_ = 0;
    coordinates_ = coordinate;
    cell_ = nullptr;
    //! Logger
    console_ = spdlog::get("discontinuity_point");
  }

  //! Return coordinates
  //! \retval return coordinates
  VectorDim coordinates() const { return coordinates_; }

  //! Return cell_id
  //! \retval return cell_id_
  Index cell_id() const { return cell_id_; }

  //! Assign a cell to point
  //! \param[in] cellptr Pointer to a cell
  //! \param[in] xi Local coordinates of the point in reference cell
  bool assign_cell_xi(const std::shared_ptr<Cell<Tdim>>& cellptr,
                      const Eigen::Matrix<double, Tdim, 1>& xi);

  //! Return cell ptr status
  bool cell_ptr() const { return cell_ != nullptr; }

  //! Assign a cell to point
  //! \param[in] cellptr Pointer to a cell
  bool assign_cell(const std::shared_ptr<Cell<Tdim>>& cellptr);

  //! Assign the discontinuity type to cell
  //! \param[in] map_cells map of cells
  //! \param[in] the discontinuity id
  void assign_cell_enrich(const Map<Cell<Tdim>>& map_cells, unsigned dis_id);

  //! Compute reference coordinates in a cell
  bool compute_reference_location() noexcept;

  //! Locate particles in a cell
  //! \param[in] cells vector of cells
  //! \param[in] map_cells map of cells
  //! \param[in] the discontinuity id
  void locate_discontinuity_mesh(const Vector<Cell<Tdim>>& cells,
                                 const Map<Cell<Tdim>>& map_cells,
                                 unsigned dis_id) noexcept;

  //! Compute updated position
  void compute_updated_position(double dt, int move_direction) noexcept;

  //! Compute shape function
  void compute_shapefn() noexcept;

  //! Assign point friction coefficient
  //! \param[in] friction_coef
  void assign_friction_coef(double friction_coef) noexcept {
    friction_coef_ = friction_coef;
  }

  //! Assign cohesion
  //! \param[in] friction_coef
  void assign_cohesion(double cohesion) noexcept { cohesion_ = cohesion; }

  //! Assign tip
  //! \param[in] tip
  void assign_tip(bool tip) { tip_ = tip; }

 private:
  //! point coordinates
  VectorDim coordinates_;
  //! Cell id
  Index cell_id_{std::numeric_limits<Index>::max()};
  //! Shape functions
  Eigen::VectorXd shapefn_;
  //! Cell
  std::shared_ptr<Cell<Tdim>> cell_;
  //! Reference coordinates (in a cell)
  Eigen::Matrix<double, Tdim, 1> xi_;
  //! Vector of nodal pointers
  std::vector<std::shared_ptr<NodeBase<Tdim>>> nodes_;
  //! Logger
  std::shared_ptr<spdlog::logger> console_;
  //! friction coefficient
  double friction_coef_{0.};
  //! cohesion
  double cohesion_{0.};
  //! tip
  bool tip_{false};
};

//! To do: Struct of discontinuity line: for 2d
struct discontinuity_line {
 public:
  //! Return points indices
  Eigen::Matrix<int, 2, 1> points() const { return points_; };

 private:
  //! points index of the line
  Eigen::Matrix<int, 2, 1> points_;
};

//! Struct of discontinuity surface: triangle
template <unsigned Tdim>
struct discontinuity_surface {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Construct with points indices
  discontinuity_surface(const std::vector<mpm::Index>& points) {
    for (int i = 0; i < 3; ++i) points_[i] = points[i];
  }
  //! Return points indices
  Eigen::Matrix<mpm::Index, 3, 1> points() const { return points_; }

  //! Assign the surface center
  //! \param[in] center coordinates of the surface center
  inline void assign_center(VectorDim& center) { center_ = center; }

  //! Assign the surface normal vector
  //! \param[in] normal normal vector of the surface
  inline void assign_normal(VectorDim& normal) { normal_ = normal; }

  //! Return normal of the elements
  VectorDim normal() const { return normal_; }

  //! Return the vertical distance to the surface
  //! \param[in]  coor coordinates
  double vertical_distance(const VectorDim& coor) const {
    return (coor[0] - center_[0]) * normal_[0] +
           (coor[1] - center_[1]) * normal_[1] +
           (coor[2] - center_[2]) * normal_[2];
  };

  //! Return the vertical distance to the surface
  //! \param[in]  coor coordinates
  double ptocenter_distance(const VectorDim& coor) const {
    return (coor - center_).norm();
  };

 private:
  //！ The discontinuity id
  unsigned id_;
  //! Points indices
  Eigen::Matrix<mpm::Index, 3, 1> points_;
  //！ The center coordinates
  VectorDim center_;
  //！ The normal vector
  VectorDim normal_;
};

}  // namespace mpm

#include "discontinuity_base.tcc"
#include "discontinuity_point.tcc"

#endif  // MPM_DiscontinuityBase_H_
