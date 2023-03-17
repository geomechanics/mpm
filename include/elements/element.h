#ifndef MPM_ELEMENT_H_
#define MPM_ELEMENT_H_

#include <exception>
#include <map>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "factory.h"
#include "quadrature.h"

namespace mpm {

// Degree of Element
enum ElementDegree { Linear = 1, Quadratic = 2, Infinity = 99 };

// Element Shapefn
enum ShapefnType {
  NORMAL_MPM = 1,
  GIMP = 2,
  CPDI = 3,
  BSPLINE = 4,
  LME = 5,
  ALME = 6
};

//! Base class of shape functions
//! \brief Base class that stores the information about shape functions
//! \tparam Tdim Dimension
template <unsigned Tdim>
class Element {
 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define a matrix of size dimension
  using MatrixDim = Eigen::Matrix<double, Tdim, Tdim>;

  //! Constructor
  //! Assign variables to zero
  Element() = default;

  //! Destructor
  virtual ~Element() {}

  //! Return number of shape functions
  virtual unsigned nfunctions() const = 0;

  //! Return number of local shape functions
  virtual unsigned nfunctions_local() const = 0;

  //! Evaluate shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  virtual Eigen::VectorXd shapefn(
      const VectorDim& xi, VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Evaluate local shape functions at given coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  virtual Eigen::VectorXd shapefn_local(
      const VectorDim& xi, VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Evaluate gradient of shape functions
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  virtual Eigen::MatrixXd grad_shapefn(
      const VectorDim& xi, VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Compute Jacobian
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  virtual Eigen::Matrix<double, Tdim, Tdim> jacobian(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Compute Jacobian local
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  virtual Eigen::Matrix<double, Tdim, Tdim> jacobian_local(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Return the dN/dx at a given local coord
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  virtual Eigen::MatrixXd dn_dx(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Return the local dN/dx at a given local coord
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  virtual Eigen::MatrixXd dn_dx_local(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Evaluate the B matrix at given local coordinates for a real cell
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval bmatrix B matrix
  virtual std::vector<Eigen::MatrixXd> bmatrix(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const = 0;

  //! Return the degree of element
  virtual mpm::ElementDegree degree() const = 0;

  //! Return the shapefn type of element
  virtual mpm::ShapefnType shapefn_type() const = 0;

  //! Return nodal coordinates of a unit cell
  virtual Eigen::MatrixXd unit_cell_coordinates() const = 0;

  //! Return the side indices of a cell to calculate the cell length
  //! \retval indices Outer-indices that form the sides of the cell
  virtual Eigen::MatrixXi sides_indices() const = 0;

  //! Return the corner indices of a cell to calculate the cell volume
  //! \retval indices Outer-indices that form the cell
  virtual Eigen::VectorXi corner_indices() const = 0;

  //! Return indices of a face of an element
  //! \param[in] face_id given id of the face
  //! \retval indices Indices that make the face
  virtual Eigen::VectorXi face_indices(unsigned face_id) const = 0;

  //! Return number of faces
  virtual unsigned nfaces() const = 0;
  //! Return unit element length
  virtual double unit_element_length() const = 0;

  //! Return quadrature of the element
  virtual std::shared_ptr<mpm::Quadrature<Tdim>> quadrature(
      unsigned nquadratures) const = 0;

  //! Compute volume
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \retval volume Return the volume of cell
  virtual double compute_volume(
      const Eigen::MatrixXd& nodal_coordinates) const = 0;

  //! Return if natural coordinates can be evaluates
  virtual bool isvalid_natural_coordinates_analytical() const = 0;

  //! Compute Natural coordinates of a point (analytical)
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] point Location of the point in cell
  //! \retval xi Return the local coordinates
  virtual VectorDim natural_coordinates_analytical(
      const VectorDim& point,
      const Eigen::MatrixXd& nodal_coordinates) const = 0;

  //! Assign nodal connectivity property for bspline elements
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] nodal_properties Vector determining node type for each
  //! dimension
  //! \param[in] kernel_correction Apply Kernel correction at the boundary
  virtual void initialise_bspline_connectivity_properties(
      const Eigen::MatrixXd& nodal_coordinates,
      const std::vector<std::vector<unsigned>>& nodal_properties,
      bool kernel_correction = false) = 0;

  //! Assign nodal connectivity property for LME elements
  //! \param[in] beta Coldness function of the system in the range of [0,inf)
  //! \param[in] radius Support radius of the kernel
  //! \param[in] anisotropy Shape function anisotropy (F^{-T}F^{-1})
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  virtual void initialise_lme_connectivity_properties(
      double beta, double radius, bool anisotropy,
      const Eigen::MatrixXd& nodal_coordinates) = 0;
};

}  // namespace mpm

#endif  // MPM_ELEMENT_H_
