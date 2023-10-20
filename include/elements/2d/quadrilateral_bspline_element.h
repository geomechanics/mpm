#ifndef MPM_BSPLINE_ELEMENT_H_
#define MPM_BSPLINE_ELEMENT_H_

#include "quadrilateral_element.h"

namespace mpm {

//! Quadrilateral BSpline element class derived from Quadrilateral
//! \brief Quadrilateral BSpline element
//! \details quadrilateral BSpline element with adaptive number of nodes \n
//! Type of elements and sizes of shape function, gradient shape function,
//! B-matrix, indices are according to the number of nodes \n <pre>
//!
//! Type 1: Regular Element: nnodes = 16
//!
//!   13----------12----------11----------10
//!   |           |           |           |
//!   |           |           |           |
//!   |           |           |           |
//!   |           |           |           |
//!   14----------3-----------2-----------9
//!   |           |           |           |
//!   |           | particle  |           |
//!   |           | location  |           |
//!   |           |           |           |
//!   15----------0-----------1-----------8
//!   |           |           |           |
//!   |           |           |           |
//!   |           |           |           |
//!   |           |           |           |
//!   4-----------5-----------6-----------7
//!
//! Type 2: Boundary Element: nnodes < 16 (cell located at the mesh boundary)
//! e.g. Mesh edge
//!
//!   9-----------8-----------7-----------6
//!   |           |           |           |
//!   |           |           |           |
//!   |           |           |           |
//!   |           |           |           |
//!   10----------3-----------2-----------5
//!   |           |           |           |
//!   |           | particle  |           |
//!   |           | location  |           |
//!   |           |           |           |
//!   11----------0-----------1-----------4
//!
//! e.g. Mesh corner
//!
//!   8-----------7-----------6
//!   |           |           |
//!   |           |           |
//!   |           |           |
//!   |           |           |
//!   3-----------2-----------5
//!   |           |           |
//!   | particle  |           |
//!   | location  |           |
//!   |           |           |
//!   0-----------1-----------4
//!
//! </pre>
//!
//!
//! \tparam Tdim Dimension
//! \tparam Tpolynomial Degree of BSpline Polynomial
template <unsigned Tdim, unsigned Tpolynomial>
class QuadrilateralBSplineElement : public QuadrilateralElement<2, 4> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define a matrix of size dimension
  using MatrixDim = Eigen::Matrix<double, Tdim, Tdim>;

  //! constructor with number of shape functions
  QuadrilateralBSplineElement() : QuadrilateralElement<2, 4>() {
    static_assert(Tdim == 2, "Invalid dimension for a BSpline element");
    static_assert((Tpolynomial == 2),
                  "Specified number of polynomial order is not defined");

    if (Tpolynomial == 2)
      BSplineKnotVector = {{-1.5, -0.5, 0.5, 1.5}, {0.0, 0.0, 0.0, 0.5},
                           {-1.0, -0.5, 0.5, 1.5}, {-1.5, -0.5, 0.5, 1.0},
                           {-0.5, 0.0, 0.0, 0.0},  {0.0, 0.0, 0.5, 1.5},
                           {-1.5, -0.5, 0.0, 0.0}};

    //! Logger
    std::string logger = "quadrilateral_bspline::<" + std::to_string(Tdim) +
                         ", P" + std::to_string(Tpolynomial) + ">";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Evaluate shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn(const VectorDim& xi, VectorDim& particle_size,
                          const MatrixDim& deformation_gradient) const override;

  //! Evaluate local shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn_local(
      const VectorDim& xi, VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const override;

  //! Evaluate gradient of shape functions
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval grad_shapefn Gradient of shape function of a given cell
  Eigen::MatrixXd grad_shapefn(
      const VectorDim& xi, VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const override;

  //! Compute Jacobian
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const override;

  //! Return the dN/dx at a given local coord
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  Eigen::MatrixXd dn_dx(const VectorDim& xi,
                        const Eigen::MatrixXd& nodal_coordinates,
                        VectorDim& particle_size,
                        const MatrixDim& deformation_gradient) const override;

  //! Return the local dN/dx at a given local coord
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  Eigen::MatrixXd dn_dx_local(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const override;

  //! Compute Jacobian local
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian_local(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const override;

  //! Evaluate the B matrix at given local coordinates for a real cell
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval bmatrix B matrix
  std::vector<Eigen::MatrixXd> bmatrix(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const override;

  //! Return the type of shape function
  mpm::ShapefnType shapefn_type() const override {
    return mpm::ShapefnType::BSPLINE;
  }

  //! Return number of shape functions
  unsigned nfunctions() const override { return nconnectivity_; }

  //! Return number of local shape functions
  unsigned nfunctions_local() const override { return 4; }

  //! Return if natural coordinates can be evaluates
  bool isvalid_natural_coordinates_analytical() const override { return false; }

  //! Compute Natural coordinates of a point (analytical)
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] point Location of the point in cell
  //! \retval xi Return the local coordinates
  VectorDim natural_coordinates_analytical(
      const VectorDim& point,
      const Eigen::MatrixXd& nodal_coordinates) const override;

  //! Assign nodal connectivity property for bspline elements
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] nodal_properties Vector determining node type for each
  //! dimension
  //! \param[in] kernel_correction Apply Kernel correction at the boundary
  void initialise_bspline_connectivity_properties(
      const Eigen::MatrixXd& nodal_coordinates,
      const std::vector<std::vector<unsigned>>& nodal_properties,
      bool kernel_correction = false) override;

  //! Return the degree of shape function
  mpm::ElementDegree degree() const override {
    return mpm::ElementDegree::Quadratic;
  };

 private:
  //! Compute B-Spline Basis Function using the recursive De Boor's algorithm
  //! for single direction
  //! \param[in] point_coord point coordinate in one direction
  //! \param[in] nodal_coord nodal coordinate in one direction
  //! dimension
  //! \param[in] node_type Node type associated with direction
  //! \param[in] poly_order Polynomial degree
  //! \param[in] index Index associated to local BSplineKnotVector
  double kernel(double point_coord, double nodal_coord, unsigned node_type,
                unsigned poly_order, unsigned index = 0) const;

  //! Compute B-Spline Basis Function using the close-form equation
  //! \param[in] point_coord point coordinate in one direction
  //! \param[in] nodal_coord nodal coordinate in one direction
  //! dimension
  double kernel(double point_coord, double nodal_coord) const;

  //! Compute B-Spline Basis Function Gradient using the recursive De Boor's
  //! algorithm for single direction
  //! \param[in] point_coord point coordinate in one direction
  //! \param[in] nodal_coord nodal coordinate in one direction dimension
  //! \param[in] node_type Node type associated with direction
  //! \param[in] poly_order Polynomial degree
  //! \param[in] index Index associated to local BSplineKnotVector
  double gradient(double point_coord, double nodal_coord, unsigned node_type,
                  unsigned poly_order, unsigned index = 0) const;

  //! Compute B-Spline Basis Function Gradient using the close-form equation
  //! \param[in] point_coord point coordinate in one direction
  //! \param[in] nodal_coord nodal coordinate in one direction dimension
  double gradient(double point_coord, double nodal_coord) const;

  //! Function that returns BSpline knot vector
  //! The order of the vectors are:
  //! Regular = 0,
  //! LowerBoundary = 1,
  //! LowerIntermediate = 2,
  //! UpperIntermediate = 3
  //! UpperBoundary = 4,
  //! LowerBoundaryVirtual = 5, (automatically defined)
  //! UpperBoundaryVirtual = 6 (automatically defined)
  std::vector<double> knot(unsigned node_type) const {
    return BSplineKnotVector[node_type];
  }

  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Number of connectivity
  unsigned nconnectivity_{4};
  //! Spacing length
  double spacing_length_;
  //! Nodal coordinates vector (n_connectivity_ x Tdim)
  Eigen::MatrixXd nodal_coordinates_;
  //! Nodal type matrix (n_connectivity_ x Tdim)
  std::vector<std::vector<unsigned>> node_type_;
  //! BSpline knot vector for different node type
  std::vector<std::vector<double>> BSplineKnotVector;
  //! Boolean to identify kernel correction
  bool kernel_correction_{false};
};

}  // namespace mpm
#include "quadrilateral_bspline_element.tcc"

#endif  // MPM_BSPLINE_ELEMENT_H_