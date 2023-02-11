#ifndef MPM_BSPLINE_HEX_ELEMENT_H_
#define MPM_BSPLINE_HEX_ELEMENT_H_

#include "hexahedron_element.h"

namespace mpm {

//! Hexahedron BSpline element class derived from Hexahedron
//! \brief Hexahedron BSpline element
//! \details Hexahedron BSpline element, see online document for
//! details \n

//! \tparam Tdim Dimension
//! \tparam Tpolynomial Degree of BSpline Polynomial
template <unsigned Tdim, unsigned Tpolynomial>
class HexahedronBSplineElement : public HexahedronElement<3, 8> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define a matrix of size dimension
  using MatrixDim = Eigen::Matrix<double, Tdim, Tdim>;

  //! constructor with number of shape functions
  HexahedronBSplineElement() : HexahedronElement<3, 8>() {
    static_assert(Tdim == 3, "Invalid dimension for a BSpline element");
    static_assert((Tpolynomial == 2),
                  "Specified number of polynomial order is not defined");

    if (Tpolynomial == 2)
      BSplineKnotVector = {{-1.5, -0.5, 0.5, 1.5}, {0.0, 0.0, 0.0, 0.5},
                           {-1.0, -0.5, 0.5, 1.5}, {-1.5, -0.5, 0.5, 1.0},
                           {-0.5, 0.0, 0.0, 0.0},  {0.0, 0.0, 0.5, 1.5},
                           {-1.5, -0.5, 0.0, 0.0}};

    //! Logger
    std::string logger = "hex_bspline::<" + std::to_string(Tdim) + ", P" +
                         std::to_string(Tpolynomial) + ">";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Evaluate shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn(const VectorDim& xi, VectorDim& particle_size,
                          const MatrixDim& deformation_gradient) const override;

  //! Evaluate gradient of shape functions
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval grad_shapefn Gradient of shape function of a given cell
  Eigen::MatrixXd grad_shapefn(
      const VectorDim& xi, VectorDim& particle_size,
      const MatrixDim& deformation_gradient) const override;

  //! Evaluate local shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn_local(
      const VectorDim& xi, VectorDim& particle_size,
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

  //! Compute Jacobian
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian(
      const Eigen::Matrix<double, 3, 1>& xi,
      const Eigen::MatrixXd& nodal_coordinates,
      Eigen::Matrix<double, 3, 1>& particle_size,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient) const override;

  //! Compute Jacobian local
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian_local(
      const Eigen::Matrix<double, 3, 1>& xi,
      const Eigen::MatrixXd& nodal_coordinates,
      Eigen::Matrix<double, 3, 1>& particle_size,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient) const override;
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
  unsigned nfunctions_local() const override { return 8; }

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

  //! Compute B-Spline Basis Function Gradient using the recursive De Boor's
  //! algorithm for single direction
  //! \param[in] point_coord point coordinate in one direction
  //! \param[in] nodal_coord nodal coordinate in one direction dimension
  //! \param[in] node_type Node type associated with direction
  //! \param[in] poly_order Polynomial degree
  //! \param[in] index Index associated to local BSplineKnotVector
  double gradient(double point_coord, double nodal_coord, unsigned node_type,
                  unsigned poly_order, unsigned index = 0) const;

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

  //! Function to check if particle is lying on the region where kernel
  //! correction is necessary
  //! \param[in] xi given local coordinates
  bool kernel_correction_region(const VectorDim& xi) const;

  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Number of connectivity
  unsigned nconnectivity_{8};
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
#include "hexahedron_bspline_element.tcc"

#endif  // MPM_BSPLINE_HEX_ELEMENT_H_
