#ifndef MPM_LME_TRI_ELEMENT_H
#define MPM_LME_TRI_ELEMENT_H

#include "triangle_element.h"

namespace mpm {

//! Quadrilateral Local Maximum Entropy element class derived from Quadrilateral
//! \brief Quadrilateral LME element
//! \details quadrilateral LME element with adaptive number of nodes \n
//! Type of elements and sizes of shape function, gradient shape function,
//! B-matrix, indices are according to the number of nodes \n <pre>
//!
//! Type 1: Trinagular Element: nnodes = 36
//!
//!   30---------31----------32----------33----------34----------35
//!   |  \        |  \        |  \        |  \        |  \        |
//!   |    \      |    \      |    \      |    \      |    \      |
//!   |      \    |      \    |      \    |      \    |      \    |
//!   |        \  |        \  |        \  |        \  |        \  |
//!   24---------25----------26----------27----------28----------29
//!   |  \        |  \        |  \        |  \        |  \        |
//!   |    \      |    \      |    \      |    \      |    \      |
//!   |      \    |      \    |      \    |      \    |      \    |
//!   |        \  |        \  |        \  |        \  |        \  |
//!   20---------21-----------3-----------2----------22----------23
//!   |  \        |  \        |  \        |  \        |  \        |
//!   |    \      |    \      |  particle |    \      |    \      |
//!   |      \    |      \    |  location |      \    |      \    |
//!   |        \  |        \  |         \ |        \  |        \  |
//!   16---------17-----------0-----------1----------18----------19
//!   |  \        |  \        |  \        |  \        |  \        |
//!   |    \      |    \      |    \      |    \      |    \      |
//!   |      \    |      \    |      \    |      \    |      \    |
//!   |        \  |        \  |        \  |        \  |        \  |
//!   10---------11----------12----------13----------14----------15
//!   |  \        |  \        |  \        |  \        |  \        |
//!   |    \      |    \      |    \      |    \      |    \      |
//!   |      \    |      \    |      \    |      \    |      \    |
//!   |        \  |        \  |        \  |        \  |        \  |
//!   4-----------5-----------6-----------7-----------8-----------9
//!
//!
//! </pre>
//!
//!
//! \tparam Tdim Dimension
template <unsigned Tdim>
class TriangleLMEElement : public TriangleElement<2, 3> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define a matrix of size dimension
  using MatrixDim = Eigen::Matrix<double, Tdim, Tdim>;

  //! constructor with number of shape functions
  TriangleLMEElement() : TriangleElement<2, 3>() {
    static_assert(Tdim == 2, "Invalid dimension for a LME element");

    //! Logger
    std::string logger = "triangle_LME::<" + std::to_string(Tdim) + ">";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Evaluate shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn(const VectorDim& xi, VectorDim& lambda,
                          const MatrixDim& deformation_gradient) const override;

  //! Evaluate local shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn_local(
      const VectorDim& xi, VectorDim& lambda,
      const MatrixDim& deformation_gradient) const override;

  //! Evaluate gradient of shape functions
  //! \param[in] xi given local coordinates
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval grad_shapefn Gradient of shape function of a given cell
  Eigen::MatrixXd grad_shapefn(
      const VectorDim& xi, VectorDim& lambda,
      const MatrixDim& deformation_gradient) const override;

  //! Compute Jacobian
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& lambda, const MatrixDim& deformation_gradient) const override;

  //! Return the dN/dx at a given local coord
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  Eigen::MatrixXd dn_dx(const VectorDim& xi,
                        const Eigen::MatrixXd& nodal_coordinates,
                        VectorDim& lambda,
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
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian_local(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& lambda, const MatrixDim& deformation_gradient) const override;

  //! Evaluate the B matrix at given local coordinates for a real cell
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval bmatrix B matrix
  std::vector<Eigen::MatrixXd> bmatrix(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& lambda, const MatrixDim& deformation_gradient) const override;

  //! Return the type of shape function
  mpm::ShapefnType shapefn_type() const override {
    return (anisotropy_) ? mpm::ShapefnType::ALME : mpm::ShapefnType::LME;
  }

  //! Return number of shape functions
  unsigned nfunctions() const override { return nconnectivity_; }

  //! Return number of local shape functions
  unsigned nfunctions_local() const override { return 3; }

  //! Return if natural coordinates can be evaluates
  bool isvalid_natural_coordinates_analytical() const override { return false; }

  //! Compute Natural coordinates of a point (analytical)
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] point Location of the point in cell
  //! \retval xi Return the local coordinates
  VectorDim natural_coordinates_analytical(
      const VectorDim& point,
      const Eigen::MatrixXd& nodal_coordinates) const override;

  //! Assign nodal connectivity property for LME elements
  //! \param[in] beta Coldness function of the system in the range of [0,inf)
  //! \param[in] radius Support radius of the kernel
  //! \param[in] anisotropy Shape function anisotropy (F^{-T}F^{-1})
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  void initialise_lme_connectivity_properties(
      double beta, double radius, bool anisotropy,
      const Eigen::MatrixXd& nodal_coordinates) override;

  //! Return the degree of shape function
  mpm::ElementDegree degree() const override {
    return mpm::ElementDegree::Infinity;
  };

 private:
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Number of connectivity
  unsigned nconnectivity_{3};
  //! Beta parameter with range of [0,inf)
  double beta_;
  //! Support radius
  double support_radius_;
  //! Anisotropy parameter
  bool anisotropy_{false};
  //! Apply preconditioner
  bool preconditioner_{false};
  //! Nodal coordinates vector (n_connectivity_ x Tdim)
  Eigen::MatrixXd nodal_coordinates_;
};

}  // namespace mpm
#include "triangle_lme_element.tcc"

#endif  // MPM_LME_TRI_ELEMENT_H
