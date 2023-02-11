#ifndef MPM_LME_HEX_ELEMENT_H_
#define MPM_LME_HEX_ELEMENT_H_

#include "hexahedron_element.h"

namespace mpm {

//! Hexahedron LME element class derived from Hexahedron
//! \brief Hexahedron LME element
//! \details Hexahedron LME element with adaptive number of nodes \n

//! \tparam Tdim Dimension
template <unsigned Tdim>
class HexahedronLMEElement : public HexahedronElement<3, 8> {

 public:
  //! Define a vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define a matrix of size dimension
  using MatrixDim = Eigen::Matrix<double, Tdim, Tdim>;

  //! constructor with number of shape functions
  HexahedronLMEElement() : HexahedronElement<3, 8>() {
    static_assert(Tdim == 3, "Invalid dimension for a BSpline element");

    //! Logger
    std::string logger = "hex_LME::<" + std::to_string(Tdim) + ">";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Evaluate shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn(const VectorDim& xi, VectorDim& lambda,
                          const MatrixDim& deformation_gradient) const override;

  //! Evaluate gradient of shape functions
  //! \param[in] xi given local coordinates
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval grad_shapefn Gradient of shape function of a given cell
  Eigen::MatrixXd grad_shapefn(
      const VectorDim& xi, VectorDim& lambda,
      const MatrixDim& deformation_gradient) const override;

  //! Evaluate local shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn_local(
      const VectorDim& xi, VectorDim& lambda,
      const MatrixDim& deformation_gradient) const override;

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
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  Eigen::MatrixXd dn_dx_local(
      const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
      VectorDim& lambda, const MatrixDim& deformation_gradient) const override;

  //! Compute Jacobian
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian(
      const Eigen::Matrix<double, 3, 1>& xi,
      const Eigen::MatrixXd& nodal_coordinates,
      Eigen::Matrix<double, 3, 1>& lambda,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient) const override;

  //! Compute Jacobian local
  //! \param[in] xi given local coordinates
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \param[in] lambda Lagrange multiplier
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval jacobian Jacobian matrix
  Eigen::Matrix<double, Tdim, Tdim> jacobian_local(
      const Eigen::Matrix<double, 3, 1>& xi,
      const Eigen::MatrixXd& nodal_coordinates,
      Eigen::Matrix<double, 3, 1>& lambda,
      const Eigen::Matrix<double, 3, 3>& deformation_gradient) const override;

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
  unsigned nfunctions_local() const override { return 8; }

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

  //! Logger
  std::unique_ptr<spdlog::logger> console_;
  //! Number of connectivity
  unsigned nconnectivity_{8};
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
#include "hexahedron_lme_element.tcc"

#endif  // MPM_LME_HEX_ELEMENT_H_
