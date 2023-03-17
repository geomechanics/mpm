#ifndef MPM_TRIANGLE_ELEMENT_H_
#define MPM_TRIANGLE_ELEMENT_H_

#include "element.h"
#include "logger.h"

namespace mpm {

//! Triangle element class derived from Element class
//! \brief Triangle element
//! \details 3-noded and 6-noded triangle element \n
//! Shapte function, gradient shape function, B-matrix, indices \n
//! 3-node Triangle ELement \n
//! <pre>
//!   2 0
//!     |`\
//!     |  `\
//!     |    `\
//!     |      `\
//!     |        `\
//!   0 0----------0 1
//! </pre>
//! 6-node Triangle Element
//! <pre>
//!   2 0
//!     |`\
//!     |  `\
//!   5 0    `0 4
//!     |      `\
//!     |        `\
//!   0 0-----0----0 1
//!           3
//! </pre>
//!
//!
//! \tparam Tdim Dimension
//! \tparam Tnfunctions Number of functions
template <unsigned Tdim, unsigned Tnfunctions>
class TriangleElement : public Element<Tdim> {

 public:
  //! Define vector of size dimension
  using VectorDim = Eigen::Matrix<double, Tdim, 1>;

  //! Define a matrix of size dimension
  using MatrixDim = Eigen::Matrix<double, Tdim, Tdim>;

  //! constructor with number of shape functions
  TriangleElement() : mpm::Element<Tdim>() {
    static_assert(Tdim == 2, "Invalid dimension for a triangular element");
    static_assert((Tnfunctions == 3 || Tnfunctions == 6),
                  "Specified number of shape funcions is not defined");

    //! Logger
    std::string logger = "triangular::<" + std::to_string(Tdim) + ", " +
                         std::to_string(Tnfunctions) + ">";
    console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
  }

  //! Return number of shape functions
  unsigned nfunctions() const override { return Tnfunctions; }

  //! Return number of local shape functions
  unsigned nfunctions_local() const override { return Tnfunctions; }

  //! Evaluate shape functions at given local coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn Shape function of a given cell
  Eigen::VectorXd shapefn(const VectorDim& xi, VectorDim& particle_size,
                          const MatrixDim& deformation_gradient) const override;

  //! Evaluate local shape functions at given coordinates
  //! \param[in] xi given local coordinates
  //! \param[in] particle_size Particle size
  //! \param[in] deformation_gradient Deformation gradient
  //! \retval shapefn_local Shape function of a given cell
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

  //! Return the degree of shape function
  mpm::ElementDegree degree() const override;

  //! Return the type of shape function
  mpm::ShapefnType shapefn_type() const override {
    return mpm::ShapefnType::NORMAL_MPM;
  }

  //! Return nodal coordinates of a unit cell
  Eigen::MatrixXd unit_cell_coordinates() const override;

  //! Return the side indices of a cell to calculate the cell length
  //! \retval indices Outer-indices that form the sides of the cell
  Eigen::MatrixXi sides_indices() const override;

  //! Return the corner indices of a cell to calculate the cell volume
  //! \retval indices Outer-indices that form the cell
  Eigen::VectorXi corner_indices() const override;

  //! Return indices of a face of an element
  //! \param[in] face_id given id of the face
  //! \retval indices Indices that make the face
  Eigen::VectorXi face_indices(unsigned face_id) const override;

  //! Return the number of faces in a triangle
  unsigned nfaces() const override { return 3; }

  //! Return unit element length
  double unit_element_length() const override { return 1.; }

  //! Return quadrature of the element
  std::shared_ptr<mpm::Quadrature<Tdim>> quadrature(
      unsigned nquadratures = 1) const override;

  //! Compute volume
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  //! \retval volume Return the volume of cell
  double compute_volume(
      const Eigen::MatrixXd& nodal_coordinates) const override;

  //! Return if natural coordinates can be evaluates
  bool isvalid_natural_coordinates_analytical() const override;

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

  //! Assign nodal connectivity property for LME elements
  //! \param[in] beta Coldness function of the system in the range of [0,inf)
  //! \param[in] radius Support radius of the kernel
  //! \param[in] anisotropy Shape function anisotropy (F^{-T}F^{-1})
  //! \param[in] nodal_coordinates Coordinates of nodes forming the cell
  void initialise_lme_connectivity_properties(
      double beta, double radius, bool anisotropy,
      const Eigen::MatrixXd& nodal_coordinates) override;

 private:
  //! Logger
  std::unique_ptr<spdlog::logger> console_;
};

}  // namespace mpm
#include "triangle_element.tcc"

#endif  // MPM_TRIANGLE_ELEMENT_H_
