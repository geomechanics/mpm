// 4-node (Linear) Tetrahedron Element
//!       3
//!       *
//!      /|\
//!     / | \
//!    /  |  \
//! 2 *. -|- .* 1
//!      `*´
//!       0

//! Return shape function of a 4-noded tetrahedron, with particle size and
//! deformation gradient
//! \param[in] xi Coordinates of point of interest \retval
//! shapefn Shape function of a given cell
template <>
inline Eigen::VectorXd mpm::TetrahedronElement<3, 4>::shapefn(
    const Eigen::Matrix<double, 3, 1>& xi,
    Eigen::Matrix<double, 3, 1>& particle_size,
    const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  // 8-noded
  Eigen::Matrix<double, 4, 1> shapefn;
  shapefn(0) = 1 - xi(0) - xi(1) - xi(2);
  shapefn(1) = xi(0);
  shapefn(2) = xi(1);
  shapefn(3) = xi(2);
  return shapefn;
}

//! Return gradient of shape functions of a 4-noded tetrahedron, with particle
//! size and deformation gradient
//! \param[in] xi Coordinates of point of interest
//! \retval grad_shapefn Gradient of shape function of a given cell
template <>
inline Eigen::MatrixXd mpm::TetrahedronElement<3, 4>::grad_shapefn(
    const Eigen::Matrix<double, 3, 1>& xi,
    Eigen::Matrix<double, 3, 1>& particle_size,
    const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  Eigen::Matrix<double, 4, 3> grad_shapefn;  //(Nk, xi(k))
  grad_shapefn(0, 0) = -1;
  grad_shapefn(1, 0) = 1;
  grad_shapefn(2, 0) = 0;
  grad_shapefn(3, 0) = 0;

  grad_shapefn(0, 1) = -1;
  grad_shapefn(1, 1) = 0;
  grad_shapefn(2, 1) = 1;
  grad_shapefn(3, 1) = 0;

  grad_shapefn(0, 2) = -1;
  grad_shapefn(1, 2) = 0;
  grad_shapefn(2, 2) = 0;
  grad_shapefn(3, 2) = 1;

  return grad_shapefn;
}

//! Return nodal coordinates of a unit cell
template <>
inline Eigen::MatrixXd mpm::TetrahedronElement<3, 4>::unit_cell_coordinates()
    const {
  // Coordinates of a unit cell
  Eigen::Matrix<double, 4, 3> unit_cell;
  // clang-format off
  unit_cell << 0., 0., 0.,
               1., 0., 0.,
               0., 1., 0.,
               0., 0., 1.;
    // cppcheck-suppress *
  // clang-format on
  return unit_cell;
}

//! Return local shape functions of a Tetrahedron Element at a given local
//! coordinate, with particle size and deformation gradient
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::VectorXd
    mpm::TetrahedronElement<Tdim, Tnfunctions>::shapefn_local(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {
  return this->shapefn(xi, particle_size, deformation_gradient);
}

//! Compute Jacobian
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::TetrahedronElement<Tdim, Tnfunctions>::jacobian(
        const Eigen::Matrix<double, 3, 1>& xi,
        const Eigen::MatrixXd& nodal_coordinates,
        Eigen::Matrix<double, 3, 1>& particle_size,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  // Get gradient shape functions
  const Eigen::MatrixXd grad_shapefn =
      this->grad_shapefn(xi, particle_size, deformation_gradient);
  try {
    // Check if dimensions are correct
    if ((grad_shapefn.rows() != nodal_coordinates.rows()) ||
        (xi.size() != nodal_coordinates.cols()))
      throw std::runtime_error(
          "Jacobian calculation: Incorrect dimension of xi and "
          "nodal_coordinates");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return Eigen::Matrix<double, Tdim, Tdim>::Zero();
  }

  // Jacobian
  return (grad_shapefn.transpose() * nodal_coordinates);
}

//! Compute Jacobian local with particle size and deformation gradient
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::TetrahedronElement<Tdim, Tnfunctions>::jacobian_local(
        const Eigen::Matrix<double, 3, 1>& xi,
        const Eigen::MatrixXd& nodal_coordinates,
        Eigen::Matrix<double, 3, 1>& particle_size,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  // Jacobian dx_i/dxi_j
  return this->jacobian(xi, nodal_coordinates, particle_size,
                        deformation_gradient);
}

//! Compute Jacobian
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::MatrixXd mpm::TetrahedronElement<Tdim, Tnfunctions>::dn_dx(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  Eigen::MatrixXd grad_sf =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  // Jacobian dx_i/dxi_j
  Eigen::Matrix<double, Tdim, Tdim> jacobian =
      (grad_sf.transpose() * nodal_coordinates);

  // Gradient shapefn of the cell
  // dN/dx = [J]^-1 * dN/dxi
  return grad_sf * (jacobian.inverse()).transpose();
}

//! Compute local dn_dx
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::MatrixXd mpm::TetrahedronElement<Tdim, Tnfunctions>::dn_dx_local(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  return this->dn_dx(xi, nodal_coordinates, particle_size,
                     deformation_gradient);
}

//! Compute Bmatrix
template <unsigned Tdim, unsigned Tnfunctions>
inline std::vector<Eigen::MatrixXd>
    mpm::TetrahedronElement<Tdim, Tnfunctions>::bmatrix(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  Eigen::MatrixXd grad_sf =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  // B-Matrix
  std::vector<Eigen::MatrixXd> bmatrix;
  bmatrix.reserve(Tnfunctions);

  try {
    // Check if matrices dimensions are correct
    if ((grad_sf.rows() != nodal_coordinates.rows()) ||
        (xi.rows() != nodal_coordinates.cols()))
      throw std::runtime_error(
          "BMatrix - Jacobian calculation: Incorrect dimension of xi and "
          "nodal_coordinates");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return bmatrix;
  }

  // Jacobian dx_i/dxi_j
  Eigen::Matrix<double, Tdim, Tdim> jacobian =
      (grad_sf.transpose() * nodal_coordinates);

  // Gradient shapefn of the cell
  // dN/dx = [J]^-1 * dN/dxi
  Eigen::MatrixXd grad_shapefn = grad_sf * (jacobian.inverse()).transpose();

  for (unsigned i = 0; i < Tnfunctions; ++i) {
    // clang-format off
    Eigen::Matrix<double, 6, Tdim> bi;
    bi(0, 0) = grad_shapefn(i, 0); bi(0, 1) = 0.;                 bi(0, 2) = 0.;
    bi(1, 0) = 0.;                 bi(1, 1) = grad_shapefn(i, 1); bi(1, 2) = 0.;
    bi(2, 0) = 0.;                 bi(2, 1) = 0.;                 bi(2, 2) = grad_shapefn(i, 2);
    bi(3, 0) = grad_shapefn(i, 1); bi(3, 1) = grad_shapefn(i, 0); bi(3, 2) = 0.;
    bi(4, 0) = 0.;                 bi(4, 1) = grad_shapefn(i, 2); bi(4, 2) = grad_shapefn(i, 1);
    bi(5, 0) = grad_shapefn(i, 2); bi(5, 1) = 0.;                 bi(5, 2) = grad_shapefn(i, 0);
    // clang-format on
    bmatrix.push_back(bi);
  }
  return bmatrix;
}

//! Return the degree of element
//! 4-noded tetrahedron
template <>
inline mpm::ElementDegree mpm::TetrahedronElement<3, 4>::degree() const {
  return mpm::ElementDegree::Linear;
}

//! Return the indices of a cell sides
//! \retval indices Sides that form the cell
//! \tparam Tdim Dimension
//! \tparam Tnfunctions Number of shape functions
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::MatrixXi
    mpm::TetrahedronElement<Tdim, Tnfunctions>::sides_indices() const {
  Eigen::Matrix<int, 6, 2> indices;
  // clang-format off
  indices << 0, 1,
             1, 2,
             2, 3,
             3, 0,
             1, 3,
             0, 2;
  // clang-format on
  return indices;
}

//! Return the corner indices of a cell to calculate the cell volume
//! \retval indices Outer-indices that form the cell
//! \tparam Tdim Dimension
//! \tparam Tnfunctions Number of shape functions
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::VectorXi
    mpm::TetrahedronElement<Tdim, Tnfunctions>::corner_indices() const {
  Eigen::Matrix<int, 4, 1> indices;
  // cppcheck-suppress *
  indices << 0, 1, 2, 3;
  return indices;
}

//! Return indices of a face of the element
//! 4-noded tetrahedron
template <>
inline Eigen::VectorXi mpm::TetrahedronElement<3, 4>::face_indices(
    unsigned face_id) const {

  //! Face ids and its associated nodal indices
  const std::map<unsigned, Eigen::Matrix<int, 3, 1>> face_indices_tetrahedron{
      {0, Eigen::Matrix<int, 3, 1>(0, 1, 2)},
      {1, Eigen::Matrix<int, 3, 1>(0, 1, 3)},
      {2, Eigen::Matrix<int, 3, 1>(0, 2, 3)},
      {3, Eigen::Matrix<int, 3, 1>(1, 2, 3)}};

  return face_indices_tetrahedron.at(face_id);
}

//! Return quadrature
template <unsigned Tdim, unsigned Tnfunctions>
inline std::shared_ptr<mpm::Quadrature<Tdim>>
    mpm::TetrahedronElement<Tdim, Tnfunctions>::quadrature(
        unsigned nquadratures) const {
  switch (nquadratures) {
    case 1:
      return Factory<mpm::Quadrature<Tdim>>::instance()->create("QTET1");
      break;
    case 2:
      return Factory<mpm::Quadrature<Tdim>>::instance()->create("QTET2");
      break;
    default:
      return Factory<mpm::Quadrature<Tdim>>::instance()->create("QTET1");
      break;
  }
}

//! Compute volume
//! \param[in] nodal_coordinates Coordinates of nodes forming the cell
//! \retval volume Return the volume of cell
template <unsigned Tdim, unsigned Tnfunctions>
inline double mpm::TetrahedronElement<Tdim, Tnfunctions>::compute_volume(
    const Eigen::MatrixXd& nodal_coordinates) const {

  // Temporary variables for node coordinants (xk, yk, zk)
  const double x1 = nodal_coordinates(0, 0);
  const double x2 = nodal_coordinates(1, 0);
  const double x3 = nodal_coordinates(2, 0);
  const double x4 = nodal_coordinates(3, 0);
  const double y1 = nodal_coordinates(0, 1);
  const double y2 = nodal_coordinates(1, 1);
  const double y3 = nodal_coordinates(2, 1);
  const double y4 = nodal_coordinates(3, 1);
  const double z1 = nodal_coordinates(0, 2);
  const double z2 = nodal_coordinates(1, 2);
  const double z3 = nodal_coordinates(2, 2);
  const double z4 = nodal_coordinates(3, 2);

  const double volume =
      (1.0 / 6) * (x1 * y3 * z2 - x1 * y2 * z3 + x2 * y1 * z3 - x2 * y3 * z1 -
                   x3 * y1 * z2 + x3 * y2 * z1 + x1 * y2 * z4 - x1 * y4 * z2 -
                   x2 * y1 * z4 + x2 * y4 * z1 + x4 * y1 * z2 - x4 * y2 * z1 -
                   x1 * y3 * z4 + x1 * y4 * z3 + x3 * y1 * z4 - x3 * y4 * z1 -
                   x4 * y1 * z3 + x4 * y3 * z1 + x2 * y3 * z4 - x2 * y4 * z3 -
                   x3 * y2 * z4 + x3 * y4 * z2 + x4 * y2 * z3 - x4 * y3 * z2);
  return volume;
}

//! Compute natural coordinates of a point (analytical)
template <unsigned Tdim, unsigned Tnfunctions>
inline Eigen::Matrix<double, Tdim, 1>
    mpm::TetrahedronElement<Tdim, Tnfunctions>::natural_coordinates_analytical(
        const VectorDim& point,
        const Eigen::MatrixXd& nodal_coordinates) const {
  // Local point coordinates
  Eigen::Matrix<double, Tdim, 1> xi;
  xi.fill(std::numeric_limits<double>::max());

  // Assemble Ainv
  // 1. Temporary variables for node coordinants (xk, yk, zk)
  const double x1 = nodal_coordinates(0, 0);
  const double x2 = nodal_coordinates(1, 0);
  const double x3 = nodal_coordinates(2, 0);
  const double x4 = nodal_coordinates(3, 0);
  const double y1 = nodal_coordinates(0, 1);
  const double y2 = nodal_coordinates(1, 1);
  const double y3 = nodal_coordinates(2, 1);
  const double y4 = nodal_coordinates(3, 1);
  const double z1 = nodal_coordinates(0, 2);
  const double z2 = nodal_coordinates(1, 2);
  const double z3 = nodal_coordinates(2, 2);
  const double z4 = nodal_coordinates(3, 2);

  // 2. Volume of linear tetrahedron, multuiplied by 6
  const double tetrahedron_6xV = 6.0 * this->compute_volume(nodal_coordinates);

  // 3. Assembled matrix in cpp numbering; with (1/(6*V)); without first row
  Eigen::Matrix<double, 3, 4> Ainv;
  Ainv(0, 0) = (x1 * y4 * z3 - x1 * y3 * z4 + x3 * y1 * z4 - x3 * y4 * z1 -
                x4 * y1 * z3 + x4 * y3 * z1);
  Ainv(0, 1) = (y1 * z3 - y3 * z1 - y1 * z4 + y4 * z1 + y3 * z4 - y4 * z3);
  Ainv(0, 2) = (x3 * z1 - x1 * z3 + x1 * z4 - x4 * z1 - x3 * z4 + x4 * z3);
  Ainv(0, 3) = (x1 * y3 - x3 * y1 - x1 * y4 + x4 * y1 + x3 * y4 - x4 * y3);
  Ainv(1, 0) = (x1 * y2 * z4 - x1 * y4 * z2 - x2 * y1 * z4 + x2 * y4 * z1 +
                x4 * y1 * z2 - x4 * y2 * z1);
  Ainv(1, 1) = (y2 * z1 - y1 * z2 + y1 * z4 - y4 * z1 - y2 * z4 + y4 * z2);
  Ainv(1, 2) = (x1 * z2 - x2 * z1 - x1 * z4 + x4 * z1 + x2 * z4 - x4 * z2);
  Ainv(1, 3) = (x2 * y1 - x1 * y2 + x1 * y4 - x4 * y1 - x2 * y4 + x4 * y2);
  Ainv(2, 0) = (x1 * y3 * z2 - x1 * y2 * z3 + x2 * y1 * z3 - x2 * y3 * z1 -
                x3 * y1 * z2 + x3 * y2 * z1);
  Ainv(2, 1) = (y1 * z2 - y2 * z1 - y1 * z3 + y3 * z1 + y2 * z3 - y3 * z2);
  Ainv(2, 2) = (x2 * z1 - x1 * z2 + x1 * z3 - x3 * z1 - x2 * z3 + x3 * z2);
  Ainv(2, 3) = (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2);
  Ainv *= (1 / tetrahedron_6xV);

  // Output point in natural coordinates (3x4 Ainv matrix multiplied
  // by 3x1 point in global coordinates)
  xi(0) = Ainv(0, 0) + Ainv(0, 1) * point(0) + Ainv(0, 2) * point(1) +
          Ainv(0, 3) * point(2);
  xi(1) = Ainv(1, 0) + Ainv(1, 1) * point(0) + Ainv(1, 2) * point(1) +
          Ainv(1, 3) * point(2);
  xi(2) = Ainv(2, 0) + Ainv(2, 1) * point(0) + Ainv(2, 2) * point(1) +
          Ainv(2, 3) * point(2);

  return xi;
}

//! Assign nodal connectivity property for bspline elements
template <unsigned Tdim, unsigned Tnfunctions>
void mpm::TetrahedronElement<Tdim, Tnfunctions>::
    initialise_bspline_connectivity_properties(
        const Eigen::MatrixXd& nodal_coordinates,
        const std::vector<std::vector<unsigned>>& nodal_properties,
        bool kernel_correction) {
  throw std::runtime_error(
      "Function to initialise bspline connectivity is not implemented for "
      "Tet<Tdim, Tnfunctions> ");
}

//! Assign nodal connectivity property for LME elements
template <unsigned Tdim, unsigned Tnfunctions>
void mpm::TetrahedronElement<Tdim, Tnfunctions>::
    initialise_lme_connectivity_properties(
        double beta, double radius, bool anisotropy,
        const Eigen::MatrixXd& nodal_coordinates) {
  throw std::runtime_error(
      "Function to initialise lme connectivity is not implemented for "
      "Tet<Tdim, Tnfunctions> ");
}
