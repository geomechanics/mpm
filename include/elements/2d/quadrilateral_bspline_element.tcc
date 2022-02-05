//! Assign nodal connectivity property for bspline elements
template <unsigned Tdim, unsigned Tpolynomial>
void mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::
    initialise_bspline_connectivity_properties(
        const Eigen::MatrixXd& nodal_coordinates,
        const std::vector<std::vector<unsigned>>& nodal_properties) {
  assert(nodal_coordinates.rows() == nodal_properties.size());

  this->nconnectivity_ = nodal_coordinates.rows();
  this->nodal_coordinates_ = nodal_coordinates;
  this->node_type_ = nodal_properties;

  //! Uniform spacing length in 2D
  this->spacing_length_ =
      std::abs(nodal_coordinates(1, 0) - nodal_coordinates(0, 0));
}

//! Return shape functions of a Quadrilateral BSpline Element at a given
//! local coordinate
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::VectorXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::shapefn(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store shape functions
  Eigen::VectorXd shapefn =
      Eigen::VectorXd::Constant(this->nconnectivity_, 1.0);

  if (this->nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, particle_size,
                                                       deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord;
    pcoord.setZero();
    auto local_shapefn =
        this->shapefn_local(xi, particle_size, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Compute shape function following a multiplicative rule
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      //! Loop over dimension
      for (unsigned i = 0; i < Tdim; ++i) {
        double N = this->kernel(pcoord[i], nodal_coordinates_.row(n)[i],
                                this->node_type_[n][i], Tpolynomial);
        switch (this->node_type_[n][i]) {
          case 1:
            N += this->kernel(pcoord[i], nodal_coordinates_.row(n)[i], 5,
                              Tpolynomial);
            break;
          case 4:
            N += this->kernel(pcoord[i], nodal_coordinates_.row(n)[i], 6,
                              Tpolynomial);
            break;
        }

        shapefn[n] = shapefn[n] * N;
      }
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return shapefn;
  }
  return shapefn;
}

//! Return gradient of shape functions of a Quadrilateral BSpline Element at a
//! given local coordinate
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::grad_shapefn(
        const Eigen::Matrix<double, Tdim, 1>& xi,
        Eigen::Matrix<double, Tdim, 1>& particle_size,
        const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store grad shape functions
  Eigen::MatrixXd grad_shapefn(this->nconnectivity_, Tdim);

  if (this->nconnectivity_ == 4)
    return mpm::QuadrilateralElement<Tdim, 4>::grad_shapefn(
        xi, particle_size, deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord;
    pcoord.setZero();
    auto local_shapefn =
        this->shapefn_local(xi, particle_size, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Compute the shape function gradient following a multiplicative rule
    for (unsigned n = 0; n < this->nconnectivity_; ++n)
      //! Loop over dimension
      for (unsigned i = 0; i < Tdim; ++i) {
        double dN_dx = this->gradient(pcoord[i], nodal_coordinates_.row(n)[i],
                                      (this->node_type_[n])[i], Tpolynomial);
        switch (this->node_type_[n][i]) {
          case 1:
            dN_dx += this->gradient(pcoord[i], nodal_coordinates_.row(n)[i], 5,
                                    Tpolynomial);
            break;
          case 4:
            dN_dx += this->gradient(pcoord[i], nodal_coordinates_.row(n)[i], 6,
                                    Tpolynomial);
            break;
        }

        for (unsigned j = 0; j < Tdim; ++j) {
          if (j != i) {
            double N = this->kernel(pcoord[j], nodal_coordinates_.row(n)[j],
                                    this->node_type_[n][j], Tpolynomial);
            switch (this->node_type_[n][j]) {
              case 1:
                N += this->kernel(pcoord[j], nodal_coordinates_.row(n)[j], 5,
                                  Tpolynomial);
                break;
              case 4:
                N += this->kernel(pcoord[j], nodal_coordinates_.row(n)[j], 6,
                                  Tpolynomial);
                break;
            }
            dN_dx = dN_dx * N;
          }
        }

        grad_shapefn(n, i) = dN_dx;
      }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return grad_shapefn;
  }
  return grad_shapefn;
}

//! Compute dn_dx
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::MatrixXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::dn_dx(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  return this->grad_shapefn(xi, particle_size, deformation_gradient);
}

//! Return the B-matrix of a Quadrilateral Element at a given local
//! coordinate for a real cell
template <unsigned Tdim, unsigned Tpolynomial>
inline std::vector<Eigen::MatrixXd>
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::bmatrix(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {

  // Get gradient shape functions
  Eigen::MatrixXd grad_sf =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  // B-Matrix
  std::vector<Eigen::MatrixXd> bmatrix;
  bmatrix.reserve(this->nconnectivity_);

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

  for (unsigned i = 0; i < this->nconnectivity_; ++i) {
    Eigen::Matrix<double, 3, Tdim> bi;
    // clang-format off
          bi(0, 0) = grad_shapefn(i, 0); bi(0, 1) = 0.;
          bi(1, 0) = 0.;                 bi(1, 1) = grad_shapefn(i, 1);
          bi(2, 0) = grad_shapefn(i, 1); bi(2, 1) = grad_shapefn(i, 0);
          bmatrix.push_back(bi);
    // clang-format on
  }
  return bmatrix;
}

//! Return local shape functions of a BSpline Quadrilateral Element at a given
//! local coordinate, with particle size and deformation gradient
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::VectorXd
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::shapefn_local(
        const VectorDim& xi, VectorDim& particle_size,
        const MatrixDim& deformation_gradient) const {
  return mpm::QuadrilateralElement<Tdim, 4>::shapefn(xi, particle_size,
                                                     deformation_gradient);
}

//! Compute Jacobian with particle size and deformation gradient
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::jacobian(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {

  // Get gradient shape functions
  const Eigen::MatrixXd grad_shapefn =
      this->grad_shapefn(xi, particle_size, deformation_gradient);

  try {
    // Check if matrices dimensions are correct
    if ((grad_shapefn.rows() != nodal_coordinates.rows()) ||
        (xi.size() != nodal_coordinates.cols()))
      throw std::runtime_error(
          "Jacobian calculation: Incorrect dimension of xi and "
          "nodal_coordinates");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return Eigen::Matrix<double, Tdim, Tdim>::Zero();
  }

  // Jacobian dx_i/dxi_j
  return (grad_shapefn.transpose() * nodal_coordinates);
}

//! Compute Jacobian local with particle size and deformation gradient
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::jacobian_local(
        const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
        VectorDim& particle_size, const MatrixDim& deformation_gradient) const {
  // Jacobian dx_i/dxi_j
  return mpm::QuadrilateralElement<2, 4>::jacobian(
      xi, nodal_coordinates, particle_size, deformation_gradient);
}

//! Compute natural coordinates of a point (analytical)
template <unsigned Tdim, unsigned Tpolynomial>
inline Eigen::Matrix<double, Tdim, 1> mpm::QuadrilateralBSplineElement<
    Tdim, Tpolynomial>::natural_coordinates_analytical(const VectorDim& point,
                                                       const Eigen::MatrixXd&
                                                           nodal_coordinates)
    const {
  // Local point coordinates
  Eigen::Matrix<double, 2, 1> xi;
  xi.fill(std::numeric_limits<double>::max());
  throw std::runtime_error(
      "Analytical solution for QuadBSpline<Tdim, Tpolynomial> has "
      "not been "
      "implemented");
  return xi;
}

//! Compute B-Spline Basis Function using the recursive De Boor's algorithm
//! for single direction
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::kernel(
    double point_coord, double nodal_coord, unsigned node_type,
    unsigned poly_order, unsigned index) const {
  double value = 0.0;
  //! Compute knot coordinate
  Eigen::VectorXd knot_vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      (this->knot(node_type)).data(), (this->knot(node_type)).size());
  const Eigen::VectorXd one =
      Eigen::VectorXd::Constant(knot_vector.size(), 1.0);
  const Eigen::VectorXd knot_coord =
      nodal_coord * one + spacing_length_ * knot_vector;

  if (poly_order == 0) {
    if (point_coord >= knot_coord[index] &&
        point_coord < knot_coord[index + 1]) {
      value = 1.0;
    } else {
      value = 0.0;
    }
  } else {
    const double den_a = (knot_coord[index + poly_order] - knot_coord[index]);
    double a = (point_coord - knot_coord(index)) / den_a;
    if (den_a < std::numeric_limits<double>::epsilon()) a = 0;

    const double den_b =
        (knot_coord[index + poly_order + 1] - knot_coord[index + 1]);
    double b = (knot_coord[index + poly_order + 1] - point_coord) / den_b;
    if (den_b < std::numeric_limits<double>::epsilon()) b = 0;

    value = a * this->kernel(point_coord, nodal_coord, node_type,
                             poly_order - 1, index) +
            b * this->kernel(point_coord, nodal_coord, node_type,
                             poly_order - 1, index + 1);
  }
  return value;
}

//! Compute B-Spline Basis Function Gradient using the recursive De Boor's
//! algorithm for single direction
template <unsigned Tdim, unsigned Tpolynomial>
double mpm::QuadrilateralBSplineElement<Tdim, Tpolynomial>::gradient(
    double point_coord, double nodal_coord, unsigned node_type,
    unsigned poly_order, unsigned index) const {
  double value = 0;
  Eigen::VectorXd knot_vector = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
      (this->knot(node_type)).data(), (this->knot(node_type)).size());
  const Eigen::VectorXd one =
      Eigen::VectorXd::Constant(knot_vector.size(), 1.0);
  const Eigen::VectorXd knot_coord =
      nodal_coord * one + spacing_length_ * knot_vector;

  const double den_a = (knot_coord[index + poly_order] - knot_coord[index]);
  double a = poly_order / den_a;
  if (den_a < std::numeric_limits<double>::epsilon()) a = 0;

  const double den_b =
      (knot_coord[index + poly_order + 1] - knot_coord[index + 1]);
  double b = poly_order / den_b;
  if (den_b < std::numeric_limits<double>::epsilon()) b = 0;

  value = a * this->kernel(point_coord, nodal_coord, node_type, poly_order - 1,
                           index) -
          b * this->kernel(point_coord, nodal_coord, node_type, poly_order - 1,
                           index + 1);

  return value;
}