//! Assign nodal connectivity property for LME elements
template <unsigned Tdim>
void mpm::HexahedronLMEElement<Tdim>::initialise_lme_connectivity_properties(
    double beta, double radius, bool anisotropy,
    const Eigen::MatrixXd& nodal_coordinates) {
  this->nconnectivity_ = nodal_coordinates.rows();
  this->nodal_coordinates_ = nodal_coordinates;
  this->beta_ = beta;
  this->anisotropy_ = anisotropy;
  this->support_radius_ = radius;

  //! Uniform spacing length in 3D
  const double spacing_length =
      std::abs(nodal_coordinates(1, 0) - nodal_coordinates(0, 0));
  const double gamma = beta * spacing_length * spacing_length;
  if (gamma > 6.0) this->preconditioner_ = true;
}

//! Return shape functions of a Hexahedron LME Element at a given
//! local coordinate
template <unsigned Tdim>
inline Eigen::VectorXd mpm::HexahedronLMEElement<Tdim>::shapefn(
    const Eigen::Matrix<double, Tdim, 1>& xi,
    Eigen::Matrix<double, Tdim, 1>& lambda,
    const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store shape functions
  Eigen::VectorXd shapefn =
      Eigen::VectorXd::Constant(this->nconnectivity_, 1.0);

  if (this->nconnectivity_ == 8)
    return mpm::HexahedronElement<Tdim, 8>::shapefn(xi, lambda,
                                                    deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord =
        Eigen::Matrix<double, Tdim, 1>::Zero();
    auto local_shapefn = this->shapefn_local(xi, lambda, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Create relative coordinate vector
    const auto& rel_coordinates =
        (-nodal_coordinates_.transpose()).colwise() + pcoord;

    //! Create metric tensor
    Eigen::Matrix<double, Tdim, Tdim> metric =
        Eigen::Matrix<double, Tdim, Tdim>::Identity();

    if (anisotropy_) {
      //! Anisotropic metric tensor
      const auto& inverse_deformation_gradient = deformation_gradient.inverse();
      metric = inverse_deformation_gradient.transpose() *
               inverse_deformation_gradient;
    }

    //! Compute particle-node distance once as a vector
    Eigen::VectorXd distance =
        Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      distance(n) = std::sqrt((rel_coordinates.col(n)).transpose() *
                              (metric * rel_coordinates.col(n)));
    }

    //! Compute functional f in each connectivity
    Eigen::VectorXd f = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    double sum_exp_f = 0.;
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_) {
        f(n) = -beta_ * distance(n) * distance(n) +
               lambda.dot(rel_coordinates.col(n));
        sum_exp_f += std::exp(f(n));
      }
    }

    //! Compute p in each connectivity
    Eigen::VectorXd p = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_)
        p(n) = std::exp(f(n)) / sum_exp_f;
    }

    //! Compute vector r
    VectorDim r = VectorDim::Zero();
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      r.noalias() += p(n) * (rel_coordinates.col(n));
    }

    //! Begin regularized Newton-Raphson iteration
    const double tolerance = 1.e-12;
    if (r.norm() > tolerance) {
      bool convergence = false;
      unsigned it = 1;
      const unsigned max_it = 100;
      while (!convergence) {

        //! Compute matrix J
        Eigen::Matrix3d J = -r * r.transpose();
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          J.noalias() += p(n) * ((rel_coordinates.col(n)) *
                                 (rel_coordinates.col(n)).transpose());
        }

        //! Add preconditioner for J (Mathieu Foca, PhD Thesis)
        if (this->preconditioner_) J.diagonal().array() += r.norm();

        //! Compute Delta lambda
        const auto olambda = lambda;
        const auto& dlambda = J.inverse() * (-r);
        lambda = lambda + dlambda;

        //! Reevaluate f, p, and r
        //! Compute functional f in each connectivity
        sum_exp_f = 0.;
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          if (distance(n) < this->support_radius_) {
            f(n) = -beta_ * distance(n) * distance(n) +
                   lambda.dot(rel_coordinates.col(n));
            sum_exp_f += std::exp(f(n));
          }
        }

        //! Compute p in each connectivity
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          if (distance(n) < this->support_radius_)
            p(n) = std::exp(f(n)) / sum_exp_f;
        }

        //! Compute vector r
        r.setZero();
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          r.noalias() += p(n) * (rel_coordinates.col(n));
        }

        //! Check convergence
        if (r.norm() < tolerance) {
          convergence = true;
        } else if ((lambda - olambda).norm() < tolerance) {
          convergence = true;
        } else if (it == max_it) {
          //! Abort simulation if r.norm() is too big
          if (r.norm() > 1.e-3)
            throw std::runtime_error(
                "LME shapefn: the LME Newton-Raphson iteration unable to "
                "converge");

          //! Check condition number
          Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
          const double rcond =
              svd.singularValues()(svd.singularValues().size() - 1) /
              svd.singularValues()(0);
          if (rcond < 1E-8)
            console_->warn("LME shapefn: the LME Hessian matrix is singular!");
          convergence = true;
        }
        it++;
      }
    }

    // Assign shape function
    shapefn = p;

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return shapefn;
  }
  return shapefn;
}

//! Return gradient of shape functions of a Hexahedron LME Element at a
//! given local coordinate
template <unsigned Tdim>
inline Eigen::MatrixXd mpm::HexahedronLMEElement<Tdim>::grad_shapefn(
    const Eigen::Matrix<double, Tdim, 1>& xi,
    Eigen::Matrix<double, Tdim, 1>& lambda,
    const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {

  //! To store grad shape functions
  Eigen::MatrixXd grad_shapefn(this->nconnectivity_, Tdim);

  if (this->nconnectivity_ == 8)
    return mpm::HexahedronElement<Tdim, 8>::grad_shapefn(xi, lambda,
                                                         deformation_gradient);

  try {
    //! Convert local coordinates to real coordinates
    Eigen::Matrix<double, Tdim, 1> pcoord =
        Eigen::Matrix<double, Tdim, 1>::Zero();
    auto local_shapefn = this->shapefn_local(xi, lambda, deformation_gradient);
    for (unsigned i = 0; i < local_shapefn.size(); ++i)
      pcoord.noalias() +=
          local_shapefn(i) * nodal_coordinates_.row(i).transpose();

    //! Create relative coordinate vector
    const auto& rel_coordinates =
        (-nodal_coordinates_.transpose()).colwise() + pcoord;

    //! Create metric tensor
    Eigen::Matrix<double, Tdim, Tdim> metric =
        Eigen::Matrix<double, Tdim, Tdim>::Identity();

    if (anisotropy_) {
      //! Anisotropic metric tensor
      const auto& inverse_deformation_gradient = deformation_gradient.inverse();
      metric = inverse_deformation_gradient.transpose() *
               inverse_deformation_gradient;
    }

    //! Compute particle-node distance once as a vector
    Eigen::VectorXd distance =
        Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      distance(n) = std::sqrt((rel_coordinates.col(n)).transpose() *
                              (metric * rel_coordinates.col(n)));
    }

    //! Compute functional f in each connectivity
    Eigen::VectorXd f = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    double sum_exp_f = 0.;
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_) {
        f(n) = -beta_ * distance(n) * distance(n) +
               lambda.dot(rel_coordinates.col(n));
        sum_exp_f += std::exp(f(n));
      }
    }

    //! Compute p in each connectivity
    Eigen::VectorXd p = Eigen::VectorXd::Constant(this->nconnectivity_, 0.0);
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      if (distance(n) < this->support_radius_)
        p(n) = std::exp(f(n)) / sum_exp_f;
    }

    //! Compute vector r
    VectorDim r = VectorDim::Zero();
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      r.noalias() += p(n) * (rel_coordinates.col(n));
    }

    //! Compute matrix J
    Eigen::Matrix3d J = -r * r.transpose();
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      J.noalias() += p(n) * ((rel_coordinates.col(n)) *
                             (rel_coordinates.col(n)).transpose());
    }

    //! Add preconditioner for J (Mathieu Foca, PhD Thesis)
    if (this->preconditioner_) J.diagonal().array() += r.norm();

    //! Begin Newton-Raphson iteration
    const double tolerance = 1.e-12;
    if (r.norm() > tolerance) {
      bool convergence = false;
      unsigned it = 1;
      unsigned max_it = 100;
      while (!convergence) {
        //! Compute Delta lambda
        const auto olambda = lambda;
        const auto& dlambda = J.inverse() * (-r);
        lambda = lambda + dlambda;

        //! Reevaluate f, p, and r
        //! Compute functional f in each connectivity
        sum_exp_f = 0.;
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          if (distance(n) < this->support_radius_) {
            f(n) = -beta_ * distance(n) * distance(n) +
                   lambda.dot(rel_coordinates.col(n));
            sum_exp_f += std::exp(f(n));
          }
        }

        //! Compute p in each connectivity
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          if (distance(n) < this->support_radius_)
            p(n) = std::exp(f(n)) / sum_exp_f;
        }

        //! Compute vector r
        r.setZero();
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          r.noalias() += p(n) * (rel_coordinates.col(n));
        }

        //! Compute matrix J
        J = -r * r.transpose();
        for (unsigned n = 0; n < this->nconnectivity_; ++n) {
          J.noalias() += p(n) * ((rel_coordinates.col(n)) *
                                 (rel_coordinates.col(n)).transpose());
        }

        //! Add preconditioner for J (Mathieu Foca, PhD Thesis)
        if (this->preconditioner_) J.diagonal().array() += r.norm();

        //! Check convergence
        if (r.norm() < tolerance) {
          convergence = true;
        } else if ((lambda - olambda).norm() < tolerance) {
          convergence = true;
        } else if (it == max_it) {
          //! Abort simulation if r.norm() is too big
          if (r.norm() > 1.e-3)
            throw std::runtime_error(
                "LME grad_shapefn: the LME Newton-Raphson iteration unable to "
                "converge");

          //! Check condition number
          Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
          const double rcond =
              svd.singularValues()(svd.singularValues().size() - 1) /
              svd.singularValues()(0);
          if (rcond < 1E-8)
            console_->warn(
                "LME grad_shapefn: the LME Hessian matrix is singular!");
          convergence = true;
        }
        it++;
      }
    }

    // Compute shape function gradient
    for (unsigned n = 0; n < this->nconnectivity_; ++n) {
      const VectorDim grad_p = -p(n) * J.inverse() * (rel_coordinates.col(n));
      grad_shapefn.row(n) = grad_p.transpose();
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return grad_shapefn;
  }
  return grad_shapefn;
}

//! Return local shape functions of a LME Hexahedron Element at a given local
//! coordinate, with particle size and deformation gradient
template <unsigned Tdim>
inline Eigen::VectorXd mpm::HexahedronLMEElement<Tdim>::shapefn_local(
    const Eigen::Matrix<double, Tdim, 1>& xi,
    Eigen::Matrix<double, Tdim, 1>& lambda,
    const Eigen::Matrix<double, Tdim, Tdim>& deformation_gradient) const {
  return mpm::HexahedronElement<Tdim, 8>::shapefn(xi, lambda,
                                                  deformation_gradient);
}

//! Compute Jacobian
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::HexahedronLMEElement<Tdim>::jacobian(
        const Eigen::Matrix<double, 3, 1>& xi,
        const Eigen::MatrixXd& nodal_coordinates,
        Eigen::Matrix<double, 3, 1>& lambda,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  // Jacobian dx_i/dxi_j local
  return this->jacobian_local(xi, nodal_coordinates.block(0, 0, 8, 3), lambda,
                              deformation_gradient);
}

//! Compute dn_dx
template <unsigned Tdim>
inline Eigen::MatrixXd mpm::HexahedronLMEElement<Tdim>::dn_dx(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    VectorDim& lambda, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  return this->grad_shapefn(xi, lambda, deformation_gradient);
}

//! Compute local dn_dx
template <unsigned Tdim>
inline Eigen::MatrixXd mpm::HexahedronLMEElement<Tdim>::dn_dx_local(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    VectorDim& lambda, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  Eigen::MatrixXd grad_sf = mpm::HexahedronElement<Tdim, 8>::grad_shapefn(
      xi, lambda, deformation_gradient);

  // Jacobian dx_i/dxi_j
  Eigen::Matrix<double, Tdim, Tdim> jacobian =
      (grad_sf.transpose() * nodal_coordinates.block(0, 0, 8, Tdim));

  // Gradient shapefn of the cell
  // dN/dx = [J]^-1 * dN/dxi
  const Eigen::MatrixXd dn_dx = grad_sf * (jacobian.inverse()).transpose();

  // Compute dN/dx local
  Eigen::MatrixXd dn_dx_local(this->nconnectivity_, Tdim);
  dn_dx_local.setZero();
  dn_dx_local.block(0, 0, 8, Tdim) = dn_dx;

  return dn_dx_local;
}

//! Compute Jacobian local with particle size and deformation gradient
template <unsigned Tdim>
inline Eigen::Matrix<double, Tdim, Tdim>
    mpm::HexahedronLMEElement<Tdim>::jacobian_local(
        const Eigen::Matrix<double, 3, 1>& xi,
        const Eigen::MatrixXd& nodal_coordinates,
        Eigen::Matrix<double, 3, 1>& lambda,
        const Eigen::Matrix<double, 3, 3>& deformation_gradient) const {
  // Jacobian dx_i/dxi_j
  return mpm::HexahedronElement<Tdim, 8>::jacobian(
      xi, nodal_coordinates, lambda, deformation_gradient);
}

//! Compute Bmatrix
template <unsigned Tdim>
inline std::vector<Eigen::MatrixXd> mpm::HexahedronLMEElement<Tdim>::bmatrix(
    const VectorDim& xi, const Eigen::MatrixXd& nodal_coordinates,
    VectorDim& lambda, const MatrixDim& deformation_gradient) const {
  // Get gradient shape functions
  Eigen::MatrixXd grad_shapefn =
      this->grad_shapefn(xi, lambda, deformation_gradient);

  // B-Matrix
  std::vector<Eigen::MatrixXd> bmatrix;
  bmatrix.reserve(this->nconnectivity_);

  try {
    // Check if matrices dimensions are correct
    if ((grad_shapefn.rows() != nodal_coordinates.rows()) ||
        (xi.rows() != nodal_coordinates.cols()))
      throw std::runtime_error(
          "BMatrix - Jacobian calculation: Incorrect dimension of xi and "
          "nodal_coordinates");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    return bmatrix;
  }

  for (unsigned i = 0; i < this->nconnectivity_; ++i) {
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
