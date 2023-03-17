// Hexahedron element test
#include <memory>

#include "catch.hpp"

#include "hexahedron_element.h"

//! \brief Check hexahedron element class
TEST_CASE("Hexahedron elements are checked", "[hex][element][3D]") {
  const unsigned Dim = 3;
  const double Tolerance = 1.E-7;

  Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d zero_matrix = Eigen::Matrix3d::Zero();

  //! Check for 8 noded element
  SECTION("Hexahedron element with eight nodes") {
    const unsigned nfunctions = 8;
    std::shared_ptr<mpm::Element<Dim>> hex =
        std::make_shared<mpm::HexahedronElement<Dim, nfunctions>>();

    // Check degree
    REQUIRE(hex->degree() == mpm::ElementDegree::Linear);
    REQUIRE(hex->shapefn_type() == mpm::ShapefnType::NORMAL_MPM);

    // Coordinates is (0, 0, 0)
    SECTION("Eight noded hexahedron element for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);
      REQUIRE(hex->nfunctions() == nfunctions);
      REQUIRE(hex->nfunctions_local() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(-0.125).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.125).epsilon(Tolerance));

      REQUIRE(gradsf(0, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(0.125).epsilon(Tolerance));
    }

    // Coordinates is (-1, -1, -1);
    SECTION("Eight noded hexahedron element for coordinates(-1, -1, -1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1., -1., -1.;
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);
      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 2) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (1, 1, 1)
    SECTION("Eight noded hexahedron element for coordinates(1, 1, 1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1., 1., 1.;
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(-0.5).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (0, 0, 0)
    SECTION(
        "Eight noded local sf hexahedron element for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = hex->shapefn_local(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));
    }

    SECTION("Eight noded hexahedron shapefn with deformation gradient") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();
      auto shapefn = hex->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(-0.125).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.125).epsilon(Tolerance));

      REQUIRE(gradsf(0, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(0.125).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION(
        "Eight noded hexahedron Jacobian for local coordinates(0.5,0.5,0.5)") {
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2., 1., 0.5,
                4., 2., 1.0,
                2., 4., 1.0,
                1., 3., 0.5,
                2., 1., 1.5,
                4., 2., 2.0,
                2., 4., 2.0,
                1., 3., 1.5;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5, 0.25,
                 -0.875, 1.0, 0.00,
                  0.000, 0.0, 0.50;
      // clang-format on

      // Get Jacobian
      auto jac = hex->jacobian(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check local Jacobian
    SECTION(
        "Eight noded hexahedron local Jacobian for local "
        "coordinates(0.5,0.5,0.5)") {
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2., 1., 0.5,
                4., 2., 1.0,
                2., 4., 1.0,
                1., 3., 0.5,
                2., 1., 1.5,
                4., 2., 2.0,
                2., 4., 2.0,
                1., 3., 1.5;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5, 0.25,
                 -0.875, 1.0, 0.00,
                  0.000, 0.0, 0.50;
      // clang-format on

      // Get Jacobian
      auto jac = hex->jacobian_local(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION("Eight noded hexahedron Jacobian with deformation gradient") {
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2., 1., 0.5,
                4., 2., 1.0,
                2., 4., 1.0,
                1., 3., 0.5,
                2., 1., 1.5,
                4., 2., 2.0,
                2., 4., 2.0,
                1., 3., 1.5;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5, 0.5;
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5, 0.25,
                 -0.875, 1.0, 0.00,
                  0.000, 0.0, 0.50;
      // clang-format on

      // Get Jacobian
      auto jac = hex->jacobian(xi, coords, psize, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Coordinates is (0, 0, 0)
    SECTION("Eight noded hexahedron B-matrix cell for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                1., 1., 0.,
                0., 1., 0.,
                0., 0., 1.,
                1., 0., 1.,
                1., 1., 1.,
                0., 1., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = hex->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = hex->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
      }

      // Check dN/dx local
      auto dn_dx_local = hex->dn_dx_local(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx_local.rows() == nfunctions);
      REQUIRE(dn_dx_local.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx_local(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx_local(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(dn_dx_local(i, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 1) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 2) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 0) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 2) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (0, 0, 0)
    SECTION("Eight noded hexahedron B-matrix cell for xi(0.5, 0.5, 0.5)") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5, 0.5;

      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                1., 1., 0.,
                0., 1., 0.,
                0., 0., 1.,
                1., 0., 1.,
                1., 1., 1.,
                0., 1., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = hex->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = hex->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 1) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 2) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 0) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 2) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (-0.5, -0.5, -0.5)
    SECTION("Eight noded hexahedron B-matrix cell xi(-0.5, -0.5, -0.5)") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << -0.5, -0.5, -0.5;

      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                1., 1., 0.,
                0., 1., 0.,
                0., 0., 1.,
                1., 0., 1.,
                1., 1., 1.,
                0., 1., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = hex->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = hex->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 1) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 2) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 0) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 2) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    SECTION("Eight noded hexahedron B-matrix with deformation gradient") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                1., 1., 0.,
                0., 1., 0.,
                0., 0., 1.,
                1., 0., 1.,
                1., 1., 1.,
                0., 1., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = hex->bmatrix(xi, coords, psize, defgrad);

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(xi, psize, defgrad);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = hex->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(3, 2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 1) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(4, 2) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 0) == Approx(gradsf(i, 2)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(5, 2) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    SECTION("Eight noded hexahedron B-matrix and Jacobian failure") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      Eigen::Matrix<double, 7, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                1., 1., 0.,
                0., 1., 0.,
                0., 0., 1.,
                1., 0., 1.,
                1., 1., 1.;
      // clang-format on
      // Get B-Matrix
      hex->bmatrix(xi, coords, zero, zero_matrix);
      hex->jacobian(xi, coords, zero, zero_matrix);
    }

    // Ni Nj matrix of a cell
    SECTION("Eight noded hexahedron ni-nj-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);

      xi << -one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 8);
    }

    // Laplace matrix of a cell
    SECTION("Eight noded hexahedron laplace-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);

      xi << -one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 8);

      // Nodal coordinates
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2., 1., 0.5,
                4., 2., 1.0,
                2., 4., 1.0,
                1., 3., 0.5,
                2., 1., 1.5,
                4., 2., 2.0,
                2., 4., 2.0,
                1., 3., 1.5;
      // clang-format on
    }

    SECTION("Eight noded hexahedron coordinates of unit cell") {
      const unsigned nfunctions = 8;
      // Coordinates of a unit cell
      Eigen::Matrix<double, nfunctions, Dim> unit_cell;
      // clang-format off
      unit_cell << -1., -1., -1.,
                    1., -1., -1.,
                    1.,  1., -1.,
                   -1.,  1., -1.,
                   -1., -1.,  1.,
                    1., -1.,  1.,
                    1.,  1.,  1.,
                   -1.,  1.,  1.;
      // clang-format on

      auto coordinates = hex->unit_cell_coordinates();
      REQUIRE(coordinates.rows() == nfunctions);
      REQUIRE(coordinates.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {  // Iterate through nfunctions
        for (unsigned j = 0; j < Dim; ++j) {       // Dimension
          REQUIRE(coordinates(i, j) ==
                  Approx(unit_cell(i, j)).epsilon(Tolerance));
        }
      }
    }

    SECTION("Eight noded hexahedron element for sides indices") {
      // Check for sides indices
      Eigen::MatrixXi indices = hex->sides_indices();
      REQUIRE(indices.rows() == 12);
      REQUIRE(indices.cols() == 2);
      REQUIRE(indices(0, 0) == 0);
      REQUIRE(indices(0, 1) == 1);

      REQUIRE(indices(1, 0) == 1);
      REQUIRE(indices(1, 1) == 2);

      REQUIRE(indices(2, 0) == 2);
      REQUIRE(indices(2, 1) == 3);

      REQUIRE(indices(3, 0) == 3);
      REQUIRE(indices(3, 1) == 0);

      REQUIRE(indices(4, 0) == 4);
      REQUIRE(indices(4, 1) == 5);

      REQUIRE(indices(5, 0) == 5);
      REQUIRE(indices(5, 1) == 6);

      REQUIRE(indices(6, 0) == 6);
      REQUIRE(indices(6, 1) == 7);

      REQUIRE(indices(7, 0) == 7);
      REQUIRE(indices(7, 1) == 4);

      REQUIRE(indices(8, 0) == 0);
      REQUIRE(indices(8, 1) == 4);

      REQUIRE(indices(9, 0) == 1);
      REQUIRE(indices(9, 1) == 5);

      REQUIRE(indices(10, 0) == 2);
      REQUIRE(indices(10, 1) == 6);

      REQUIRE(indices(11, 0) == 3);
      REQUIRE(indices(11, 1) == 7);
    }

    SECTION("Eight noded hexahedron element for corner indices") {
      // Check for volume indices
      Eigen::VectorXi indices = hex->corner_indices();
      REQUIRE(indices.size() == 8);
      REQUIRE(indices(0) == 0);
      REQUIRE(indices(1) == 1);
      REQUIRE(indices(2) == 2);
      REQUIRE(indices(3) == 3);
      REQUIRE(indices(4) == 4);
      REQUIRE(indices(5) == 5);
      REQUIRE(indices(6) == 6);
      REQUIRE(indices(7) == 7);
    }

    SECTION("Eight noded hexahedron shape function for face indices") {
      // Check for face indices
      Eigen::Matrix<int, 6, 4> indices;
      // clang-format off
      indices << 0, 1, 5, 4,
                 5, 1, 2, 6,
                 7, 6, 2, 3,
                 0, 4, 7, 3,
                 1, 0, 3, 2,
                 4, 5, 6, 7;
      // clang-format on

      // Check for all face indices
      for (unsigned i = 0; i < indices.rows(); ++i) {
        const auto check_indices = hex->face_indices(i);
        REQUIRE(check_indices.rows() == 4);
        REQUIRE(check_indices.cols() == 1);

        for (unsigned j = 0; j < indices.cols(); ++j)
          REQUIRE(check_indices(j) == indices(i, j));
      }

      // Check number of faces
      REQUIRE(hex->nfaces() == 6);
    }
  }

  // 20-Node (Serendipity) Hexahedron Element
  //! Check for 8 noded element
  SECTION("Hexahedron element with twenty nodes") {
    const unsigned nfunctions = 20;
    std::shared_ptr<mpm::Element<Dim>> hex =
        std::make_shared<mpm::HexahedronElement<Dim, nfunctions>>();

    // Check degree
    REQUIRE(hex->degree() == mpm::ElementDegree::Quadratic);
    REQUIRE(hex->shapefn_type() == mpm::ShapefnType::NORMAL_MPM);

    // Coordinates is (0, 0, 0)
    SECTION("Twenty noded hexahedron element for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      // Edge nodes
      // N0 = 0.125*(1 - xi(0))(1. - xi(1))(1 - xi(2))(-xi(0) - xi(1) -xi(2) -2)
      // N1 = 0.125*(1 + xi(0))(1. - xi(1))(1 - xi(2))(+xi(0) - xi(1) -xi(2) -2)
      // N2 = 0.125*(1 + xi(0))(1. + xi(1))(1 - xi(2))(+xi(0) + xi(1) -xi(2) -2)
      // N3 = 0.125*(1 - xi(0))(1. + xi(1))(1 - xi(2))(-xi(0) + xi(1) -xi(2) -2)
      // N4 = 0.125*(1 - xi(0))(1. - xi(1))(1 + xi(2))(-xi(0) - xi(1) +xi(2) -2)
      // N5 = 0.125*(1 + xi(0))(1. - xi(1))(1 + xi(2))(+xi(0) - xi(1) +xi(2) -2)
      // N6 = 0.125*(1 + xi(0))(1. + xi(1))(1 + xi(2))(+xi(0) + xi(1) +xi(2) -2)
      // N7 = 0.125*(1 - xi(0))(1. + xi(1))(1 + xi(2))(-xi(0) + xi(1) +xi(2) -2)
      REQUIRE(shapefn(0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(-0.25).epsilon(Tolerance));

      // Midside nodes
      // N8 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 - xi(2))
      // N9 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N10 = 0.25*(1 - xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N11 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N12 = 0.25*(1 + xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N13 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 - xi(2))
      // N14 = 0.25*(1 + xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N15 = 0.25*(1 - xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N16 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 + xi(2))
      // N17 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N18 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N19 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 + xi(2))
      REQUIRE(shapefn(8) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(16) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(17) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(18) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(19) == Approx(0.25).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      // Derivatives with respect to eta-direction
      // Edge nodes
      // G(0,0) = 0.125*(1. - xi(1))(1 - xi(2))(2*xi(0) + xi(1) + xi(2) + 1.)
      // G(1,0) = 0.125*(1. - xi(1))(1 - xi(2))(2*xi(0) - xi(1) - xi(2) - 1.)
      // G(2,0) = 0.125*(1. + xi(1))(1 - xi(2))(2*xi(0) + xi(1) - xi(2) - 1.)
      // G(3,0) = 0.125*(1. + xi(1))(1 - xi(2))(2*xi(0) - xi(1) + xi(2) + 1.)
      // G(4,0) = 0.125*(1. - xi(1))(1 + xi(2))(2*xi(0) + xi(1) - xi(2) + 1.)
      // G(5,0) = 0.125*(1. - xi(1))(1 + xi(2))(2*xi(0) - xi(1) + xi(2) - 1.)
      // G(6,0) = 0.125*(1. + xi(1))(1 + xi(2))(2*xi(0) + xi(1) + xi(2) - 1.)
      // G(7,0) = 0.125*(1. + xi(1))(1 + xi(2))(2*xi(0) - xi(1) - xi(2) + 1.)
      REQUIRE(gradsf(0, 0) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(+0.125).epsilon(Tolerance));

      // Midside nodes
      // G(8, 0) = -0.5*xi(0)*(1 - xi(1))(1 - xi(2));
      // G(9, 0) = -0.25*(1 - xi(1)xi(1))(1 - xi(2));
      // G(10, 0) = -0.25*(1 - xi(2)xi(2))(1 - xi(1));
      // G(11, 0) = 0.25*(1 - xi(1)xi(1))(1 - xi(2));
      // G(12, 0) = 0.25*(1 - xi(2)xi(2))(1 - xi(1));
      // G(13, 0) = -0.5*xi(0)(1 + xi(1))(1 - xi(2));
      // G(14, 0) = 0.25*(1 - xi(2)xi(2))(1 + xi(1));
      // G(15, 0) = -0.25*(1 - xi(2)xi(2))(1 + xi(1));
      // G(16, 0) = -0.5*xi(0)(1 - xi(1))(1 + xi(2));
      // G(17, 0) = -0.25*(1 - xi(1)xi(1))(1 + xi(2));
      // G(18, 0) = 0.25*(1 - xi(1)xi(1))(1 + xi(2));
      // G(19, 0) = -0.5*xi(0)(1 + xi(1))(1 + xi(2));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(16, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(17, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(18, 0) == Approx(+0.25).epsilon(Tolerance));
      REQUIRE(gradsf(19, 0) == Approx(0.0).epsilon(Tolerance));

      // Derivatives with respect to psi-direction
      // Edge nodes
      // G(0,1) = 0.125*(1. - xi(0))(1 - xi(2))(+xi(0) + 2*xi(1) + xi(2) + 1.)
      // G(1,1) = 0.125*(1. + xi(0))(1 - xi(2))(-xi(0) + 2*xi(1) + xi(2) + 1.)
      // G(2,1) = 0.125*(1. + xi(0))(1 - xi(2))(+xi(0) + 2*xi(1) - xi(2) - 1.)
      // G(3,1) = 0.125*(1. - xi(0))(1 - xi(2))(-xi(0) + 2*xi(1) - xi(2) - 1.)
      // G(4,1) = 0.125*(1. - xi(0))(1 + xi(2))(+xi(0) + 2*xi(1) - xi(2) + 1.)
      // G(5,1) = 0.125*(1. + xi(0))(1 + xi(2))(-xi(0) + 2*xi(1) - xi(2) + 1.)
      // G(6,1) = 0.125*(1. + xi(0))(1 + xi(2))(+xi(0) + 2*xi(1) + xi(2) - 1.)
      // G(7,1) = 0.125*(1. - xi(0))(1 + xi(2))(-xi(0) + 2*xi(1) + xi(2) - 1.)
      REQUIRE(gradsf(0, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(-0.125).epsilon(Tolerance));

      // Midside nodes
      // G(8, 1) = -0.25*(1 - xi(0)xi(0))(1 - xi(2));
      // G(9, 1) = -0.5*xi(1)(1 - xi(0))(1 - xi(2));
      // G(10, 1) = -0.25*(1 - xi(2)xi(2))(1 - xi(0));
      // G(11, 1) = -0.5*xi(1)(1 + xi(0))(1 - xi(2));
      // G(12, 1) = -0.25*(1 - xi(2)xi(2))(1 + xi(0));
      // G(13, 1) = 0.25*(1 - xi(0)xi(0))(1 - xi(2));
      // G(14, 1) = 0.25*(1 - xi(2)xi(2))(1 + xi(0));
      // G(15, 1) = 0.25*(1 - xi(2)xi(2))(1 - xi(0));
      // G(16, 1) = -0.25*(1 - xi(0)xi(0))(1 + xi(2));
      // G(17, 1) = -0.5*xi(1)(1 - xi(0))(1 + xi(2));
      // G(18, 1) = -0.5*xi(1)(1 + xi(0))(1 + xi(2));
      // G(19, 1) = 0.25*(1 - xi(0)xi(0))(1 + xi(2));
      REQUIRE(gradsf(8, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(16, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(17, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(18, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(19, 1) == Approx(0.25).epsilon(Tolerance));

      // Derivatives with respect to mu-direction
      // Edge nodes
      // G(0,2) = 0.125*(1. - xi(0))(1 - xi(1))(+xi(0) + xi(1) + 2*xi(2) + 1.)
      // G(1,2) = 0.125*(1. + xi(0))(1 - xi(1))(-xi(0) + xi(1) + 2*xi(2) + 1.)
      // G(2,2) = 0.125*(1. + xi(0))(1 + xi(1))(-xi(0) - xi(1) + 2*xi(2) + 1.)
      // G(3,2) = 0.125*(1. - xi(0))(1 + xi(1))(+xi(0) - xi(1) + 2*xi(2) + 1.)
      // G(4,2) = 0.125*(1. - xi(0))(1 - xi(1))(-xi(0) - xi(1) + 2*xi(2) - 1.)
      // G(5,2) = 0.125*(1. + xi(0))(1 - xi(1))(+xi(0) - xi(1) + 2*xi(2) - 1.)
      // G(6,2) = 0.125*(1. + xi(0))(1 + xi(1))(+xi(0) + xi(1) + 2*xi(2) - 1.)
      // G(7,2) = 0.125*(1. - xi(0))(1 + xi(1))(-xi(0) + xi(1) + 2*xi(2) - 1.)
      REQUIRE(gradsf(0, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(-0.125).epsilon(Tolerance));

      // Midside nodes
      // G(8, 2) = -0.25*(1 - xi(0)xi(0))(1 - xi(1));
      // G(9, 2) = -0.25*(1 - xi(1)xi(1))(1 - xi(0));
      // G(10, 2) = -0.5*xi(2)(1 - xi(0))(1 - xi(1));
      // G(11, 2) = -0.25*(1 - xi(1)xi(1))(1 + xi(0));
      // G(12, 2) = -0.5*xi(2)(1 + xi(0))(1 - xi(1));
      // G(13, 2) = -0.25*(1 - xi(0)xi(0))(1 + xi(1));
      // G(14, 2) = -0.5*xi(2)(1 + xi(0))(1 + xi(1));
      // G(15, 2) = -0.5*xi(2)(1 - xi(0))(1 + xi(1));
      // G(16, 2) = 0.25*(1 - xi(0)xi(0))(1 - xi(1));
      // G(17, 2) = 0.25*(1 - xi(1)xi(1))(1 - xi(0));
      // G(18, 2) = 0.25*(1 - xi(1)xi(1))(1 + xi(0));
      // G(19, 2) = 0.25*(1 - xi(0)xi(0))(1 + xi(1));
      REQUIRE(gradsf(8, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(9, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(10, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(12, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(14, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(16, 2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(17, 2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(18, 2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(19, 2) == Approx(0.25).epsilon(Tolerance));
    }

    // Coordinates is (-0.5, -0.5, -0,5)
    SECTION(
        "Twenty noded hexahedron element for coordinates(-0.5, "
        "-0.5, -0.5)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -0.5, -0.5, -0.5;
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      // Edge nodes
      // N0 = 0.125*(1 - xi(0))(1. - xi(1))(1 - xi(2))(-xi(0) - xi(1) -xi(2) -2)
      // N1 = 0.125*(1 + xi(0))(1. - xi(1))(1 - xi(2))(+xi(0) - xi(1) -xi(2) -2)
      // N2 = 0.125*(1 + xi(0))(1. + xi(1))(1 - xi(2))(+xi(0) + xi(1) -xi(2) -2)
      // N3 = 0.125*(1 - xi(0))(1. + xi(1))(1 - xi(2))(-xi(0) + xi(1) -xi(2) -2)
      // N4 = 0.125*(1 - xi(0))(1. - xi(1))(1 + xi(2))(-xi(0) - xi(1) +xi(2) -2)
      // N5 = 0.125*(1 + xi(0))(1. - xi(1))(1 + xi(2))(+xi(0) - xi(1) +xi(2) -2)
      // N6 = 0.125*(1 + xi(0))(1. + xi(1))(1 + xi(2))(+xi(0) + xi(1) +xi(2) -2)
      // N7 = 0.125*(1 - xi(0))(1. + xi(1))(1 + xi(2))(-xi(0) + xi(1) +xi(2) -2)
      REQUIRE(shapefn(0) == Approx(-0.2109375).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.2109375).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.1171875).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.2109375).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(-0.2109375).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(-0.1171875).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(-0.0546875).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(-0.1171875).epsilon(Tolerance));

      // Midside nodes
      // N8 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 - xi(2))
      // N9 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N10 = 0.25*(1 - xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N11 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N12 = 0.25*(1 + xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N13 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 - xi(2))
      // N14 = 0.25*(1 + xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N15 = 0.25*(1 - xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N16 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 + xi(2))
      // N17 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N18 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N19 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 + xi(2))
      REQUIRE(shapefn(8) == Approx(0.421875).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.421875).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.421875).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.046875).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(16) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(17) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(18) == Approx(0.046875).epsilon(Tolerance));
      REQUIRE(shapefn(19) == Approx(0.046875).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      // Derivatives with respect to eta-direction
      // Edge nodes
      // G(0,0) = 0.125*(1. - xi(1))(1 - xi(2))(2*xi(0) + xi(1) + xi(2) + 1.)
      // G(1,0) = 0.125*(1. - xi(1))(1 - xi(2))(2*xi(0) - xi(1) - xi(2) - 1.)
      // G(2,0) = 0.125*(1. + xi(1))(1 - xi(2))(2*xi(0) + xi(1) - xi(2) - 1.)
      // G(3,0) = 0.125*(1. + xi(1))(1 - xi(2))(2*xi(0) - xi(1) + xi(2) + 1.)
      // G(4,0) = 0.125*(1. - xi(1))(1 + xi(2))(2*xi(0) + xi(1) - xi(2) + 1.)
      // G(5,0) = 0.125*(1. - xi(1))(1 + xi(2))(2*xi(0) - xi(1) + xi(2) - 1.)
      // G(6,0) = 0.125*(1. + xi(1))(1 + xi(2))(2*xi(0) + xi(1) + xi(2) - 1.)
      // G(7,0) = 0.125*(1. + xi(1))(1 + xi(2))(2*xi(0) - xi(1) - xi(2) + 1.)
      REQUIRE(gradsf(0, 0) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.03125).epsilon(Tolerance));

      // Midside nodes
      // G(8, 0) = -0.5*xi(0)*(1 - xi(1))(1 - xi(2));
      // G(9, 0) = -0.25*(1 - xi(1)xi(1))(1 - xi(2));
      // G(10, 0) = -0.25*(1 - xi(2)xi(2))(1 - xi(1));
      // G(11, 0) = 0.25*(1 - xi(1)xi(1))(1 - xi(2));
      // G(12, 0) = 0.25*(1 - xi(2)xi(2))(1 - xi(1));
      // G(13, 0) = -0.5*xi(0)(1 + xi(1))(1 - xi(2));
      // G(14, 0) = 0.25*(1 - xi(2)xi(2))(1 + xi(1));
      // G(15, 0) = -0.25*(1 - xi(2)xi(2))(1 + xi(1));
      // G(16, 0) = -0.5*xi(0)(1 - xi(1))(1 + xi(2));
      // G(17, 0) = -0.25*(1 - xi(1)xi(1))(1 + xi(2));
      // G(18, 0) = 0.25*(1 - xi(1)xi(1))(1 + xi(2));
      // G(19, 0) = -0.5*xi(0)(1 + xi(1))(1 + xi(2));
      REQUIRE(gradsf(8, 0) == Approx(0.5625).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(16, 0) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(17, 0) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(18, 0) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(19, 0) == Approx(0.0625).epsilon(Tolerance));

      // Derivatives with respect to psi-direction
      // Edge nodes
      // G(0,1) = 0.125*(1. - xi(0))(1 - xi(2))(+xi(0) + 2*xi(1) + xi(2) + 1.)
      // G(1,1) = 0.125*(1. + xi(0))(1 - xi(2))(-xi(0) + 2*xi(1) + xi(2) + 1.)
      // G(2,1) = 0.125*(1. + xi(0))(1 - xi(2))(+xi(0) + 2*xi(1) - xi(2) - 1.)
      // G(3,1) = 0.125*(1. - xi(0))(1 - xi(2))(-xi(0) + 2*xi(1) - xi(2) - 1.)
      // G(4,1) = 0.125*(1. - xi(0))(1 + xi(2))(+xi(0) + 2*xi(1) - xi(2) + 1.)
      // G(5,1) = 0.125*(1. + xi(0))(1 + xi(2))(-xi(0) + 2*xi(1) - xi(2) + 1.)
      // G(6,1) = 0.125*(1. + xi(0))(1 + xi(2))(+xi(0) + 2*xi(1) + xi(2) - 1.)
      // G(7,1) = 0.125*(1. - xi(0))(1 + xi(2))(-xi(0) + 2*xi(1) + xi(2) - 1.)
      REQUIRE(gradsf(0, 1) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.03125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(-0.1875).epsilon(Tolerance));

      // Midside nodes
      // G(8, 1) = -0.25*(1 - xi(0)xi(0))(1 - xi(2));
      // G(9, 1) = -0.5*xi(1)(1 - xi(0))(1 - xi(2));
      // G(10, 1) = -0.25*(1 - xi(2)xi(2))(1 - xi(0));
      // G(11, 1) = -0.5*xi(1)(1 + xi(0))(1 - xi(2));
      // G(12, 1) = -0.25*(1 - xi(2)xi(2))(1 + xi(0));
      // G(13, 1) = 0.25*(1 - xi(0)xi(0))(1 - xi(2));
      // G(14, 1) = 0.25*(1 - xi(2)xi(2))(1 + xi(0));
      // G(15, 1) = 0.25*(1 - xi(2)xi(2))(1 - xi(0));
      // G(16, 1) = -0.25*(1 - xi(0)xi(0))(1 + xi(2));
      // G(17, 1) = -0.5*xi(1)(1 - xi(0))(1 + xi(2));
      // G(18, 1) = -0.5*xi(1)(1 + xi(0))(1 + xi(2));
      // G(19, 1) = 0.25*(1 - xi(0)xi(0))(1 + xi(2));
      REQUIRE(gradsf(8, 1) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.5625).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(16, 1) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(17, 1) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(18, 1) == Approx(0.0625).epsilon(Tolerance));
      REQUIRE(gradsf(19, 1) == Approx(0.09375).epsilon(Tolerance));

      // Derivatives with respect to mu-direction
      // Edge nodes
      // G(0,2) = 0.125*(1. - xi(0))(1 - xi(1))(+xi(0) + xi(1) + 2*xi(2) + 1.)
      // G(1,2) = 0.125*(1. + xi(0))(1 - xi(1))(-xi(0) + xi(1) + 2*xi(2) + 1.)
      // G(2,2) = 0.125*(1. + xi(0))(1 + xi(1))(-xi(0) - xi(1) + 2*xi(2) + 1.)
      // G(3,2) = 0.125*(1. - xi(0))(1 + xi(1))(+xi(0) - xi(1) + 2*xi(2) + 1.)
      // G(4,2) = 0.125*(1. - xi(0))(1 - xi(1))(-xi(0) - xi(1) + 2*xi(2) - 1.)
      // G(5,2) = 0.125*(1. + xi(0))(1 - xi(1))(+xi(0) - xi(1) + 2*xi(2) - 1.)
      // G(6,2) = 0.125*(1. + xi(0))(1 + xi(1))(+xi(0) + xi(1) + 2*xi(2) - 1.)
      // G(7,2) = 0.125*(1. - xi(0))(1 + xi(1))(-xi(0) + xi(1) + 2*xi(2) - 1.)
      REQUIRE(gradsf(0, 2) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(0.).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(0.03125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(-0.1875).epsilon(Tolerance));

      // Midside nodes
      // G(8, 2) = -0.25*(1 - xi(0)xi(0))(1 - xi(1));
      // G(9, 2) = -0.25*(1 - xi(1)xi(1))(1 - xi(0));
      // G(10, 2) = -0.5*xi(2)(1 - xi(0))(1 - xi(1));
      // G(11, 2) = -0.25*(1 - xi(1)xi(1))(1 + xi(0));
      // G(12, 2) = -0.5*xi(2)(1 + xi(0))(1 - xi(1));
      // G(13, 2) = -0.25*(1 - xi(0)xi(0))(1 + xi(1));
      // G(14, 2) = -0.5*xi(2)(1 + xi(0))(1 + xi(1));
      // G(15, 2) = -0.5*xi(2)(1 - xi(0))(1 + xi(1));
      // G(16, 2) = 0.25*(1 - xi(0)xi(0))(1 - xi(1));
      // G(17, 2) = 0.25*(1 - xi(1)xi(1))(1 - xi(0));
      // G(18, 2) = 0.25*(1 - xi(1)xi(1))(1 + xi(0));
      // G(19, 2) = 0.25*(1 - xi(0)xi(0))(1 + xi(1));
      REQUIRE(gradsf(8, 2) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(9, 2) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(10, 2) == Approx(0.5625).epsilon(Tolerance));
      REQUIRE(gradsf(11, 2) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(12, 2) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(13, 2) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(14, 2) == Approx(0.0625).epsilon(Tolerance));
      REQUIRE(gradsf(15, 2) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(16, 2) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(17, 2) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(18, 2) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(19, 2) == Approx(0.09375).epsilon(Tolerance));
    }
    // Coordinates is (0.5, 0.5, 0,5)
    SECTION(
        "Twenty noded hexahedron element for coordinates(0.5, "
        "0.5, 0.5)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 0.5, 0.5, 0.5;
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      // Edge nodes
      // N0 = 0.125*(1 - xi(0))(1. - xi(1))(1 - xi(2))(-xi(0) - xi(1) -xi(2) -2)
      // N1 = 0.125*(1 + xi(0))(1. - xi(1))(1 - xi(2))(+xi(0) - xi(1) -xi(2) -2)
      // N2 = 0.125*(1 + xi(0))(1. + xi(1))(1 - xi(2))(+xi(0) + xi(1) -xi(2) -2)
      // N3 = 0.125*(1 - xi(0))(1. + xi(1))(1 - xi(2))(-xi(0) + xi(1) -xi(2) -2)
      // N4 = 0.125*(1 - xi(0))(1. - xi(1))(1 + xi(2))(-xi(0) - xi(1) +xi(2) -2)
      // N5 = 0.125*(1 + xi(0))(1. - xi(1))(1 + xi(2))(+xi(0) - xi(1) +xi(2) -2)
      // N6 = 0.125*(1 + xi(0))(1. + xi(1))(1 + xi(2))(+xi(0) + xi(1) +xi(2) -2)
      // N7 = 0.125*(1 - xi(0))(1. + xi(1))(1 + xi(2))(-xi(0) + xi(1) +xi(2) -2)
      REQUIRE(shapefn(0) == Approx(-0.0546875).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.1171875).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.2109375).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.1171875).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(-0.1171875).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(-0.2109375).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(-0.2109375).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(-0.2109375).epsilon(Tolerance));

      // Midside nodes
      // N8 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 - xi(2))
      // N9 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N10 = 0.25*(1 - xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N11 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N12 = 0.25*(1 + xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N13 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 - xi(2))
      // N14 = 0.25*(1 + xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N15 = 0.25*(1 - xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N16 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 + xi(2))
      // N17 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N18 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N19 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 + xi(2))
      REQUIRE(shapefn(8) == Approx(0.046875).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.046875).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.046875).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.421875).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(16) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(17) == Approx(0.140625).epsilon(Tolerance));
      REQUIRE(shapefn(18) == Approx(0.421875).epsilon(Tolerance));
      REQUIRE(shapefn(19) == Approx(0.421875).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      // Derivatives with respect to eta-direction
      // Edge nodes
      // G(0,0) = 0.125*(1. - xi(1))(1 - xi(2))(2*xi(0) + xi(1) + xi(2) + 1.)
      // G(1,0) = 0.125*(1. - xi(1))(1 - xi(2))(2*xi(0) - xi(1) - xi(2) - 1.)
      // G(2,0) = 0.125*(1. + xi(1))(1 - xi(2))(2*xi(0) + xi(1) - xi(2) - 1.)
      // G(3,0) = 0.125*(1. + xi(1))(1 - xi(2))(2*xi(0) - xi(1) + xi(2) + 1.)
      // G(4,0) = 0.125*(1. - xi(1))(1 + xi(2))(2*xi(0) + xi(1) - xi(2) + 1.)
      // G(5,0) = 0.125*(1. - xi(1))(1 + xi(2))(2*xi(0) - xi(1) + xi(2) - 1.)
      // G(6,0) = 0.125*(1. + xi(1))(1 + xi(2))(2*xi(0) + xi(1) + xi(2) - 1.)
      // G(7,0) = 0.125*(1. + xi(1))(1 + xi(2))(2*xi(0) - xi(1) - xi(2) + 1.)
      REQUIRE(gradsf(0, 0) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(-0.03125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.28125).epsilon(Tolerance));

      // Midside nodes
      // G(8, 0) = -0.5*xi(0)*(1 - xi(1))(1 - xi(2));
      // G(9, 0) = -0.25*(1 - xi(1)xi(1))(1 - xi(2));
      // G(10, 0) = -0.25*(1 - xi(2)xi(2))(1 - xi(1));
      // G(11, 0) = 0.25*(1 - xi(1)xi(1))(1 - xi(2));
      // G(12, 0) = 0.25*(1 - xi(2)xi(2))(1 - xi(1));
      // G(13, 0) = -0.5*xi(0)(1 + xi(1))(1 - xi(2));
      // G(14, 0) = 0.25*(1 - xi(2)xi(2))(1 + xi(1));
      // G(15, 0) = -0.25*(1 - xi(2)xi(2))(1 + xi(1));
      // G(16, 0) = -0.5*xi(0)(1 - xi(1))(1 + xi(2));
      // G(17, 0) = -0.25*(1 - xi(1)xi(1))(1 + xi(2));
      // G(18, 0) = 0.25*(1 - xi(1)xi(1))(1 + xi(2));
      // G(19, 0) = -0.5*xi(0)(1 + xi(1))(1 + xi(2));
      REQUIRE(gradsf(8, 0) == Approx(-0.0625).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(16, 0) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(17, 0) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(18, 0) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(19, 0) == Approx(-0.5625).epsilon(Tolerance));

      // Derivatives with respect to psi-direction
      // Edge nodes
      // G(0,1) = 0.125*(1. - xi(0))(1 - xi(2))(+xi(0) + 2*xi(1) + xi(2) + 1.)
      // G(1,1) = 0.125*(1. + xi(0))(1 - xi(2))(-xi(0) + 2*xi(1) + xi(2) + 1.)
      // G(2,1) = 0.125*(1. + xi(0))(1 - xi(2))(+xi(0) + 2*xi(1) - xi(2) - 1.)
      // G(3,1) = 0.125*(1. - xi(0))(1 - xi(2))(-xi(0) + 2*xi(1) - xi(2) - 1.)
      // G(4,1) = 0.125*(1. - xi(0))(1 + xi(2))(+xi(0) + 2*xi(1) - xi(2) + 1.)
      // G(5,1) = 0.125*(1. + xi(0))(1 + xi(2))(-xi(0) + 2*xi(1) - xi(2) + 1.)
      // G(6,1) = 0.125*(1. + xi(0))(1 + xi(2))(+xi(0) + 2*xi(1) + xi(2) - 1.)
      // G(7,1) = 0.125*(1. - xi(0))(1 + xi(2))(-xi(0) + 2*xi(1) + xi(2) - 1.)
      REQUIRE(gradsf(0, 1) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(-0.03125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));

      // Midside nodes
      // G(8, 1) = -0.25*(1 - xi(0)xi(0))(1 - xi(2));
      // G(9, 1) = -0.5*xi(1)(1 - xi(0))(1 - xi(2));
      // G(10, 1) = -0.25*(1 - xi(2)xi(2))(1 - xi(0));
      // G(11, 1) = -0.5*xi(1)(1 + xi(0))(1 - xi(2));
      // G(12, 1) = -0.25*(1 - xi(2)xi(2))(1 + xi(0));
      // G(13, 1) = 0.25*(1 - xi(0)xi(0))(1 - xi(2));
      // G(14, 1) = 0.25*(1 - xi(2)xi(2))(1 + xi(0));
      // G(15, 1) = 0.25*(1 - xi(2)xi(2))(1 - xi(0));
      // G(16, 1) = -0.25*(1 - xi(0)xi(0))(1 + xi(2));
      // G(17, 1) = -0.5*xi(1)(1 - xi(0))(1 + xi(2));
      // G(18, 1) = -0.5*xi(1)(1 + xi(0))(1 + xi(2));
      // G(19, 1) = 0.25*(1 - xi(0)xi(0))(1 + xi(2));
      REQUIRE(gradsf(8, 1) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(-0.0625).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(16, 1) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(17, 1) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(18, 1) == Approx(-0.5625).epsilon(Tolerance));
      REQUIRE(gradsf(19, 1) == Approx(0.28125).epsilon(Tolerance));

      // Derivatives with respect to mu-direction
      // Edge nodes
      // G(0,2) = 0.125*(1. - xi(0))(1 - xi(1))(+xi(0) + xi(1) + 2*xi(2) + 1.)
      // G(1,2) = 0.125*(1. + xi(0))(1 - xi(1))(-xi(0) + xi(1) + 2*xi(2) + 1.)
      // G(2,2) = 0.125*(1. + xi(0))(1 + xi(1))(-xi(0) - xi(1) + 2*xi(2) + 1.)
      // G(3,2) = 0.125*(1. - xi(0))(1 + xi(1))(+xi(0) - xi(1) + 2*xi(2) + 1.)
      // G(4,2) = 0.125*(1. - xi(0))(1 - xi(1))(-xi(0) - xi(1) + 2*xi(2) - 1.)
      // G(5,2) = 0.125*(1. + xi(0))(1 - xi(1))(+xi(0) - xi(1) + 2*xi(2) - 1.)
      // G(6,2) = 0.125*(1. + xi(0))(1 + xi(1))(+xi(0) + xi(1) + 2*xi(2) - 1.)
      // G(7,2) = 0.125*(1. - xi(0))(1 + xi(1))(-xi(0) + xi(1) + 2*xi(2) - 1.)
      REQUIRE(gradsf(0, 2) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(-0.03125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(0.0).epsilon(Tolerance));

      // Midside nodes
      // G(8, 2) = -0.25*(1 - xi(0)xi(0))(1 - xi(1));
      // G(9, 2) = -0.25*(1 - xi(1)xi(1))(1 - xi(0));
      // G(10, 2) = -0.5*xi(2)(1 - xi(0))(1 - xi(1));
      // G(11, 2) = -0.25*(1 - xi(1)xi(1))(1 + xi(0));
      // G(12, 2) = -0.5*xi(2)(1 + xi(0))(1 - xi(1));
      // G(13, 2) = -0.25*(1 - xi(0)xi(0))(1 + xi(1));
      // G(14, 2) = -0.5*xi(2)(1 + xi(0))(1 + xi(1));
      // G(15, 2) = -0.5*xi(2)(1 - xi(0))(1 + xi(1));
      // G(16, 2) = 0.25*(1 - xi(0)xi(0))(1 - xi(1));
      // G(17, 2) = 0.25*(1 - xi(1)xi(1))(1 - xi(0));
      // G(18, 2) = 0.25*(1 - xi(1)xi(1))(1 + xi(0));
      // G(19, 2) = 0.25*(1 - xi(0)xi(0))(1 + xi(1));
      REQUIRE(gradsf(8, 2) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(9, 2) == Approx(-0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(10, 2) == Approx(-0.0625).epsilon(Tolerance));
      REQUIRE(gradsf(11, 2) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(12, 2) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(13, 2) == Approx(-0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(14, 2) == Approx(-0.5625).epsilon(Tolerance));
      REQUIRE(gradsf(15, 2) == Approx(-0.1875).epsilon(Tolerance));
      REQUIRE(gradsf(16, 2) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(17, 2) == Approx(0.09375).epsilon(Tolerance));
      REQUIRE(gradsf(18, 2) == Approx(0.28125).epsilon(Tolerance));
      REQUIRE(gradsf(19, 2) == Approx(0.28125).epsilon(Tolerance));
    }

    // Coordinates is (0, 0, 0)
    SECTION(
        "Twenty noded local sf hexahedron element for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = hex->shapefn_local(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      // Edge nodes
      // N0 = 0.125*(1 - xi(0))(1. - xi(1))(1 - xi(2))(-xi(0) - xi(1) -xi(2) -2)
      // N1 = 0.125*(1 + xi(0))(1. - xi(1))(1 - xi(2))(+xi(0) - xi(1) -xi(2) -2)
      // N2 = 0.125*(1 + xi(0))(1. + xi(1))(1 - xi(2))(+xi(0) + xi(1) -xi(2) -2)
      // N3 = 0.125*(1 - xi(0))(1. + xi(1))(1 - xi(2))(-xi(0) + xi(1) -xi(2) -2)
      // N4 = 0.125*(1 - xi(0))(1. - xi(1))(1 + xi(2))(-xi(0) - xi(1) +xi(2) -2)
      // N5 = 0.125*(1 + xi(0))(1. - xi(1))(1 + xi(2))(+xi(0) - xi(1) +xi(2) -2)
      // N6 = 0.125*(1 + xi(0))(1. + xi(1))(1 + xi(2))(+xi(0) + xi(1) +xi(2) -2)
      // N7 = 0.125*(1 - xi(0))(1. + xi(1))(1 + xi(2))(-xi(0) + xi(1) +xi(2) -2)
      REQUIRE(shapefn(0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(-0.25).epsilon(Tolerance));

      // Midside nodes
      // N8 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 - xi(2))
      // N9 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N10 = 0.25*(1 - xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N11 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 - xi(2))
      // N12 = 0.25*(1 + xi(0))(1 - xi(1))(1 - xi(2)^2)
      // N13 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 - xi(2))
      // N14 = 0.25*(1 + xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N15 = 0.25*(1 - xi(0))(1 + xi(1))(1 - xi(2)^2)
      // N16 = 0.25*(1 - xi(0)^2)(1 - xi(1))(1 + xi(2))
      // N17 = 0.25*(1 - xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N18 = 0.25*(1 + xi(0))(1 - xi(1)^2)(1 + xi(2))
      // N19 = 0.25*(1 - xi(0)^2)(1 + xi(1))(1 + xi(2))
      REQUIRE(shapefn(8) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(16) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(17) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(18) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(19) == Approx(0.25).epsilon(Tolerance));
    }

    SECTION("Twenty noded hexahedron element with grad deformation") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      auto shapefn = hex->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      // Edge nodes
      REQUIRE(shapefn(0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(-0.25).epsilon(Tolerance));

      // Midside nodes
      REQUIRE(shapefn(8) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(16) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(17) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(18) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(19) == Approx(0.25).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = hex->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      // Derivatives with respect to eta-direction
      // Edge nodes
      REQUIRE(gradsf(0, 0) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(+0.125).epsilon(Tolerance));

      // Midside nodes
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(16, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(17, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(18, 0) == Approx(+0.25).epsilon(Tolerance));
      REQUIRE(gradsf(19, 0) == Approx(0.0).epsilon(Tolerance));

      // Derivatives with respect to psi-direction
      // Edge nodes
      REQUIRE(gradsf(0, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(-0.125).epsilon(Tolerance));

      // Midside nodes
      REQUIRE(gradsf(8, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(16, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(17, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(18, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(19, 1) == Approx(0.25).epsilon(Tolerance));

      // Derivatives with respect to mu-direction
      // Edge nodes
      REQUIRE(gradsf(0, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(+0.125).epsilon(Tolerance));
      REQUIRE(gradsf(4, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(5, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(6, 2) == Approx(-0.125).epsilon(Tolerance));
      REQUIRE(gradsf(7, 2) == Approx(-0.125).epsilon(Tolerance));

      // Midside nodes
      REQUIRE(gradsf(8, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(9, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(10, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(12, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(14, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(16, 2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(17, 2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(18, 2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(19, 2) == Approx(0.25).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION("20 noded hexahedron Jacobian for local coordinates(0.5,0.5,0.5)") {
      Eigen::Matrix<double, 20, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0, 0.50,
                4.0, 2.0, 1.00,
                2.0, 4.0, 1.00,
                1.0, 3.0, 0.50,
                2.0, 1.0, 1.50,
                4.0, 2.0, 2.00,
                2.0, 4.0, 2.00,
                1.0, 3.0, 1.50,
                3.0, 1.5, 0.75,
                1.5, 2.0, 0.50,
                2.0, 1.0, 1.00,
                3.0, 3.0, 1.00,
                4.0, 2.0, 1.50,
                1.5, 3.5, 0.75,
                2.0, 4.0, 1.50,
                1.0, 3.0, 1.00,
                3.0, 1.5, 1.75,
                1.5, 2.0, 1.50,
                4.0, 2.0, 1.50,
                1.5, 3.5, 1.75;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.90625, 0.21875, 0.109375,
                 -1.43750, 1.56250, 0.281250,
                  0.28125,-0.28125, 0.359375;
      // clang-format on

      // Get Jacobian
      auto jac = hex->jacobian(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION(
        "20 noded hexahedron local Jacobian for local "
        "coordinates(0.5,0.5,0.5)") {
      Eigen::Matrix<double, 20, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0, 0.50,
                4.0, 2.0, 1.00,
                2.0, 4.0, 1.00,
                1.0, 3.0, 0.50,
                2.0, 1.0, 1.50,
                4.0, 2.0, 2.00,
                2.0, 4.0, 2.00,
                1.0, 3.0, 1.50,
                3.0, 1.5, 0.75,
                1.5, 2.0, 0.50,
                2.0, 1.0, 1.00,
                3.0, 3.0, 1.00,
                4.0, 2.0, 1.50,
                1.5, 3.5, 0.75,
                2.0, 4.0, 1.50,
                1.0, 3.0, 1.00,
                3.0, 1.5, 1.75,
                1.5, 2.0, 1.50,
                4.0, 2.0, 1.50,
                1.5, 3.5, 1.75;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.90625, 0.21875, 0.109375,
                 -1.43750, 1.56250, 0.281250,
                  0.28125,-0.28125, 0.359375;
      // clang-format on

      // Get Jacobian
      auto jac = hex->jacobian_local(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian with deformation gradient
    SECTION("20 noded hexahedron Jacobian with deformation gradient") {
      Eigen::Matrix<double, 20, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0, 0.50,
                4.0, 2.0, 1.00,
                2.0, 4.0, 1.00,
                1.0, 3.0, 0.50,
                2.0, 1.0, 1.50,
                4.0, 2.0, 2.00,
                2.0, 4.0, 2.00,
                1.0, 3.0, 1.50,
                3.0, 1.5, 0.75,
                1.5, 2.0, 0.50,
                2.0, 1.0, 1.00,
                3.0, 3.0, 1.00,
                4.0, 2.0, 1.50,
                1.5, 3.5, 0.75,
                2.0, 4.0, 1.50,
                1.0, 3.0, 1.00,
                3.0, 1.5, 1.75,
                1.5, 2.0, 1.50,
                4.0, 2.0, 1.50,
                1.5, 3.5, 1.75;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5, 0.5;

      Eigen::Matrix<double, Dim, 1> psize;
      psize << 0.25, 0.5, 0.75;
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.90625, 0.21875, 0.109375,
                 -1.43750, 1.56250, 0.281250,
                  0.28125,-0.28125, 0.359375;
      // clang-format on

      // Get Jacobian
      auto jac = hex->jacobian(xi, coords, psize, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    SECTION("20-noded hexahedron B-matrix and Jacobian failure") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      Eigen::Matrix<double, 7, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                1., 1., 0.,
                0., 1., 0.,
                0., 0., 1.,
                1., 0., 1.,
                1., 1., 1.;
      // clang-format on
      // Get B-Matrix
      hex->bmatrix(xi, coords, zero, zero_matrix);
      hex->jacobian(xi, coords, zero, zero_matrix);
    }

    // Ni Nj matrix of a cell
    SECTION("20-noded hexahedron ni-nj-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);

      xi << -one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 8);
    }

    // Laplace matrix of a cell
    SECTION("20-noded hexahedron laplace-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);

      xi << -one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 8);

      // Nodal coordinates
      Eigen::Matrix<double, 20, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0, 0.50,
                4.0, 2.0, 1.00,
                2.0, 4.0, 1.00,
                1.0, 3.0, 0.50,
                2.0, 1.0, 1.50,
                4.0, 2.0, 2.00,
                2.0, 4.0, 2.00,
                1.0, 3.0, 1.50,
                3.0, 1.5, 0.75,
                1.5, 2.0, 0.50,
                2.0, 1.0, 1.00,
                3.0, 3.0, 1.00,
                4.0, 2.0, 1.50,
                1.5, 3.5, 0.75,
                2.0, 4.0, 1.50,
                1.0, 3.0, 1.00,
                3.0, 1.5, 1.75,
                1.5, 2.0, 1.50,
                4.0, 2.0, 1.50,
                1.5, 3.5, 1.75;
      // clang-format on
    }

    SECTION("20-noded hexahedron coordinates of unit cell") {
      const unsigned nfunctions = 20;

      // Coordinates of a unit cell
      Eigen::Matrix<double, nfunctions, Dim> unit_cell;
      // clang-format off
      unit_cell << -1., -1., -1.,
                    1., -1., -1.,
                    1.,  1., -1.,
                   -1.,  1., -1.,
                   -1., -1.,  1.,
                    1., -1.,  1.,
                    1.,  1.,  1.,
                   -1.,  1.,  1.,
                    0., -1., -1.,
                   -1.,  0., -1.,
                   -1., -1.,  0.,
                    1.,  0., -1.,
                    1., -1.,  0.,
                    0.,  1., -1.,
                    1.,  1.,  0.,
                   -1.,  1.,  0.,
                    0., -1.,  1.,
                   -1.,  0.,  1.,
                    1.,  0.,  1.,
                    0.,  1.,  1.;
      // clang-format on

      auto coordinates = hex->unit_cell_coordinates();
      REQUIRE(coordinates.rows() == nfunctions);
      REQUIRE(coordinates.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {  // Iterate through nfunctions
        for (unsigned j = 0; j < Dim; ++j) {       // Dimension
          REQUIRE(coordinates(i, j) ==
                  Approx(unit_cell(i, j)).epsilon(Tolerance));
        }
      }
    }

    SECTION("20-noded hexahedron element for sides indices") {
      // Check for sides indices
      Eigen::MatrixXi indices = hex->sides_indices();
      REQUIRE(indices.rows() == 12);
      REQUIRE(indices.cols() == 2);
      REQUIRE(indices(0, 0) == 0);
      REQUIRE(indices(0, 1) == 1);

      REQUIRE(indices(1, 0) == 1);
      REQUIRE(indices(1, 1) == 2);

      REQUIRE(indices(2, 0) == 2);
      REQUIRE(indices(2, 1) == 3);

      REQUIRE(indices(3, 0) == 3);
      REQUIRE(indices(3, 1) == 0);

      REQUIRE(indices(4, 0) == 4);
      REQUIRE(indices(4, 1) == 5);

      REQUIRE(indices(5, 0) == 5);
      REQUIRE(indices(5, 1) == 6);

      REQUIRE(indices(6, 0) == 6);
      REQUIRE(indices(6, 1) == 7);

      REQUIRE(indices(7, 0) == 7);
      REQUIRE(indices(7, 1) == 4);

      REQUIRE(indices(8, 0) == 0);
      REQUIRE(indices(8, 1) == 4);

      REQUIRE(indices(9, 0) == 1);
      REQUIRE(indices(9, 1) == 5);

      REQUIRE(indices(10, 0) == 2);
      REQUIRE(indices(10, 1) == 6);

      REQUIRE(indices(11, 0) == 3);
      REQUIRE(indices(11, 1) == 7);
    }

    SECTION("20-noded hexahedron element for corner indices") {
      // Check for volume indices
      Eigen::VectorXi indices = hex->corner_indices();
      REQUIRE(indices.size() == 8);
      REQUIRE(indices(0) == 0);
      REQUIRE(indices(1) == 1);
      REQUIRE(indices(2) == 2);
      REQUIRE(indices(3) == 3);
      REQUIRE(indices(4) == 4);
      REQUIRE(indices(5) == 5);
      REQUIRE(indices(6) == 6);
      REQUIRE(indices(7) == 7);
    }

    SECTION("20-noded noded hexahedron shape function for face indices") {
      // Check for face indices
      Eigen::Matrix<int, 6, 8> indices;
      // clang-format off
      indices << 0, 1, 5, 4,  8, 12, 16, 10,
                 5, 1, 2, 6, 12, 11, 14, 18,
                 7, 6, 2, 3, 19, 14, 13, 15,
                 0, 4, 7, 3, 10, 17, 15,  9,
                 1, 0, 3, 2,  8,  9, 13, 11,
                 4, 5, 6, 7, 16, 18, 19, 17;
      // clang-format on

      // Check for all face indices
      for (unsigned i = 0; i < indices.rows(); ++i) {
        const auto check_indices = hex->face_indices(i);
        REQUIRE(check_indices.rows() == 8);
        REQUIRE(check_indices.cols() == 1);

        for (unsigned j = 0; j < indices.cols(); ++j)
          REQUIRE(check_indices(j) == indices(i, j));
      }

      // Check number of faces
      REQUIRE(hex->nfaces() == 6);
    }
    SECTION("Hexahedron element length") {
      // Check element length
      REQUIRE(hex->unit_element_length() == Approx(2).epsilon(Tolerance));
    }

    SECTION("Nonlocal functions check fail") {
      // Check illegal functions
      Eigen::MatrixXd error;
      REQUIRE_THROWS(hex->initialise_bspline_connectivity_properties(
          error, std::vector<std::vector<unsigned>>()));
      REQUIRE_THROWS(
          hex->initialise_lme_connectivity_properties(0.0, 0.0, true, error));
    }
  }
}
