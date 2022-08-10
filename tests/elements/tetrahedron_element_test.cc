// Tetrahedron element test
#include <memory>

#include "catch.hpp"

#include "tetrahedron_element.h"

//! \brief Check tetrahedron element class
TEST_CASE("Tetrahedron elements are checked", "[tet][element][3D]") {
  const unsigned Dim = 3;
  const double Tolerance = 1.E-7;

  Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d zero_matrix = Eigen::Matrix3d::Zero();

  //! Check for 4 noded element
  SECTION("Tetrahedron element with four nodes") {
    const unsigned nfunctions = 4;
    std::shared_ptr<mpm::Element<Dim>> tet =
        std::make_shared<mpm::TetrahedronElement<Dim, nfunctions>>();

    // Check degree
    REQUIRE(tet->degree() == mpm::ElementDegree::Linear);
    REQUIRE(tet->shapefn_type() == mpm::ShapefnType::NORMAL_MPM);

    // Coordinates is (0, 0, 0)
    SECTION("Four noded tetrahedron element for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = tet->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = tet->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 2) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(1.0).epsilon(Tolerance));
    }

    // Coordinates is (1.0/3, 1.0/3, 1.0/3);
    SECTION(
        "Four noded tetrahedron element for coordinates (1.0/3, 1.0/3, "
        "1.0/3)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1.0 / 3, 1.0 / 3, 1.0 / 3;
      auto shapefn = tet->shapefn(coords, zero, zero_matrix);
      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(1.0 / 3).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(1.0 / 3).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(1.0 / 3).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = tet->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 2) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(1.0).epsilon(Tolerance));
    }

    // Coordinates is (0, 0, 0)
    SECTION(
        "Four noded local sf tetrahedron element for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = tet->shapefn_local(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
    }

    SECTION("Four noded tetrahedron shapefn with deformation gradient") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();
      auto shapefn = tet->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = tet->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 2) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 2) == Approx(1.0).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION("Four noded tetrahedron Jacobian for local coordinates(0,0,0)") {
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 1.0 / 3, 1.0 / 3, 1.0 / 3;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 1., 0., 0.,
                  0., 1., 0.,
                  0., 0., 1.;
      // clang-format on

      // Get Jacobian
      auto jac = tet->jacobian(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check local Jacobian
    SECTION(
        "Four noded tetrahedron local Jacobian for local coordinates(0,0,0.)") {
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 1., 0., 0.,
                  0., 1., 0.,
                  0., 0., 1.;
      // clang-format on

      // Get Jacobian
      auto jac = tet->jacobian_local(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION("Four noded tetrahedron Jacobian with deformation gradient") {
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 1., 0., 0.,
                  0., 1., 0.,
                  0., 0., 1.;
      // clang-format on

      // Get Jacobian
      auto jac = tet->jacobian(xi, coords, psize, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Coordinates is (0, 0, 0)
    SECTION("Four noded tetrahedron B-matrix cell for coordinates(0, 0, 0)") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = tet->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = tet->grad_shapefn(xi, zero, zero_matrix);

      // Check dN/dx
      auto dn_dx = tet->dn_dx(xi, coords, zero, zero_matrix);
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

    // Coordinates is (0, 0, 0)
    SECTION("Four noded tetrahedron B-matrix cell for xi(0., 0., 0.)") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = tet->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = tet->grad_shapefn(xi, zero, zero_matrix);

      // Check dN/dx
      auto dn_dx = tet->dn_dx(xi, coords, zero, zero_matrix);
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

    // Coordinates is (1.0/3, 1.0/3, 1.0/3)
    SECTION("Four noded tetrahedron B-matrix cell xi(1.0/3, 1.0/3, 1.0/3)") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 1.0 / 3, 1.0 / 3, 1.0 / 3;

      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = tet->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = tet->grad_shapefn(xi, zero, zero_matrix);

      // Check dN/dx
      auto dn_dx = tet->dn_dx(xi, coords, zero, zero_matrix);
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

    SECTION("Four noded tetrahedron B-matrix with deformation gradient") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 1.0 / 3, 1.0 / 3, 1.0 / 3;

      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = tet->bmatrix(xi, coords, psize, defgrad);

      // Check gradient of shape functions
      auto gradsf = tet->grad_shapefn(xi, psize, defgrad);

      // Check dN/dx
      auto dn_dx = tet->dn_dx(xi, coords, zero, zero_matrix);
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

    SECTION("Four noded tetrahedron B-matrix and Jacobian failure") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0., 0.;

      Eigen::Matrix<double, 3, Dim> coords;
      // clang-format off
      coords << 0., 0., 0.,
                1., 0., 0.,
                0., 1., 0.;
      // clang-format on
      // Get B-Matrix
      auto bmatrix = tet->bmatrix(xi, coords, zero, zero_matrix);
      auto jacobian = tet->jacobian(xi, coords, zero, zero_matrix);
    }

    // Ni Nj matrix of a cell
    SECTION("Four noded tetrahedron ni-nj-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.58541020, 0.13819660, 0.13819660;
      xi_s.emplace_back(xi);
      xi << 0.13819660, 0.58541020, 0.13819660;
      xi_s.emplace_back(xi);
      xi << 0.13819660, 0.13819660, 0.58541020;
      xi_s.emplace_back(xi);
      xi << 0.13819660, 0.13819660, 0.13819660;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);

      // Get Ni Nj matrix
      const auto ni_nj_matrix = tet->ni_nj_matrix(xi_s);

      // Check size of ni-nj-matrix
      REQUIRE(ni_nj_matrix.rows() == nfunctions);
      REQUIRE(ni_nj_matrix.cols() == nfunctions);

      // Sum should be equal to 1. * xi_s.size()
      REQUIRE(ni_nj_matrix.sum() ==
              Approx(1. * xi_s.size()).epsilon(Tolerance));

      Eigen::Matrix<double, 4, 4> mass;
      // clang-format off
      mass << 0.4, 0.2, 0.2, 0.2,
              0.2, 0.4, 0.2, 0.2,
              0.2, 0.2, 0.4, 0.2,
              0.2, 0.2, 0.2, 0.4;
      // clang-format on

      // auxiliary matrices for checking its multiplication by scalar
      auto ni_nj_matrix_unit = 1.0 * ni_nj_matrix;
      auto ni_nj_matrix_zero = 0.0 * ni_nj_matrix;
      auto ni_nj_matrix_negative = -2.0 * ni_nj_matrix;
      double scalar = 21.65489;
      auto ni_nj_matrix_scalar = scalar * ni_nj_matrix;

      for (unsigned i = 0; i < nfunctions; ++i) {
        for (unsigned j = 0; j < nfunctions; ++j) {
          REQUIRE(ni_nj_matrix(i, j) == Approx(mass(i, j)).epsilon(Tolerance));
          // check multiplication by unity;
          REQUIRE(ni_nj_matrix_unit(i, j) ==
                  Approx(1.0 * mass(i, j)).epsilon(Tolerance));
          // check multiplication by zero;
          REQUIRE(ni_nj_matrix_zero(i, j) ==
                  Approx(0.0 * mass(i, j)).epsilon(Tolerance));
          // check multiplication by negative number;
          REQUIRE(ni_nj_matrix_negative(i, j) ==
                  Approx(-2.0 * mass(i, j)).epsilon(Tolerance));
          // check multiplication by an arbitrary scalar;
          REQUIRE(ni_nj_matrix_scalar(i, j) ==
                  Approx(scalar * mass(i, j)).epsilon(Tolerance));
        }
      }
    }

    // Laplace matrix of a cell
    SECTION("Four noded tetrahedron laplace-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.58541020, 0.13819660, 0.13819660;
      xi_s.emplace_back(xi);
      xi << 0.13819660, 0.58541020, 0.13819660;
      xi_s.emplace_back(xi);
      xi << 0.13819660, 0.13819660, 0.58541020;
      xi_s.emplace_back(xi);
      xi << 0.13819660, 0.13819660, 0.13819660;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);

      // Nodal coordinates
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 1., 1., 1.,
                1., 0., 0.,
                0., 1., 0.,
                0., 0., 1.;
      // clang-format on

      // Get laplace matrix
      const auto laplace_matrix = tet->laplace_matrix(xi_s, coords);

      // Check size of laplace-matrix
      REQUIRE(laplace_matrix.rows() == nfunctions);
      REQUIRE(laplace_matrix.cols() == nfunctions);

      // Sum should be equal to 0.
      REQUIRE(laplace_matrix.sum() == Approx(0.).epsilon(Tolerance));

      Eigen::Matrix<double, 4, 4> laplace;
      // clang-format off
      laplace <<   3., -1., -1., -1.,
                  -1.,  3., -1., -1.,
                  -1., -1.,  3., -1.,
                  -1., -1., -1.,  3.;
      // clang-format on
      for (unsigned i = 0; i < nfunctions; ++i)
        for (unsigned j = 0; j < nfunctions; ++j)
          REQUIRE(laplace_matrix(i, j) ==
                  Approx(laplace(i, j)).epsilon(Tolerance));
    }

    SECTION("Four noded tetrahedron coordinates of unit cell") {
      const unsigned nfunctions = 4;
      // Coordinates of a unit cell
      Eigen::Matrix<double, nfunctions, Dim> unit_cell;
      // clang-format off
      unit_cell << 0., 0., 0.,
                   1., 0., 0.,
                   0., 1., 0.,
                   0., 0., 1.;
      // clang-format on

      auto coordinates = tet->unit_cell_coordinates();
      REQUIRE(coordinates.rows() == nfunctions);
      REQUIRE(coordinates.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {  // Iterate through nfunctions
        for (unsigned j = 0; j < Dim; ++j) {       // Dimension
          REQUIRE(coordinates(i, j) ==
                  Approx(unit_cell(i, j)).epsilon(Tolerance));
        }
      }
    }

    SECTION("Four noded tetrahedron element for sides indices") {
      // Check for sides indices
      Eigen::MatrixXi indices = tet->sides_indices();
      REQUIRE(indices.rows() == 6);
      REQUIRE(indices.cols() == 2);
      REQUIRE(indices(0, 0) == 0);
      REQUIRE(indices(0, 1) == 1);

      REQUIRE(indices(1, 0) == 1);
      REQUIRE(indices(1, 1) == 2);

      REQUIRE(indices(2, 0) == 2);
      REQUIRE(indices(2, 1) == 3);

      REQUIRE(indices(3, 0) == 3);
      REQUIRE(indices(3, 1) == 0);

      REQUIRE(indices(4, 0) == 1);
      REQUIRE(indices(4, 1) == 3);

      REQUIRE(indices(5, 0) == 0);
      REQUIRE(indices(5, 1) == 2);
    }

    SECTION("Four noded tetrahedron element for corner indices") {
      // Check for volume indices
      Eigen::VectorXi indices = tet->corner_indices();
      REQUIRE(indices.size() == 4);
      REQUIRE(indices(0) == 0);
      REQUIRE(indices(1) == 1);
      REQUIRE(indices(2) == 2);
      REQUIRE(indices(3) == 3);
    }

    SECTION("Four noded tetrahedron shape function for face indices") {
      // Check for face indices
      Eigen::Matrix<int, 4, 3> indices;
      // clang-format off
      indices << 0, 1, 2,
                 0, 1, 3,
                 0, 2, 3,
                 1, 2, 3;
      // clang-format on

      // Check for all face indices
      for (unsigned i = 0; i < indices.rows(); ++i) {
        const auto check_indices = tet->face_indices(i);
        REQUIRE(check_indices.rows() == 3);
        REQUIRE(check_indices.cols() == 1);

        for (unsigned j = 0; j < indices.cols(); ++j)
          REQUIRE(check_indices(j) == indices(i, j));
      }

      // Check number of faces
      REQUIRE(tet->nfaces() == 4);
    }

    // Global Point Coordinates (1/3, 1/3, 1/3)
    // Nodal Coordinates [0 0 0; 1 0 0; 0 1 0; 0 0 1]
    SECTION("Four noded tetrahedron natural coordinates of a point") {

      // Nodal coords
      Eigen::Matrix<double, 4, Dim> nodal_coords;
      // clang-format off
      nodal_coords << 0., 0., 0.,
                      1., 0., 0.,
                      0., 1., 0.,
                      0., 0., 1.;
      // clang-format on

      // Point coords (global)
      Eigen::Matrix<double, Dim, 1> point_coords;
      point_coords << 1.0 / 3, 1.0 / 3, 1.0 / 3;

      // Check xi
      auto xi = tet->natural_coordinates_analytical(point_coords, nodal_coords);
      REQUIRE(xi.size() == 3);
      REQUIRE(xi(0) == Approx(1.0 / 3).epsilon(Tolerance));
      REQUIRE(xi(1) == Approx(1.0 / 3).epsilon(Tolerance));
      REQUIRE(xi(2) == Approx(1.0 / 3).epsilon(Tolerance));
    }
  }
}
