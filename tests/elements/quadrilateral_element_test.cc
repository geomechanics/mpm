// Quadrilateral element test
#include <memory>

#include "catch.hpp"

#include "quadrilateral_element.h"

//! \brief Check quadrilateral element class
TEST_CASE("Quadrilateral elements are checked", "[quad][element][2D]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-7;

  Eigen::Vector2d zero = Eigen::Vector2d::Zero();
  const Eigen::Matrix2d zero_matrix = Eigen::Matrix2d::Zero();

  //! Check for 4 noded element
  SECTION("Quadrilateral element with four nodes") {
    const unsigned nfunctions = 4;
    std::shared_ptr<mpm::Element<Dim>> quad =
        std::make_shared<mpm::QuadrilateralElement<Dim, nfunctions>>();

    // Check degree
    REQUIRE(quad->degree() == mpm::ElementDegree::Linear);
    REQUIRE(quad->shapefn_type() == mpm::ShapefnType::NORMAL_MPM);

    // Coordinates is (0,0)
    SECTION("Four noded quadrilateral element for coordinates(0,0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);
      REQUIRE(quad->nfunctions() == nfunctions);
      REQUIRE(quad->nfunctions_local() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(0, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.25).epsilon(Tolerance));
    }

    // Coordinates is (-1, -1);
    SECTION("Four noded quadrilateral element for coordinates(-1, -1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1., -1.;
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(0, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.5).epsilon(Tolerance));
    }

    // Coordinates is (1,1)
    SECTION("Four noded quadrilateral element for coordinates(1,1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1., 1.;
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.5).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (0,0)
    SECTION("Four noded local sf quadrilateral element for coordinates(0,0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn_local(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));
    }

    // Check shapefn with deformation gradient
    SECTION(
        "Four noded quadrilateral element shapefn with deformation gradient") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();
      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(0, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.25).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION(
        "Four noded quadrilateral Jacobian for local coordinates(0.5,0.5)") {
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 2., 1.,
                4., 2.,
                2., 4.,
                1., 3.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check local Jacobian
    SECTION(
        "Four noded quadrilateral local Jacobian for local "
        "coordinates(0.5,0.5)") {
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 2., 1.,
                4., 2.,
                2., 4.,
                1., 3.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian_local(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION("Four noded quadrilateral Jacobian with deformation gradient") {
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 2., 1.,
                4., 2.,
                2., 4.,
                1., 3.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian(xi, coords, psize, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Coordinates is (0,0)
    SECTION("Four noded quadrilateral B-matrix cell for coordinates(0,0)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi.setZero();

      // Nodal coordinates
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check dN/dx local
      Eigen::Matrix<double, nfunctions, Dim> dndx_local;
      dndx_local << -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5;
      auto dn_dx_local = quad->dn_dx_local(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx_local.rows() == nfunctions);
      REQUIRE(dn_dx_local.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i)
        for (unsigned j = 0; j < dn_dx_local.cols(); ++j)
          REQUIRE(dn_dx_local(i, j) ==
                  Approx(dndx_local(i, j)).epsilon(Tolerance));

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (0.5,0.5)
    SECTION("Four noded quadrilateral B-matrix cell for coordinates(0.5,0.5)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (-0.5,-0.5)
    SECTION(
        "Four noded quadrilateral B-matrix cell for coordinates(-0.5,-0.5)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << -0.5, -0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Check BMatrix with deformation gradient
    SECTION(
        "Four noded quadrilateral B-matrix cell with deformation gradient") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, psize, defgrad);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, psize, defgrad);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    SECTION("Four noded quadrilateral B-matrix and Jacobian failure") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0.;

      Eigen::Matrix<double, 3, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.;
      // clang-format on
      // Get B-Matrix
      quad->bmatrix(xi, coords, zero, zero_matrix);
      quad->jacobian(xi, coords, zero, zero_matrix);
    }

    // Ni Nj matrix of a cell
    SECTION("Four noded quadrilateral ni-nj-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);
    }

    // Laplace matrix of a cell
    SECTION("Four noded quadrilateral laplace-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);

      Eigen::Matrix<double, 4, Dim> coords;
      // clang-format off
      coords << 2., 1.,
                4., 2.,
                2., 4.,
                1., 3.;
      // clang-format on
    }

    SECTION("Four noded quadrilateral coordinates of unit cell") {
      const unsigned nfunctions = 4;

      // Coordinates of a unit cell
      Eigen::Matrix<double, nfunctions, Dim> unit_cell;
      // clang-format off
      unit_cell << -1., -1.,
                    1., -1.,
                    1.,  1.,
                   -1.,  1.;
      // clang-format on

      auto coordinates = quad->unit_cell_coordinates();
      REQUIRE(coordinates.rows() == nfunctions);
      REQUIRE(coordinates.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {  // Iterate through nfunctions
        for (unsigned j = 0; j < Dim; ++j) {       // Dimension
          REQUIRE(coordinates(i, j) ==
                  Approx(unit_cell(i, j)).epsilon(Tolerance));
        }
      }
    }

    SECTION("Four noded quadrilateral element for side indices") {
      // Check for sides indices
      Eigen::MatrixXi indices = quad->sides_indices();
      REQUIRE(indices.rows() == 4);
      REQUIRE(indices.cols() == 2);

      REQUIRE(indices(0, 0) == 0);
      REQUIRE(indices(0, 1) == 1);

      REQUIRE(indices(1, 0) == 1);
      REQUIRE(indices(1, 1) == 2);

      REQUIRE(indices(2, 0) == 2);
      REQUIRE(indices(2, 1) == 3);

      REQUIRE(indices(3, 0) == 3);
      REQUIRE(indices(3, 1) == 0);
    }

    SECTION("Four noded quadrilateral element for corner indices") {
      // Check for corner indices
      Eigen::VectorXi indices = quad->corner_indices();
      REQUIRE(indices.size() == 4);
      REQUIRE(indices(0) == 0);
      REQUIRE(indices(1) == 1);
      REQUIRE(indices(2) == 2);
      REQUIRE(indices(3) == 3);
    }

    SECTION("Four noded quadrilateral shape function for face indices") {
      // Check for face indices
      Eigen::Matrix<int, 4, 2> indices;
      // clang-format off
      indices << 0, 1,
                 1, 2,
                 2, 3,
                 3, 0;
      // clang-format on
      // Check for all face indices
      for (unsigned i = 0; i < indices.rows(); ++i) {
        const auto check_indices = quad->face_indices(i);
        REQUIRE(check_indices.rows() == 2);
        REQUIRE(check_indices.cols() == 1);

        for (unsigned j = 0; j < indices.cols(); ++j)
          REQUIRE(check_indices(j) == indices(i, j));
      }

      // Check number of faces
      REQUIRE(quad->nfaces() == 4);
    }
  }

  //! Check for 8 noded element
  SECTION("Quadrilateral element with eight nodes") {
    const unsigned nfunctions = 8;
    std::shared_ptr<mpm::Element<Dim>> quad =
        std::make_shared<mpm::QuadrilateralElement<Dim, nfunctions>>();

    // Check degree
    REQUIRE(quad->degree() == mpm::ElementDegree::Quadratic);
    REQUIRE(quad->shapefn_type() == mpm::ShapefnType::NORMAL_MPM);

    // Coordinates is (0,0)
    SECTION("Eight noded quadrilateral element for coordinates(0,0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.5).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(-0.5).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (-1,-1)
    SECTION("Eight noded quadrilateral element for coordinates(-1,-1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1., -1.;
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

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
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-1.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(4, 0) == Approx(2.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-1.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(-0.5).epsilon(Tolerance));

      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(2.0).epsilon(Tolerance));
    }

    // Coordinates is (1,1)
    SECTION("Eight noded quadrilateral element for coordinates(1, 1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1., 1.;
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(1.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(-2.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(1.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(-2.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (0,0)
    SECTION("Eight noded local sf quadrilateral element for coordinates(0,0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn_local(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.5).epsilon(Tolerance));
    }

    // Coordinates is (0,0)
    SECTION(
        "Eight noded quadrilateral element shapefn with deformation gradient") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();
      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.5).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(-0.5).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION(
        "Eight noded quadrilateral Jacobian for local coordinates(0.5,0.5)") {
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check local Jacobian
    SECTION(
        "Eight noded quadrilateral local Jacobian for local "
        "coordinates(0.5,0.5)") {
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian_local(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION("Eight noded quadrilateral Jacobian with deformation gradient") {
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian(xi, coords, psize, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Coordinates is (0,0)
    SECTION("Eight noded quadrilateral B-matrix cell for coordinates(0,0)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi.setZero();

      // Nodal coordinates
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (0.5,0.5)
    SECTION(
        "Eight noded quadrilateral B-matrix cell for coordinates(0.5,0.5)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (-0.5,-0.5)
    SECTION(
        "Eight noded quadrilateral B-matrix cell for coordinates(-0.5,-0.5)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << -0.5, -0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Check Bmatrix with deformation gradient
    SECTION("Eight noded quadrilateral B-matrix with deformation gradient") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> psize;
      psize << 0.5, 0.5;
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad << 0.5, 0.5, 0.5, 0.5;

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, psize, defgrad);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, psize, defgrad);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    SECTION("Eight noded quadrilateral B-matrix and Jacobian failure") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0.;

      Eigen::Matrix<double, 3, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.;
      // clang-format on
      // Get B-Matrix
      quad->bmatrix(xi, coords, zero, zero_matrix);
      quad->jacobian(xi, coords, zero, zero_matrix);
    }

    // Ni Nj matrix of a cell
    SECTION("Eight noded quadrilateral ni-nj-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);
    }

    // Laplace matrix of a cell
    SECTION("Four noded quadrilateral laplace-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);

      Eigen::Matrix<double, 8, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0;
      // clang-format on
    }

    SECTION("Eight noded quadrilateral coordinates of unit cell") {
      const unsigned nfunctions = 8;

      // Coordinates of a unit cell
      Eigen::Matrix<double, nfunctions, Dim> unit_cell;
      // clang-format off
      unit_cell << -1., -1.,
                    1., -1.,
                    1.,  1.,
                   -1.,  1.,
                    0., -1.,
                    1.,  0.,
                    0.,  1.,
                   -1.,  0.;
      // clang-format on

      auto coordinates = quad->unit_cell_coordinates();
      REQUIRE(coordinates.rows() == nfunctions);
      REQUIRE(coordinates.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {  // Iterate through nfunctions
        for (unsigned j = 0; j < Dim; ++j) {       // Dimension
          REQUIRE(coordinates(i, j) ==
                  Approx(unit_cell(i, j)).epsilon(Tolerance));
        }
      }
    }

    SECTION("Eight noded quadrilateral element for side indices") {
      // Check for sides indices
      Eigen::MatrixXi indices = quad->sides_indices();
      REQUIRE(indices.rows() == 4);
      REQUIRE(indices.cols() == 2);

      REQUIRE(indices(0, 0) == 0);
      REQUIRE(indices(0, 1) == 1);

      REQUIRE(indices(1, 0) == 1);
      REQUIRE(indices(1, 1) == 2);

      REQUIRE(indices(2, 0) == 2);
      REQUIRE(indices(2, 1) == 3);

      REQUIRE(indices(3, 0) == 3);
      REQUIRE(indices(3, 1) == 0);
    }

    SECTION("Eight noded quadrilateral element for corner indices") {
      // Check for corner indices
      Eigen::VectorXi indices = quad->corner_indices();
      REQUIRE(indices.size() == 4);
      REQUIRE(indices(0) == 0);
      REQUIRE(indices(1) == 1);
      REQUIRE(indices(2) == 2);
      REQUIRE(indices(3) == 3);
    }

    SECTION("Eight noded quadrilateral shape function for face indices") {
      // Check for face indices
      Eigen::Matrix<int, 4, 3> indices;
      // clang-format off
      indices << 0, 1, 4,
                 1, 2, 5,
                 2, 3, 6,
                 3, 0, 7;
      // clang-format on
      // Check for all face indices
      for (unsigned i = 0; i < indices.rows(); ++i) {
        const auto check_indices = quad->face_indices(i);
        REQUIRE(check_indices.rows() == 3);
        REQUIRE(check_indices.cols() == 1);

        for (unsigned j = 0; j < indices.cols(); ++j)
          REQUIRE(check_indices(j) == indices(i, j));
      }

      // Check number of faces
      REQUIRE(quad->nfaces() == 4);
    }
  }

  //! Check for 9 noded element
  SECTION("Quadrilateral element with nine nodes") {
    const unsigned nfunctions = 9;
    std::shared_ptr<mpm::Element<Dim>> quad =
        std::make_shared<mpm::QuadrilateralElement<Dim, nfunctions>>();

    // Check degree
    REQUIRE(quad->degree() == mpm::ElementDegree::Quadratic);
    REQUIRE(quad->shapefn_type() == mpm::ShapefnType::NORMAL_MPM);

    // Coordinates is (0,0)
    SECTION("Nine noded quadrilateral element for coordinates(0,0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(1.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (-1,-1)
    SECTION("Nine noded quadrilateral element for coordinates(-1,-1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1., -1.;
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

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
      REQUIRE(shapefn(8) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-1.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(2.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-1.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(2.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (1,1)
    SECTION("Nine noded quadrilateral element for coordinates(1, 1)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1., 1.;
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(1.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(-2.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(1.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(-2.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (0,0)
    SECTION("Nine noded local sf quadrilateral element for coordinates(0,0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn_local(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(1.0).epsilon(Tolerance));
    }

    SECTION("Nine noded quadrilateral element with deformation gradient") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      Eigen::Matrix<double, Dim, 1> psize;
      psize << 0.5, 0.1;
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad << -0.25, 0.1, -0.25, 0.1;

      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(1.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Ni Nj matrix of a cell
    SECTION("Nine noded quadrilateral ni-nj-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);
    }

    // Laplace matrix of a cell
    SECTION("Nine noded quadrilateral laplace-matrix") {
      std::vector<Eigen::Matrix<double, Dim, 1>> xi_s;

      Eigen::Matrix<double, Dim, 1> xi;
      const double one_by_sqrt3 = std::fabs(1 / std::sqrt(3));
      xi << -one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, -one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);
      xi << -one_by_sqrt3, one_by_sqrt3;
      xi_s.emplace_back(xi);

      REQUIRE(xi_s.size() == 4);

      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0,
                2.25, 2.5;
      // clang-format on
    }

    // Check Jacobian
    SECTION(
        "Nine noded quadrilateral Jacobian for local coordinates(0.5,0.5)") {
      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0,
                2.25, 2.5;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check local Jacobian
    SECTION(
        "Nine noded quadrilateral local Jacobian for local "
        "coordinates(0.5,0.5)") {
      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0,
                2.25, 2.5;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian_local(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian with deformation gradient
    SECTION("Nine noded quadrilateral Jacobian with deformation gradient") {
      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 2.0, 1.0,
                4.0, 2.0,
                2.0, 4.0,
                1.0, 3.0,
                3.0, 1.5,
                3.0, 3.0,
                1.5, 3.5,
                1.5, 2.0,
                2.25, 2.5;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      Eigen::Matrix<double, Dim, 1> psize;
      psize << -0.5, -0.5;
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad << -0.5, -0.5, -0.5, -0.5;

      // Jacobian result
      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 0.625, 0.5,
                 -0.875, 1.0;
      // clang-format on

      // Get Jacobian
      auto jac = quad->jacobian(xi, coords, psize, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Coordinates is (0,0)
    SECTION("Nine noded quadrilateral B-matrix cell for coordinates(0,0)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi.setZero();

      // Nodal coordinates
      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5,
                0.5, 0.5;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (0.5,0.5)
    SECTION("Nine noded quadrilateral B-matrix cell for coordinates(0.5,0.5)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5,
                0.5, 0.5;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    // Coordinates is (-0.5,-0.5)
    SECTION(
        "Nine noded quadrilateral B-matrix cell for coordinates(-0.5,-0.5)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << -0.5, -0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5,
                0.5, 0.5;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, zero, zero_matrix);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, zero, zero_matrix);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    SECTION("Nine noded quadrilateral B-matrix with deformation gradient)") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << -0.5, -0.5;

      Eigen::Matrix<double, Dim, 1> psize;
      psize << -0.5, -0.5;
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad << -0.5, -0.5, -0.5, -0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 9, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0.5, 0.,
                1.0, 0.5,
                0.5, 1.,
                0., 0.5,
                0.5, 0.5;
      // clang-format on

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, psize, defgrad);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, psize, defgrad);
      gradsf *= 2.;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, zero_matrix);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check size of B-matrix
      REQUIRE(bmatrix.size() == nfunctions);

      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
        REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
      }
    }

    SECTION("Nine noded quadrilateral B-matrix and Jacobian failure") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0.;

      Eigen::Matrix<double, 3, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.;
      // clang-format on
      // Get B-Matrix
      quad->bmatrix(xi, coords, zero, zero_matrix);
      quad->jacobian(xi, coords, zero, zero_matrix);
    }

    SECTION("Nine noded quadrilateral coordinates of unit cell") {
      const unsigned nfunctions = 9;

      // Coordinates of a unit cell
      Eigen::Matrix<double, nfunctions, Dim> unit_cell;
      // clang-format off
      unit_cell << -1., -1.,
                    1., -1.,
                    1.,  1.,
                   -1.,  1.,
                    0., -1.,
                    1.,  0.,
                    0.,  1.,
                   -1.,  0.,
                    0.,  0.;
      // clang-format on

      auto coordinates = quad->unit_cell_coordinates();
      REQUIRE(coordinates.rows() == nfunctions);
      REQUIRE(coordinates.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {  // Iterate through nfunctions
        for (unsigned j = 0; j < Dim; ++j) {       // Dimension
          REQUIRE(coordinates(i, j) ==
                  Approx(unit_cell(i, j)).epsilon(Tolerance));
        }
      }
    }

    SECTION("Nine noded quadrilateral element for side indices") {
      // Check for sides indices
      Eigen::MatrixXi indices = quad->sides_indices();
      REQUIRE(indices.rows() == 4);
      REQUIRE(indices.cols() == 2);

      REQUIRE(indices(0, 0) == 0);
      REQUIRE(indices(0, 1) == 1);

      REQUIRE(indices(1, 0) == 1);
      REQUIRE(indices(1, 1) == 2);

      REQUIRE(indices(2, 0) == 2);
      REQUIRE(indices(2, 1) == 3);

      REQUIRE(indices(3, 0) == 3);
      REQUIRE(indices(3, 1) == 0);
    }

    SECTION("Nine noded quadrilateral element for corner indices") {
      // Check for corner indices
      Eigen::VectorXi indices = quad->corner_indices();
      REQUIRE(indices.size() == 4);
      REQUIRE(indices(0) == 0);
      REQUIRE(indices(1) == 1);
      REQUIRE(indices(2) == 2);
      REQUIRE(indices(3) == 3);
    }

    SECTION("Nine noded quadrilateral shape function for face indices") {
      // Check for face indices
      Eigen::Matrix<int, 4, 3> indices;
      // clang-format off
      indices << 0, 1, 4,
                 1, 2, 5,
                 2, 3, 6,
                 3, 0, 7;
      // clang-format on

      // Check for all face indices
      for (unsigned i = 0; i < indices.rows(); ++i) {
        const auto check_indices = quad->face_indices(i);
        REQUIRE(check_indices.rows() == 3);
        REQUIRE(check_indices.cols() == 1);

        for (unsigned j = 0; j < indices.cols(); ++j)
          REQUIRE(check_indices(j) == indices(i, j));
      }

      // Check number of faces
      REQUIRE(quad->nfaces() == 4);
    }

    SECTION("Quadrilateral element length") {
      // Check element length
      REQUIRE(quad->unit_element_length() == Approx(2).epsilon(Tolerance));
    }

    SECTION("Nonlocal functions check fail") {
      // Check illegal functions
      Eigen::MatrixXd error;
      REQUIRE_THROWS(quad->initialise_bspline_connectivity_properties(
          error, std::vector<std::vector<unsigned>>()));
      REQUIRE_THROWS(
          quad->initialise_lme_connectivity_properties(0.0, 0.0, true, error));
    }
  }
}
