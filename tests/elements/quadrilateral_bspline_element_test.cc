// Quadrilateral element test
#include <memory>

#include "catch.hpp"

#include "quadrilateral_bspline_element.h"

//! \brief Check quadrilateral bspline element class
TEST_CASE("Quadrilateral bspline elements are checked",
          "[quad][element][2D][bspline]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-6;

  Eigen::Vector2d zero = Eigen::Vector2d::Zero();
  const Eigen::Matrix2d zero_matrix = Eigen::Matrix2d::Zero();

  //! Check for center element nodes
  SECTION("Quadratic Quadrilateral BSpline Element") {
    const unsigned npolynomials = 2;
    std::shared_ptr<mpm::Element<Dim>> quad =
        std::make_shared<mpm::QuadrilateralBSplineElement<Dim, npolynomials>>();

    // Check degree and shapefn type
    REQUIRE(quad->degree() == mpm::ElementDegree::Quadratic);
    REQUIRE(quad->shapefn_type() == mpm::ShapefnType::BSPLINE);

    // Coordinates is (0,0) before upgraded
    SECTION("2D BSpline element for coordinates(0,0) before upgrade") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == 4);

      REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == 4);
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

    // Coordinates is (1,1)
    SECTION("2D BSpline element for coordinates(1,1) before upgrade") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1., 1.;
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == 4);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == 4);
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

    // Initialising upgrade properties
    SECTION("2D BSpline element regular element - nnodes = 16") {
      Eigen::Matrix<double, 16, Dim> nodal_coords;
      nodal_coords << -1., -1., 1., -1., 1., 1., -1., 1., -3., -3., -1., -3.,
          1., -3., 3., -3., -3., -1., 3., -1., -3., 1., 3., 1., -3., 3., -1.,
          3., 1., 3., 3., 3.;

      SECTION("2D BSpline element regular element no support") {
        std::vector<std::vector<unsigned>> nodal_props{
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0},
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};

        REQUIRE_NOTHROW(quad->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props));

        // Coordinates is (0,0) after upgrade
        SECTION("2D BSpline element for coordinates(0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = quad->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 16);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0.).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0.).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 16);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 16, Dim> gradsf_ans;
          gradsf_ans << -0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25, 0,
              0, -0, 0, 0, 0, 0, 0, 0, -0, 0, -0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0,
              0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (1,1) after upgrade
        SECTION("2D BSpline element for coordinates(1,1) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 1., 1.;
          auto shapefn = quad->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 16);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.015625).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.5625).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0.015625).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0.015625).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0.015625).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 16);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 16, Dim> gradsf_ans;
          gradsf_ans << -0.03125, -0.03125, 0, -0.1875, 0, 0, -0.1875, 0, 0, 0,
              -0, 0, 0, 0, 0, 0, 0, -0, 0.03125, -0.03125, 0, 0, 0.1875, 0, 0,
              0, -0.03125, 0.03125, 0, 0.1875, 0.03125, 0.03125;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (0,0)
        SECTION(
            "Four noded local sf quadrilateral element for coordinates(0,0)") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = quad->shapefn_local(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 4);

          REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));
        }

        // Check Jacobian
        SECTION("16-noded quadrilateral Jacobian with deformation gradient") {
          Eigen::Matrix<double, 16, Dim> coords;
          // clang-format off
          // clang-format off
          coords << -1., -1.,
                    1., -1.,
                    1.,  1.,
                    -1.,  1.,
                    -3., -3.,
                    -1., -3.,
                    1., -3.,
                    3., -3.,
                    3., -1.,
                    3.,  1.,
                    3.,  3.,
                    1.,  3.,
                    -1.,  3.,
                    -3.,  3.,
                    -3.,  1.,
                    -3., -1.;
          // clang-format on

          Eigen::Matrix<double, Dim, 1> psize;
          psize.setZero();
          Eigen::Matrix<double, Dim, Dim> defgrad;
          defgrad.setZero();

          Eigen::Matrix<double, Dim, 1> xi;
          xi << 0., 0.;

          Eigen::Matrix<double, Dim, Dim> jacobian;
          // clang-format off
          jacobian << 1.0, 0,
                      0, 1.0;
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
      }
    }

    SECTION("2D BSpline element mesh edge - nnodes = 12") {
      Eigen::Matrix<double, 12, Dim> nodal_coords;
      nodal_coords << -1., -1., 1., -1., 1., 1., -1., 1., 3., -1, 3., 1, 3., 3.,
          1., 3., -1., 3., -3., 3, -3., 1, -3., -1.;

      std::vector<std::vector<unsigned>> nodal_props{
          {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 1}, {0, 2},
          {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 2}, {0, 1}};

      REQUIRE_NOTHROW(quad->initialise_bspline_connectivity_properties(
          nodal_coords, nodal_props));

      // Coordinates is (0,0) after upgrade
      SECTION("2D BSpline element for coordinates(0,0) after upgrade") {
        Eigen::Matrix<double, Dim, 1> coords;
        coords.setZero();
        auto shapefn = quad->shapefn(coords, zero, zero_matrix);

        // Check shape function
        REQUIRE(shapefn.size() == 12);
        REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

        REQUIRE(shapefn(0) == Approx(0.3333333333).epsilon(Tolerance));
        REQUIRE(shapefn(1) == Approx(0.3333333333).epsilon(Tolerance));
        REQUIRE(shapefn(2) == Approx(0.1666666667).epsilon(Tolerance));
        REQUIRE(shapefn(3) == Approx(0.1666666667).epsilon(Tolerance));
        REQUIRE(shapefn(4) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(5) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(6) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(7) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(8) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(9) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(10) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(11) == Approx(0.).epsilon(Tolerance));

        // Check gradient of shape functions
        auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
        REQUIRE(gradsf.rows() == 12);
        REQUIRE(gradsf.cols() == Dim);

        Eigen::Matrix<double, 12, Dim> gradsf_ans;
        gradsf_ans << -0.3333333333, -0.3333333333, 0.3333333333, -0.3333333333,
            0.1666666667, 0.3333333333, -0.1666666667, 0.3333333333, 0, -0, 0,
            0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, -0;

        for (unsigned i = 0; i < gradsf.rows(); ++i)
          for (unsigned j = 0; j < gradsf.cols(); ++j)
            REQUIRE(gradsf(i, j) ==
                    Approx(gradsf_ans(i, j)).epsilon(Tolerance));
      }

      // Coordinates is (0.5,-0.5) after upgrade
      SECTION("2D BSpline element for coordinates(0.5,-0.5) after upgrade") {
        Eigen::Matrix<double, Dim, 1> coords;
        coords << 0.5, -0.5;
        auto shapefn = quad->shapefn(coords, zero, zero_matrix);

        // Check shape function
        REQUIRE(shapefn.size() == 12);
        REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

        REQUIRE(shapefn(0) == Approx(0.257812).epsilon(Tolerance));
        REQUIRE(shapefn(1) == Approx(0.630208).epsilon(Tolerance));
        REQUIRE(shapefn(2) == Approx(0.0572917).epsilon(Tolerance));
        REQUIRE(shapefn(3) == Approx(0.0234375).epsilon(Tolerance));
        REQUIRE(shapefn(4) == Approx(0.0286458).epsilon(Tolerance));
        REQUIRE(shapefn(5) == Approx(0.00260417).epsilon(Tolerance));
        REQUIRE(shapefn(6) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(7) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));

        // Check gradient of shape functions
        auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
        REQUIRE(gradsf.rows() == 12);
        REQUIRE(gradsf.cols() == Dim);

        Eigen::Matrix<double, 12, Dim> gradsf_ans;
        gradsf_ans << -0.34375, -0.09375, 0.229167, -0.229167, 0.0208333,
            0.229167, -0.03125, 0.09375, 0.114583, -0.0104167, 0.0104167,
            0.0104167, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, -0;

        for (unsigned i = 0; i < gradsf.rows(); ++i)
          for (unsigned j = 0; j < gradsf.cols(); ++j)
            REQUIRE(gradsf(i, j) ==
                    Approx(gradsf_ans(i, j)).epsilon(Tolerance));
      }
    }

    SECTION("2D BSpline element mesh corner - nnodes = 9") {
      Eigen::Matrix<double, 9, Dim> nodal_coords;
      nodal_coords << -1., -1., 1., -1., 1., 1., -1., 1., 3., -1, 3., 1, 3., 3.,
          1., 3., -1., 3.;

      std::vector<std::vector<unsigned>> nodal_props{{1, 1}, {2, 1}, {2, 2},
                                                     {1, 2}, {0, 1}, {0, 2},
                                                     {0, 0}, {2, 0}, {1, 0}};

      REQUIRE_NOTHROW(quad->initialise_bspline_connectivity_properties(
          nodal_coords, nodal_props));

      // Coordinates is (0,0) after upgrade
      SECTION("2D BSpline element for coordinates(0,0) after upgrade") {
        Eigen::Matrix<double, Dim, 1> coords;
        coords.setZero();
        auto shapefn = quad->shapefn(coords, zero, zero_matrix);

        // Check shape function
        REQUIRE(shapefn.size() == 9);
        REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

        REQUIRE(shapefn(0) == Approx(0.444444).epsilon(Tolerance));
        REQUIRE(shapefn(1) == Approx(0.222222).epsilon(Tolerance));
        REQUIRE(shapefn(2) == Approx(0.111111).epsilon(Tolerance));
        REQUIRE(shapefn(3) == Approx(0.222222).epsilon(Tolerance));
        REQUIRE(shapefn(4) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(5) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(6) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(7) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));

        // Check gradient of shape functions
        auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
        REQUIRE(gradsf.rows() == 9);
        REQUIRE(gradsf.cols() == Dim);

        Eigen::Matrix<double, 9, Dim> gradsf_ans;
        gradsf_ans << -0.444444, -0.444444, 0.444444, -0.222222, 0.222222,
            0.222222, -0.222222, 0.444444, 0, -0, 0, 0, 0, 0, 0, 0, -0, 0;

        for (unsigned i = 0; i < gradsf.rows(); ++i)
          for (unsigned j = 0; j < gradsf.cols(); ++j)
            REQUIRE(gradsf(i, j) ==
                    Approx(gradsf_ans(i, j)).epsilon(Tolerance));
      }

      // Coordinates is (0.5,0.5) after upgrade
      SECTION("2D BSpline element for coordinates(0.5,0.5) after upgrade") {
        Eigen::Matrix<double, Dim, 1> coords;
        coords << 0.5, 0.5;
        auto shapefn = quad->shapefn(coords, zero, zero_matrix);

        // Check shape function
        REQUIRE(shapefn.size() == 9);
        REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

        REQUIRE(shapefn(0) == Approx(0.140625).epsilon(Tolerance));
        REQUIRE(shapefn(1) == Approx(0.222656).epsilon(Tolerance));
        REQUIRE(shapefn(2) == Approx(0.352539).epsilon(Tolerance));
        REQUIRE(shapefn(3) == Approx(0.222656).epsilon(Tolerance));
        REQUIRE(shapefn(4) == Approx(0.0117188).epsilon(Tolerance));
        REQUIRE(shapefn(5) == Approx(0.0185547).epsilon(Tolerance));
        REQUIRE(shapefn(6) == Approx(0.000976562).epsilon(Tolerance));
        REQUIRE(shapefn(7) == Approx(0.0185547).epsilon(Tolerance));
        REQUIRE(shapefn(8) == Approx(0.0117188).epsilon(Tolerance));

        // Check gradient of shape functions
        auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
        REQUIRE(gradsf.rows() == 9);
        REQUIRE(gradsf.cols() == Dim);

        Eigen::Matrix<double, 9, Dim> gradsf_ans;
        gradsf_ans << -0.1875, -0.1875, 0.140625, -0.296875, 0.222656, 0.222656,
            -0.296875, 0.140625, 0.046875, -0.015625, 0.0742188, 0.0117188,
            0.00390625, 0.00390625, 0.0117188, 0.0742188, -0.015625, 0.046875;

        for (unsigned i = 0; i < gradsf.rows(); ++i)
          for (unsigned j = 0; j < gradsf.cols(); ++j)
            REQUIRE(gradsf(i, j) ==
                    Approx(gradsf_ans(i, j)).epsilon(Tolerance));

        // Check dn_dx of shape functions
        auto dn_dx = quad->dn_dx(coords, zero, zero, zero_matrix);
        REQUIRE(dn_dx.rows() == 9);
        REQUIRE(dn_dx.cols() == Dim);

        for (unsigned i = 0; i < dn_dx.rows(); ++i)
          for (unsigned j = 0; j < dn_dx.cols(); ++j)
            REQUIRE(dn_dx(i, j) == Approx(gradsf_ans(i, j)).epsilon(Tolerance));
      }

      // Coordinates is (-1,-1)
      SECTION(
          "Four noded local sf quadrilateral element for coordinates(-1,-1)") {
        Eigen::Matrix<double, Dim, 1> coords;
        coords << -1., -1.;
        auto shapefn = quad->shapefn_local(coords, zero, zero_matrix);

        // Check shape function
        REQUIRE(shapefn.size() == 4);

        REQUIRE(shapefn(0) == Approx(1.0).epsilon(Tolerance));
        REQUIRE(shapefn(1) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(2) == Approx(0.).epsilon(Tolerance));
        REQUIRE(shapefn(3) == Approx(0.).epsilon(Tolerance));
      }
    }
  }
}