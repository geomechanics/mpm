// Quadrilateral element test
#include <memory>

#include "catch.hpp"

#include "quadrilateral_lme_element.h"

//! \brief Check quadrilateral lme element class
TEST_CASE("Quadrilateral lme elements are checked",
          "[quad][element][2D][lme]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-6;

  Eigen::Vector2d zero = Eigen::Vector2d::Zero();
  Eigen::Matrix2d zero_matrix = Eigen::Matrix2d::Zero();

  //! Check for center element nodes
  SECTION("Quadratic Quadrilateral LME Element") {
    std::shared_ptr<mpm::Element<Dim>> quad =
        std::make_shared<mpm::QuadrilateralLMEElement<Dim>>();

    // Check degree and shapefn type
    REQUIRE(quad->degree() == mpm::ElementDegree::Infinity);
    REQUIRE(quad->shapefn_type() == mpm::ShapefnType::LME);

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
        double gamma = 20.0;
        double h = 2.0;

        // Calculate beta
        double beta = gamma / (h * h);

        // Calculate support radius automatically
        double tol0 = 1.e-10;
        double r = h * std::sqrt(-std::log(tol0) / gamma);
        unsigned anisotropy = 0;

        REQUIRE_NOTHROW(quad->initialise_lme_connectivity_properties(
            beta, r, anisotropy, nodal_coords));

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
      }
    }
  }
}