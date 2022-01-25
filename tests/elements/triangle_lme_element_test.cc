// Triangular element test
#include <memory>

#include "catch.hpp"

#include "triangle_lme_element.h"

//! \brief Check triangle lme element class
TEST_CASE("Triangle lme elements are checked", "[tri][element][2D][lme]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-6;

  Eigen::Vector2d zero = Eigen::Vector2d::Zero();

  //! Check for center element nodes
  SECTION("Linear Triangle LME Element") {
    std::shared_ptr<mpm::Element<Dim>> tri =
        std::make_shared<mpm::TriangleLMEElement<Dim>>();

    // Check degree and shapefn type
    REQUIRE(tri->degree() == mpm::ElementDegree::Infinity);
    REQUIRE(tri->shapefn_type() == mpm::ShapefnType::LME);

    // Coordinates is (0,0) before upgraded
    SECTION("2D LME for coordinates in the barycentre before upgrade") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 0.16666666666, 0.16666666666;
      auto shapefn = tri->shapefn(coords, zero, zero);

      // Check shape function
      REQUIRE(shapefn.size() == 3);

      REQUIRE(shapefn(0) == Approx(0.66666666666).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.16666666666).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.16666666666).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = tri->grad_shapefn(coords, zero, zero);
      REQUIRE(gradsf.rows() == 3);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(0, 1) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(1.0).epsilon(Tolerance));
    }

    // Initialising upgrade properties
    SECTION("2D triangle LME element regular element - nnodes = 16") {
      Eigen::Matrix<double, 16, Dim> nodal_coords;
      nodal_coords << 0., 0., 1., 0., 0., 1., 1., 1., 2., 0., 2., 1., 2., 2.,
          1., 2., 0., 2., -1., 2., -1., 1., -1., 0., -1., -1., 0., -1., 1., -1.,
          2., -1.;

      SECTION("2D triangle LME regular element no support") {
        double gamma = 3;
        double h = 1.0;

        // Calculate beta
        double beta = gamma / (h * h);

        // Calculate support radius automatically
        double tol0 = 1.e-10;
        double r = h * std::sqrt(-std::log(tol0) / gamma);

        REQUIRE_NOTHROW(
            tri->initialise_lme_connectivity_properties(beta, r, nodal_coords));

        // Coordinates is (0,0) after upgrade
        SECTION("2D BSpline element for coordinates(0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 1. / 3., 1. / 3.;
          auto shapefn = tri->shapefn(coords, zero, zero);

          // Check shape function
          REQUIRE(shapefn.size() == 16);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(1. / 3.).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(1. / 3.).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(1. / 3.).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
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
          auto gradsf = tri->grad_shapefn(coords, zero, zero);
          REQUIRE(gradsf.rows() == 16);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 16, Dim> gradsf_ans;
          gradsf_ans << -1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0, -0, 0,
              0, 0, 0, 0, 0, -0, 0, -0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }
      }
    }
  }
}