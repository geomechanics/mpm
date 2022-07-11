// Tetrahedron quadrature test
#include <cmath>

#include <limits>
#include <memory>

#include "catch.hpp"
#include "tetrahedron_quadrature.h"

//! \brief Check TetrahedronQuadratures class
TEST_CASE("Tetrahedron quadratures are checked",
          "[tetquadrature][tet][quadrature][3D]") {
  const unsigned Dim = 3;
  const double Tolerance = 1.E-7;

  //! Check for a single point quadrature function
  SECTION("Hexahedron with a single quadrature") {
    const unsigned Nquadratures = 1;

    auto quad =
        std::make_shared<mpm::TetrahedronQuadrature<Dim, Nquadratures>>();

    // Check quadratures
    auto points = quad->quadratures();

    // Check size
    REQUIRE(points.rows() == 3);
    REQUIRE(points.cols() == 1);

    // Check quadrature points
    REQUIRE(points(0, 0) == Approx(0.25).epsilon(Tolerance));
    REQUIRE(points(1, 0) == Approx(0.25).epsilon(Tolerance));
    REQUIRE(points(2, 0) == Approx(0.25).epsilon(Tolerance));

    // Check weights
    auto weights = quad->weights();

    // Check size
    REQUIRE(weights.size() == 1);

    // Check weights
    REQUIRE(weights(0) == Approx(1.0 / 6).epsilon(Tolerance));
  }

  //! Check for eight quadrature points
  SECTION("Hexahedron with eight quadratures") {
    const unsigned Nquadratures = 4;

    auto quad =
        std::make_shared<mpm::TetrahedronQuadrature<Dim, Nquadratures>>();

    // Check quadratures
    auto points = quad->quadratures();

    // Check size
    REQUIRE(points.rows() == 3);
    REQUIRE(points.cols() == 4);

    // Check quadrature points
    REQUIRE(points(0, 0) == Approx(0.58541020).epsilon(Tolerance));
    REQUIRE(points(1, 0) == Approx(0.13819660).epsilon(Tolerance));
    REQUIRE(points(2, 0) == Approx(0.13819660).epsilon(Tolerance));

    REQUIRE(points(0, 1) == Approx(0.13819660).epsilon(Tolerance));
    REQUIRE(points(1, 1) == Approx(0.58541020).epsilon(Tolerance));
    REQUIRE(points(2, 1) == Approx(0.13819660).epsilon(Tolerance));

    REQUIRE(points(0, 2) == Approx(0.13819660).epsilon(Tolerance));
    REQUIRE(points(1, 2) == Approx(0.13819660).epsilon(Tolerance));
    REQUIRE(points(2, 2) == Approx(0.58541020).epsilon(Tolerance));

    REQUIRE(points(0, 3) == Approx(0.13819660).epsilon(Tolerance));
    REQUIRE(points(1, 3) == Approx(0.13819660).epsilon(Tolerance));
    REQUIRE(points(2, 3) == Approx(0.13819660).epsilon(Tolerance));

    // Check weights
    auto weights = quad->weights();

    // Check size
    REQUIRE(weights.size() == 4);

    // Check weights
    REQUIRE(weights(0) == Approx(1.0 / 24).epsilon(Tolerance));
    REQUIRE(weights(1) == Approx(1.0 / 24).epsilon(Tolerance));
    REQUIRE(weights(2) == Approx(1.0 / 24).epsilon(Tolerance));
    REQUIRE(weights(3) == Approx(1.0 / 24).epsilon(Tolerance));

    // Check Sum
    REQUIRE((weights(0) + weights(1) + weights(2) + weights(3)) ==
            Approx(1.0 / 6).epsilon(Tolerance));
  }
}
