// hexahedron element test
#include <memory>

#include "catch.hpp"

#include "hexahedron_bspline_element.h"

//! \brief Check hexahedron element class
TEST_CASE("Hexahedron bspline elements are checked",
          "[hex][element][3D][bspline]") {
  const unsigned Dim = 3;
  const double Tolerance = 1.E-6;

  Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d zero_matrix = Eigen::Matrix3d::Zero();

  //! Check for center element nodes
  SECTION("Quadratic Hexahedron BSpline Element") {
    const unsigned npolynomials = 2;
    std::shared_ptr<mpm::Element<Dim>> hex =
        std::make_shared<mpm::HexahedronBSplineElement<Dim, npolynomials>>();

    // Check degree
    REQUIRE(hex->degree() == mpm::ElementDegree::Quadratic);
    REQUIRE(hex->shapefn_type() == mpm::ShapefnType::BSPLINE);

    // Coordinates is (0,0,0) before upgrade
    SECTION("3D BSpline element for coordinate (0.,0.,0.) before upgrade") {

      // Coordinate location of point (x,y)
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == 8);
      REQUIRE(hex->nfunctions() == 8);
      REQUIRE(hex->nfunctions_local() == 8);

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
      REQUIRE(gradsf.rows() == 8);
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
    SECTION(
        "3D BSpline element for coordinate (-1., -1., -1.) before upgrade") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1., -1., -1.;
      auto shapefn = hex->shapefn(coords, zero, zero_matrix);
      // Check shape function
      REQUIRE(shapefn.size() == 8);

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
      REQUIRE(gradsf.rows() == 8);
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

    // Initialising upgrade properties
    SECTION("3D BSpline element regular element - nnodes = 64") {
      Eigen::Matrix<double, 64, Dim> nodal_coords;
      // clang-format off
      nodal_coords <<-1,  -1, -1,
                      1,  -1, -1,
                      1,  1,  -1,
                      -1,  1,  -1,
                      -1, -1, 1,
                      1, -1, 1,
                      1,  1,  1,
                      -1, 1,  1,
                      -3,  3, -3,
                      -1,  3, -3,
                      1,  3, -3,
                      3,  3, -3,
                      -3,  3, -1,
                      -1,  3, -1,
                      1,  3, -1,
                      3,  3, -1,
                      -3,  3,  1,
                      -1,  3,  1,
                      1,  3,  1,
                      3,  3,  1,
                      -3,  3,  3,
                      -1,  3,  3,
                      1,  3,  3,
                      3,  3,  3,
                      -3,  1, -3,
                      -1,  1, -3,
                      1,  1, -3,
                      3,  1, -3,
                      -3,  1, -1,
                      3,  1, -1,
                      -3,  1,  1,
                      3,  1,  1,
                      -3,  1,  3,
                      -1,  1,  3,
                      1,  1,  3,
                      3,  1,  3,
                      -3, -1, -3,
                      -1, -1, -3,
                      1, -1, -3,
                      3, -1, -3,
                      -3, -1, -1,
                      3, -1, -1,
                      -3, -1,  1,
                      3, -1,  1,
                      -3, -1,  3,
                      -1, -1,  3,
                      1, -1,  3,
                      3, -1,  3,
                      -3, -3, -3,
                      -1, -3, -3,
                      1, -3, -3,
                      3, -3, -3,
                      -3, -3, -1,
                      -1, -3, -1,
                      1, -3, -1,
                      3, -3, -1,
                      -3, -3,  1,
                      -1, -3,  1,
                      1, -3,  1,
                      3, -3,  1,
                      -3, -3,  3,
                      -1, -3,  3,
                      1, -3,  3,
                      3, -3,  3;
      // clang-format on

      SECTION("3D BSpline element regular element no support") {
        std::vector<std::vector<unsigned>> nodal_props{
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props));

        // Coordinates is (0,0,0) after upgrade
        SECTION("3D BSpline element for coordinates(0,0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 64);
          REQUIRE(hex->nfunctions() == 64);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(48) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(49) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(50) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(51) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(52) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(53) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(54) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(55) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(56) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(57) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(58) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(59) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(60) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(61) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(62) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(63) == Approx(0).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 64);
          REQUIRE(gradsf.cols() == Dim);

          REQUIRE(gradsf(0, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(0, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(0, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(8, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(8, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(8, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(9, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(9, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(9, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(10, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(10, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(10, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(11, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(11, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(11, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(12, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(12, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(12, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(13, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(13, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(13, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(14, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(14, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(14, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(15, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(15, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(15, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(16, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(16, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(16, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(17, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(17, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(17, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(18, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(18, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(18, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(19, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(19, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(19, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(20, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(20, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(20, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(21, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(21, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(21, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(22, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(22, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(22, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(23, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(23, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(23, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(24, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(24, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(24, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(25, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(25, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(25, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(26, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(26, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(26, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(27, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(27, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(27, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(28, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(28, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(28, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(29, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(29, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(29, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(30, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(30, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(30, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(31, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(31, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(31, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(32, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(32, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(32, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(33, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(33, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(33, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(34, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(34, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(34, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(35, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(35, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(35, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(36, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(36, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(36, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(37, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(37, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(37, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(38, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(38, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(38, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(39, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(39, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(39, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(40, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(40, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(40, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(41, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(41, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(41, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(42, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(42, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(42, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(43, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(43, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(43, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(44, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(44, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(44, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(45, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(45, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(45, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(46, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(46, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(46, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(47, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(47, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(47, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(48, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(48, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(48, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(49, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(49, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(49, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(50, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(50, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(50, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(51, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(51, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(51, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(52, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(52, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(52, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(53, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(53, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(53, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(54, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(54, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(54, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(55, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(55, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(55, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(56, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(56, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(56, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(57, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(57, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(57, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(58, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(58, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(58, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(59, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(59, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(59, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(60, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(60, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(60, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(61, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(61, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(61, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(62, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(62, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(62, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(63, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(63, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(63, 2) == Approx(0).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 64);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0543823).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.132935).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0543823).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0222473).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.132935).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.324951).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.132935).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0543823).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0.00247192).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0.00247192).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0.000274658).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0.0147705).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0.0147705).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0.000671387).epsilon(Tolerance));
          REQUIRE(shapefn(48) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(49) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(50) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(51) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(52) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(53) == Approx(0.00247192).epsilon(Tolerance));
          REQUIRE(shapefn(54) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(55) == Approx(0.000274658).epsilon(Tolerance));
          REQUIRE(shapefn(56) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(57) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(58) == Approx(0.0147705).epsilon(Tolerance));
          REQUIRE(shapefn(59) == Approx(0.000671387).epsilon(Tolerance));
          REQUIRE(shapefn(60) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(61) == Approx(0.000274658).epsilon(Tolerance));
          REQUIRE(shapefn(62) == Approx(0.000671387).epsilon(Tolerance));
          REQUIRE(shapefn(63) == Approx(3.05176e-05).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 64);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 64, Dim> gradsf_ans;
          gradsf_ans << -0.0725098, -0.0197754, -0.0725098, 0.0483398,
              -0.0483398, -0.177246, 0.0197754, 0.0725098, -0.0725098,
              -0.0296631, 0.0296631, -0.0296631, -0.177246, -0.0483398,
              0.0483398, 0.118164, -0.118164, 0.118164, 0.0483398, 0.177246,
              0.0483398, -0.0725098, 0.0725098, 0.0197754, 0, 0, 0, -0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, 0, 0, 0,
              -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0.0098877,
              0.0032959, -0.0032959, 0, 0, 0, 0.0241699, 0.00805664, 0.00219727,
              0, 0, 0, -0.0032959, 0.0032959, 0.0098877, 0.00219727, 0.00805664,
              0.0241699, 0.00109863, 0.000366211, 0.00109863, 0, -0, 0, -0, -0,
              0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0.0241699, -0.00219727,
              -0.00805664, 0, -0, 0, 0.059082, -0.00537109, 0.00537109, 0, -0,
              0, -0.00805664, -0.00219727, 0.0241699, 0.00537109, -0.00537109,
              0.059082, 0.00268555, -0.000244141, 0.00268555, 0, -0, 0, -0, -0,
              0, 0, -0, 0, 0, -0, 0, 0, -0, -0, -0.0032959, -0.0098877,
              -0.0032959, 0.00219727, -0.0241699, -0.00805664, 0.00109863,
              -0.00109863, -0.000366211, 0, -0, 0, -0.00805664, -0.0241699,
              0.00219727, 0.00537109, -0.059082, 0.00537109, 0.00268555,
              -0.00268555, 0.000244141, 0, -0, 0, -0.000366211, -0.00109863,
              0.00109863, 0.000244141, -0.00268555, 0.00268555, 0.00012207,
              -0.00012207, 0.00012207;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (0,0,0)
        SECTION(
            "Eight noded local sf hexahedron element for coordinates(0,0,0)") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn_local(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 8);

          REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));
        }

        // Check Jacobian
        SECTION("64-noded hexrilateral Jacobian with deformation gradient") {
          Eigen::Matrix<double, 64, Dim> coords;
          coords << -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1,
              -1, 1, 1, 1, 1, -1, 1, 1, -3, 3, -3, -1, 3, -3, 1, 3, -3, 3, 3,
              -3, -3, 3, -1, -1, 3, -1, 1, 3, -1, 3, 3, -1, -3, 3, 1, -1, 3, 1,
              1, 3, 1, 3, 3, 1, -3, 3, 3, -1, 3, 3, 1, 3, 3, 3, 3, 3, -3, 1, -3,
              -1, 1, -3, 1, 1, -3, 3, 1, -3, -3, 1, -1, 3, 1, -1, -3, 1, 1, 3,
              1, 1, -3, 1, 3, -1, 1, 3, 1, 1, 3, 3, 1, 3, -3, -1, -3, -1, -1,
              -3, 1, -1, -3, 3, -1, -3, -3, -1, -1, 3, -1, -1, -3, -1, 1, 3, -1,
              1, -3, -1, 3, -1, -1, 3, 1, -1, 3, 3, -1, 3, -3, -3, -3, -1, -3,
              -3, 1, -3, -3, 3, -3, -3, -3, -3, -1, -1, -3, -1, 1, -3, -1, 3,
              -3, -1, -3, -3, 1, -1, -3, 1, 1, -3, 1, 3, -3, 1, -3, -3, 3, -1,
              -3, 3, 1, -3, 3, 3, -3, 3;

          Eigen::Matrix<double, Dim, 1> psize;
          psize.setZero();
          Eigen::Matrix<double, Dim, Dim> defgrad;
          defgrad.setZero();

          Eigen::Matrix<double, Dim, 1> xi;
          xi << 0., 0., 0.;

          Eigen::Matrix<double, Dim, Dim> jacobian;
          // clang-format off
          jacobian << 1., 0., 0.,
                      0., 1., 0.,
                      0., 0., 1;
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
        SECTION("64 noded hexahedron B-matrix cell for coordinates(0, 0, 0)") {
          Eigen::Matrix<double, Dim, 1> xi;
          xi << 0.0, 0.0, 0.0;

          Eigen::Matrix<double, 64, Dim> coords;
          coords << -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1,
              -1, 1, 1, 1, 1, -1, 1, 1, -3, 3, -3, -1, 3, -3, 1, 3, -3, 3, 3,
              -3, -3, 3, -1, -1, 3, -1, 1, 3, -1, 3, 3, -1, -3, 3, 1, -1, 3, 1,
              1, 3, 1, 3, 3, 1, -3, 3, 3, -1, 3, 3, 1, 3, 3, 3, 3, 3, -3, 1, -3,
              -1, 1, -3, 1, 1, -3, 3, 1, -3, -3, 1, -1, 3, 1, -1, -3, 1, 1, 3,
              1, 1, -3, 1, 3, -1, 1, 3, 1, 1, 3, 3, 1, 3, -3, -1, -3, -1, -1,
              -3, 1, -1, -3, 3, -1, -3, -3, -1, -1, 3, -1, -1, -3, -1, 1, 3, -1,
              1, -3, -1, 3, -1, -1, 3, 1, -1, 3, 3, -1, 3, -3, -3, -3, -1, -3,
              -3, 1, -3, -3, 3, -3, -3, -3, -3, -1, -1, -3, -1, 1, -3, -1, 3,
              -3, -1, -3, -3, 1, -1, -3, 1, 1, -3, 1, 3, -3, 1, -3, -3, 3, -1,
              -3, 3, 1, -3, 3, 3, -3, 3;

          // Get B-Matrix
          auto bmatrix = hex->bmatrix(xi, coords, zero, zero_matrix);

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(xi, zero, zero_matrix);

          // Check dN/dx
          auto dn_dx = hex->dn_dx(xi, coords, zero, zero_matrix);
          REQUIRE(dn_dx.rows() == 64);
          REQUIRE(dn_dx.cols() == Dim);
          for (unsigned i = 0; i < 64; ++i) {
            REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(dn_dx(i, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
          }

          // Check dN/dx local
          auto dn_dx_local = hex->dn_dx_local(xi, coords, zero, zero_matrix);
          REQUIRE(dn_dx_local.rows() == 64);
          REQUIRE(dn_dx_local.cols() == Dim);
          for (unsigned i = 0; i < 64; ++i) {
            REQUIRE(dn_dx_local(i, 0) ==
                    Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(dn_dx_local(i, 1) ==
                    Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(dn_dx_local(i, 2) ==
                    Approx(gradsf(i, 2)).epsilon(Tolerance));
          }

          // Check size of B-matrix
          REQUIRE(bmatrix.size() == 64);

          for (unsigned i = 0; i < 64; ++i) {
            REQUIRE(bmatrix.at(i)(0, 0) ==
                    Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(0, 2) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 1) ==
                    Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 2) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 2) ==
                    Approx(gradsf(i, 2)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(3, 0) ==
                    Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(3, 1) ==
                    Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(3, 2) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(4, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(4, 1) ==
                    Approx(gradsf(i, 2)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(4, 2) ==
                    Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(5, 0) ==
                    Approx(gradsf(i, 2)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(5, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(5, 2) ==
                    Approx(gradsf(i, 0)).epsilon(Tolerance));
          }
        }

        // Initialise BSpline with kernel correction equal to true
        // There should be no difference in regular nodes
        bool kernel_correction = true;
        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props, kernel_correction));

        // Coordinates is (0,0,0) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0,0,0) after upgrade - kernel "
            "correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 64);
          REQUIRE(hex->nfunctions() == 64);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(48) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(49) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(50) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(51) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(52) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(53) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(54) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(55) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(56) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(57) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(58) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(59) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(60) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(61) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(62) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(63) == Approx(0).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 64);
          REQUIRE(gradsf.cols() == Dim);

          REQUIRE(gradsf(0, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(0, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(0, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(8, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(8, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(8, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(9, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(9, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(9, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(10, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(10, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(10, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(11, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(11, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(11, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(12, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(12, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(12, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(13, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(13, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(13, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(14, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(14, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(14, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(15, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(15, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(15, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(16, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(16, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(16, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(17, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(17, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(17, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(18, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(18, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(18, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(19, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(19, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(19, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(20, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(20, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(20, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(21, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(21, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(21, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(22, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(22, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(22, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(23, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(23, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(23, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(24, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(24, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(24, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(25, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(25, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(25, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(26, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(26, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(26, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(27, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(27, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(27, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(28, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(28, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(28, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(29, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(29, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(29, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(30, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(30, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(30, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(31, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(31, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(31, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(32, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(32, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(32, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(33, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(33, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(33, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(34, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(34, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(34, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(35, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(35, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(35, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(36, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(36, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(36, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(37, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(37, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(37, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(38, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(38, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(38, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(39, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(39, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(39, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(40, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(40, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(40, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(41, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(41, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(41, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(42, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(42, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(42, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(43, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(43, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(43, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(44, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(44, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(44, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(45, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(45, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(45, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(46, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(46, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(46, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(47, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(47, 1) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(47, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(48, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(48, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(48, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(49, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(49, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(49, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(50, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(50, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(50, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(51, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(51, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(51, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(52, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(52, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(52, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(53, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(53, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(53, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(54, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(54, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(54, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(55, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(55, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(55, 2) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(56, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(56, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(56, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(57, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(57, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(57, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(58, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(58, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(58, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(59, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(59, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(59, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(60, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(60, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(60, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(61, 0) == Approx(-0).epsilon(Tolerance));
          REQUIRE(gradsf(61, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(61, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(62, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(62, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(62, 2) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(63, 0) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(63, 1) == Approx(0).epsilon(Tolerance));
          REQUIRE(gradsf(63, 2) == Approx(0).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after upgrade - "
            "kernel correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 64);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0543823).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.132935).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0543823).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0222473).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.132935).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.324951).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.132935).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0543823).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0.00247192).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0.00247192).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0.000274658).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0.0147705).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0.0147705).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0.000671387).epsilon(Tolerance));
          REQUIRE(shapefn(48) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(49) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(50) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(51) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(52) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(53) == Approx(0.00247192).epsilon(Tolerance));
          REQUIRE(shapefn(54) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(55) == Approx(0.000274658).epsilon(Tolerance));
          REQUIRE(shapefn(56) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(57) == Approx(0.00604248).epsilon(Tolerance));
          REQUIRE(shapefn(58) == Approx(0.0147705).epsilon(Tolerance));
          REQUIRE(shapefn(59) == Approx(0.000671387).epsilon(Tolerance));
          REQUIRE(shapefn(60) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(61) == Approx(0.000274658).epsilon(Tolerance));
          REQUIRE(shapefn(62) == Approx(0.000671387).epsilon(Tolerance));
          REQUIRE(shapefn(63) == Approx(3.05176e-05).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 64);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 64, Dim> gradsf_ans;
          gradsf_ans << -0.0725098, -0.0197754, -0.0725098, 0.0483398,
              -0.0483398, -0.177246, 0.0197754, 0.0725098, -0.0725098,
              -0.0296631, 0.0296631, -0.0296631, -0.177246, -0.0483398,
              0.0483398, 0.118164, -0.118164, 0.118164, 0.0483398, 0.177246,
              0.0483398, -0.0725098, 0.0725098, 0.0197754, 0, 0, 0, -0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, 0, 0, 0,
              -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0.0098877,
              0.0032959, -0.0032959, 0, 0, 0, 0.0241699, 0.00805664, 0.00219727,
              0, 0, 0, -0.0032959, 0.0032959, 0.0098877, 0.00219727, 0.00805664,
              0.0241699, 0.00109863, 0.000366211, 0.00109863, 0, -0, 0, -0, -0,
              0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0.0241699, -0.00219727,
              -0.00805664, 0, -0, 0, 0.059082, -0.00537109, 0.00537109, 0, -0,
              0, -0.00805664, -0.00219727, 0.0241699, 0.00537109, -0.00537109,
              0.059082, 0.00268555, -0.000244141, 0.00268555, 0, -0, 0, -0, -0,
              0, 0, -0, 0, 0, -0, 0, 0, -0, -0, -0.0032959, -0.0098877,
              -0.0032959, 0.00219727, -0.0241699, -0.00805664, 0.00109863,
              -0.00109863, -0.000366211, 0, -0, 0, -0.00805664, -0.0241699,
              0.00219727, 0.00537109, -0.059082, 0.00537109, 0.00268555,
              -0.00268555, 0.000244141, 0, -0, 0, -0.000366211, -0.00109863,
              0.00109863, 0.000244141, -0.00268555, 0.00268555, 0.00012207,
              -0.00012207, 0.00012207;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (0,0,0)
        SECTION(
            "Eight noded local sf hexahedron element for coordinates(0,0,0) - "
            "kernel correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn_local(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 8);

          REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));
        }

        // Check Jacobian
        SECTION(
            "64-noded hexrilateral Jacobian with deformation gradient - kernel "
            "correction") {
          Eigen::Matrix<double, 64, Dim> coords;
          coords << -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1,
              -1, 1, 1, 1, 1, -1, 1, 1, -3, 3, -3, -1, 3, -3, 1, 3, -3, 3, 3,
              -3, -3, 3, -1, -1, 3, -1, 1, 3, -1, 3, 3, -1, -3, 3, 1, -1, 3, 1,
              1, 3, 1, 3, 3, 1, -3, 3, 3, -1, 3, 3, 1, 3, 3, 3, 3, 3, -3, 1, -3,
              -1, 1, -3, 1, 1, -3, 3, 1, -3, -3, 1, -1, 3, 1, -1, -3, 1, 1, 3,
              1, 1, -3, 1, 3, -1, 1, 3, 1, 1, 3, 3, 1, 3, -3, -1, -3, -1, -1,
              -3, 1, -1, -3, 3, -1, -3, -3, -1, -1, 3, -1, -1, -3, -1, 1, 3, -1,
              1, -3, -1, 3, -1, -1, 3, 1, -1, 3, 3, -1, 3, -3, -3, -3, -1, -3,
              -3, 1, -3, -3, 3, -3, -3, -3, -3, -1, -1, -3, -1, 1, -3, -1, 3,
              -3, -1, -3, -3, 1, -1, -3, 1, 1, -3, 1, 3, -3, 1, -3, -3, 3, -1,
              -3, 3, 1, -3, 3, 3, -3, 3;

          Eigen::Matrix<double, Dim, 1> psize;
          psize.setZero();
          Eigen::Matrix<double, Dim, Dim> defgrad;
          defgrad.setZero();

          Eigen::Matrix<double, Dim, 1> xi;
          xi << 0., 0., 0.;

          Eigen::Matrix<double, Dim, Dim> jacobian;
          // clang-format off
          jacobian << 1., 0., 0.,
                      0., 1., 0.,
                      0., 0., 1;
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
        SECTION(
            "64 noded hexahedron B-matrix cell for coordinates(0, 0, 0) - "
            "kernel correction") {
          Eigen::Matrix<double, Dim, 1> xi;
          xi << 0.0, 0.0, 0.0;

          Eigen::Matrix<double, 64, Dim> coords;
          coords << -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1,
              -1, 1, 1, 1, 1, -1, 1, 1, -3, 3, -3, -1, 3, -3, 1, 3, -3, 3, 3,
              -3, -3, 3, -1, -1, 3, -1, 1, 3, -1, 3, 3, -1, -3, 3, 1, -1, 3, 1,
              1, 3, 1, 3, 3, 1, -3, 3, 3, -1, 3, 3, 1, 3, 3, 3, 3, 3, -3, 1, -3,
              -1, 1, -3, 1, 1, -3, 3, 1, -3, -3, 1, -1, 3, 1, -1, -3, 1, 1, 3,
              1, 1, -3, 1, 3, -1, 1, 3, 1, 1, 3, 3, 1, 3, -3, -1, -3, -1, -1,
              -3, 1, -1, -3, 3, -1, -3, -3, -1, -1, 3, -1, -1, -3, -1, 1, 3, -1,
              1, -3, -1, 3, -1, -1, 3, 1, -1, 3, 3, -1, 3, -3, -3, -3, -1, -3,
              -3, 1, -3, -3, 3, -3, -3, -3, -3, -1, -1, -3, -1, 1, -3, -1, 3,
              -3, -1, -3, -3, 1, -1, -3, 1, 1, -3, 1, 3, -3, 1, -3, -3, 3, -1,
              -3, 3, 1, -3, 3, 3, -3, 3;

          // Get B-Matrix
          auto bmatrix = hex->bmatrix(xi, coords, zero, zero_matrix);

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(xi, zero, zero_matrix);

          // Check dN/dx
          auto dn_dx = hex->dn_dx(xi, coords, zero, zero_matrix);
          REQUIRE(dn_dx.rows() == 64);
          REQUIRE(dn_dx.cols() == Dim);
          for (unsigned i = 0; i < 64; ++i) {
            REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(dn_dx(i, 2) == Approx(gradsf(i, 2)).epsilon(Tolerance));
          }

          // Check size of B-matrix
          REQUIRE(bmatrix.size() == 64);

          for (unsigned i = 0; i < 64; ++i) {
            REQUIRE(bmatrix.at(i)(0, 0) ==
                    Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(0, 2) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 1) ==
                    Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 2) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 2) ==
                    Approx(gradsf(i, 2)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(3, 0) ==
                    Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(3, 1) ==
                    Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(3, 2) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(4, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(4, 1) ==
                    Approx(gradsf(i, 2)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(4, 2) ==
                    Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(5, 0) ==
                    Approx(gradsf(i, 2)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(5, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(5, 2) ==
                    Approx(gradsf(i, 0)).epsilon(Tolerance));
          }
        }
      }
    }

    // Initialising upgrade properties
    SECTION("3D BSpline element regular element - nnodes = 48") {
      Eigen::Matrix<double, 48, Dim> nodal_coords;
      // clang-format off
      nodal_coords <<-1,  -1, -1,
                      1,  -1, -1,
                      1,  1,  -1,
                      -1,  1,  -1,
                      -1, -1, 1,
                      1, -1, 1,
                      1,  1,  1,
                      -1, 1,  1,
                      -3,  3, -3,
                      -1,  3, -3,
                      1,  3, -3,
                      3,  3, -3,
                      -3,  3, -1,
                      -1,  3, -1,
                      1,  3, -1,
                      3,  3, -1,
                      -3,  3,  1,
                      -1,  3,  1,
                      1,  3,  1,
                      3,  3,  1,
                      -3,  3,  3,
                      -1,  3,  3,
                      1,  3,  3,
                      3,  3,  3,
                      -3,  1, -3,
                      -1,  1, -3,
                      1,  1, -3,
                      3,  1, -3,
                      -3,  1, -1,
                      3,  1, -1,
                      -3,  1,  1,
                      3,  1,  1,
                      -3,  1,  3,
                      -1,  1,  3,
                      1,  1,  3,
                      3,  1,  3,
                      -3, -1, -3,
                      -1, -1, -3,
                      1, -1, -3,
                      3, -1, -3,
                      -3, -1, -1,
                      3, -1, -1,
                      -3, -1,  1,
                      3, -1,  1,
                      -3, -1,  3,
                      -1, -1,  3,
                      1, -1,  3,
                      3, -1,  3;
      // clang-format on

      SECTION("3D BSpline element regular element lower y-axis support") {
        std::vector<std::vector<unsigned>> nodal_props{
            {0, 1, 0}, {0, 1, 0}, {0, 2, 0}, {0, 2, 0}, {0, 1, 0}, {0, 1, 0},
            {0, 2, 0}, {0, 2, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
            {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 2, 0},
            {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 2, 0},
            {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
            {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};

        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props));

        // Coordinates is (0,0,0) after upgrade
        SECTION("3D BSpline element for coordinates(0,0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 48);
          REQUIRE(hex->nfunctions() == 48);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.166667).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.166667).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0833333).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0833333).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.166667).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.166667).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.0833333).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0833333).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 48);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 48, Dim> gradsf_ans;
          gradsf_ans << -0.166667, -0.166667, -0.166667, 0.166667, -0.166667,
              -0.166667, 0.0833333, 0.166667, -0.0833333, -0.0833333, 0.166667,
              -0.0833333, -0.166667, -0.166667, 0.166667, 0.166667, -0.166667,
              0.166667, 0.0833333, 0.166667, 0.0833333, -0.0833333, 0.166667,
              0.0833333, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, -0, 0,
              -0, 0, 0, -0, 0, 0, -0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, -0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, -0, 0, -0, -0, 0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0,
              -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, 0, -0, -0, 0, 0, -0, 0, 0, -0,
              0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 48);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0725098).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.177246).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0161133).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0065918).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.177246).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.433268).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.039388).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0161133).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0.000732422).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0.00179036).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0.000732422).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0.00179036).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(8.13802e-05).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0.00805664).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0.019694).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0.00805664).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0.019694).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0.000895182).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 48);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 48, Dim> gradsf_ans;
          gradsf_ans << -0.0966797, -0.0263672, -0.0966797, 0.0644531,
              -0.0644531, -0.236328, 0.00585938, 0.0644531, -0.0214844,
              -0.00878906, 0.0263672, -0.00878906, -0.236328, -0.0644531,
              0.0644531, 0.157552, -0.157552, 0.157552, 0.0143229, 0.157552,
              0.0143229, -0.0214844, 0.0644531, 0.00585938, 0, 0, 0, -0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, 0, 0,
              0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0.00292969,
              0.00292969, -0.000976562, 0, 0, 0, 0.00716146, 0.00716146,
              0.000651042, 0, 0, 0, -0.000976562, 0.00292969, 0.00292969,
              0.000651042, 0.00716146, 0.00716146, 0.000325521, 0.000325521,
              0.000325521, 0, -0, 0, -0, -0, 0, 0, -0, 0, 0, -0, 0, 0, -0, -0,
              0.0322266, -0.00292969, -0.0107422, 0, -0, 0, 0.078776,
              -0.00716146, 0.00716146, 0, -0, 0, -0.0107422, -0.00292969,
              0.0322266, 0.00716146, -0.00716146, 0.078776, 0.00358073,
              -0.000325521, 0.00358073;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Initialise BSpline with kernel correction equal to true
        bool kernel_correction = true;
        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props, kernel_correction));

        // Coordinates is (0,0,0) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0,0,0) after upgrade - kernel "
            "correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 48);
          REQUIRE(hex->nfunctions() == 48);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0).epsilon(Tolerance));

          // Check linear reproduction property
          Eigen::Matrix<double, Dim, 1> rep_coords;
          rep_coords.setZero();
          for (unsigned i = 0; i < shapefn.size(); i++) {
            rep_coords.noalias() +=
                shapefn(i) * nodal_coords.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(rep_coords(j) == Approx(coords(j)).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 48);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 48, Dim> gradsf_ans;
          gradsf_ans << -0.166667, -0.125, -0.166667, 0.166667, -0.125,
              -0.166667, 0.0833333, 0.125, -0.0833333, -0.0833333, 0.125,
              -0.0833333, -0.166667, -0.125, 0.166667, 0.166667, -0.125,
              0.166667, 0.0833333, 0.125, 0.0833333, -0.0833333, 0.125,
              0.0833333, -0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0, -0,
              -0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0,
              0, -0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, -0, -0, 0, -0, 0,
              0, -0, 0, 0, -0, -0, 0, -0, 0, 0, -0, -0, 0, 0, 0, 0, 0, -0, 0, 0,
              -0, 0, 0, 0, 0, 0, 0, 0, 0, -0, -0, -0, -0, -0, -0, 0, -0, -0, 0,
              -0, -0, -0, -0, -0, 0, -0, -0, -0, -0, 0, 0, -0, 0, -0, -0, 0, -0,
              -0, 0, 0, -0, 0, 0, -0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check zero gradient sum property
          Eigen::Matrix<double, Dim, 1> grad_sum;
          grad_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            grad_sum.noalias() += gradsf.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(grad_sum(j) == Approx(0.0).epsilon(Tolerance));

          // Check identity tensor product property
          Eigen::Matrix<double, Dim, Dim> identity =
              Eigen::Matrix<double, Dim, Dim>::Identity();
          Eigen::Matrix<double, Dim, Dim> identity_sum;
          identity_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            identity_sum.noalias() +=
                nodal_coords.row(i).transpose() * gradsf.row(i);
          }
          for (unsigned j = 0; j < Dim; j++)
            for (unsigned k = 0; k < Dim; k++)
              REQUIRE(identity_sum(j, k) ==
                      Approx(identity(j, k)).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after upgrade - "
            "kernel correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 48);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0593262).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.14502).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0483398).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0197754).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.14502).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.354492).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.118164).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0483398).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0.00219727).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0.00537109).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0.00219727).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0.00537109).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0.000244141).epsilon(Tolerance));
          REQUIRE(shapefn(36) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(37) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(38) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(39) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(40) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(41) == Approx(0.0065918).epsilon(Tolerance));
          REQUIRE(shapefn(42) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(43) == Approx(0.0161133).epsilon(Tolerance));
          REQUIRE(shapefn(44) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(45) == Approx(0.0065918).epsilon(Tolerance));
          REQUIRE(shapefn(46) == Approx(0.0161133).epsilon(Tolerance));
          REQUIRE(shapefn(47) == Approx(0.000732422).epsilon(Tolerance));

          // Check linear reproduction property
          Eigen::Matrix<double, Dim, 1> rep_coords;
          rep_coords.setZero();
          for (unsigned i = 0; i < shapefn.size(); i++) {
            rep_coords.noalias() +=
                shapefn(i) * nodal_coords.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(rep_coords(j) == Approx(coords(j)).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 48);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 48, Dim> gradsf_ans;
          gradsf_ans << -0.108765, -0.0395508, -0.108765, 0.088623, -0.0966797,
              -0.265869, 0.00805664, 0.0966797, -0.0241699, -0.0098877,
              0.0395508, -0.0098877, -0.265869, -0.0966797, 0.088623, 0.216634,
              -0.236328, 0.216634, 0.019694, 0.236328, 0.019694, -0.0241699,
              0.0966797, 0.00805664, -0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0,
              -0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0, 0, -0, 0, 0, 0,
              0, 0, 0, 0, 0, -0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, -0,
              -0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0, -0, 0.00183105, 0.00439453,
              -0.00109863, -0, 0, 0, 0.00447591, 0.0107422, 0.000895182, -0, 0,
              0, -0.00109863, 0.00439453, 0.00183105, 0.000895182, 0.0107422,
              0.00447591, 0.000203451, 0.000488281, 0.000203451, -0, -0, -0, -0,
              -0, -0, 0, -0, -0, 0, -0, -0, -0, -0, -0, 0.0201416, -0.00439453,
              -0.012085, -0, -0, 0, 0.049235, -0.0107422, 0.00984701, -0, -0, 0,
              -0.012085, -0.00439453, 0.0201416, 0.00984701, -0.0107422,
              0.049235, 0.00223796, -0.000488281, 0.00223796;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check zero gradient sum property
          Eigen::Matrix<double, Dim, 1> grad_sum;
          grad_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            grad_sum.noalias() += gradsf.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(grad_sum(j) == Approx(0.0).epsilon(Tolerance));

          // Check identity tensor product property
          Eigen::Matrix<double, Dim, Dim> identity =
              Eigen::Matrix<double, Dim, Dim>::Identity();
          Eigen::Matrix<double, Dim, Dim> identity_sum;
          identity_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            identity_sum.noalias() +=
                nodal_coords.row(i).transpose() * gradsf.row(i);
          }
          for (unsigned j = 0; j < Dim; j++)
            for (unsigned k = 0; k < Dim; k++)
              REQUIRE(identity_sum(j, k) ==
                      Approx(identity(j, k)).epsilon(Tolerance));
        }
      }
    }

    // Initialising upgrade properties
    SECTION("3D BSpline element regular element - nnodes = 36") {
      Eigen::Matrix<double, 36, Dim> nodal_coords;
      // clang-format off
      nodal_coords <<-1,  -1, -1,
                      1,  -1, -1,
                      1,  1,  -1,
                      -1,  1,  -1,
                      -1, -1, 1,
                      1, -1, 1,
                      1,  1,  1,
                      -1, 1,  1,
                      -3,  3, -1,
                      -1,  3, -1,
                      1,  3, -1,
                      3,  3, -1,
                      -3,  3,  1,
                      -1,  3,  1,
                      1,  3,  1,
                      3,  3,  1,
                      -3,  3,  3,
                      -1,  3,  3,
                      1,  3,  3,
                      3,  3,  3,
                      -3,  1, -1,
                      3,  1, -1,
                      -3,  1,  1,
                      3,  1,  1,
                      -3,  1,  3,
                      -1,  1,  3,
                      1,  1,  3,
                      3,  1,  3,
                      -3, -1, -1,
                      3, -1, -1,
                      -3, -1,  1,
                      3, -1,  1,
                      -3, -1,  3,
                      -1, -1,  3,
                      1, -1,  3,
                      3, -1,  3;
      // clang-format on

      SECTION("3D BSpline element regular element lower y-axis support") {
        std::vector<std::vector<unsigned>> nodal_props{
            {0, 1, 1}, {0, 1, 1}, {0, 2, 1}, {0, 2, 1}, {0, 1, 2}, {0, 1, 2},
            {0, 2, 2}, {0, 2, 2}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
            {0, 0, 2}, {0, 0, 2}, {0, 0, 2}, {0, 0, 2}, {0, 0, 0}, {0, 0, 0},
            {0, 0, 0}, {0, 0, 0}, {0, 2, 1}, {0, 2, 1}, {0, 2, 2}, {0, 2, 2},
            {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 2, 0}, {0, 1, 1}, {0, 1, 1},
            {0, 1, 2}, {0, 1, 2}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};

        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props));

        // Coordinates is (0,0,0) after upgrade
        SECTION("3D BSpline element for coordinates(0,0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 36);
          REQUIRE(hex->nfunctions() == 36);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.222222).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.222222).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.0555556).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0555556).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 36);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 36, Dim> gradsf_ans;
          gradsf_ans << -0.222222, -0.222222, -0.222222, 0.222222, -0.222222,
              -0.222222, 0.111111, 0.222222, -0.111111, -0.111111, 0.222222,
              -0.111111, -0.111111, -0.111111, 0.222222, 0.111111, -0.111111,
              0.222222, 0.0555556, 0.111111, 0.111111, -0.0555556, 0.111111,
              0.111111, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, 0, 0, 0, -0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              -0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, -0, -0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, 0, -0, -0, 0,
              0, -0, 0, 0, -0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 36);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0966797).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.236328).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0214844).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.00878906).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.153076).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.374186).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.0340169).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.013916).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0.000976562).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0.00154622).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0.000732422).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0.00179036).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(8.13802e-05).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0.0107422).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0.0170085).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0.00805664).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0.019694).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0.000895182).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 36);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 36, Dim> gradsf_ans;
          gradsf_ans << -0.128906, -0.0351562, -0.128906, 0.0859375, -0.0859375,
              -0.315104, 0.0078125, 0.0859375, -0.0286458, -0.0117188,
              0.0351562, -0.0117188, -0.204102, -0.0556641, 0.0966797, 0.136068,
              -0.136068, 0.236328, 0.0123698, 0.136068, 0.0214844, -0.0185547,
              0.0556641, 0.00878906, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, 0,
              0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, -0, 0.00390625, 0.00390625, -0.00130208, 0, 0, 0,
              0.0061849, 0.0061849, 0.000976562, 0, 0, 0, -0.000976562,
              0.00292969, 0.00292969, 0.000651042, 0.00716146, 0.00716146,
              0.000325521, 0.000325521, 0.000325521, 0, -0, -0, 0.0429688,
              -0.00390625, -0.0143229, 0, -0, 0, 0.0680339, -0.0061849,
              0.0107422, 0, -0, 0, -0.0107422, -0.00292969, 0.0322266,
              0.00716146, -0.00716146, 0.078776, 0.00358073, -0.000325521,
              0.00358073;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Initialise BSpline with kernel correction equal to true
        bool kernel_correction = true;
        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props, kernel_correction));

        // Coordinates is (0,0,0) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0,0,0) after upgrade - kernel "
            "correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 36);
          REQUIRE(hex->nfunctions() == 36);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.138889).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.138889).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.138889).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.138889).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.111111).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0).epsilon(Tolerance));

          // Check linear reproduction property
          Eigen::Matrix<double, Dim, 1> rep_coords;
          rep_coords.setZero();
          for (unsigned i = 0; i < shapefn.size(); i++) {
            rep_coords.noalias() +=
                shapefn(i) * nodal_coords.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(rep_coords(j) == Approx(coords(j)).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 36);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 36, Dim> gradsf_ans;
          gradsf_ans << -0.222222, -0.166667, -0.166667, 0.222222, -0.166667,
              -0.166667, 0.111111, 0.166667, -0.0833333, -0.111111, 0.166667,
              -0.0833333, -0.111111, -0.0833333, 0.166667, 0.111111, -0.0833333,
              0.166667, 0.0555556, 0.0833333, 0.0833333, -0.0555556, 0.0833333,
              0.0833333, -0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0, 0, -0,
              0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0, -0,
              0, -0, 0, 0, -0, -0, 0, 0, 0, 0, 0, -0, 0, 0, -0, 0, 0, 0, 0, 0,
              0, 0, 0, -0, -0, -0, 0, -0, -0, -0, -0, 0, 0, -0, 0, -0, -0, 0,
              -0, -0, 0, 0, -0, 0, 0, -0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check zero gradient sum property
          Eigen::Matrix<double, Dim, 1> grad_sum;
          grad_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            grad_sum.noalias() += gradsf.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(grad_sum(j) == Approx(0.0).epsilon(Tolerance));

          // Check identity tensor product property
          Eigen::Matrix<double, Dim, Dim> identity =
              Eigen::Matrix<double, Dim, Dim>::Identity();
          Eigen::Matrix<double, Dim, Dim> identity_sum;
          identity_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            identity_sum.noalias() +=
                nodal_coords.row(i).transpose() * gradsf.row(i);
          }
          for (unsigned j = 0; j < Dim; j++)
            for (unsigned k = 0; k < Dim; k++)
              REQUIRE(identity_sum(j, k) ==
                      Approx(identity(j, k)).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after upgrade - "
            "kernel correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 36);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0584547).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.142889).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0598649).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0244902).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.142368).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.34801).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.105856).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0433048).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0.00272113).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0.00481164).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0.00251755).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0.006154).epsilon(Tolerance));
          REQUIRE(shapefn(27) == Approx(0.000279727).epsilon(Tolerance));
          REQUIRE(shapefn(28) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(29) == Approx(0.00649497).epsilon(Tolerance));
          REQUIRE(shapefn(30) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(31) == Approx(0.0158187).epsilon(Tolerance));
          REQUIRE(shapefn(32) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(33) == Approx(0.0101149).epsilon(Tolerance));
          REQUIRE(shapefn(34) == Approx(0.0247252).epsilon(Tolerance));
          REQUIRE(shapefn(35) == Approx(0.00112387).epsilon(Tolerance));

          // Check linear reproduction property
          Eigen::Matrix<double, Dim, 1> rep_coords;
          rep_coords.setZero();
          for (unsigned i = 0; i < shapefn.size(); i++) {
            rep_coords.noalias() +=
                shapefn(i) * nodal_coords.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(rep_coords(j) == Approx(coords(j)).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 36);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 36, Dim> gradsf_ans;
          gradsf_ans << -0.14502, -0.0527344, -0.110117, 0.118164, -0.128906,
              -0.269174, 0.0107422, 0.128906, -0.0244703, -0.0131836, 0.0527344,
              -0.0100106, -0.229614, -0.0834961, 0.0913268, 0.187093, -0.204102,
              0.223243, 0.0170085, 0.204102, 0.0202948, -0.020874, 0.0834961,
              0.00830244, -0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, -0, -0, 0, 0,
              -0, 0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, 0,
              -0, 0, -0, 0.00244141, 0.00585937, -0.00111229, -0, 0, 0,
              0.00386556, 0.00927734, 0.000922493, -0, 0, 0, -0.00109863,
              0.00439453, 0.00170816, 0.000895182, 0.0107422, 0.00417549,
              0.000203451, 0.000488281, 0.000189795, -0, -0, -0, 0.0268555,
              -0.00585938, -0.0122352, -0, -0, 0, 0.0425212, -0.00927734,
              0.0101474, -0, -0, 0, -0.012085, -0.00439453, 0.0187897,
              0.00984701, -0.0107422, 0.0459304, 0.00223796, -0.000488281,
              0.00208775;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check zero gradient sum property
          Eigen::Matrix<double, Dim, 1> grad_sum;
          grad_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            grad_sum.noalias() += gradsf.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(grad_sum(j) == Approx(0.0).epsilon(Tolerance));

          // Check identity tensor product property
          Eigen::Matrix<double, Dim, Dim> identity =
              Eigen::Matrix<double, Dim, Dim>::Identity();
          Eigen::Matrix<double, Dim, Dim> identity_sum;
          identity_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            identity_sum.noalias() +=
                nodal_coords.row(i).transpose() * gradsf.row(i);
          }
          for (unsigned j = 0; j < Dim; j++)
            for (unsigned k = 0; k < Dim; k++)
              REQUIRE(identity_sum(j, k) ==
                      Approx(identity(j, k)).epsilon(Tolerance));
        }
      }
    }

    // Initialising upgrade properties
    SECTION("3D BSpline element regular element - nnodes = 27") {
      Eigen::Matrix<double, 27, Dim> nodal_coords;
      // clang-format off
        nodal_coords <<-1,  -1, -1,
                        1,  -1, -1,
                        1,  1,  -1,
                        -1,  1,  -1,
                        -1, -1, 1,
                        1, -1, 1,
                        1,  1,  1,
                        -1, 1,  1,
                        -3,  3, -1,
                        -1,  3, -1,
                        1,  3, -1,
                        -3,  3,  1,
                        -1,  3,  1,
                        1,  3,  1,
                        -3,  3,  3,
                        -1,  3,  3,
                        1,  3,  3,
                        -3,  1, -1,
                        -3,  1,  1,
                        -3,  1,  3,
                        -1,  1,  3,
                        1,  1,  3,
                        -3, -1, -1,
                        -3, -1,  1,
                        -3, -1,  3,
                        -1, -1,  3,
                        1, -1,  3;
      // clang-format on

      SECTION("3D BSpline element regular element lower y-axis support") {
        std::vector<std::vector<unsigned>> nodal_props{
            {3, 1, 1}, {4, 1, 1}, {4, 2, 1}, {3, 2, 1}, {3, 1, 2}, {4, 1, 2},
            {4, 2, 2}, {3, 2, 2}, {0, 0, 1}, {3, 0, 1}, {4, 0, 1}, {0, 0, 2},
            {3, 0, 2}, {4, 0, 2}, {0, 0, 0}, {3, 0, 0}, {4, 0, 0}, {0, 2, 1},
            {0, 2, 2}, {0, 2, 0}, {3, 2, 0}, {4, 2, 0}, {0, 1, 1}, {0, 1, 2},
            {0, 1, 0}, {3, 1, 0}, {4, 1, 0}};

        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props));

        // Coordinates is (0,0,0) after upgrade
        SECTION("3D BSpline element for coordinates(0,0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 27);
          REQUIRE(hex->nfunctions() == 27);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.148148).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.296296).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.148148).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0740741).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.0740741).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.148148).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.0740741).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.037037).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 27);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 27, Dim> gradsf_ans;
          gradsf_ans << -0.296296, -0.148148, -0.148148, 0.296296, -0.296296,
              -0.296296, 0.148148, 0.296296, -0.148148, -0.148148, 0.148148,
              -0.0740741, -0.148148, -0.0740741, 0.148148, 0.148148, -0.148148,
              0.296296, 0.0740741, 0.148148, 0.148148, -0.0740741, 0.0740741,
              0.0740741, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, 0, -0, 0, 0, 0, 0,
              0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, -0, 0,
              0, 0, 0, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, -0, -0, 0, 0, -0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after "
            "upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 27);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0286458).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.315104).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0286458).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.00260417).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.0453559).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.498915).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.0453559).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.00412326).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0.000217014).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0.00238715).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0.00238715).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0.0262587).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 27);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 27, Dim> gradsf_ans;
          gradsf_ans << -0.114583, -0.0104167, -0.0381944, 0.114583, -0.114583,
              -0.420139, 0.0104167, 0.114583, -0.0381944, -0.0104167, 0.0104167,
              -0.00347222, -0.181424, -0.0164931, 0.0286458, 0.181424,
              -0.181424, 0.315104, 0.0164931, 0.181424, 0.0286458, -0.0164931,
              0.0164931, 0.00260417, 0, 0, -0, -0, 0, -0, 0, 0, -0, 0, 0, 0, -0,
              0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0,
              0, 0, -0.000868056, 0.000868056, 0.000868056, 0.000868056,
              0.00954861, 0.00954861, 0, -0, -0, 0, -0, 0, 0, -0, 0,
              -0.00954861, -0.000868056, 0.00954861, 0.00954861, -0.00954861,
              0.105035;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Initialise BSpline with kernel correction equal to true
        bool kernel_correction = true;
        REQUIRE_NOTHROW(hex->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props, kernel_correction));

        // Coordinates is (0,0,0) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0,0,0) after upgrade - kernel "
            "correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 27);
          REQUIRE(hex->nfunctions() == 27);
          REQUIRE(hex->nfunctions_local() == 8);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.148148).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.0740741).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.148148).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.12963).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.12963).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.148148).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.12963).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0925926).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0).epsilon(Tolerance));

          // Check linear reproduction property
          Eigen::Matrix<double, Dim, 1> rep_coords;
          rep_coords.setZero();
          for (unsigned i = 0; i < shapefn.size(); i++) {
            rep_coords.noalias() +=
                shapefn(i) * nodal_coords.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(rep_coords(j) == Approx(coords(j)).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 27);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 27, Dim> gradsf_ans;
          gradsf_ans << -0.222222, -0.111111, -0.111111, 0.222222, -0.222222,
              -0.222222, 0.111111, 0.222222, -0.111111, -0.111111, 0.111111,
              -0.0555556, -0.111111, -0.0555556, 0.111111, 0.111111, -0.111111,
              0.222222, 0.0555556, 0.111111, 0.111111, -0.0555556, 0.0555556,
              0.0555556, -0, 0, -0, -0, 0, -0, 0, 0, -0, -0, 0, 0, -0, 0, 0, 0,
              0, 0, -0, 0, 0, -0, 0, 0, 0, 0, 0, -0, 0, -0, -0, 0, 0, -0, 0, 0,
              -0, 0, 0, 0, 0, 0, -0, -0, -0, -0, -0, 0, -0, -0, 0, -0, -0, 0, 0,
              -0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check zero gradient sum property
          Eigen::Matrix<double, Dim, 1> grad_sum;
          grad_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            grad_sum.noalias() += gradsf.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(grad_sum(j) == Approx(0.0).epsilon(Tolerance));

          // Check identity tensor product property
          Eigen::Matrix<double, Dim, Dim> identity =
              Eigen::Matrix<double, Dim, Dim>::Identity();
          Eigen::Matrix<double, Dim, Dim> identity_sum;
          identity_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            identity_sum.noalias() +=
                nodal_coords.row(i).transpose() * gradsf.row(i);
          }
          for (unsigned j = 0; j < Dim; j++)
            for (unsigned k = 0; k < Dim; k++)
              REQUIRE(identity_sum(j, k) ==
                      Approx(identity(j, k)).epsilon(Tolerance));
        }

        // Coordinates is (0.5,-0.5,0.5) after upgrade
        SECTION(
            "3D BSpline element for coordinates(0.5,-0.5,0.5) after "
            "upgrade - kernel correction") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.5, -0.5, 0.5;
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 27);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.0746116).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.133227).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.0746116).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.0124647).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0.132895).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0.373302).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0.132895).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0.0210776).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(16) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(17) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(18) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(19) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(20) == Approx(0.00117997).epsilon(Tolerance));
          REQUIRE(shapefn(21) == Approx(0.0077713).epsilon(Tolerance));
          REQUIRE(shapefn(22) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(23) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(24) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(25) == Approx(0.0077713).epsilon(Tolerance));
          REQUIRE(shapefn(26) == Approx(0.0281927).epsilon(Tolerance));

          // Check linear reproduction property
          Eigen::Matrix<double, Dim, 1> rep_coords;
          rep_coords.setZero();
          for (unsigned i = 0; i < shapefn.size(); i++) {
            rep_coords.noalias() +=
                shapefn(i) * nodal_coords.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(rep_coords(j) == Approx(coords(j)).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 27);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 27, Dim> gradsf_ans;
          gradsf_ans << -0.171875, -0.015625, -0.0326271, 0.171875, -0.171875,
              -0.358898, 0.015625, 0.171875, -0.0326271, -0.015625, 0.015625,
              -0.0029661, -0.272135, -0.0247396, 0.0270598, 0.272135, -0.272135,
              0.297658, 0.0247396, 0.272135, 0.0270598, -0.0247396, 0.0247396,
              0.00245998, -0, 0, -0, -0, 0, -0, 0, 0, -0, -0, 0, 0, -0, 0, 0, 0,
              0, 0, -0, 0, 0, -0, 0, 0, 0, 0, 0, -0, 0, -0, -0, 0, 0, -0, 0, 0,
              -0.00130208, 0.00130208, 0.000506121, 0.00130208, 0.0143229,
              0.00556733, -0, -0, -0, -0, -0, 0, -0, -0, 0, -0.0143229,
              -0.00130208, 0.00556733, 0.0143229, -0.0143229, 0.0612406;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check zero gradient sum property
          Eigen::Matrix<double, Dim, 1> grad_sum;
          grad_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            grad_sum.noalias() += gradsf.row(i).transpose();
          }
          for (unsigned j = 0; j < Dim; j++)
            REQUIRE(grad_sum(j) == Approx(0.0).epsilon(Tolerance));

          // Check identity tensor product property
          Eigen::Matrix<double, Dim, Dim> identity =
              Eigen::Matrix<double, Dim, Dim>::Identity();
          Eigen::Matrix<double, Dim, Dim> identity_sum;
          identity_sum.setZero();
          for (unsigned i = 0; i < gradsf.rows(); i++) {
            identity_sum.noalias() +=
                nodal_coords.row(i).transpose() * gradsf.row(i);
          }
          for (unsigned j = 0; j < Dim; j++)
            for (unsigned k = 0; k < Dim; k++)
              REQUIRE(identity_sum(j, k) ==
                      Approx(identity(j, k)).epsilon(Tolerance));
        }
      }
    }
  }
}
