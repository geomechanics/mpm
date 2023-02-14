// hexahedron element test
#include <memory>

#include "catch.hpp"

#include "hexahedron_lme_element.h"

//! \brief Check hexahedron element class
TEST_CASE("Hexahedron lme elements are checked", "[hex][element][3D][lme]") {
  const unsigned Dim = 3;
  const double Tolerance = 1.E-6;

  Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  const Eigen::Matrix3d zero_matrix = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d def_gradient = Eigen::Matrix3d::Identity();

  //! Check for center element nodes
  SECTION("Quadratic Hexahedron LME Element") {
    std::shared_ptr<mpm::Element<Dim>> hex =
        std::make_shared<mpm::HexahedronLMEElement<Dim>>();

    // Check degree
    REQUIRE(hex->degree() == mpm::ElementDegree::Infinity);
    REQUIRE(hex->shapefn_type() == mpm::ShapefnType::LME);

    // Coordinates is (0,0,0) before upgrade
    SECTION("3D LME element for coordinate (0.,0.,0.) before upgrade") {

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
    SECTION("3D LME element for coordinate (-1., -1., -1.) before upgrade") {
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
    SECTION("3D ALME element regular element - nnodes = 64") {
      Eigen::Matrix<double, 64, Dim> nodal_coords;
      nodal_coords << -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, -1,
          -1, 1, -1, 1, -1, -1, 1, -3, 3, -3, -1, 3, -3, 1, 3, -3, 3, 3, -3, -3,
          3, -1, -1, 3, -1, 1, 3, -1, 3, 3, -1, -3, 3, 1, -1, 3, 1, 1, 3, 1, 3,
          3, 1, -3, 3, 3, -1, 3, 3, 1, 3, 3, 3, 3, 3, -3, 1, -3, -1, 1, -3, 1,
          1, -3, 3, 1, -3, -3, 1, -1, 3, 1, -1, -3, 1, 1, 3, 1, 1, -3, 1, 3, -1,
          1, 3, 1, 1, 3, 3, 1, 3, -3, -1, -3, -1, -1, -3, 1, -1, -3, 3, -1, -3,
          -3, -1, -1, 3, -1, -1, -3, -1, 1, 3, -1, 1, -3, -1, 3, -1, -1, 3, 1,
          -1, 3, 3, -1, 3, -3, -3, -3, -1, -3, -3, 1, -3, -3, 3, -3, -3, -3, -3,
          -1, -1, -3, -1, 1, -3, -1, 3, -3, -1, -3, -3, 1, -1, -3, 1, 1, -3, 1,
          3, -3, 1, -3, -3, 3, -1, -3, 3, 1, -3, 3, 3, -3, 3;

      SECTION("3D ALME element regular element no support") {
        double gamma = 20.0;
        double h = 2.0;

        // Calculate beta
        double beta = gamma / (h * h);

        // Calculate support radius automatically
        double tol0 = 1.e-10;
        double r = h * std::sqrt(-std::log(tol0) / gamma);
        unsigned anisotropy = false;

        REQUIRE_NOTHROW(hex->initialise_lme_connectivity_properties(
            beta, r, anisotropy, nodal_coords));

        // Coordinates is (0,0,0) after upgrade
        SECTION("3D ALME element for coordinates(0,0,0)") {
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
          zero.setZero();
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 64);
          REQUIRE(gradsf.cols() == Dim);

          REQUIRE(gradsf(0, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(0, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(0, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(1, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(2, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 1) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(3, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(4, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(5, 2) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 0) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 1) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(6, 2) == Approx(0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 0) == Approx(-0.125).epsilon(Tolerance));
          REQUIRE(gradsf(7, 1) == Approx(-0.125).epsilon(Tolerance));
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

        // Check Jacobian
        SECTION("64-noded hexrilateral Jacobian with deformation gradient") {
          Eigen::Matrix<double, 64, Dim> coords;
          coords << -1., 1., -1, 1., 1., -1, 1., 1., 1, -1., 1., 1, -1., -1.,
              -1, 1., -1., -1, 1., -1., 1, -1., -1., 1, -3., 3, -3, -1., 3, -3,
              1., 3, -3, 3., 3, -3, -3., 3, -1, -1., 3, -1, 1., 3, -1, 3., 3,
              -1, -3., 3, 1, -1., 3, 1, 1., 3, 1, 3., 3, 1, -3., 3, 3, -1., 3,
              3, 1., 3, 3, 3., 3, 3, -3., 1., -3, -1., 1., -3, 1., 1., -3, 3.,
              1., -3, -3., 1., -1, 3., 1., -1, -3., 1., 1, 3., 1., 1, -3., 1.,
              3, -1., 1., 3, 1., 1., 3, 3., 1., 3, -3., -1., -3, -1., -1., -3,
              1., -1., -3, 3., -1., -3, -3., -1., -1, 3., -1., -1, -3., -1., 1,
              3., -1., 1, -3., -1., 3, -1., -1., 3, 1., -1., 3, 3., -1., 3, -3.,
              -3., -3, -1., -3., -3, 1., -3., -3, 3., -3., -3, -3., -3., -1,
              -1., -3., -1, 1., -3., -1, 3., -3., -1, -3., -3., 1, -1., -3., 1,
              1., -3., 1, 3., -3., 1, -3., -3., 3, -1., -3., 3, 1., -3., 3, 3.,
              -3., 3;

          Eigen::Matrix<double, Dim, 1> psize;
          psize.setZero();
          Eigen::Matrix<double, Dim, Dim> defgrad;
          defgrad.setZero();

          Eigen::Matrix<double, Dim, 1> xi;
          xi << 0., 0., 0.;

          Eigen::Matrix<double, Dim, Dim> jacobian;
          // clang-format off
          jacobian << 1., 0., 0.,
                      0., 0., 1.,
                      0., -1., 0.;
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
          coords << -1., 1., -1, 1., 1., -1, 1., 1., 1, -1., 1., 1, -1., -1.,
              -1, 1., -1., -1, 1., -1., 1, -1., -1., 1, -3., 3, -3, -1., 3, -3,
              1., 3, -3, 3., 3, -3, -3., 3, -1, -1., 3, -1, 1., 3, -1, 3., 3,
              -1, -3., 3, 1, -1., 3, 1, 1., 3, 1, 3., 3, 1, -3., 3, 3, -1., 3,
              3, 1., 3, 3, 3., 3, 3, -3., 1., -3, -1., 1., -3, 1., 1., -3, 3.,
              1., -3, -3., 1., -1, 3., 1., -1, -3., 1., 1, 3., 1., 1, -3., 1.,
              3, -1., 1., 3, 1., 1., 3, 3., 1., 3, -3., -1., -3, -1., -1., -3,
              1., -1., -3, 3., -1., -3, -3., -1., -1, 3., -1., -1, -3., -1., 1,
              3., -1., 1, -3., -1., 3, -1., -1., 3, 1., -1., 3, 3., -1., 3, -3.,
              -3., -3, -1., -3., -3, 1., -3., -3, 3., -3., -3, -3., -3., -1,
              -1., -3., -1, 1., -3., -1, 3., -3., -1, -3., -3., 1, -1., -3., 1,
              1., -3., 1, 3., -3., 1, -3., -3., 3, -1., -3., 3, 1., -3., 3, 3.,
              -3., 3;

          // Get B-Matrix
          auto bmatrix = hex->bmatrix(xi, coords, zero, zero_matrix);

          // Check gradient of shape functions
          auto gradsf = hex->grad_shapefn(xi, zero, zero_matrix);

          // Check dN/dx
          auto dn_dx = hex->dn_dx(xi, coords, zero, zero_matrix);
          REQUIRE(dn_dx.rows() == 64);
          REQUIRE(dn_dx.cols() == Dim);
          for (unsigned i = 0; i < 8; ++i) {
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
      }

      SECTION("3D ALME element regular element no support") {
        double gamma = 2.0;
        double h = 2.0;

        // Calculate beta
        double beta = gamma / (h * h);

        // Calculate support radius automatically
        double tol0 = 1.e-10;
        double r = h * std::sqrt(-std::log(tol0) / gamma);
        unsigned anisotropy = false;

        REQUIRE_NOTHROW(hex->initialise_lme_connectivity_properties(
            beta, r, anisotropy, nodal_coords));

        // Coordinates is (0,0,0) after upgrade
        SECTION("3D ALME element for coordinates(0.1, 0.1, 0.1)") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.1, 0.1, 0.1;
          zero.setZero();
          auto shapefn = hex->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 64);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          Eigen::Matrix<double, 64, 1> shapefn_ans;
          shapefn_ans << 8.983914627171446510e-02, 1.070346289463408179e-01,
              1.275213786986738873e-01, 1.070346289463408457e-01,
              1.070346289463408179e-01, 1.275213786986739151e-01,
              1.519293539417322925e-01, 1.275213786986738873e-01,
              3.264042151607778662e-07, 2.123207197712485166e-05,
              2.529595438229431816e-05, 5.519907923351278668e-07,
              2.123207197712485166e-05, 1.381112312596077590e-03,
              1.645461361184892947e-03, 3.590611790290974381e-05,
              2.529595438229436221e-05, 1.645461361184892947e-03,
              1.960407612370831075e-03, 4.277865681200847085e-05,
              5.519907923351278668e-07, 3.590611790290974381e-05,
              4.277865681200854539e-05, 9.334862133219627325e-07,
              2.123207197712485166e-05, 1.381112312596079975e-03,
              1.645461361184891646e-03, 3.590611790290974381e-05,
              1.381112312596077590e-03, 2.335635522838428336e-03,
              1.645461361184892947e-03, 2.782683183395450940e-03,
              3.590611790290974381e-05, 2.335635522838432239e-03,
              2.782683183395450940e-03, 6.072178467775909955e-05,
              2.529595438229431816e-05, 1.645461361184891646e-03,
              1.960407612370832810e-03, 4.277865681200847085e-05,
              1.645461361184892947e-03, 2.782683183395448338e-03,
              1.960407612370832810e-03, 3.315297110116567376e-03,
              4.277865681200854539e-05, 2.782683183395450940e-03,
              3.315297110116567376e-03, 7.234411680946502472e-05,
              5.519907923351278668e-07, 3.590611790290974381e-05,
              4.277865681200847085e-05, 9.334862133219677088e-07,
              3.590611790290974381e-05, 2.335635522838428336e-03,
              2.782683183395448338e-03, 6.072178467775909955e-05,
              4.277865681200854539e-05, 2.782683183395450940e-03,
              3.315297110116567376e-03, 7.234411680946502472e-05,
              9.334862133219627325e-07, 6.072178467775909955e-05,
              7.234411680946514669e-05, 1.578643199419772499e-06;

          for (unsigned i = 0; i < 64; ++i)
            REQUIRE(shapefn(i) == Approx(shapefn_ans(i)).epsilon(Tolerance));

          // Check gradient of shape functions
          zero.setZero();
          auto gradsf = hex->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 64);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 64, Dim> gradsf_ans;
          gradsf_ans << -8.682152672399913451e-02, 8.682152672399914839e-02,
              -8.682152672399913451e-02, 8.463225922813281954e-02,
              1.034394279454956961e-01, -1.034394279454956544e-01,
              1.008311280694525469e-01, 1.232380454182198104e-01,
              1.008311280694525330e-01, -1.034394279454956683e-01,
              1.034394279454956961e-01, 8.463225922813283342e-02,
              -1.034394279454956683e-01, -8.463225922813283342e-02,
              -1.034394279454956544e-01, 1.008311280694525330e-01,
              -1.008311280694525608e-01, -1.232380454182197965e-01,
              1.201305091047213114e-01, -1.201305091047213530e-01,
              1.201305091047213253e-01, -1.232380454182197826e-01,
              -1.008311280694525747e-01, 1.008311280694525330e-01,
              -8.889689035948416356e-07, 8.889689035948419532e-07,
              -8.889689035948417414e-07, -2.051890496591112653e-05,
              5.782600490393138522e-05, -5.782600490393136489e-05,
              2.000150595914818160e-05, 6.889407608151038975e-05,
              -6.889407608151037620e-05, 1.406367910358171960e-06,
              1.503358800727701594e-06, -1.503358800727700747e-06,
              -5.782600490393144621e-05, 5.782600490393146654e-05,
              -2.051890496591116380e-05, -1.334721939523378234e-03,
              3.761489102293158557e-03, -1.334721939523378451e-03,
              1.301065961928016745e-03, 4.481449424418723453e-03,
              -1.590191731245353029e-03, 9.148198249932056243e-05,
              9.779108474065303837e-05, -3.470006232732848219e-05,
              -6.889407608151048462e-05, 6.889407608151049817e-05,
              2.000150595914821210e-05, -1.590191731245352595e-03,
              4.481449424418723453e-03, 1.301065961928016311e-03,
              1.550093898360221500e-03, 5.339212316574095361e-03,
              1.550093898360221066e-03, 1.089919089666715714e-04,
              1.165085923436834381e-04, 3.382507519655323719e-05,
              -1.503358800727703499e-06, 1.503358800727703923e-06,
              1.406367910358174501e-06, -3.470006232732847541e-05,
              9.779108474065302482e-05, 9.148198249932054887e-05,
              3.382507519655325075e-05, 1.165085923436834110e-04,
              1.089919089666715579e-04, 2.378345931503573391e-06,
              2.542369788848647914e-06, 2.378345931503572967e-06,
              -5.782600490393136489e-05, 2.051890496591112653e-05,
              -5.782600490393136489e-05, -1.334721939523381053e-03,
              1.334721939523381053e-03, -3.761489102293164195e-03,
              1.301065961928015010e-03, 1.590191731245352161e-03,
              -4.481449424418718248e-03, 9.148198249932054887e-05,
              3.470006232732850252e-05, -9.779108474065299772e-05,
              -3.761489102293157256e-03, 1.334721939523378451e-03,
              -1.334721939523378451e-03, 5.950756598161134094e-03,
              2.257183537233534848e-03, -2.257183537233533547e-03,
              -4.481449424418721718e-03, 1.590191731245353029e-03,
              1.301065961928016311e-03, 7.089749300463794694e-03,
              2.689215251900061582e-03, 2.200267024281866990e-03,
              -9.779108474065299772e-05, 3.470006232732848896e-05,
              9.148198249932054887e-05, -2.257183537233538751e-03,
              2.257183537233539618e-03, 5.950756598161147971e-03,
              2.200267024281867424e-03, 2.689215251900061148e-03,
              7.089749300463794694e-03, 1.547075976923626924e-04,
              5.868219222813762136e-05, 1.547075976923626924e-04,
              -6.889407608151037620e-05, -2.000150595914819177e-05,
              -6.889407608151037620e-05, -1.590191731245351944e-03,
              -1.301065961928015661e-03, -4.481449424418718248e-03,
              1.550093898360220632e-03, -1.550093898360221283e-03,
              -5.339212316574093627e-03, 1.089919089666715579e-04,
              -3.382507519655323719e-05, -1.165085923436833974e-04,
              -4.481449424418721718e-03, -1.301065961928016962e-03,
              -1.590191731245353245e-03, 7.089749300463794694e-03,
              -2.200267024281866990e-03, -2.689215251900060281e-03,
              -5.339212316574093627e-03, -1.550093898360221933e-03,
              1.550093898360220849e-03, 8.446748630074931916e-03,
              -2.621404747264633340e-03, 2.621404747264633774e-03,
              -1.165085923436833703e-04, -3.382507519655325752e-05,
              1.089919089666715579e-04, -2.689215251900060281e-03,
              -2.200267024281867857e-03, 7.089749300463794694e-03,
              2.621404747264633340e-03, -2.621404747264633774e-03,
              8.446748630074931916e-03, 1.843190969791636132e-04,
              -5.720247837284385894e-05, 1.843190969791635861e-04,
              -1.503358800727700958e-06, -1.406367910358172383e-06,
              -1.503358800727700747e-06, -3.470006232732849574e-05,
              -9.148198249932056243e-05, -9.779108474065301127e-05,
              3.382507519655322364e-05, -1.089919089666715714e-04,
              -1.165085923436833974e-04, 2.378345931503585673e-06,
              -2.378345931503586096e-06, -2.542369788848661043e-06,
              -9.779108474065301127e-05, -9.148198249932057598e-05,
              -3.470006232732848896e-05, -2.257183537233534414e-03,
              -5.950756598161134961e-03, -2.257183537233533547e-03,
              2.200267024281866123e-03, -7.089749300463795562e-03,
              -2.689215251900060281e-03, 1.547075976923626653e-04,
              -1.547075976923626924e-04, -5.868219222813758748e-05,
              -1.165085923436833974e-04, -1.089919089666715985e-04,
              3.382507519655323719e-05, -2.689215251900061148e-03,
              -7.089749300463795562e-03, 2.200267024281866990e-03,
              2.621404747264632473e-03, -8.446748630074933650e-03,
              2.621404747264633340e-03, 1.843190969791635861e-04,
              -1.843190969791636132e-04, 5.720247837284386571e-05,
              -2.542369788848647067e-06, -2.378345931503573814e-06,
              2.378345931503572967e-06, -5.868219222813760103e-05,
              -1.547075976923627195e-04, 1.547075976923626653e-04,
              5.720247837284385216e-05, -1.843190969791636403e-04,
              1.843190969791635861e-04, 4.022083644143324285e-06,
              -4.022083644143325133e-06, 4.022083644143324285e-06;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }
      }
    }

    SECTION("3D ALME element anisotropic element - nnodes = 216") {
      Eigen::Matrix<double, 216, Dim> nodal_coords;

      nodal_coords << 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1,
          -1, -1, -1, -1, 1, -1, -1, -3, -3, -3, -5, -3, -3, -5, -5, -3, -3, -5,
          -3, -3, -3, -5, -5, -3, -5, -5, -5, -5, -3, -5, -5, -3, -3, 1, -5, -3,
          1, -5, -5, 1, -3, -5, 1, -3, -3, -1, -5, -3, -1, -5, -5, -1, -3, -5,
          -1, -3, -3, 5, -5, -3, 5, -5, -5, 5, -3, -5, 5, -3, -3, 3, -5, -3, 3,
          -5, -5, 3, -3, -5, 3, -3, 1, -3, -5, 1, -3, -5, -1, -3, -3, -1, -3,
          -3, 1, -5, -5, 1, -5, -5, -1, -5, -3, -1, -5, -3, 1, 1, -5, 1, 1, -5,
          -1, 1, -3, -1, 1, -3, 1, -1, -5, 1, -1, -5, -1, -1, -3, -1, -1, -3, 1,
          5, -5, 1, 5, -5, -1, 5, -3, -1, 5, -3, 1, 3, -5, 1, 3, -5, -1, 3, -3,
          -1, 3, -3, 5, -3, -5, 5, -3, -5, 3, -3, -3, 3, -3, -3, 5, -5, -5, 5,
          -5, -5, 3, -5, -3, 3, -5, -3, 5, 1, -5, 5, 1, -5, 3, 1, -3, 3, 1, -3,
          5, -1, -5, 5, -1, -5, 3, -1, -3, 3, -1, -3, 5, 5, -5, 5, 5, -5, 3, 5,
          -3, 3, 5, -3, 5, 3, -5, 5, 3, -5, 3, 3, -3, 3, 3, 1, -3, -3, -1, -3,
          -3, -1, -5, -3, 1, -5, -3, 1, -3, -5, -1, -3, -5, -1, -5, -5, 1, -5,
          -5, 1, -3, 1, -1, -3, 1, -1, -5, 1, 1, -5, 1, 1, -3, -1, -1, -3, -1,
          -1, -5, -1, 1, -5, -1, 1, -3, 5, -1, -3, 5, -1, -5, 5, 1, -5, 5, 1,
          -3, 3, -1, -3, 3, -1, -5, 3, 1, -5, 3, 1, 1, -3, -1, 1, -3, -1, -1,
          -3, 1, -1, -3, 1, 1, -5, -1, 1, -5, -1, -1, -5, 1, -1, -5, 1, 1, 5,
          -1, 1, 5, -1, -1, 5, 1, -1, 5, 1, 1, 3, -1, 1, 3, -1, -1, 3, 1, -1, 3,
          1, 5, -3, -1, 5, -3, -1, 3, -3, 1, 3, -3, 1, 5, -5, -1, 5, -5, -1, 3,
          -5, 1, 3, -5, 1, 5, 1, -1, 5, 1, -1, 3, 1, 1, 3, 1, 1, 5, -1, -1, 5,
          -1, -1, 3, -1, 1, 3, -1, 1, 5, 5, -1, 5, 5, -1, 3, 5, 1, 3, 5, 1, 5,
          3, -1, 5, 3, -1, 3, 3, 1, 3, 3, 5, -3, -3, 3, -3, -3, 3, -5, -3, 5,
          -5, -3, 5, -3, -5, 3, -3, -5, 3, -5, -5, 5, -5, -5, 5, -3, 1, 3, -3,
          1, 3, -5, 1, 5, -5, 1, 5, -3, -1, 3, -3, -1, 3, -5, -1, 5, -5, -1, 5,
          -3, 5, 3, -3, 5, 3, -5, 5, 5, -5, 5, 5, -3, 3, 3, -3, 3, 3, -5, 3, 5,
          -5, 3, 5, 1, -3, 3, 1, -3, 3, -1, -3, 5, -1, -3, 5, 1, -5, 3, 1, -5,
          3, -1, -5, 5, -1, -5, 5, 1, 1, 3, 1, 1, 3, -1, 1, 5, -1, 1, 5, 1, -1,
          3, 1, -1, 3, -1, -1, 5, -1, -1, 5, 1, 5, 3, 1, 5, 3, -1, 5, 5, -1, 5,
          5, 1, 3, 3, 1, 3, 3, -1, 3, 5, -1, 3, 5, 5, -3, 3, 5, -3, 3, 3, -3, 5,
          3, -3, 5, 5, -5, 3, 5, -5, 3, 3, -5, 5, 3, -5, 5, 5, 1, 3, 5, 1, 3, 3,
          1, 5, 3, 1, 5, 5, -1, 3, 5, -1, 3, 3, -1, 5, 3, -1, 5, 5, 5, 3, 5, 5,
          3, 3, 5, 5, 3, 5, 5, 5, 3, 3, 5, 3, 3, 3, 3, 5, 3, 3;

      double gamma = 0.5;
      double h = 2.0;

      // Calculate beta
      double beta = gamma / (h * h);

      // Calculate support radius automatically
      double tol0 = 1.e-10;
      double r = h * std::sqrt(-std::log(tol0) / gamma);
      unsigned anisotropy = true;

      REQUIRE_NOTHROW(hex->initialise_lme_connectivity_properties(
          beta, r, anisotropy, nodal_coords));

      // Coordinates is (0,0,0) after upgrade
      def_gradient(0, 1) = 0.5;
      def_gradient(0, 2) = 0.5;
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 0.0, 0.0, 0.0;
      auto shapefn = hex->shapefn(coords, zero, def_gradient);

      // Check shape function
      REQUIRE(shapefn.size() == 216);
      REQUIRE(hex->nfunctions() == 216);
      REQUIRE(hex->nfunctions_local() == 8);
      REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

      Eigen::VectorXd shapefn_ans = Eigen::VectorXd::Constant(216, 1.0);
      shapefn_ans << 5.019554348373359703e-02, 3.044513610382312135e-02,
          4.429741164794527564e-02, 4.429741164794527564e-02,
          4.429741164794527564e-02, 4.429741164794527564e-02,
          5.019554348373359703e-02, 3.044513610382312135e-02,
          6.793228094586798577e-03, 4.120301117788124565e-03,
          8.113353954493729512e-04, 8.113353954493729512e-04,
          8.113353954493729512e-04, 8.113353954493729512e-04,
          1.244223126692554048e-04, 7.546594738625509498e-05,
          1.120013965626295155e-02, 2.499088955186733532e-03,
          8.113353954493729512e-04, 2.205438262235663843e-03,
          1.629610704238744490e-02, 5.995002752023497437e-03,
          1.515774072669964345e-03, 2.499088955186733532e-03,
          1.244223126692554048e-04, 1.021320536423814445e-05,
          5.466734893019912842e-06, 4.039401080250588943e-05,
          2.205438262235663843e-03, 2.984736118805267151e-04,
          1.244223126692554048e-04, 5.576221187959881512e-04,
          1.120013965626295155e-02, 2.499088955186733532e-03,
          5.995002752023497437e-03, 1.629610704238744490e-02,
          2.205438262235663843e-03, 8.113353954493729512e-04,
          1.515774072669964345e-03, 2.499088955186733532e-03,
          6.793228094586798577e-03, 5.576221187959881512e-04,
          2.205438262235663843e-03, 1.629610704238744490e-02,
          1.629610704238744490e-02, 2.205438262235663843e-03,
          6.793228094586798577e-03, 3.044513610382312135e-02,
          2.776237055192899095e-05, 8.383509482690837696e-07,
          5.466734893019912842e-06, 1.098023055430299402e-04,
          8.113353954493729512e-04, 4.039401080250588943e-05,
          2.051377134475035877e-04, 2.499088955186733532e-03,
          1.244223126692554048e-04, 1.021320536423814445e-05,
          2.984736118805267151e-04, 2.205438262235663843e-03,
          4.039401080250588943e-05, 5.466734893019912842e-06,
          1.244223126692554048e-04, 5.576221187959881512e-04,
          2.776237055192899095e-05, 8.383509482690837696e-07,
          4.039401080250588943e-05, 8.113353954493729512e-04,
          1.098023055430299402e-04, 5.466734893019912842e-06,
          2.051377134475035877e-04, 2.499088955186733532e-03,
          4.173903597773644601e-08, 4.636788061862452377e-10,
          3.683456996721946813e-08, 2.011099377476593536e-06,
          2.011099377476593536e-06, 3.683456996721946813e-08,
          2.278874148551255536e-06, 7.546594738625509498e-05,
          9.193634482718187262e-04, 4.120301117788124565e-03,
          2.984736118805267151e-04, 4.039401080250588943e-05,
          4.039401080250588943e-05, 2.984736118805267151e-04,
          1.683872892604804491e-05, 1.382207040722861302e-06,
          1.120013965626295155e-02, 1.846590848609274554e-02,
          2.205438262235663843e-03, 8.113353954493729512e-04,
          5.995002752023497437e-03, 1.629610704238744490e-02,
          1.515774072669964345e-03, 3.382149115836868836e-04,
          9.193634482718187262e-04, 5.576221187959881512e-04,
          1.098023055430299402e-04, 1.098023055430299402e-04,
          5.995002752023497437e-03, 5.995002752023497437e-03,
          9.193634482718187262e-04, 5.576221187959881512e-04,
          1.120013965626295155e-02, 1.846590848609274554e-02,
          1.629610704238744490e-02, 5.995002752023497437e-03,
          8.113353954493729512e-04, 2.205438262235663843e-03,
          1.515774072669964345e-03, 3.382149115836868836e-04,
          1.515774072669964345e-03, 3.382149115836868836e-04,
          8.113353954493729512e-04, 2.205438262235663843e-03,
          1.629610704238744490e-02, 5.995002752023497437e-03,
          1.120013965626295155e-02, 1.846590848609274554e-02,
          9.193634482718187262e-04, 5.576221187959881512e-04,
          5.995002752023497437e-03, 5.995002752023497437e-03,
          1.098023055430299402e-04, 1.098023055430299402e-04,
          9.193634482718187262e-04, 5.576221187959881512e-04,
          1.515774072669964345e-03, 3.382149115836868836e-04,
          5.995002752023497437e-03, 1.629610704238744490e-02,
          2.205438262235663843e-03, 8.113353954493729512e-04,
          1.120013965626295155e-02, 1.846590848609274554e-02,
          1.683872892604804491e-05, 1.382207040722861302e-06,
          4.039401080250588943e-05, 2.984736118805267151e-04,
          2.984736118805267151e-04, 4.039401080250588943e-05,
          9.193634482718187262e-04, 4.120301117788124565e-03,
          2.278874148551255536e-06, 7.546594738625509498e-05,
          2.011099377476593536e-06, 3.683456996721946813e-08,
          3.683456996721946813e-08, 2.011099377476593536e-06,
          4.173903597773644601e-08, 4.636788061862452377e-10,
          2.051377134475035877e-04, 2.499088955186733532e-03,
          1.098023055430299402e-04, 5.466734893019912842e-06,
          4.039401080250588943e-05, 8.113353954493729512e-04,
          2.776237055192899095e-05, 8.383509482690837696e-07,
          1.244223126692554048e-04, 5.576221187959881512e-04,
          4.039401080250588943e-05, 5.466734893019912842e-06,
          2.984736118805267151e-04, 2.205438262235663843e-03,
          1.244223126692554048e-04, 1.021320536423814445e-05,
          2.051377134475035877e-04, 2.499088955186733532e-03,
          8.113353954493729512e-04, 4.039401080250588943e-05,
          5.466734893019912842e-06, 1.098023055430299402e-04,
          2.776237055192899095e-05, 8.383509482690837696e-07,
          6.793228094586798577e-03, 3.044513610382312135e-02,
          1.629610704238744490e-02, 2.205438262235663843e-03,
          2.205438262235663843e-03, 1.629610704238744490e-02,
          6.793228094586798577e-03, 5.576221187959881512e-04,
          1.515774072669964345e-03, 2.499088955186733532e-03,
          2.205438262235663843e-03, 8.113353954493729512e-04,
          5.995002752023497437e-03, 1.629610704238744490e-02,
          1.120013965626295155e-02, 2.499088955186733532e-03,
          1.244223126692554048e-04, 5.576221187959881512e-04,
          2.205438262235663843e-03, 2.984736118805267151e-04,
          5.466734893019912842e-06, 4.039401080250588943e-05,
          1.244223126692554048e-04, 1.021320536423814445e-05,
          1.515774072669964345e-03, 2.499088955186733532e-03,
          1.629610704238744490e-02, 5.995002752023497437e-03,
          8.113353954493729512e-04, 2.205438262235663843e-03,
          1.120013965626295155e-02, 2.499088955186733532e-03,
          1.244223126692554048e-04, 7.546594738625509498e-05,
          8.113353954493729512e-04, 8.113353954493729512e-04,
          8.113353954493729512e-04, 8.113353954493729512e-04,
          6.793228094586798577e-03, 4.120301117788124565e-03;

      for (unsigned i = 0; i < 216; ++i)
        REQUIRE(shapefn(i) == Approx(shapefn_ans(i)).epsilon(Tolerance));

      // Check gradient of shape functions
      zero.setZero();
      auto gradsf = hex->grad_shapefn(coords, zero, def_gradient);
      REQUIRE(gradsf.rows() == 216);
      REQUIRE(gradsf.cols() == Dim);

      Eigen::Matrix<double, 216, Dim> gradsf_ans;
      gradsf_ans << 8.028344871708081344e-04, 1.278577816067815072e-02,
          1.278577816067815072e-02, -1.572532355459026546e-02,
          1.537415637445875040e-02, 1.537415637445875040e-02,
          -1.179435485611927091e-02, -5.746278417446468514e-03,
          1.683213432534940920e-02, 1.179435485611927091e-02,
          -1.683213432534940920e-02, 5.746278417446468514e-03,
          1.179435485611927091e-02, 5.746278417446468514e-03,
          -1.683213432534940920e-02, -1.179435485611927091e-02,
          1.683213432534940920e-02, -5.746278417446468514e-03,
          -8.028344871708081344e-04, -1.278577816067815072e-02,
          -1.278577816067815072e-02, 1.572532355459026546e-02,
          -1.537415637445875040e-02, -1.537415637445875040e-02,
          -3.259554981401462169e-04, -5.191100726327626223e-03,
          -5.191100726327626223e-03, -2.391793788346387867e-03,
          -2.117416523437615737e-03, -2.117416523437615737e-03,
          -2.679275880245821496e-04, -9.318981848705192982e-04,
          -5.183601303926759765e-04, 1.641147009159557905e-04,
          -1.134942718452209946e-03, -7.214046639743666245e-04,
          1.641147009159557905e-04, -7.214046639743665161e-04,
          -1.134942718452210163e-03, -2.679275880245821496e-04,
          -5.183601303926758681e-04, -9.318981848705192982e-04,
          -9.950138662887307342e-06, -1.584638772307706532e-04,
          -1.584638772307706532e-04, 3.415112865577679967e-05,
          -1.149992835755195265e-04, -1.149992835755195265e-04,
          -6.143296416892995299e-03, -5.758666153511818667e-03,
          5.658767350090673776e-03, -2.701538651180367730e-03,
          -6.595109076554083859e-04, 1.888062858487465608e-03,
          -6.740166551879634999e-04, -7.290659935554507894e-04,
          5.115481698780797179e-04, -6.577545227232704621e-04,
          -2.533739108032272928e-03, 8.385998285580127665e-04,
          -4.860185067727599943e-03, -1.041580654694705330e-02,
          -2.109672684589136617e-03, -4.980346876797456766e-03,
          -2.331453642624411841e-03, 7.242022409550121984e-04,
          -8.798917622541961190e-04, -1.551544805892025609e-03,
          -6.363208846791123291e-06, -1.199123265025560977e-04,
          -3.183486117337512664e-03, -6.359123511946386705e-04,
          -1.305216340595888021e-04, -3.286767839004937881e-05,
          2.208047801442140500e-04, -1.615247794010064018e-05,
          -1.419898651429649501e-07, 2.068071354456309485e-05,
          -7.277695110612270198e-06, -3.545735468622911533e-06,
          1.038625199726020198e-05, -3.226517846593650416e-05,
          -3.630863036477560815e-05, 6.663560659023366490e-05,
          -1.761619054729374537e-03, -8.582710692905650080e-04,
          2.514067867299720795e-03, -3.973486895446900688e-04,
          -4.145844870928057602e-05, 4.149379964440234729e-04,
          -1.345016895247437311e-04, -9.625322928235764144e-05,
          1.574192292519057941e-04, -3.058567187119053859e-04,
          -5.709273767072963694e-04, 5.659537081514999035e-04,
          -6.143296416892995299e-03, 5.658767350090673776e-03,
          -5.758666153511818667e-03, -2.701538651180367730e-03,
          1.888062858487465608e-03, -6.595109076554083859e-04,
          -4.980346876797456766e-03, 7.242022409550121984e-04,
          -2.331453642624411841e-03, -4.860185067727599943e-03,
          -2.109672684589136184e-03, -1.041580654694705330e-02,
          -6.577545227232704621e-04, 8.385998285580128749e-04,
          -2.533739108032272928e-03, -6.740166551879634999e-04,
          5.115481698780798264e-04, -7.290659935554507894e-04,
          -8.798917622541961190e-04, -6.363208846791038587e-06,
          -1.551544805892025609e-03, -1.199123265025560977e-04,
          -6.359123511946384536e-04, -3.183486117337512664e-03,
          -7.126239758956587171e-03, 5.130509039184096183e-03,
          5.130509039184096183e-03, -8.818954139806679414e-04,
          5.606881584691366261e-04, 5.606881584691366261e-04,
          -2.936031757882291184e-03, 8.177741755759157386e-04,
          1.941887154439344339e-03, -1.301670202034052939e-02,
          1.964320783280738749e-03, 1.027045464563865716e-02,
          -1.301670202034052939e-02, 1.027045464563865716e-02,
          1.964320783280738749e-03, -2.936031757882291184e-03,
          1.941887154439344339e-03, 8.177741755759157386e-04,
          -7.343543424383353231e-03, 1.669775221632345584e-03,
          1.669775221632345584e-03, -1.669921101687779283e-02,
          -1.357765510122502001e-04, -1.357765510122502001e-04,
          -4.301891685582150311e-05, 2.790775372595277783e-05,
          5.620874187495086339e-05, -1.745487003666260052e-06,
          1.052546576808505500e-06, 1.907162367174292589e-06,
          -1.001390172206718672e-05, 3.393732072519627702e-06,
          1.175292455204949623e-05, -1.426640272397911263e-04,
          4.068584138807525459e-05, 2.085847105837052332e-04,
          -1.054152500574188044e-03, 7.141680189265264128e-04,
          1.127706073404370060e-03, -7.399328159353288261e-05,
          4.566532405958962537e-05, 6.625417145059149624e-05,
          -3.244311920668656378e-04, 1.017068518650117225e-04,
          3.108244409531300389e-04, -2.621597100178663358e-03,
          6.136219151886419506e-04, 3.161195681331515402e-03,
          -1.305216340595888021e-04, 2.208047801442140771e-04,
          -3.286767839004936525e-05, -1.615247794010064018e-05,
          2.068071354456309485e-05, -1.419898651429638119e-07,
          -3.973486895446900688e-04, 4.149379964440234729e-04,
          -4.145844870928057602e-05, -1.761619054729374320e-03,
          2.514067867299721228e-03, -8.582710692905650080e-04,
          -3.226517846593649738e-05, 6.663560659023366490e-05,
          -3.630863036477560138e-05, -7.277695110612270198e-06,
          1.038625199726020198e-05, -3.545735468622911109e-06,
          -1.345016895247437311e-04, 1.574192292519057941e-04,
          -9.625322928235764144e-05, -3.058567187119052774e-04,
          5.659537081514999035e-04, -5.709273767072963694e-04,
          -4.301891685582150311e-05, 5.620874187495086339e-05,
          2.790775372595277783e-05, -1.745487003666260052e-06,
          1.907162367174292589e-06, 1.052546576808505500e-06,
          -7.399328159353288261e-05, 6.625417145059149624e-05,
          4.566532405958962537e-05, -1.054152500574188044e-03,
          1.127706073404370060e-03, 7.141680189265264128e-04,
          -1.426640272397911263e-04, 2.085847105837052332e-04,
          4.068584138807526814e-05, -1.001390172206718672e-05,
          1.175292455204949623e-05, 3.393732072519628125e-06,
          -3.244311920668656378e-04, 3.108244409531300389e-04,
          1.017068518650117225e-04, -2.621597100178663358e-03,
          3.161195681331515402e-03, 6.136219151886419506e-04,
          -8.556752092121144220e-08, 9.494109924490714060e-08,
          9.494109924490714060e-08, -1.197481796077310639e-09,
          1.170740450875256526e-09, 1.170740450875256526e-09,
          -8.590955418387018303e-08, 6.962455132782635646e-08,
          8.839914995523511391e-08, -3.619576964488194971e-06,
          3.298074619811187113e-06, 4.323132972482512233e-06,
          -3.619576964488194971e-06, 4.323132972482512233e-06,
          3.298074619811187113e-06, -8.590955418387018303e-08,
          8.839914995523511391e-08, 6.962455132782635646e-08,
          -4.744725603877412937e-06, 4.022661519978067569e-06,
          4.022661519978067569e-06, -1.169375399691701209e-04,
          1.143261707327957832e-04, 1.143261707327957832e-04,
          9.350228230815148169e-04, -1.162698285370870801e-03,
          -1.162698285370870801e-03, 1.996389781698581622e-03,
          -4.179706972910831805e-03, -4.179706972910832673e-03,
          2.193139002788562712e-04, -4.922180025727455064e-04,
          -3.400858541883109115e-04, 5.119102778925625275e-05,
          -7.672345486749495159e-05, -5.613460747649310782e-05,
          5.119102778925625275e-05, -5.613460747649310782e-05,
          -7.672345486749495159e-05, 2.193139002788562712e-04,
          -3.400858541883109115e-04, -4.922180025727456148e-04,
          1.658689844661160950e-05, -2.987386341452249230e-05,
          -2.987386341452249230e-05, 2.097571332227821604e-06,
          -2.798106724110892900e-06, -2.798106724110892900e-06,
          5.785023241502785032e-03, -1.136455280731949770e-02,
          5.288069628299405238e-05, -2.953463024935582926e-04,
          -1.411570766178916504e-02, 4.708457812404548241e-03,
          5.166581804296458596e-04, -3.085671374035325182e-03,
          2.866675625549607292e-04, 6.221102116336502119e-04,
          -1.338199594300522625e-03, -9.758543086699221264e-05,
          4.596807253478658699e-03, -6.832375990273371891e-03,
          -3.776720106693947417e-03, 3.817616279166087187e-03,
          -1.449406502325351932e-02, -6.187931160895601343e-03,
          7.344307572516519007e-04, -2.310219063977434970e-03,
          -7.650374669322004655e-04, 3.439756736377720279e-04,
          -6.001211040767850686e-04, -2.553444868696752861e-04,
          1.470442655444261597e-05, -7.030203177292217859e-04,
          1.171379709134111628e-03, -2.880193476343812778e-04,
          -2.868530460558807340e-04, 8.500280388029153762e-04,
          -2.923528277198104598e-05, -1.261761795987296662e-04,
          1.536552690606536587e-04, 2.923528277198104598e-05,
          -1.536552690606536587e-04, 1.261761795987296662e-04,
          1.596192355046019027e-03, -5.333637549977295923e-03,
          3.833330100760975762e-03, -1.596192355046019027e-03,
          -3.833330100760975762e-03, 5.333637549977295923e-03,
          -1.470442655444261597e-05, -1.171379709134111628e-03,
          7.030203177292216775e-04, 2.880193476343812778e-04,
          -8.500280388029155931e-04, 2.868530460558806798e-04,
          5.785023241502785032e-03, 5.288069628299405238e-05,
          -1.136455280731949770e-02, -2.953463024935582926e-04,
          4.708457812404548241e-03, -1.411570766178916504e-02,
          3.817616279166087187e-03, -6.187931160895601343e-03,
          -1.449406502325351932e-02, 4.596807253478658699e-03,
          -3.776720106693947417e-03, -6.832375990273371891e-03,
          6.221102116336502119e-04, -9.758543086699217198e-05,
          -1.338199594300522625e-03, 5.166581804296458596e-04,
          2.866675625549608376e-04, -3.085671374035325182e-03,
          7.344307572516519007e-04, -7.650374669322003570e-04,
          -2.310219063977434970e-03, 3.439756736377720279e-04,
          -2.553444868696752319e-04, -6.001211040767850686e-04,
          -7.344307572516519007e-04, 7.650374669322003570e-04,
          2.310219063977434970e-03, -3.439756736377720279e-04,
          2.553444868696752319e-04, 6.001211040767850686e-04,
          -6.221102116336502119e-04, 9.758543086699217198e-05,
          1.338199594300522625e-03, -5.166581804296458596e-04,
          -2.866675625549608376e-04, 3.085671374035325182e-03,
          -3.817616279166087187e-03, 6.187931160895601343e-03,
          1.449406502325351932e-02, -4.596807253478658699e-03,
          3.776720106693947417e-03, 6.832375990273371891e-03,
          -5.785023241502785032e-03, -5.288069628299405238e-05,
          1.136455280731949770e-02, 2.953463024935582926e-04,
          -4.708457812404548241e-03, 1.411570766178916504e-02,
          1.470442655444261597e-05, 1.171379709134111628e-03,
          -7.030203177292216775e-04, -2.880193476343812778e-04,
          8.500280388029155931e-04, -2.868530460558806798e-04,
          -1.596192355046019027e-03, 5.333637549977295923e-03,
          -3.833330100760975762e-03, 1.596192355046019027e-03,
          3.833330100760975762e-03, -5.333637549977295923e-03,
          2.923528277198104598e-05, 1.261761795987296662e-04,
          -1.536552690606536587e-04, -2.923528277198104598e-05,
          1.536552690606536587e-04, -1.261761795987296662e-04,
          -1.470442655444261597e-05, 7.030203177292217859e-04,
          -1.171379709134111628e-03, 2.880193476343812778e-04,
          2.868530460558807340e-04, -8.500280388029153762e-04,
          -7.344307572516519007e-04, 2.310219063977434970e-03,
          7.650374669322004655e-04, -3.439756736377720279e-04,
          6.001211040767850686e-04, 2.553444868696752861e-04,
          -4.596807253478658699e-03, 6.832375990273371891e-03,
          3.776720106693947417e-03, -3.817616279166087187e-03,
          1.449406502325351932e-02, 6.187931160895601343e-03,
          -5.166581804296458596e-04, 3.085671374035325182e-03,
          -2.866675625549607292e-04, -6.221102116336502119e-04,
          1.338199594300522625e-03, 9.758543086699221264e-05,
          -5.785023241502785032e-03, 1.136455280731949770e-02,
          -5.288069628299405238e-05, 2.953463024935582926e-04,
          1.411570766178916504e-02, -4.708457812404548241e-03,
          -1.658689844661160950e-05, 2.987386341452249230e-05,
          2.987386341452249230e-05, -2.097571332227821604e-06,
          2.798106724110892900e-06, 2.798106724110892900e-06,
          -5.119102778925625275e-05, 5.613460747649310782e-05,
          7.672345486749495159e-05, -2.193139002788562712e-04,
          3.400858541883109115e-04, 4.922180025727456148e-04,
          -2.193139002788562712e-04, 4.922180025727455064e-04,
          3.400858541883109115e-04, -5.119102778925625275e-05,
          7.672345486749495159e-05, 5.613460747649310782e-05,
          -9.350228230815148169e-04, 1.162698285370870801e-03,
          1.162698285370870801e-03, -1.996389781698581622e-03,
          4.179706972910831805e-03, 4.179706972910832673e-03,
          4.744725603877412937e-06, -4.022661519978067569e-06,
          -4.022661519978067569e-06, 1.169375399691701209e-04,
          -1.143261707327957832e-04, -1.143261707327957832e-04,
          3.619576964488194971e-06, -4.323132972482512233e-06,
          -3.298074619811187113e-06, 8.590955418387018303e-08,
          -8.839914995523511391e-08, -6.962455132782635646e-08,
          8.590955418387018303e-08, -6.962455132782635646e-08,
          -8.839914995523511391e-08, 3.619576964488194971e-06,
          -3.298074619811187113e-06, -4.323132972482512233e-06,
          8.556752092121144220e-08, -9.494109924490714060e-08,
          -9.494109924490714060e-08, 1.197481796077310639e-09,
          -1.170740450875256526e-09, -1.170740450875256526e-09,
          3.244311920668656378e-04, -3.108244409531300389e-04,
          -1.017068518650117225e-04, 2.621597100178663358e-03,
          -3.161195681331515402e-03, -6.136219151886419506e-04,
          1.426640272397911263e-04, -2.085847105837052332e-04,
          -4.068584138807526814e-05, 1.001390172206718672e-05,
          -1.175292455204949623e-05, -3.393732072519628125e-06,
          7.399328159353288261e-05, -6.625417145059149624e-05,
          -4.566532405958962537e-05, 1.054152500574188044e-03,
          -1.127706073404370060e-03, -7.141680189265264128e-04,
          4.301891685582150311e-05, -5.620874187495086339e-05,
          -2.790775372595277783e-05, 1.745487003666260052e-06,
          -1.907162367174292589e-06, -1.052546576808505500e-06,
          1.345016895247437311e-04, -1.574192292519057941e-04,
          9.625322928235764144e-05, 3.058567187119052774e-04,
          -5.659537081514999035e-04, 5.709273767072963694e-04,
          3.226517846593649738e-05, -6.663560659023366490e-05,
          3.630863036477560138e-05, 7.277695110612270198e-06,
          -1.038625199726020198e-05, 3.545735468622911109e-06,
          3.973486895446900688e-04, -4.149379964440234729e-04,
          4.145844870928057602e-05, 1.761619054729374320e-03,
          -2.514067867299721228e-03, 8.582710692905650080e-04,
          1.305216340595888021e-04, -2.208047801442140771e-04,
          3.286767839004936525e-05, 1.615247794010064018e-05,
          -2.068071354456309485e-05, 1.419898651429638119e-07,
          3.244311920668656378e-04, -1.017068518650117225e-04,
          -3.108244409531300389e-04, 2.621597100178663358e-03,
          -6.136219151886419506e-04, -3.161195681331515402e-03,
          1.054152500574188044e-03, -7.141680189265264128e-04,
          -1.127706073404370060e-03, 7.399328159353288261e-05,
          -4.566532405958962537e-05, -6.625417145059149624e-05,
          1.001390172206718672e-05, -3.393732072519627702e-06,
          -1.175292455204949623e-05, 1.426640272397911263e-04,
          -4.068584138807525459e-05, -2.085847105837052332e-04,
          4.301891685582150311e-05, -2.790775372595277783e-05,
          -5.620874187495086339e-05, 1.745487003666260052e-06,
          -1.052546576808505500e-06, -1.907162367174292589e-06,
          7.343543424383353231e-03, -1.669775221632345584e-03,
          -1.669775221632345584e-03, 1.669921101687779283e-02,
          1.357765510122502001e-04, 1.357765510122502001e-04,
          1.301670202034052939e-02, -1.027045464563865716e-02,
          -1.964320783280738749e-03, 2.936031757882291184e-03,
          -1.941887154439344339e-03, -8.177741755759157386e-04,
          2.936031757882291184e-03, -8.177741755759157386e-04,
          -1.941887154439344339e-03, 1.301670202034052939e-02,
          -1.964320783280738749e-03, -1.027045464563865716e-02,
          7.126239758956587171e-03, -5.130509039184096183e-03,
          -5.130509039184096183e-03, 8.818954139806679414e-04,
          -5.606881584691366261e-04, -5.606881584691366261e-04,
          8.798917622541961190e-04, 6.363208846791038587e-06,
          1.551544805892025609e-03, 1.199123265025560977e-04,
          6.359123511946384536e-04, 3.183486117337512664e-03,
          6.577545227232704621e-04, -8.385998285580128749e-04,
          2.533739108032272928e-03, 6.740166551879634999e-04,
          -5.115481698780798264e-04, 7.290659935554507894e-04,
          4.980346876797456766e-03, -7.242022409550121984e-04,
          2.331453642624411841e-03, 4.860185067727599943e-03,
          2.109672684589136184e-03, 1.041580654694705330e-02,
          6.143296416892995299e-03, -5.658767350090673776e-03,
          5.758666153511818667e-03, 2.701538651180367730e-03,
          -1.888062858487465608e-03, 6.595109076554083859e-04,
          1.345016895247437311e-04, 9.625322928235764144e-05,
          -1.574192292519057941e-04, 3.058567187119053859e-04,
          5.709273767072963694e-04, -5.659537081514999035e-04,
          1.761619054729374537e-03, 8.582710692905650080e-04,
          -2.514067867299720795e-03, 3.973486895446900688e-04,
          4.145844870928057602e-05, -4.149379964440234729e-04,
          7.277695110612270198e-06, 3.545735468622911533e-06,
          -1.038625199726020198e-05, 3.226517846593650416e-05,
          3.630863036477560815e-05, -6.663560659023366490e-05,
          1.305216340595888021e-04, 3.286767839004937881e-05,
          -2.208047801442140500e-04, 1.615247794010064018e-05,
          1.419898651429649501e-07, -2.068071354456309485e-05,
          8.798917622541961190e-04, 1.551544805892025609e-03,
          6.363208846791123291e-06, 1.199123265025560977e-04,
          3.183486117337512664e-03, 6.359123511946386705e-04,
          4.860185067727599943e-03, 1.041580654694705330e-02,
          2.109672684589136617e-03, 4.980346876797456766e-03,
          2.331453642624411841e-03, -7.242022409550121984e-04,
          6.740166551879634999e-04, 7.290659935554507894e-04,
          -5.115481698780797179e-04, 6.577545227232704621e-04,
          2.533739108032272928e-03, -8.385998285580127665e-04,
          6.143296416892995299e-03, 5.758666153511818667e-03,
          -5.658767350090673776e-03, 2.701538651180367730e-03,
          6.595109076554083859e-04, -1.888062858487465608e-03,
          9.950138662887307342e-06, 1.584638772307706532e-04,
          1.584638772307706532e-04, -3.415112865577679967e-05,
          1.149992835755195265e-04, 1.149992835755195265e-04,
          -1.641147009159557905e-04, 7.214046639743665161e-04,
          1.134942718452210163e-03, 2.679275880245821496e-04,
          5.183601303926758681e-04, 9.318981848705192982e-04,
          2.679275880245821496e-04, 9.318981848705192982e-04,
          5.183601303926759765e-04, -1.641147009159557905e-04,
          1.134942718452209946e-03, 7.214046639743666245e-04,
          3.259554981401462169e-04, 5.191100726327626223e-03,
          5.191100726327626223e-03, 2.391793788346387867e-03,
          2.117416523437615737e-03, 2.117416523437615737e-03;

      for (unsigned i = 0; i < gradsf.rows(); ++i)
        for (unsigned j = 0; j < gradsf.cols(); ++j)
          REQUIRE(gradsf(i, j) == Approx(gradsf_ans(i, j)).epsilon(Tolerance));
    }

    SECTION("3D LME element evaluation at the edge of the mesh") {
      Eigen::Matrix<double, 216, Dim> nodal_coords;

      nodal_coords << 5, 5, 5, 3, 5, 5, 3, 3, 5, 5, 3, 5, 5, 5, 3, 3, 5, 3, 3,
          3, 3, 5, 3, 3, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1,
          1, -1, -1, -1, -1, 1, -1, -1, -3, -3, -3, -5, -3, -3, -5, -5, -3, -3,
          -5, -3, -3, -3, -5, -5, -3, -5, -5, -5, -5, -3, -5, -5, -3, -3, 1, -5,
          -3, 1, -5, -5, 1, -3, -5, 1, -3, -3, -1, -5, -3, -1, -5, -5, -1, -3,
          -5, -1, -3, -3, 5, -5, -3, 5, -5, -5, 5, -3, -5, 5, -3, -3, 3, -5, -3,
          3, -5, -5, 3, -3, -5, 3, -3, 1, -3, -5, 1, -3, -5, -1, -3, -3, -1, -3,
          -3, 1, -5, -5, 1, -5, -5, -1, -5, -3, -1, -5, -3, 1, 1, -5, 1, 1, -5,
          -1, 1, -3, -1, 1, -3, 1, -1, -5, 1, -1, -5, -1, -1, -3, -1, -1, -3, 1,
          5, -5, 1, 5, -5, -1, 5, -3, -1, 5, -3, 1, 3, -5, 1, 3, -5, -1, 3, -3,
          -1, 3, -3, 5, -3, -5, 5, -3, -5, 3, -3, -3, 3, -3, -3, 5, -5, -5, 5,
          -5, -5, 3, -5, -3, 3, -5, -3, 5, 1, -5, 5, 1, -5, 3, 1, -3, 3, 1, -3,
          5, -1, -5, 5, -1, -5, 3, -1, -3, 3, -1, -3, 5, 5, -5, 5, 5, -5, 3, 5,
          -3, 3, 5, -3, 5, 3, -5, 5, 3, -5, 3, 3, -3, 3, 3, 1, -3, -3, -1, -3,
          -3, -1, -5, -3, 1, -5, -3, 1, -3, -5, -1, -3, -5, -1, -5, -5, 1, -5,
          -5, 1, -3, 1, -1, -3, 1, -1, -5, 1, 1, -5, 1, 1, -3, -1, -1, -3, -1,
          -1, -5, -1, 1, -5, -1, 1, -3, 5, -1, -3, 5, -1, -5, 5, 1, -5, 5, 1,
          -3, 3, -1, -3, 3, -1, -5, 3, 1, -5, 3, 1, 1, -3, -1, 1, -3, -1, -1,
          -3, 1, -1, -3, 1, 1, -5, -1, 1, -5, -1, -1, -5, 1, -1, -5, 1, 1, 5,
          -1, 1, 5, -1, -1, 5, 1, -1, 5, 1, 1, 3, -1, 1, 3, -1, -1, 3, 1, -1, 3,
          1, 5, -3, -1, 5, -3, -1, 3, -3, 1, 3, -3, 1, 5, -5, -1, 5, -5, -1, 3,
          -5, 1, 3, -5, 1, 5, 1, -1, 5, 1, -1, 3, 1, 1, 3, 1, 1, 5, -1, -1, 5,
          -1, -1, 3, -1, 1, 3, -1, 1, 5, 5, -1, 5, 5, -1, 3, 5, 1, 3, 5, 1, 5,
          3, -1, 5, 3, -1, 3, 3, 1, 3, 3, 5, -3, -3, 3, -3, -3, 3, -5, -3, 5,
          -5, -3, 5, -3, -5, 3, -3, -5, 3, -5, -5, 5, -5, -5, 5, -3, 1, 3, -3,
          1, 3, -5, 1, 5, -5, 1, 5, -3, -1, 3, -3, -1, 3, -5, -1, 5, -5, -1, 5,
          -3, 5, 3, -3, 5, 3, -5, 5, 5, -5, 5, 5, -3, 3, 3, -3, 3, 3, -5, 3, 5,
          -5, 3, 5, 1, -3, 3, 1, -3, 3, -1, -3, 5, -1, -3, 5, 1, -5, 3, 1, -5,
          3, -1, -5, 5, -1, -5, 5, 1, 1, 3, 1, 1, 3, -1, 1, 5, -1, 1, 5, 1, -1,
          3, 1, -1, 3, -1, -1, 5, -1, -1, 5, 1, 5, 3, 1, 5, 3, -1, 5, 5, -1, 5,
          5, 1, 3, 3, 1, 3, 3, -1, 3, 5, -1, 3, 5, 5, -3, 3, 5, -3, 3, 3, -3, 5,
          3, -3, 5, 5, -5, 3, 5, -5, 3, 3, -5, 5, 3, -5, 5, 5, 1, 3, 5, 1, 3, 3,
          1, 5, 3, 1, 5, 5, -1, 3, 5, -1, 3, 3, -1, 5, 3, -1;

      double gamma = 0.5;
      double h = 2.0;

      // Calculate beta
      double beta = gamma / (h * h);

      // Calculate support radius automatically
      double tol0 = 1.e-10;
      double r = h * std::sqrt(-std::log(tol0) / gamma);
      unsigned anisotropy = false;

      REQUIRE_NOTHROW(hex->initialise_lme_connectivity_properties(
          beta, r, anisotropy, nodal_coords));

      // Coordinates is (0,0,0) after upgrade
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1.0, -1.0, -1.0;
      auto shapefn = hex->shapefn(coords, zero, def_gradient);

      // Check shape function
      REQUIRE(shapefn.size() == 216);
      REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

      // Check gradient of shape functions
      zero.setZero();
      auto gradsf = hex->grad_shapefn(coords, zero, def_gradient);
      REQUIRE(gradsf.rows() == 216);
      REQUIRE(gradsf.cols() == Dim);
    }
  }
}
