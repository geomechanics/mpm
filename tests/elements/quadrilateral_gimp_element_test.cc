// Quadrilateral element test
#include <memory>

#include "catch.hpp"

#include "quadrilateral_gimp_element.h"

//! \brief Check quadrilateral element class
TEST_CASE("Quadrilateral gimp elements are checked",
          "[quad][element][2D][gimp]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-7;

  Eigen::Vector2d zero = Eigen::Vector2d::Zero();
  Eigen::Matrix<double, Dim, Dim> defgrad;
  defgrad.setZero();

  //! Check for center element nodes
  SECTION("16 Node Quadrilateral GIMP Element") {
    const unsigned nfunctions = 16;
    std::shared_ptr<mpm::Element<Dim>> quad =
        std::make_shared<mpm::QuadrilateralGIMPElement<Dim, nfunctions>>();

    // Check degree
    REQUIRE(quad->degree() == mpm::ElementDegree::Linear);
    REQUIRE(quad->shapefn_type() == mpm::ShapefnType::GIMP);

    // Coordinates is (0,0) Size is (0,0)
    SECTION(
        "16 Node quadrilateral element matrix for coordinate(0,0), Size "
        "(0,0)") {

      // Coordinate location of point (x,y)
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      // Particle size (x,y)
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      // Deformation gradient
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);
      REQUIRE(quad->nfunctions() == nfunctions);
      REQUIRE(quad->nfunctions_local() == 4);

      REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.25).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (-1,-1) Size is (0,0)
    SECTION(
        "16 Node quadrilateral element matrix for coordinate(-1,-1), Size "
        "(0,0)") {
      // Coordinate location of point (x,y)
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1, -1;
      // Particle size (x,y)
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      // Deformation gradient
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);
      REQUIRE(quad->nfunctions() == nfunctions);
      REQUIRE(quad->nfunctions_local() == 4);

      REQUIRE(shapefn(0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(-0.5).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (1,1) Size is (0,0)
    SECTION(
        "16 Node quadrilateral element matrix for coordinate(1,1), Size "
        "(1,1)") {
      // Coordinate location of point (x,y)
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1, 1;
      // Particle size (x,y)
      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      // Deformation gradient
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);
      REQUIRE(quad->nfunctions() == nfunctions);
      REQUIRE(quad->nfunctions_local() == 4);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.5).epsilon(Tolerance));

      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.5).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.5).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Coordinates is (-0.8,-0.8) Size is (0.25,0.25)
    SECTION(
        "16 Node quadrilateral element matrix for coordinate(-0.8,-0.8), Size "
        "(0.25,0.25)") {
      // Location of point (x,y)
      Eigen::Matrix<double, Dim, 1> coords;
      coords << -0.8, -0.8;
      // Size of particle (x,y)
      Eigen::Matrix<double, Dim, 1> psize;
      psize << 0.5, 0.5;
      // Deformarion gradient
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);
      REQUIRE(quad->nfunctions() == nfunctions);
      REQUIRE(quad->nfunctions_local() == 4);

      REQUIRE(shapefn(0) == Approx(0.80550625).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.090871875).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.0102515625).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.090871875).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(1.5625e-06).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.001121875).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0001265625).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.0001265625).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.001121875).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.359).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.403875).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0455625).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.0405).epsilon(Tolerance));

      REQUIRE(gradsf(4, 0) == Approx(-6.25e-05).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(-0.0005).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0005625).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(-0.0050625).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(-0.044875).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-0.359).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.0405).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.0455625).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.403875).epsilon(Tolerance));

      REQUIRE(gradsf(4, 1) == Approx(-6.25e-05).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(-0.044875).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(-0.0050625).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.0005625).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(-0.0005).epsilon(Tolerance));
    }

    // Coordinates is (0.8,0.8) Size is (0.25,0.25)
    SECTION(
        "16 Node quadrilateral element matrix for coordinate(0.8,0.8), Size "
        "(0.25,0.25)") {
      // Location of point (x,y)
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 0.8, 0.8;
      // Size of particle (x,y)
      Eigen::Matrix<double, Dim, 1> psize;
      psize << 0.5, 0.5;
      // Deformarion gradient
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      auto shapefn = quad->shapefn(coords, psize, defgrad);

      // Check shape function
      REQUIRE(shapefn.size() == nfunctions);
      REQUIRE(quad->nfunctions() == nfunctions);
      REQUIRE(quad->nfunctions_local() == 4);

      REQUIRE(shapefn(0) == Approx(0.0102515625).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.090871875).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.80550625).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.090871875).epsilon(Tolerance));
      REQUIRE(shapefn(4) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(5) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(6) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(7) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(8) == Approx(0.0001265625).epsilon(Tolerance));
      REQUIRE(shapefn(9) == Approx(0.001121875).epsilon(Tolerance));
      REQUIRE(shapefn(10) == Approx(1.5625e-06).epsilon(Tolerance));
      REQUIRE(shapefn(11) == Approx(0.001121875).epsilon(Tolerance));
      REQUIRE(shapefn(12) == Approx(0.0001265625).epsilon(Tolerance));
      REQUIRE(shapefn(13) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(14) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(15) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, psize, defgrad);
      REQUIRE(gradsf.rows() == nfunctions);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-0.0455625).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(0.0405).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.359).epsilon(Tolerance));
      REQUIRE(gradsf(3, 0) == Approx(-0.403875).epsilon(Tolerance));

      REQUIRE(gradsf(4, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 0) == Approx(0.0050625).epsilon(Tolerance));
      REQUIRE(gradsf(9, 0) == Approx(0.044875).epsilon(Tolerance));
      REQUIRE(gradsf(10, 0) == Approx(6.25e-05).epsilon(Tolerance));
      REQUIRE(gradsf(11, 0) == Approx(0.0005).epsilon(Tolerance));
      REQUIRE(gradsf(12, 0) == Approx(-0.0005625).epsilon(Tolerance));
      REQUIRE(gradsf(13, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 0) == Approx(0.0).epsilon(Tolerance));

      REQUIRE(gradsf(0, 1) == Approx(-0.0455625).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(-0.403875).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(0.359).epsilon(Tolerance));
      REQUIRE(gradsf(3, 1) == Approx(0.0405).epsilon(Tolerance));

      REQUIRE(gradsf(4, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(5, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(6, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(7, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(8, 1) == Approx(-0.0005625).epsilon(Tolerance));
      REQUIRE(gradsf(9, 1) == Approx(0.0005).epsilon(Tolerance));
      REQUIRE(gradsf(10, 1) == Approx(6.25e-05).epsilon(Tolerance));
      REQUIRE(gradsf(11, 1) == Approx(0.044875).epsilon(Tolerance));
      REQUIRE(gradsf(12, 1) == Approx(0.0050625).epsilon(Tolerance));
      REQUIRE(gradsf(13, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(14, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(15, 1) == Approx(0.0).epsilon(Tolerance));
    }

    // Shapefn and grad check fail
    SECTION("16 Node quadrilateral element check fail") {
      // Location of point (x,y)
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 20.0, 6500.0;
      // Size of particle (x,y)
      Eigen::Matrix<double, Dim, 1> psize;
      psize << 0.5, 0.5;

      quad->shapefn(coords, psize, defgrad);
      quad->grad_shapefn(coords, psize, defgrad);
    }

    // Coordinates is (0,0)
    SECTION("Four noded local sf quadrilateral element for coordinates(0,0)") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn_local(coords, zero, defgrad);

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
      auto jac = quad->jacobian_local(xi, coords, zero, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check BMatrix with deformation gradient
    SECTION(
        "Four noded quadrilateral B-matrix cell with deformation gradient") {
      // Reference coordinates
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.5, 0.5;

      // Nodal coordinates
      Eigen::Matrix<double, 16, Dim> coords;
      // clang-format off
      coords <<  1.,1.,
                 2.,1.,
                 2.,2.,
                 1.,2.,
                 0,0,
                 1.,0,
                 2.,0,
                 3.,0,
                 3.,1.,
                 3.,2.,
                 3.,3.,
                 2.,3.,
                 1.,3.,
                 0,3.,
                 0,2.,
                 0,1.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      // Get B-Matrix
      auto bmatrix = quad->bmatrix(xi, coords, psize, defgrad);

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(xi, psize, defgrad);
      gradsf *= 2;

      // Check dN/dx
      auto dn_dx = quad->dn_dx(xi, coords, zero, defgrad);
      REQUIRE(dn_dx.rows() == nfunctions);
      REQUIRE(dn_dx.cols() == Dim);
      for (unsigned i = 0; i < nfunctions; ++i) {
        REQUIRE(dn_dx(i, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
        REQUIRE(dn_dx(i, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
      }

      // Check dN/dx local
      Eigen::Matrix<double, nfunctions, Dim> dndx_local;
      dndx_local << -0.25, -0.25, 0.25, -0.75, 0.75, 0.75, -0.75, 0.25, 0, 0,
          -0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, -0;
      auto dn_dx_local = quad->dn_dx_local(xi, coords, zero, defgrad);
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

    SECTION("4 noded GIMP quadrilateral B-matrix and Jacobian failure") {
      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0., 0.;

      Eigen::Matrix<double, 7, Dim> coords;
      // clang-format off
      coords << 0., 0.,
                1., 0.,
                1., 1.,
                0., 1.,
                0., 0.,
                1., 0.,
                1., 1.;
      // clang-format on
      // Get B-Matrix
      quad->bmatrix(xi, coords, zero, defgrad);
      quad->jacobian(xi, coords, zero, defgrad);
    }

    SECTION("Center cell gimp element length") {
      // Check element length
      REQUIRE(quad->unit_element_length() == Approx(2).epsilon(Tolerance));
    }
  }
}
