// Quadrilateral element test
#include <iostream>
#include <memory>

#include "catch.hpp"

#include "quadrilateral_bspline_element.h"

//! \brief Check quadrilateral bspline element class
TEST_CASE("Quadrilateral bspline elements are checked",
          "[quad][element][2D][bspline]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-6;

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
      auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
                                   Eigen::Vector2d::Zero());

      // Check shape function
      REQUIRE(shapefn.size() == 4);

      REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
                                       Eigen::Vector2d::Zero());
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
      auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
                                   Eigen::Vector2d::Zero());

      // Check shape function
      REQUIRE(shapefn.size() == 4);

      REQUIRE(shapefn(0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(shapefn(3) == Approx(0.0).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
                                       Eigen::Vector2d::Zero());
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
      nodal_coords << -1., -1., 1., -1., -1., 1., 1., 1., -3., -3., -1., -3.,
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
          auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
                                       Eigen::Vector2d::Zero());

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
          auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
                                           Eigen::Vector2d::Zero());
          REQUIRE(gradsf.rows() == 16);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 16, Dim> gradsf_ans;
          gradsf_ans << -0.25, -0.25, 0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0,
              0, -0, 0, 0, 0, 0, 0, 0, -0, 0, -0, 0, 0, 0, 0, 0, 0, -0, 0, 0, 0,
              0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }

        // Coordinates is (1,1) after upgrade
        SECTION("2D BSpline element for coordinates(0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 1., 1.;
          auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
                                       Eigen::Vector2d::Zero());

          // Check shape function
          REQUIRE(shapefn.size() == 16);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(0.015625).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(0.5625).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(4) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(5) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(6) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(7) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(8) == Approx(0.015625).epsilon(Tolerance));
          REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(10) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
          REQUIRE(shapefn(12) == Approx(0.015625).epsilon(Tolerance));
          REQUIRE(shapefn(13) == Approx(0.09375).epsilon(Tolerance));
          REQUIRE(shapefn(14) == Approx(0.015625).epsilon(Tolerance));
          REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));

          // Check gradient of shape functions
          auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
                                           Eigen::Vector2d::Zero());
          REQUIRE(gradsf.rows() == 16);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 16, Dim> gradsf_ans;
          gradsf_ans << 0, -0.1875, 0.03125, -0.03125, 0, 0, 0.1875, 0, -0, 0,
              0, 0, 0, 0, 0, 0, -0.03125, -0.03125, 0, -0, -0.1875, 0, 0, 0,
              -0.03125, 0.03125, 0, 0.1875, 0.03125, 0.03125, 0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        }
      }

      SECTION("2D BSpline element regular element north west support") {
        std::vector<std::vector<unsigned>> nodal_props{
            {2, 0}, {0, 0}, {0, 3}, {2, 3}, {1, 0}, {2, 0}, {0, 0}, {0, 0},
            {1, 0}, {0, 0}, {1, 3}, {0, 3}, {1, 4}, {2, 4}, {0, 4}, {0, 4}};

        REQUIRE_NOTHROW(quad->initialise_bspline_connectivity_properties(
            nodal_coords, nodal_props));

        // // Coordinates is (0,0) after upgrade
        // SECTION("2D BSpline element for coordinates(0,0) after upgrade") {
        //   Eigen::Matrix<double, Dim, 1> coords;
        //   coords.setZero();
        //   auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
        //                                Eigen::Vector2d::Zero());

        //   // Check shape function
        //   REQUIRE(shapefn.size() == 16);
        //   REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

        // REQUIRE(shapefn(0) == Approx(0.25).epsilon(Tolerance));
        // REQUIRE(shapefn(1) == Approx(0.25).epsilon(Tolerance));
        // REQUIRE(shapefn(2) == Approx(0.25).epsilon(Tolerance));
        // REQUIRE(shapefn(3) == Approx(0.25).epsilon(Tolerance));
        // REQUIRE(shapefn(4) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(5) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(6) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(7) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(8) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(9) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(10) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(11) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(12) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(13) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(14) == Approx(0.).epsilon(Tolerance));
        // REQUIRE(shapefn(15) == Approx(0.).epsilon(Tolerance));

        // // Check gradient of shape functions
        // auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
        //                                  Eigen::Vector2d::Zero());
        // REQUIRE(gradsf.rows() == 16);
        // REQUIRE(gradsf.cols() == Dim);

        // Eigen::Matrix<double, 16, Dim> gradsf_ans;
        // gradsf_ans << -0.25, -0.25, 0.25, -0.25, -0.25, 0.25, 0.25, 0.25,
        // 0,
        //     0, -0, 0, 0, 0, 0, 0, 0, -0, 0, -0, 0, 0, 0, 0, 0, 0, -0, 0, 0,
        //     0, 0, 0;

        // for (unsigned i = 0; i < gradsf.rows(); ++i)
        //   for (unsigned j = 0; j < gradsf.cols(); ++j)
        //     REQUIRE(gradsf(i, j) ==
        //             Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        // }

        // // Coordinates is (1,1) after upgrade
        // SECTION("2D BSpline element for coordinates(0,0) after upgrade") {
        //   Eigen::Matrix<double, Dim, 1> coords;
        //   coords << 1., 1.;
        //   auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
        //                                Eigen::Vector2d::Zero());

        //   // Check shape function
        //   REQUIRE(shapefn.size() == 16);
        //   REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

        //   REQUIRE(shapefn(0) == Approx(0.09375).epsilon(Tolerance));
        //   REQUIRE(shapefn(1) == Approx(0.015625).epsilon(Tolerance));
        //   REQUIRE(shapefn(2) == Approx(0.5625).epsilon(Tolerance));
        //   REQUIRE(shapefn(3) == Approx(0.09375).epsilon(Tolerance));
        //   REQUIRE(shapefn(4) == Approx(0).epsilon(Tolerance));
        //   REQUIRE(shapefn(5) == Approx(0).epsilon(Tolerance));
        //   REQUIRE(shapefn(6) == Approx(0).epsilon(Tolerance));
        //   REQUIRE(shapefn(7) == Approx(0).epsilon(Tolerance));
        //   REQUIRE(shapefn(8) == Approx(0.015625).epsilon(Tolerance));
        //   REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
        //   REQUIRE(shapefn(10) == Approx(0.09375).epsilon(Tolerance));
        //   REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));
        //   REQUIRE(shapefn(12) == Approx(0.015625).epsilon(Tolerance));
        //   REQUIRE(shapefn(13) == Approx(0.09375).epsilon(Tolerance));
        //   REQUIRE(shapefn(14) == Approx(0.015625).epsilon(Tolerance));
        //   REQUIRE(shapefn(15) == Approx(0).epsilon(Tolerance));

        //   // Check gradient of shape functions
        //   auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
        //                                    Eigen::Vector2d::Zero());
        //   REQUIRE(gradsf.rows() == 16);
        //   REQUIRE(gradsf.cols() == Dim);

        //   Eigen::Matrix<double, 16, Dim> gradsf_ans;
        //   gradsf_ans << 0, -0.1875, 0.03125, -0.03125, 0, 0, 0.1875, 0, -0,
        //   0,
        //       0, 0, 0, 0, 0, 0, -0.03125, -0.03125, 0, -0, -0.1875, 0, 0, 0,
        //       -0.03125, 0.03125, 0, 0.1875, 0.03125, 0.03125, 0, 0;

        //   for (unsigned i = 0; i < gradsf.rows(); ++i)
        //     for (unsigned j = 0; j < gradsf.cols(); ++j)
        //       REQUIRE(gradsf(i, j) ==
        //               Approx(gradsf_ans(i, j)).epsilon(Tolerance));
        // }
      }
    }

    //! nnodes = 12 arrangement
    //!   9-----------8-----------7-----------6
    //!   |           |           |           |
    //!   |           |           |           |
    //!   |           |           |           |
    //!   |           |           |           |
    //!   10----------3-----------2-----------5
    //!   |           |           |           |
    //!   |           | particle  |           |
    //!   |           | location  |           |
    //!   |           |           |           |
    //!   11----------0-----------1-----------4

    SECTION("2D BSpline element mesh edge - nnodes = 12") {
      Eigen::Matrix<double, 12, Dim> nodal_coords;
      nodal_coords << -1., -1., 1., -1., -1., 1., 1., 1., 3., -1, 3., 1, 3., 3.,
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
        auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
                                     Eigen::Vector2d::Zero());

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
        auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
                                         Eigen::Vector2d::Zero());
        REQUIRE(gradsf.rows() == 12);
        REQUIRE(gradsf.cols() == Dim);

        Eigen::Matrix<double, 12, Dim> gradsf_ans;
        gradsf_ans << -0.3333333333, -0.3333333333, 0.3333333333, -0.3333333333,
            -0.1666666667, 0.3333333333, 0.1666666667, 0.3333333333, 0, -0, 0,
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
        auto shapefn = quad->shapefn(coords, Eigen::Vector2d::Zero(),
                                     Eigen::Vector2d::Zero());

        // Check shape function
        REQUIRE(shapefn.size() == 12);
        REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

        REQUIRE(shapefn(0) == Approx(0.350911).epsilon(Tolerance));
        REQUIRE(shapefn(1) == Approx(0.558594).epsilon(Tolerance));
        REQUIRE(shapefn(2) == Approx(0.031901).epsilon(Tolerance));
        REQUIRE(shapefn(3) == Approx(0.0507812).epsilon(Tolerance));
        REQUIRE(shapefn(4) == Approx(0.00716146).epsilon(Tolerance));
        REQUIRE(shapefn(5) == Approx(0.000651042).epsilon(Tolerance));
        REQUIRE(shapefn(6) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(7) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(8) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(9) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(10) == Approx(0).epsilon(Tolerance));
        REQUIRE(shapefn(11) == Approx(0).epsilon(Tolerance));

        // Check gradient of shape functions
        auto gradsf = quad->grad_shapefn(coords, Eigen::Vector2d::Zero(),
                                         Eigen::Vector2d::Zero());
        REQUIRE(gradsf.rows() == 12);
        REQUIRE(gradsf.cols() == Dim);

        Eigen::Matrix<double, 12, Dim> gradsf_ans;
        gradsf_ans << -0.401042, -0.127604, 0.34375, -0.203125, -0.0364583,
            0.127604, 0.03125, 0.203125, 0.0572917, -0.00260417, 0.00520833,
            0.00260417, 0, 0, 0, 0, -0, 0, 0, 0, 0, 0, 0, -0;

        for (unsigned i = 0; i < gradsf.rows(); ++i)
          for (unsigned j = 0; j < gradsf.cols(); ++j)
            REQUIRE(gradsf(i, j) ==
                    Approx(gradsf_ans(i, j)).epsilon(Tolerance));
      }
    }
  }
}