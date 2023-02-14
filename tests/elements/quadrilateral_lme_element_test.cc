// Quadrilateral element test
#include <memory>

#include "catch.hpp"

#include "quadrilateral_lme_element.h"

//! \brief Check quadrilateral lme element class
TEST_CASE("Quadrilateral lme elements are checked",
          "[quad][element][2D][lme]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-8;

  Eigen::Vector2d zero = Eigen::Vector2d::Zero();
  const Eigen::Matrix2d zero_matrix = Eigen::Matrix2d::Zero();
  Eigen::Matrix2d def_gradient = Eigen::Matrix2d::Identity();

  //! Check for center element nodes
  SECTION("Quadratic Quadrilateral LME Element") {
    std::shared_ptr<mpm::Element<Dim>> quad =
        std::make_shared<mpm::QuadrilateralLMEElement<Dim>>();

    // Check degree and shapefn type
    REQUIRE(quad->degree() == mpm::ElementDegree::Infinity);
    REQUIRE(quad->shapefn_type() == mpm::ShapefnType::LME);

    // Coordinates is (0,0) before upgraded
    SECTION("2D LME element for coordinates(0,0) before upgrade") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords.setZero();
      auto shapefn = quad->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == 4);
      REQUIRE(quad->nfunctions() == 4);
      REQUIRE(quad->nfunctions_local() == 4);

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
    SECTION("2D LME element for coordinates(1,1) before upgrade") {
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1., 1.;
      zero.setZero();
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
    SECTION("2D LME element regular element - nnodes = 16") {
      Eigen::Matrix<double, 16, Dim> nodal_coords;
      nodal_coords << -1., -1., 1., -1., 1., 1., -1., 1., -3., -3., -1., -3.,
          1., -3., 3., -3., -3., -1., 3., -1., -3., 1., 3., 1., -3., 3., -1.,
          3., 1., 3., 3., 3.;

      SECTION("2D LME element regular element no support") {
        double gamma = 20;
        double h = 2.0;

        // Calculate beta
        double beta = gamma / (h * h);

        // Calculate support radius automatically
        double tol0 = 1.e-10;
        double r = h * std::sqrt(-std::log(tol0) / gamma);
        unsigned anisotropy = false;

        REQUIRE_NOTHROW(quad->initialise_lme_connectivity_properties(
            beta, r, anisotropy, nodal_coords));

        // Coordinates is (0,0) after upgrade
        SECTION("2D LME element for coordinates(0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords.setZero();
          auto shapefn = quad->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 16);
          REQUIRE(quad->nfunctions() == 16);
          REQUIRE(quad->nfunctions_local() == 4);
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
          zero.setZero();
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

          // Check the B-matrix assembly
          auto bmatrix = quad->bmatrix(coords, nodal_coords, zero, zero_matrix);

          // Check size of B-matrix
          REQUIRE(bmatrix.size() == 16);

          for (unsigned i = 0; i < 16; ++i) {
            // clang-format off
            REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
            // clang-format on
          }
        }

        SECTION("2D LME element for coordinates(0.2, 0.2)") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 0.2, 0.2;
          zero.setZero();
          auto shapefn = quad->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 16);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          Eigen::Matrix<double, 16, 1> shapefn_ans;
          shapefn_ans << 1.599999999995444788e-01, 2.399999999998857214e-01,
              3.600000000006839951e-01, 2.399999999998857214e-01,
              0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00;

          for (unsigned i = 0; i < 16; ++i)
            REQUIRE(shapefn(i) == Approx(shapefn_ans(i)).epsilon(Tolerance));

          // Check gradient of shape functions
          zero.setZero();
          auto gradsf = quad->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 16);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 16, Dim> gradsf_ans;
          gradsf_ans << -1.999999999995252797e-01, -1.999999999995252797e-01,
              2.000000000000001499e-01, -2.999999999999998779e-01,
              3.000000000007120304e-01, 3.000000000007120304e-01,
              -2.999999999999998779e-01, 2.000000000000001499e-01,
              -0.000000000000000000e+00, -0.000000000000000000e+00,
              -0.000000000000000000e+00, -0.000000000000000000e+00,
              0.000000000000000000e+00, -0.000000000000000000e+00,
              0.000000000000000000e+00, -0.000000000000000000e+00,
              -0.000000000000000000e+00, -0.000000000000000000e+00,
              0.000000000000000000e+00, -0.000000000000000000e+00,
              -0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00,
              -0.000000000000000000e+00, 0.000000000000000000e+00,
              -0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00,
              0.000000000000000000e+00, 0.000000000000000000e+00;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check the B-matrix assembly
          auto bmatrix = quad->bmatrix(coords, nodal_coords, zero, zero_matrix);

          // Check size of B-matrix
          REQUIRE(bmatrix.size() == 16);

          for (unsigned i = 0; i < 16; ++i) {
            // clang-format off
            REQUIRE(bmatrix.at(i)(0, 0) == Approx(gradsf(i, 0)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(0, 1) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 0) == Approx(0.).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(1, 1) == Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 0) == Approx(gradsf(i, 1)).epsilon(Tolerance));
            REQUIRE(bmatrix.at(i)(2, 1) == Approx(gradsf(i, 0)).epsilon(Tolerance));
            // clang-format on
          }
        }

        // Check Jacobian
        SECTION("16-noded quadrilateral Jacobian with deformation gradient") {
          Eigen::Matrix<double, 16, Dim> coords;

          coords << -1., -1., 1., -1., 1., 1., -1., 1., -3., -3., -1., -3., 1.,
              -3., 3., -3., 3., -1., 3., 1., 3., 3., 1., 3., -1., 3., -3., 3.,
              -3., 1., -3., -1.;

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

    // Initialising upgrade properties
    SECTION("2D quadrilateral LME element element with anisotropy") {
      Eigen::Matrix<double, 100, Dim> nodal_coords;
      nodal_coords << 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -3.5, -3.5,
          -4.5, -3.5, -4.5, -4.5, -3.5, -4.5, -3.5, -1.5, -4.5, -1.5, -4.5,
          -2.5, -3.5, -2.5, -3.5, 0.5, -4.5, 0.5, -4.5, -0.5, -3.5, -0.5, -3.5,
          2.5, -4.5, 2.5, -4.5, 1.5, -3.5, 1.5, -3.5, 4.5, -4.5, 4.5, -4.5, 3.5,
          -3.5, 3.5, -1.5, -3.5, -2.5, -3.5, -2.5, -4.5, -1.5, -4.5, -1.5, -1.5,
          -2.5, -1.5, -2.5, -2.5, -1.5, -2.5, -1.5, 0.5, -2.5, 0.5, -2.5, -0.5,
          -1.5, -0.5, -1.5, 2.5, -2.5, 2.5, -2.5, 1.5, -1.5, 1.5, -1.5, 4.5,
          -2.5, 4.5, -2.5, 3.5, -1.5, 3.5, 0.5, -3.5, -0.5, -3.5, -0.5, -4.5,
          0.5, -4.5, 0.5, -1.5, -0.5, -1.5, -0.5, -2.5, 0.5, -2.5, 0.5, 2.5,
          -0.5, 2.5, -0.5, 1.5, 0.5, 1.5, 0.5, 4.5, -0.5, 4.5, -0.5, 3.5, 0.5,
          3.5, 2.5, -3.5, 1.5, -3.5, 1.5, -4.5, 2.5, -4.5, 2.5, -1.5, 1.5, -1.5,
          1.5, -2.5, 2.5, -2.5, 2.5, 0.5, 1.5, 0.5, 1.5, -0.5, 2.5, -0.5, 2.5,
          2.5, 1.5, 2.5, 1.5, 1.5, 2.5, 1.5, 2.5, 4.5, 1.5, 4.5, 1.5, 3.5, 2.5,
          3.5, 4.5, -3.5, 3.5, -3.5, 3.5, -4.5, 4.5, -4.5, 4.5, -1.5, 3.5, -1.5,
          3.5, -2.5, 4.5, -2.5, 4.5, 0.5, 3.5, 0.5, 3.5, -0.5, 4.5, -0.5, 4.5,
          2.5, 3.5, 2.5, 3.5, 1.5, 4.5, 1.5, 4.5, 4.5, 3.5, 4.5, 3.5, 3.5, 4.5,
          3.5;

      double gamma = 1.0;
      double h = 1.0;

      // Calculate beta
      double beta = gamma / (h * h);

      // Calculate support radius automatically
      double tol0 = 1.e-10;
      double r = h * std::sqrt(-std::log(tol0) / gamma);
      unsigned anisotropy = true;

      REQUIRE_NOTHROW(quad->initialise_lme_connectivity_properties(
          beta, r, anisotropy, nodal_coords));

      // Coordinates is (0,0) after upgrade
      def_gradient(0, 1) = 0.5;
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 0.0, 0.0;
      zero.setZero();
      auto shapefn = quad->shapefn(coords, zero, def_gradient);

      // Check shape function
      REQUIRE(shapefn.size() == 100);
      REQUIRE(quad->nfunctions() == 100);
      REQUIRE(quad->nfunctions_local() == 4);
      REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

      Eigen::VectorXd shapefn_ans = Eigen::VectorXd::Constant(100, 1.0);
      shapefn_ans << 2.32904581e-01, 1.41263769e-01, 2.32904581e-01,
          1.41263769e-01, 7.12460518e-08, 7.91472142e-10, 0.00000000e+00,
          0.00000000e+00, 1.74333341e-05, 2.62099577e-08, 1.58971429e-08,
          3.88990262e-06, 1.93666848e-07, 0.00000000e+00, 3.54713205e-09,
          6.41336520e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          4.80052121e-10, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 1.43101520e-06, 8.67954595e-07, 4.80052121e-10,
          0.00000000e+00, 1.91179722e-02, 1.56929872e-03, 1.28815883e-04,
          5.77312737e-04, 1.15956363e-02, 1.28815883e-04, 1.56929872e-03,
          5.19680364e-02, 3.19302652e-07, 4.80052121e-10, 8.67954595e-07,
          2.12381487e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 9.64210460e-09, 3.19302652e-07, 0.00000000e+00,
          0.00000000e+00, 7.03310893e-03, 3.15202074e-02, 3.50157875e-04,
          2.87427087e-05, 3.50157875e-04, 2.87427087e-05, 7.03310893e-03,
          3.15202074e-02, 0.00000000e+00, 0.00000000e+00, 9.64210460e-09,
          3.19302652e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 8.67954595e-07, 2.12381487e-04, 3.19302652e-07,
          4.80052121e-10, 1.56929872e-03, 5.19680364e-02, 1.15956363e-02,
          1.28815883e-04, 1.28815883e-04, 5.77312737e-04, 1.91179722e-02,
          1.56929872e-03, 4.80052121e-10, 0.00000000e+00, 1.43101520e-06,
          8.67954595e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 4.80052121e-10, 0.00000000e+00,
          0.00000000e+00, 3.54713205e-09, 6.41336520e-06, 1.93666848e-07,
          0.00000000e+00, 1.58971429e-08, 3.88990262e-06, 1.74333341e-05,
          2.62099577e-08, 0.00000000e+00, 0.00000000e+00, 7.12460518e-08,
          7.91472142e-10;

      for (unsigned i = 0; i < 100; ++i)
        REQUIRE(shapefn(i) == Approx(shapefn_ans(i)).epsilon(Tolerance));

      // Check gradient of shape functions
      zero.setZero();
      auto gradsf = quad->grad_shapefn(coords, zero, def_gradient);
      REQUIRE(gradsf.rows() == 100);
      REQUIRE(gradsf.cols() == Dim);

      Eigen::Matrix<double, 100, Dim> gradsf_ans;
      gradsf_ans << 1.16412040e-01, 1.74203790e-01, -2.11920077e-01,
          2.46972545e-01, -1.16412040e-01, -1.74203790e-01, 2.11920077e-01,
          -2.46972545e-01, -2.49274993e-07, -3.73025404e-07, -4.35213938e-09,
          -3.35219216e-09, -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
          -0.00000000e+00, -9.58743016e-05, -4.23969538e-06, -1.96561100e-07,
          1.98448945e-08, -1.03317695e-07, -2.76470212e-08, -1.75012009e-05,
          -1.06562504e-05, -1.45253489e-06, 9.19790655e-07, -0.00000000e+00,
          0.00000000e+00, -3.01500105e-08, 1.15403131e-08, -4.16857669e-05,
          1.44497932e-05, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -4.08069209e-09,
          3.47827271e-09, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, 7.17238947e-07, -1.03554354e-05, -1.30088184e-06,
          -5.41263439e-06, -2.39279336e-10, -4.19198311e-09, 0.00000000e+00,
          -0.00000000e+00, -2.86670465e-02, -4.28985535e-02, -5.49173204e-03,
          -1.95148646e-03, -3.21928400e-04, -4.81746966e-04, -2.88157477e-04,
          -2.73655243e-03, -4.05867327e-02, 3.18723860e-02, -7.08509649e-04,
          4.82930642e-04, -7.06157328e-03, 1.96590659e-03, -1.29911114e-01,
          1.31158828e-02, -1.75644058e-06, 2.47178349e-06, -3.60080589e-09,
          4.19639426e-09, -5.64215512e-06, 5.42060994e-06, -9.55826849e-04,
          1.11392461e-03, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, 4.34011378e-08, -8.90652506e-08, 7.98642968e-07,
          -2.63002299e-06, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 1.75864197e-02, -2.98525766e-02, 1.57764478e-02,
          -1.02258862e-01, 5.25539432e-04, -2.01008188e-03, 1.00624323e-04,
          -1.93750214e-04, -5.25539432e-04, 2.01008188e-03, -1.00624323e-04,
          1.93750214e-04, -1.75864197e-02, 2.98525766e-02, -1.57764478e-02,
          1.02258862e-01, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -4.34011378e-08, 8.90652506e-08, -7.98642968e-07,
          2.63002299e-06, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 5.64215512e-06, -5.42060994e-06, 9.55826849e-04,
          -1.11392461e-03, 1.75644058e-06, -2.47178349e-06, 3.60080589e-09,
          -4.19639426e-09, 7.06157328e-03, -1.96590659e-03, 1.29911114e-01,
          -1.31158828e-02, 4.05867327e-02, -3.18723860e-02, 7.08509649e-04,
          -4.82930642e-04, 3.21928400e-04, 4.81746966e-04, 2.88157477e-04,
          2.73655243e-03, 2.86670465e-02, 4.28985535e-02, 5.49173204e-03,
          1.95148646e-03, 2.39279336e-10, 4.19198311e-09, -0.00000000e+00,
          0.00000000e+00, -7.17238947e-07, 1.03554354e-05, 1.30088184e-06,
          5.41263439e-06, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 0.00000000e+00, -0.00000000e+00, 4.08069209e-09,
          -3.47827271e-09, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 3.01500105e-08, -1.15403131e-08, 4.16857669e-05,
          -1.44497932e-05, 1.45253489e-06, -9.19790655e-07, 0.00000000e+00,
          -0.00000000e+00, 1.03317695e-07, 2.76470212e-08, 1.75012009e-05,
          1.06562504e-05, 9.58743016e-05, 4.23969538e-06, 1.96561100e-07,
          -1.98448945e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 2.49274993e-07, 3.73025404e-07, 4.35213938e-09,
          3.35219216e-09;

      for (unsigned i = 0; i < gradsf.rows(); ++i)
        for (unsigned j = 0; j < gradsf.cols(); ++j)
          REQUIRE(gradsf(i, j) == Approx(gradsf_ans(i, j)).epsilon(Tolerance));

      // Check dN/dx
      auto dndx = quad->dn_dx(coords, nodal_coords, zero, def_gradient);
      REQUIRE(dndx.rows() == 100);
      REQUIRE(dndx.cols() == Dim);

      for (unsigned i = 0; i < dndx.rows(); ++i)
        for (unsigned j = 0; j < dndx.cols(); ++j)
          REQUIRE(dndx(i, j) == Approx(gradsf_ans(i, j)).epsilon(Tolerance));

      // Check dN/dx local
      Eigen::Matrix<double, 100, Dim> dndx_local;
      dndx_local << 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0;

      auto dn_dx_local =
          quad->dn_dx_local(coords, nodal_coords, zero, def_gradient);
      REQUIRE(dn_dx_local.rows() == 100);
      REQUIRE(dn_dx_local.cols() == Dim);

      for (unsigned i = 0; i < dn_dx_local.rows(); ++i)
        for (unsigned j = 0; j < dn_dx_local.cols(); ++j)
          REQUIRE(dn_dx_local(i, j) ==
                  Approx(dndx_local(i, j)).epsilon(Tolerance));
    }

    SECTION("2D quadrilateral LME element evaluation at the edge of the mesh") {
      Eigen::Matrix<double, 100, Dim> nodal_coords;
      nodal_coords << 4.5, 4.5, 3.5, 4.5, 3.5, 3.5, 4.5, 3.5, 0.5, 0.5, -0.5,
          0.5, -0.5, -0.5, 0.5, -0.5, -3.5, -3.5, -4.5, -3.5, -4.5, -4.5, -3.5,
          -4.5, -3.5, -1.5, -4.5, -1.5, -4.5, -2.5, -3.5, -2.5, -3.5, 0.5, -4.5,
          0.5, -4.5, -0.5, -3.5, -0.5, -3.5, 2.5, -4.5, 2.5, -4.5, 1.5, -3.5,
          1.5, -3.5, 4.5, -4.5, 4.5, -4.5, 3.5, -3.5, 3.5, -1.5, -3.5, -2.5,
          -3.5, -2.5, -4.5, -1.5, -4.5, -1.5, -1.5, -2.5, -1.5, -2.5, -2.5,
          -1.5, -2.5, -1.5, 0.5, -2.5, 0.5, -2.5, -0.5, -1.5, -0.5, -1.5, 2.5,
          -2.5, 2.5, -2.5, 1.5, -1.5, 1.5, -1.5, 4.5, -2.5, 4.5, -2.5, 3.5,
          -1.5, 3.5, 0.5, -3.5, -0.5, -3.5, -0.5, -4.5, 0.5, -4.5, 0.5, -1.5,
          -0.5, -1.5, -0.5, -2.5, 0.5, -2.5, 0.5, 2.5, -0.5, 2.5, -0.5, 1.5,
          0.5, 1.5, 0.5, 4.5, -0.5, 4.5, -0.5, 3.5, 0.5, 3.5, 2.5, -3.5, 1.5,
          -3.5, 1.5, -4.5, 2.5, -4.5, 2.5, -1.5, 1.5, -1.5, 1.5, -2.5, 2.5,
          -2.5, 2.5, 0.5, 1.5, 0.5, 1.5, -0.5, 2.5, -0.5, 2.5, 2.5, 1.5, 2.5,
          1.5, 1.5, 2.5, 1.5, 2.5, 4.5, 1.5, 4.5, 1.5, 3.5, 2.5, 3.5, 4.5, -3.5,
          3.5, -3.5, 3.5, -4.5, 4.5, -4.5, 4.5, -1.5, 3.5, -1.5, 3.5, -2.5, 4.5,
          -2.5, 4.5, 0.5, 3.5, 0.5, 3.5, -0.5, 4.5, -0.5, 4.5, 2.5, 3.5, 2.5,
          3.5, 1.5, 4.5, 1.5;

      double gamma = 1.0;
      double h = 1.0;

      // Calculate beta
      double beta = gamma / (h * h);

      // Calculate support radius automatically
      double tol0 = 1.e-10;
      double r = h * std::sqrt(-std::log(tol0) / gamma);
      unsigned anisotropy = false;

      REQUIRE_NOTHROW(quad->initialise_lme_connectivity_properties(
          beta, r, anisotropy, nodal_coords));

      Eigen::Matrix<double, Dim, 1> coords;
      coords << -1.0, -1.0;
      zero.setZero();
      auto shapefn = quad->shapefn(coords, zero, def_gradient);

      // Check shape function
      REQUIRE(shapefn.size() == 100);
      REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

      // Check gradient of shape functions
      zero.setZero();
      auto gradsf = quad->grad_shapefn(coords, zero, def_gradient);
      REQUIRE(gradsf.rows() == 100);
      REQUIRE(gradsf.cols() == Dim);
    }
  }
}