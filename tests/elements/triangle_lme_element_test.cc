// Triangular element test
#include <memory>

#include "catch.hpp"

#include "triangle_lme_element.h"

//! \brief Check triangle lme element class
TEST_CASE("Triangle lme elements are checked", "[tri][element][2D][lme]") {
  const unsigned Dim = 2;
  const double Tolerance = 1.E-6;

  Eigen::Vector2d zero = Eigen::Vector2d::Zero();
  const Eigen::Matrix2d zero_matrix = Eigen::Matrix2d::Zero();
  Eigen::Matrix2d def_gradient = Eigen::Matrix2d::Identity();

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
      auto shapefn = tri->shapefn(coords, zero, zero_matrix);

      // Check shape function
      REQUIRE(shapefn.size() == 3);
      REQUIRE(tri->nfunctions() == 3);
      REQUIRE(tri->nfunctions_local() == 3);

      REQUIRE(shapefn(0) == Approx(0.66666666666).epsilon(Tolerance));
      REQUIRE(shapefn(1) == Approx(0.16666666666).epsilon(Tolerance));
      REQUIRE(shapefn(2) == Approx(0.16666666666).epsilon(Tolerance));

      // Check gradient of shape functions
      auto gradsf = tri->grad_shapefn(coords, zero, zero_matrix);
      REQUIRE(gradsf.rows() == 3);
      REQUIRE(gradsf.cols() == Dim);

      REQUIRE(gradsf(0, 0) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 0) == Approx(1.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 0) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(0, 1) == Approx(-1.0).epsilon(Tolerance));
      REQUIRE(gradsf(1, 1) == Approx(0.0).epsilon(Tolerance));
      REQUIRE(gradsf(2, 1) == Approx(1.0).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION(
        "Three noded triangle Jacobian for local coordinates(0.333,0.333)") {
      Eigen::Matrix<double, 3, Dim> coords;
      // clang-format off
      coords << 2., 1.,
                4., 2.,
                2., 4.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.333, 0.333;

      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 2.0, 1.0,
                  0.0, 3.0;
      // clang-format on

      // Get Jacobian
      auto jac = tri->jacobian(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check local Jacobian
    SECTION(
        "Three noded Triangle local Jacobian for local "
        "coordinates(0.333,0.333)") {
      Eigen::Matrix<double, 3, Dim> coords;
      // clang-format off
      coords << 2., 1.,
                4., 2.,
                2., 4.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.333, 0.333;

      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 2.0, 1.0,
                  0.0, 3.0;
      // clang-format on

      // Get Jacobian
      auto jac = tri->jacobian_local(xi, coords, zero, zero_matrix);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Check Jacobian
    SECTION("Three noded triangle Jacobian with deformation gradient") {
      Eigen::Matrix<double, 3, Dim> coords;
      // clang-format off
      coords << 2., 1.,
                4., 2.,
                2., 4.;
      // clang-format on

      Eigen::Matrix<double, Dim, 1> psize;
      psize.setZero();
      Eigen::Matrix<double, Dim, Dim> defgrad;
      defgrad.setZero();

      Eigen::Matrix<double, Dim, 1> xi;
      xi << 0.333, 0.333;

      Eigen::Matrix<double, Dim, Dim> jacobian;
      // clang-format off
      jacobian << 2.0, 1.0,
                  0.0, 3.0;
      // clang-format on

      // Get Jacobian
      auto jac = tri->jacobian(xi, coords, psize, defgrad);

      // Check size of jacobian
      REQUIRE(jac.size() == jacobian.size());

      // Check Jacobian
      for (unsigned i = 0; i < Dim; ++i)
        for (unsigned j = 0; j < Dim; ++j)
          REQUIRE(jac(i, j) == Approx(jacobian(i, j)).epsilon(Tolerance));
    }

    // Initialising upgrade properties
    SECTION("2D triangle LME element regular element - nnodes = 14") {
      Eigen::Matrix<double, 14, Dim> nodal_coords;
      nodal_coords << -0.5, 0.0, 0.5, 0.0, 0.0, 0.5 * std::sqrt(3), 0.0,
          -0.5 * std::sqrt(3), 1.0, -0.5 * std::sqrt(3), 1.5, 0.0, 1.0,
          0.5 * std::sqrt(3), 0.5, std::sqrt(3), -0.5, std::sqrt(3), -1.0,
          0.5 * std::sqrt(3), -1.5, 0.0, -1.0, -0.5 * std::sqrt(3), -0.5,
          -std::sqrt(3), 0.5, -std::sqrt(3);

      SECTION("2D triangle LME regular element no support") {
        double gamma = 20;
        double h = 1.0;

        // Calculate beta
        double beta = gamma / (h * h);

        // Calculate support radius automatically
        double tol0 = 1.e-10;
        double r = h * std::sqrt(-std::log(tol0) / gamma);
        unsigned anisotropy = false;

        REQUIRE_NOTHROW(tri->initialise_lme_connectivity_properties(
            beta, r, anisotropy, nodal_coords));

        // Coordinates is (0,0) after upgrade
        SECTION("2D LME element for coordinates(0,0) after upgrade") {
          Eigen::Matrix<double, Dim, 1> coords;
          coords << 1. / 3., 1. / 3.;
          auto shapefn = tri->shapefn(coords, zero, zero_matrix);

          // Check shape function
          REQUIRE(shapefn.size() == 14);
          REQUIRE(tri->nfunctions() == 14);
          REQUIRE(tri->nfunctions_local() == 3);
          REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

          REQUIRE(shapefn(0) == Approx(1. / 3.).epsilon(Tolerance));
          REQUIRE(shapefn(1) == Approx(1. / 3.).epsilon(Tolerance));
          REQUIRE(shapefn(2) == Approx(1. / 3.).epsilon(Tolerance));
          REQUIRE(shapefn(3) == Approx(0.).epsilon(Tolerance));
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

          // Check gradient of shape functions
          zero.setZero();
          auto gradsf = tri->grad_shapefn(coords, zero, zero_matrix);
          REQUIRE(gradsf.rows() == 14);
          REQUIRE(gradsf.cols() == Dim);

          Eigen::Matrix<double, 14, Dim> gradsf_ans;
          gradsf_ans << -1.0, -1.0 / std::sqrt(3), 1.0, -1.0 / std::sqrt(3),
              0.0, 2.0 / std::sqrt(3), 0.0, 0.0, 0, 0, -0, 0, 0, 0, 0, 0, 0, -0,
              0, -0, 0, 0, 0, 0, 0, 0, -0, 0;

          for (unsigned i = 0; i < gradsf.rows(); ++i)
            for (unsigned j = 0; j < gradsf.cols(); ++j)
              REQUIRE(gradsf(i, j) ==
                      Approx(gradsf_ans(i, j)).epsilon(Tolerance));

          // Check the B-matrix assembly
          auto bmatrix = tri->bmatrix(coords, nodal_coords, zero, zero_matrix);

          // Check size of B-matrix
          REQUIRE(bmatrix.size() == 14);

          for (unsigned i = 0; i < 14; ++i) {
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
      }
    }

    // Initialising upgrade properties
    SECTION("2D triangle LME element with anisotropy") {
      Eigen::Matrix<double, 108, Dim> nodal_coords;
      nodal_coords << -0.5, 0., 0.5, 0., -1.5, 0., 0., 0.8660254, 1., 0.8660254,
          -1., 0.8660254, 0., -0.8660254, 1., -0.8660254, -1., -0.8660254, 0.5,
          1.73205081, -0.5, 1.73205081, -1.5, 1.73205081, -3.5, -3.46410162,
          -2.5, -3.46410162, -4.5, -3.46410162, -3., -2.59807621, -2.,
          -2.59807621, -4., -2.59807621, -3., -4.33012702, -2., -4.33012702,
          -4., -4.33012702, -2.5, -1.73205081, -3.5, -1.73205081, -4.5,
          -1.73205081, -3.5, 0., -2.5, 0., -4.5, 0., -3., 0.8660254, -2.,
          0.8660254, -4., 0.8660254, -3., -0.8660254, -2., -0.8660254, -4.,
          -0.8660254, -2.5, 1.73205081, -3.5, 1.73205081, -4.5, 1.73205081,
          -3.5, 3.46410162, -2.5, 3.46410162, -4.5, 3.46410162, -3., 4.33012702,
          -2., 4.33012702, -4., 4.33012702, -3., 2.59807621, -2., 2.59807621,
          -4., 2.59807621, -2.5, 5.19615242, -3.5, 5.19615242, -4.5, 5.19615242,
          -0.5, -3.46410162, 0.5, -3.46410162, -1.5, -3.46410162, 0.,
          -2.59807621, 1., -2.59807621, -1., -2.59807621, 0., -4.33012702, 1.,
          -4.33012702, -1., -4.33012702, 0.5, -1.73205081, -0.5, -1.73205081,
          -1.5, -1.73205081, -0.5, 3.46410162, 0.5, 3.46410162, -1.5,
          3.46410162, 0., 4.33012702, 1., 4.33012702, -1., 4.33012702, 0.,
          2.59807621, 1., 2.59807621, -1., 2.59807621, 0.5, 5.19615242, -0.5,
          5.19615242, -1.5, 5.19615242, 2.5, -3.46410162, 3.5, -3.46410162, 1.5,
          -3.46410162, 3., -2.59807621, 4., -2.59807621, 2., -2.59807621, 3.,
          -4.33012702, 4., -4.33012702, 2., -4.33012702, 3.5, -1.73205081, 2.5,
          -1.73205081, 1.5, -1.73205081, 2.5, 0., 3.5, 0., 1.5, 0., 3.,
          0.8660254, 4., 0.8660254, 2., 0.8660254, 3., -0.8660254, 4.,
          -0.8660254, 2., -0.8660254, 3.5, 1.73205081, 2.5, 1.73205081, 1.5,
          1.73205081, 2.5, 3.46410162, 3.5, 3.46410162, 1.5, 3.46410162, 3.,
          4.33012702, 4., 4.33012702, 2., 4.33012702, 3., 2.59807621, 4.,
          2.59807621, 2., 2.59807621, 3.5, 5.19615242, 2.5, 5.19615242, 1.5,
          5.19615242;

      double gamma = 0.5;
      double h = 1.0;

      // Calculate beta
      double beta = gamma / (h * h);

      // Calculate support radius automatically
      double tol0 = 1.e-10;
      double r = h * std::sqrt(-std::log(tol0) / gamma);
      unsigned anisotropy = true;

      REQUIRE_NOTHROW(tri->initialise_lme_connectivity_properties(
          beta, r, anisotropy, nodal_coords));

      // Coordinates is (0,0) after upgrade
      def_gradient(0, 1) = 1.0;
      Eigen::Matrix<double, Dim, 1> coords;
      coords << 0.5, 0.0;
      auto shapefn = tri->shapefn(coords, zero, def_gradient);

      // Check shape function
      REQUIRE(shapefn.size() == 108);
      REQUIRE(tri->nfunctions() == 108);
      REQUIRE(tri->nfunctions_local() == 3);
      REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

      Eigen::VectorXd shapefn_ans = Eigen::VectorXd::Constant(108, 1.0);
      shapefn_ans << 1.21707502e-01, 1.21965507e-01, 4.46789737e-02,
          6.52011885e-02, 9.42188942e-02, 1.65988706e-02, 6.52275039e-02,
          1.66760482e-02, 9.38585618e-02, 1.44307328e-02, 2.54770184e-03,
          1.65468120e-04, 3.39741729e-04, 2.14048309e-04, 1.98377085e-04,
          4.33256813e-03, 3.93614530e-03, 1.75438618e-03, 4.80891599e-06,
          7.72952493e-07, 1.10064457e-05, 2.28263577e-02, 6.40993468e-03,
          6.62180072e-04, 2.99772247e-04, 6.03385052e-03, 5.47890561e-06,
          5.35600715e-05, 1.55455855e-03, 6.78860797e-07, 9.67557359e-03,
          4.96846859e-02, 6.93164655e-04, 3.95353496e-06, 3.47506035e-08,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 7.34917662e-10,
          1.20565966e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 4.23013443e-06, 1.32688606e-07, 4.96112564e-05,
          1.61748073e-04, 7.31612961e-06, 1.31553486e-03, 9.94215618e-10,
          0.00000000e+00, 4.57050231e-08, 2.55516396e-03, 1.44118324e-02,
          2.99037155e-02, 1.32194371e-07, 4.23226503e-06, 1.51900284e-09,
          9.92211696e-10, 4.58064936e-08, 0.00000000e+00, 1.61552385e-04,
          1.31951998e-03, 7.27639542e-06, 0.00000000e+00, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.53115307e-09,
          7.45216555e-10, 0.00000000e+00, 1.21738847e-07, 0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, 3.52980400e-08, 3.99884405e-06,
          1.66657116e-04, 6.09807745e-03, 3.04249006e-04, 4.49637187e-02,
          9.79534053e-03, 7.04723224e-04, 5.00871144e-02, 5.42668311e-05,
          6.90738080e-07, 1.56841520e-03, 6.50041154e-03, 2.30507209e-02,
          3.00700182e-02, 2.15977842e-04, 3.44259272e-04, 4.98469130e-05,
          4.86059032e-06, 1.11719319e-05, 7.77956429e-07, 4.38265943e-03,
          1.78220180e-03, 3.96482560e-03, 4.51922067e-08, 5.01610931e-09,
          2.04821694e-10;

      for (unsigned i = 0; i < 108; ++i)
        REQUIRE(shapefn(i) == Approx(shapefn_ans(i)).epsilon(Tolerance));

      // Check gradient of shape functions
      zero.setZero();
      auto gradsf = tri->grad_shapefn(coords, zero, def_gradient);
      REQUIRE(gradsf.rows() == 108);
      REQUIRE(gradsf.cols() == Dim);

      Eigen::Matrix<double, 108, Dim> gradsf_ans;
      gradsf_ans << -6.14157421e-02, 6.08725772e-02, 6.15459361e-02,
          -6.10016198e-02, -6.76373835e-02, 6.70391940e-02, -5.64833544e-02,
          1.12925506e-01, 1.34678170e-02, 6.89348153e-02, -3.11316538e-02,
          4.53524980e-02, 5.65061513e-02, -1.12971083e-01, 3.12764027e-02,
          -4.55633676e-02, -1.34163104e-02, -6.86711798e-02, -1.77204939e-02,
          4.27691529e-02, -5.69972927e-03, 1.00992534e-02, -5.37182211e-04,
          8.21445548e-04, -2.28135171e-05, -1.16420524e-03, 2.01651813e-04,
          -9.47601587e-04, -2.13530059e-04, -4.81347454e-04, -1.85791508e-03,
          -9.50970991e-03, 2.28457712e-03, -1.25769502e-02, -2.52291342e-03,
          -2.09583652e-03, 6.26967853e-06, -2.72129124e-05, 1.78783634e-06,
          -5.14721033e-06, 3.24168643e-06, -5.12739246e-05, -1.80442615e-02,
          -2.19848870e-02, -1.15361917e-02, 2.38278837e-04, -1.86004485e-03,
          6.87000392e-04, -1.05889237e-03, 1.04952746e-03, -1.52239345e-02,
          1.50892930e-02, -2.48827676e-05, 2.46627028e-05, -2.08562847e-04,
          2.53493542e-04, -4.48453385e-03, 5.80250399e-03, -3.32861319e-06,
          3.89203942e-06, -2.09129009e-02, 1.22780455e-02, -5.72455472e-02,
          1.33484877e-02, -2.19778052e-03, 1.57298660e-03, -1.68249576e-05,
          2.35815822e-05, -1.82958784e-07, 2.42037683e-07, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -4.13507692e-09, 6.02396739e-09, -5.56695286e-07,
          8.67650908e-07, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, 1.25235488e-05, -2.71898835e-05, 5.26746014e-07,
          -9.85607453e-07, 9.68074683e-05, -2.69257891e-04, 4.20363829e-04,
          -8.40421014e-04, 2.63974376e-05, -4.53320085e-05, 2.09123323e-03,
          -5.51939855e-03, 4.30641177e-09, -8.60968213e-09, 0.00000000e+00,
          -0.00000000e+00, 1.51842668e-07, -3.50075988e-07, 5.71642357e-03,
          -1.01288338e-02, 1.76972849e-02, -4.27131368e-02, 6.54095883e-03,
          -5.87143120e-02, -5.24784004e-07, 9.81936288e-07, -1.25298565e-05,
          2.72035783e-05, -7.56315459e-09, 1.28025857e-08, -4.29773185e-09,
          8.59232862e-09, -1.52179777e-07, 3.50853198e-07, -0.00000000e+00,
          0.00000000e+00, -4.19855257e-04, 8.39404242e-04, -2.09756817e-03,
          5.53611836e-03, -2.62540721e-05, 4.50858086e-05, -0.00000000e+00,
          0.00000000e+00, -0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
          0.00000000e+00, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 7.62365091e-09, -1.29049913e-08, 4.19302452e-09,
          -6.10838528e-09, 0.00000000e+00, -0.00000000e+00, 5.62110889e-07,
          -8.76091527e-07, 0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
          -0.00000000e+00, 0.00000000e+00, -0.00000000e+00, 1.85840988e-07,
          -2.45850574e-07, 1.70177784e-05, -2.38518365e-05, 5.41042213e-04,
          -8.27348167e-04, 1.53859847e-02, -1.52499100e-02, 1.07470573e-03,
          -1.06520096e-03, 6.80684454e-02, -6.74664436e-02, 2.11717667e-02,
          -1.24300265e-02, 2.23442866e-03, -1.59921626e-03, 5.77092160e-02,
          -1.34566057e-02, 2.11314968e-04, -2.56838553e-04, 3.38685028e-06,
          -3.96013417e-06, 4.52450701e-03, -5.85422495e-03, 1.16990262e-02,
          -2.41642166e-04, 1.82216209e-02, 2.22009792e-02, -6.57733487e-03,
          5.90408381e-02, -2.03469598e-04, 9.56143720e-04, 2.31168683e-05,
          1.17968566e-03, -9.72673099e-05, 2.70536883e-04, -6.33704952e-06,
          2.75053295e-05, -3.29042644e-06, 5.20448479e-05, -1.79941042e-06,
          5.18053231e-06, 1.87939550e-03, 9.61965711e-03, 2.56291395e-03,
          2.12906579e-03, -2.30122345e-03, 1.26685908e-02, -7.52647760e-08,
          3.11403475e-07, -1.34164480e-08, 3.95818921e-08, -7.54544130e-10,
          1.82112380e-09;

      for (unsigned i = 0; i < gradsf.rows(); ++i)
        for (unsigned j = 0; j < gradsf.cols(); ++j)
          REQUIRE(gradsf(i, j) == Approx(gradsf_ans(i, j)).epsilon(Tolerance));

      // Check dN/dx
      zero.setZero();
      auto dn_dx = tri->dn_dx(coords, nodal_coords, zero, def_gradient);
      REQUIRE(dn_dx.rows() == 108);
      REQUIRE(dn_dx.cols() == Dim);

      for (unsigned i = 0; i < dn_dx.rows(); ++i)
        for (unsigned j = 0; j < dn_dx.cols(); ++j)
          REQUIRE(dn_dx(i, j) == Approx(gradsf_ans(i, j)).epsilon(Tolerance));

      // Check dN/dx local
      // Reset nodal coordinates as order becomes important in dn_dx_local
      nodal_coords.setZero();
      nodal_coords << -0.5, 0., 0.5, 0., 0., 1.5, 0., 0.8660254, 1., 0.8660254,
          -1., 0.8660254, 0., -0.8660254, 1., -0.8660254, -1., -0.8660254, 0.5,
          1.73205081, -0.5, 1.73205081, -1.5, 1.73205081, -3.5, -3.46410162,
          -2.5, -3.46410162, -4.5, -3.46410162, -3., -2.59807621, -2.,
          -2.59807621, -4., -2.59807621, -3., -4.33012702, -2., -4.33012702,
          -4., -4.33012702, -2.5, -1.73205081, -3.5, -1.73205081, -4.5,
          -1.73205081, -3.5, 0., -2.5, 0., -4.5, 0., -3., 0.8660254, -2.,
          0.8660254, -4., 0.8660254, -3., -0.8660254, -2., -0.8660254, -4.,
          -0.8660254, -2.5, 1.73205081, -3.5, 1.73205081, -4.5, 1.73205081,
          -3.5, 3.46410162, -2.5, 3.46410162, -4.5, 3.46410162, -3., 4.33012702,
          -2., 4.33012702, -4., 4.33012702, -3., 2.59807621, -2., 2.59807621,
          -4., 2.59807621, -2.5, 5.19615242, -3.5, 5.19615242, -4.5, 5.19615242,
          -0.5, -3.46410162, 0.5, -3.46410162, -1.5, -3.46410162, 0.,
          -2.59807621, 1., -2.59807621, -1., -2.59807621, 0., -4.33012702, 1.,
          -4.33012702, -1., -4.33012702, 0.5, -1.73205081, -0.5, -1.73205081,
          -1.5, -1.73205081, -0.5, 3.46410162, 0.5, 3.46410162, -1.5,
          3.46410162, 0., 4.33012702, 1., 4.33012702, -1., 4.33012702, 0.,
          2.59807621, 1., 2.59807621, -1., 2.59807621, 0.5, 5.19615242, -0.5,
          5.19615242, -1.5, 5.19615242, 2.5, -3.46410162, 3.5, -3.46410162, 1.5,
          -3.46410162, 3., -2.59807621, 4., -2.59807621, 2., -2.59807621, 3.,
          -4.33012702, 4., -4.33012702, 2., -4.33012702, 3.5, -1.73205081, 2.5,
          -1.73205081, 1.5, -1.73205081, 2.5, 0., 3.5, 0., 1.5, 0., 3.,
          0.8660254, 4., 0.8660254, 2., 0.8660254, 3., -0.8660254, 4.,
          -0.8660254, 2., -0.8660254, 3.5, 1.73205081, 2.5, 1.73205081, 1.5,
          1.73205081, 2.5, 3.46410162, 3.5, 3.46410162, 1.5, 3.46410162, 3.,
          4.33012702, 4., 4.33012702, 2., 4.33012702, 3., 2.59807621, 4.,
          2.59807621, 2., 2.59807621, 3.5, 5.19615242, 2.5, 5.19615242, 1.5,
          5.19615242;

      Eigen::Matrix<double, 108, Dim> dndx_local;
      dndx_local << -1, -0.333333, 1, -0.333333, 0, 0.666667, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

      auto dn_dx_local =
          tri->dn_dx_local(zero, nodal_coords, zero, def_gradient);

      REQUIRE(dn_dx_local.rows() == 108);
      REQUIRE(dn_dx_local.cols() == Dim);

      for (unsigned i = 0; i < dn_dx_local.rows(); ++i)
        for (unsigned j = 0; j < dn_dx_local.cols(); ++j)
          REQUIRE(dn_dx_local(i, j) ==
                  Approx(dndx_local(i, j)).epsilon(Tolerance));
    }

    SECTION("2D triangular LME element evaluation at the edge of the mesh") {
      Eigen::Matrix<double, 108, Dim> nodal_coords;
      nodal_coords << 3.5, 5.19615242, 2.5, 5.19615242, 1.5, 5.19615242, -0.5,
          0., 0.5, 0., -1.5, 0., 0., 0.8660254, 1., 0.8660254, -1., 0.8660254,
          0., -0.8660254, 1., -0.8660254, -1., -0.8660254, 0.5, 1.73205081,
          -0.5, 1.73205081, -1.5, 1.73205081, -3.5, -3.46410162, -2.5,
          -3.46410162, -4.5, -3.46410162, -3., -2.59807621, -2., -2.59807621,
          -4., -2.59807621, -3., -4.33012702, -2., -4.33012702, -4.,
          -4.33012702, -2.5, -1.73205081, -3.5, -1.73205081, -4.5, -1.73205081,
          -3.5, 0., -2.5, 0., -4.5, 0., -3., 0.8660254, -2., 0.8660254, -4.,
          0.8660254, -3., -0.8660254, -2., -0.8660254, -4., -0.8660254, -2.5,
          1.73205081, -3.5, 1.73205081, -4.5, 1.73205081, -3.5, 3.46410162,
          -2.5, 3.46410162, -4.5, 3.46410162, -3., 4.33012702, -2., 4.33012702,
          -4., 4.33012702, -3., 2.59807621, -2., 2.59807621, -4., 2.59807621,
          -2.5, 5.19615242, -3.5, 5.19615242, -4.5, 5.19615242, -0.5,
          -3.46410162, 0.5, -3.46410162, -1.5, -3.46410162, 0., -2.59807621, 1.,
          -2.59807621, -1., -2.59807621, 0., -4.33012702, 1., -4.33012702, -1.,
          -4.33012702, 0.5, -1.73205081, -0.5, -1.73205081, -1.5, -1.73205081,
          -0.5, 3.46410162, 0.5, 3.46410162, -1.5, 3.46410162, 0., 4.33012702,
          1., 4.33012702, -1., 4.33012702, 0., 2.59807621, 1., 2.59807621, -1.,
          2.59807621, 0.5, 5.19615242, -0.5, 5.19615242, -1.5, 5.19615242, 2.5,
          -3.46410162, 3.5, -3.46410162, 1.5, -3.46410162, 3., -2.59807621, 4.,
          -2.59807621, 2., -2.59807621, 3., -4.33012702, 4., -4.33012702, 2.,
          -4.33012702, 3.5, -1.73205081, 2.5, -1.73205081, 1.5, -1.73205081,
          2.5, 0., 3.5, 0., 1.5, 0., 3., 0.8660254, 4., 0.8660254, 2.,
          0.8660254, 3., -0.8660254, 4., -0.8660254, 2., -0.8660254, 3.5,
          1.73205081, 2.5, 1.73205081, 1.5, 1.73205081, 2.5, 3.46410162, 3.5,
          3.46410162, 1.5, 3.46410162, 3., 4.33012702, 4., 4.33012702, 2.,
          4.33012702, 3., 2.59807621, 4., 2.59807621, 2., 2.59807621;

      double gamma = 0.5;
      double h = 1.0;

      // Calculate beta
      double beta = gamma / (h * h);

      // Calculate support radius automatically
      double tol0 = 1.e-10;
      double r = h * std::sqrt(-std::log(tol0) / gamma);
      unsigned anisotropy = false;

      REQUIRE_NOTHROW(tri->initialise_lme_connectivity_properties(
          beta, r, anisotropy, nodal_coords));

      Eigen::Matrix<double, Dim, 1> coords;
      coords << 1.0, 0.0;

      // Compute shape function
      zero.setZero();
      auto shapefn = tri->shapefn(coords, zero, def_gradient);

      // Check shape function
      REQUIRE(shapefn.size() == 108);
      REQUIRE(tri->nfunctions() == 108);
      REQUIRE(tri->nfunctions_local() == 3);
      REQUIRE(shapefn.sum() == Approx(1.).epsilon(Tolerance));

      // Compute gradient of shape functions
      zero.setZero();
      auto gradsf = tri->grad_shapefn(coords, zero, def_gradient);
    }
  }
}
