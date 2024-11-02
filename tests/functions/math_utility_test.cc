#include <limits>
#include <memory>

#include "Eigen/Dense"
#include "catch.hpp"

#include "math_utility.h"

//! \brief Check math utility namespace functions
TEST_CASE("math utility is checked", "[math]") {

  // Tolerance
  const double Tolerance = 1.E-6;

  SECTION("Check for zero stresses") {

    // Initialise stress
    Eigen::Matrix<double, 6, 1> stress;
    stress.setZero();

    // Vector to matrix conversion
    double shear_multiplier = 1.0;
    auto stress_1d = mpm::math::matrix_form<1>(stress, shear_multiplier);
    REQUIRE(stress_1d.cols() == 1);
    REQUIRE(stress_1d.rows() == 1);
    for (unsigned i = 0; i < 1; i++)
      for (unsigned j = 0; j < 1; j++)
        REQUIRE(stress_1d(i, j) == Approx(0.).epsilon(Tolerance));

    auto stress_2d = mpm::math::matrix_form<2>(stress, shear_multiplier);
    REQUIRE(stress_2d.cols() == 2);
    REQUIRE(stress_2d.rows() == 2);
    for (unsigned i = 0; i < 2; i++)
      for (unsigned j = 0; j < 2; j++)
        REQUIRE(stress_2d(i, j) == Approx(0.).epsilon(Tolerance));

    auto stress_3d = mpm::math::matrix_form<3>(stress, shear_multiplier);
    REQUIRE(stress_3d.cols() == 3);
    REQUIRE(stress_3d.rows() == 3);
    for (unsigned i = 0; i < 3; i++)
      for (unsigned j = 0; j < 3; j++)
        REQUIRE(stress_3d(i, j) == Approx(0.).epsilon(Tolerance));

    stress_3d = mpm::math::matrix_form(stress, shear_multiplier);
    REQUIRE(stress_3d.cols() == 3);
    REQUIRE(stress_3d.rows() == 3);
    for (unsigned i = 0; i < 3; i++)
      for (unsigned j = 0; j < 3; j++)
        REQUIRE(stress_3d(i, j) == Approx(0.).epsilon(Tolerance));

    auto voigt_stress_1d =
        mpm::math::voigt_form<1>(stress_1d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_1d(i) == Approx(stress(i)).epsilon(Tolerance));

    auto voigt_stress_2d =
        mpm::math::voigt_form<2>(stress_2d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_2d(i) == Approx(stress(i)).epsilon(Tolerance));

    auto voigt_stress_3d =
        mpm::math::voigt_form<3>(stress_3d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_3d(i) == Approx(stress(i)).epsilon(Tolerance));

    voigt_stress_3d = mpm::math::voigt_form(stress_3d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_3d(i) == Approx(stress(i)).epsilon(Tolerance));

    // Compute principal stresses and directions from matrix form
    auto principal_stresses = mpm::math::principal_tensor(stress_3d);
    REQUIRE(principal_stresses(0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(principal_stresses(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(principal_stresses(2) == Approx(0.).epsilon(Tolerance));

    Eigen::Matrix<double, 3, 3> directors;
    directors.setZero();
    principal_stresses = mpm::math::principal_tensor(stress_3d, directors);
    REQUIRE(principal_stresses(0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(principal_stresses(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(principal_stresses(2) == Approx(0.).epsilon(Tolerance));
    REQUIRE(directors(0, 0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(directors(0, 1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(directors(0, 2) == Approx(1.).epsilon(Tolerance));
    REQUIRE(directors(1, 0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(directors(1, 1) == Approx(1.).epsilon(Tolerance));
    REQUIRE(directors(1, 2) == Approx(0.).epsilon(Tolerance));
    REQUIRE(directors(2, 0) == Approx(1.).epsilon(Tolerance));
    REQUIRE(directors(2, 1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(directors(2, 2) == Approx(0.).epsilon(Tolerance));
  }

  SECTION("Check for non-zero stresses") {

    // Initialise stress
    Eigen::Matrix<double, 6, 1> stress;
    stress(0) = -200.;
    stress(1) = -150.2;
    stress(2) = -150.2;
    stress(3) = 52.;
    stress(4) = -14.5;
    stress(5) = -33.;

    Eigen::Matrix<double, 6, 1> stress_1dim;
    stress_1dim(0) = -200.;
    stress_1dim(1) = 0.0;
    stress_1dim(2) = 0.0;
    stress_1dim(3) = 0.0;
    stress_1dim(4) = 0.0;
    stress_1dim(5) = 0.0;

    Eigen::Matrix<double, 6, 1> stress_2dim;
    stress_2dim(0) = -200.;
    stress_2dim(1) = -150.2;
    stress_2dim(2) = 0.0;
    stress_2dim(3) = 52.;
    stress_2dim(4) = 0.0;
    stress_2dim(5) = 0.0;

    Eigen::Matrix<double, 3, 3> stress_matrix;
    stress_matrix(0, 0) = -200.;
    stress_matrix(1, 1) = -150.2;
    stress_matrix(2, 2) = -150.2;
    stress_matrix(0, 1) = 52.;
    stress_matrix(1, 0) = 52.;
    stress_matrix(1, 2) = -14.5;
    stress_matrix(2, 1) = -14.5;
    stress_matrix(0, 2) = -33.;
    stress_matrix(2, 0) = -33.;

    // Vector to matrix conversion
    double shear_multiplier = 1.0;
    auto stress_1d = mpm::math::matrix_form<1>(stress, shear_multiplier);
    REQUIRE(stress_1d.cols() == 1);
    REQUIRE(stress_1d.rows() == 1);
    for (unsigned i = 0; i < 1; i++)
      for (unsigned j = 0; j < 1; j++)
        REQUIRE(stress_1d(i, j) ==
                Approx(stress_matrix(i, j)).epsilon(Tolerance));

    auto stress_2d = mpm::math::matrix_form<2>(stress, shear_multiplier);
    REQUIRE(stress_2d.cols() == 2);
    REQUIRE(stress_2d.rows() == 2);
    for (unsigned i = 0; i < 2; i++)
      for (unsigned j = 0; j < 2; j++)
        REQUIRE(stress_2d(i, j) ==
                Approx(stress_matrix(i, j)).epsilon(Tolerance));

    auto stress_3d = mpm::math::matrix_form<3>(stress, shear_multiplier);
    REQUIRE(stress_3d.cols() == 3);
    REQUIRE(stress_3d.rows() == 3);
    for (unsigned i = 0; i < 3; i++)
      for (unsigned j = 0; j < 3; j++)
        REQUIRE(stress_3d(i, j) ==
                Approx(stress_matrix(i, j)).epsilon(Tolerance));

    stress_3d = mpm::math::matrix_form(stress, shear_multiplier);
    REQUIRE(stress_3d.cols() == 3);
    REQUIRE(stress_3d.rows() == 3);
    for (unsigned i = 0; i < 3; i++)
      for (unsigned j = 0; j < 3; j++)
        REQUIRE(stress_3d(i, j) ==
                Approx(stress_matrix(i, j)).epsilon(Tolerance));

    auto voigt_stress_1d =
        mpm::math::voigt_form<1>(stress_1d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_1d(i) == Approx(stress_1dim(i)).epsilon(Tolerance));

    auto voigt_stress_2d =
        mpm::math::voigt_form<2>(stress_2d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_2d(i) == Approx(stress_2dim(i)).epsilon(Tolerance));

    auto voigt_stress_3d =
        mpm::math::voigt_form<3>(stress_3d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_3d(i) == Approx(stress(i)).epsilon(Tolerance));

    voigt_stress_3d = mpm::math::voigt_form(stress_3d, shear_multiplier);
    for (unsigned i = 0; i < 6; i++)
      REQUIRE(voigt_stress_3d(i) == Approx(stress(i)).epsilon(Tolerance));

    // Compute principal stresses and directions from matrix form
    auto principal_stresses = mpm::math::principal_tensor(stress_matrix);

    REQUIRE(principal_stresses(0) == Approx(-98.9515).epsilon(Tolerance));
    REQUIRE(principal_stresses(1) == Approx(-163.611442455).epsilon(Tolerance));
    REQUIRE(principal_stresses(2) == Approx(-237.837).epsilon(Tolerance));

    Eigen::Matrix<double, 3, 3> directors;
    directors.setZero();
    principal_stresses = mpm::math::principal_tensor(stress_matrix, directors);

    REQUIRE(principal_stresses(0) == Approx(-98.9515).epsilon(Tolerance));
    REQUIRE(principal_stresses(1) == Approx(-163.611442455).epsilon(Tolerance));
    REQUIRE(principal_stresses(2) == Approx(-237.837).epsilon(Tolerance));
    REQUIRE(directors(0, 0) == Approx(-0.5187).epsilon(Tolerance));
    REQUIRE(directors(0, 1) == Approx(0.079565).epsilon(Tolerance));
    REQUIRE(directors(0, 2) == Approx(0.851246).epsilon(Tolerance));
    REQUIRE(directors(1, 0) == Approx(-0.674829).epsilon(Tolerance));
    REQUIRE(directors(1, 1) == Approx(0.573224).epsilon(Tolerance));
    REQUIRE(directors(1, 2) == Approx(-0.464781).epsilon(Tolerance));
    REQUIRE(directors(2, 0) == Approx(0.524935).epsilon(Tolerance));
    REQUIRE(directors(2, 1) == Approx(0.815527).epsilon(Tolerance));
    REQUIRE(directors(2, 2) == Approx(0.243639).epsilon(Tolerance));

    Eigen::Matrix<double, 3, 3> eigen;
    eigen.setZero();
    eigen(0, 0) = principal_stresses(0);
    eigen(1, 1) = principal_stresses(1);
    eigen(2, 2) = principal_stresses(2);
    auto initial_stress_2 = directors * eigen * directors.transpose();
    for (unsigned i = 0; i < 3; i++)
      for (unsigned j = 0; j < 3; j++)
        REQUIRE(initial_stress_2(i, j) ==
                Approx(stress_matrix(i, j)).epsilon(Tolerance));
  }
}