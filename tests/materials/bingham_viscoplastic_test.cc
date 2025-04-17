#include <limits>
#include <vector>

#include "Eigen/Dense"
#include "catch.hpp"
#include "json.hpp"

#include "element.h"
#include "factory.h"
#include "hexahedron_element.h"
#include "material.h"
#include "mesh.h"
#include "node.h"

//! \brief Check Bingham Viscoplastic class
//! Check Bingham Viscoplastic 2D without thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 2D (without thixotropy)",
          "[material][bingham][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 0.;
  jmaterial["flocculation_parameter"] = 0.;
  jmaterial["deflocculation_rate"] = 0.;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 8);
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("rmap_niteration") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "yield_state",        "pressure",
          "rmap_niteration",    "volumetric_strain",
          "shear_stress_ratio", "lambda",
          "pgamma_dot",         "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 2;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 4;
    const double dt = 1;

    coords << -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED2Q4");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0000000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(4353.0047581362).epsilon(Tolerance));
    REQUIRE(state_vars.at("rmap_niteration") == Approx(2.).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0062500000).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0085233869).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(0.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0015145055).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0008744002).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-4407.6418537463).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-4309.2950816482).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-4342.0773390142).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<  8710752.2189319357, 8715328.2570913527, 8691948.0727941766,       0.0000000000,       0.0000000000,       0.0000000000,
                  8715328.2570913527, 8724780.3295102417, 8677919.9622158743,       0.0000000000,       0.0000000000,       0.0000000000,
                  8691948.0727941766, 8677919.9622158743, 8748160.5138074160,       0.0000000000,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,   32782.2573660535,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   32782.2573660535,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   32782.2573660535;
    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}

//! Check Bingham Viscoplastic 2D with thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 2D (with thixotropy)",
          "[material][bingham][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 1.0;
  jmaterial["flocculation_parameter"] = 1.0;
  jmaterial["deflocculation_rate"] = 0.01;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 8);
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("rmap_niteration") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "yield_state",        "pressure",
          "rmap_niteration",    "volumetric_strain",
          "shear_stress_ratio", "lambda",
          "pgamma_dot",         "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 2;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 4;
    const double dt = 1;

    coords << -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED2Q4");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0000000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(4353.0047581362).epsilon(Tolerance));
    REQUIRE(state_vars.at("rmap_niteration") == Approx(2.).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0062500000).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0172038514).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(1.0199846873).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0015012459).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0008667447).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-4463.2858570077).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-4264.7798790390).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-4330.9485383619).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<  8715521.0978077333, 8724867.7147734538, 8677639.7362362817,       0.0000000000,       0.0000000000,       0.0000000000,
                  8724867.7147734538, 8743857.8849300370, 8649302.9491139781,       0.0000000000,       0.0000000000,       0.0000000000,
                  8677639.7362362817, 8649302.9491139781, 8791085.8634672090,       0.0000000000,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,   66168.6593228985,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   66168.6593228985,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   66168.6593228985;

    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}

//! Check Bingham Viscoplastic 3D without thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 3D (without thixotropy)",
          "[material][bingham][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 0.;
  jmaterial["flocculation_parameter"] = 0.;
  jmaterial["deflocculation_rate"] = 0.;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 8);
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("rmap_niteration") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "yield_state",        "pressure",
          "rmap_niteration",    "volumetric_strain",
          "shear_stress_ratio", "lambda",
          "pgamma_dot",         "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 3;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 8;
    const double dt = 1;

    coords << -2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    coords << -2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);
    coords << 2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);
    coords << 2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);
    coords << -2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->assign_velocity_constraint(2, 0.04);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    cell->add_node(4, node4);
    cell->add_node(5, node5);
    cell->add_node(6, node6);
    cell->add_node(7, node7);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0004000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::Material<Dim>::Vector6d stress;
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(844.3429257361).epsilon(Tolerance));
    REQUIRE(state_vars.at("rmap_niteration") == Approx(2.).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0018750000).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0077634533).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(0.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0016642778).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0009608712).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-902.0711678656).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-812.4928611129).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-818.4647482297).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<  8443637.2001096904, 8445373.8017463777, 8441276.7702260334,       0.0000000000,       0.0000000000,       0.0000000000,
                  8445373.8017463777, 8471186.2051602751, 8413727.7651754487,       0.0000000000,       0.0000000000,       0.0000000000,
                  8441276.7702260334, 8413727.7651754487, 8475283.2366806176,       0.0000000000,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,   29859.4355842311,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   29859.4355842311,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   29859.4355842311;

    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}

//! Check Bingham Viscoplastic 3D with thixotropy
TEST_CASE("Bingham Viscoplastic is checked in 3D (with thixotropy)",
          "[material][bingham][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  const double dt = 1.0;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;
  jmaterial["volumetric_gamma"] = 7.0;
  jmaterial["dynamic_viscosity"] = 50.0;
  jmaterial["tau0"] = 50.0;
  jmaterial["flocculation_state"] = 1.0;
  jmaterial["flocculation_parameter"] = 1.0;
  jmaterial["deflocculation_rate"] = 0.01;
  jmaterial["rmap_absolute_tolerance"] = 1.e-10;
  jmaterial["rmap_relative_tolerance"] = 1.e-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Bingham Viscoplastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Bingham Viscoplastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  // Failed initialisation material
  SECTION("Bingham Viscoplastic failed initialisation") {
    unsigned id = 0;
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["volumetric_gamma"] = 7.0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
  }

  //! Read material properties
  SECTION("Bingham Viscoplastic check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("volumetric_gamma") ==
            Approx(jmaterial["volumetric_gamma"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("dynamic_viscosity") ==
            Approx(jmaterial["dynamic_viscosity"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("tau0") ==
            Approx(jmaterial["tau0"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_state") ==
            Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("flocculation_parameter") ==
            Approx(jmaterial["flocculation_parameter"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("deflocculation_rate") ==
            Approx(jmaterial["deflocculation_rate"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_max_iteration") ==
            Approx(jmaterial["rmap_max_iteration"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.size() == 8);
      REQUIRE(state_variables.at("yield_state") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pressure") == Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("rmap_niteration") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("volumetric_strain") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("shear_stress_ratio") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("lambda") ==
              Approx(jmaterial["flocculation_state"]).epsilon(Tolerance));
      REQUIRE(state_variables.at("pgamma_dot") ==
              Approx(0.).epsilon(Tolerance));
      REQUIRE(state_variables.at("pdstrain") == Approx(0.).epsilon(Tolerance));

      const std::vector<std::string> state_vars = {
          "yield_state",        "pressure",
          "rmap_niteration",    "volumetric_strain",
          "shear_stress_ratio", "lambda",
          "pgamma_dot",         "pdstrain"};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("Bingham Viscoplastic check stresses with strain rate") {
    unsigned id = 0;
    // Initialise material
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "BinghamViscoPlastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Add particle
    mpm::Index pid = 0;
    Eigen::Matrix<double, Dim, 1> coords;
    coords << 0.5, 0.5, 0.5;
    auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

    // Coordinates of nodes for the cell
    mpm::Index cell_id = 0;
    const unsigned Dof = 3;
    const unsigned Nphases = 1;
    const unsigned Nnodes = 8;
    const double dt = 1;

    coords << -2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2, 2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    coords << -2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);
    coords << 2, -2, -2;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);
    coords << 2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);
    coords << -2, -2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);

    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

    node0->assign_velocity_constraint(0, 0.02);
    node0->assign_velocity_constraint(1, 0.03);
    node0->assign_velocity_constraint(2, 0.04);
    node0->apply_velocity_constraints();

    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    cell->add_node(4, node4);
    cell->add_node(5, node5);
    cell->add_node(6, node6);
    cell->add_node(7, node7);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);

    particle->assign_cell(cell);
    particle->assign_material(material);
    particle->compute_shapefn();
    particle->compute_strain(dt);

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain(0) = -0.0010000;
    dstrain(1) = 0.0005000;
    dstrain(2) = 0.0004000;
    dstrain(3) = 0.0000000;
    dstrain(4) = 0.0000000;
    dstrain(5) = 0.0000000;

    // Compute updated stress
    mpm::Material<Dim>::Vector6d stress;
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress.setZero();
    auto updated_stress = material->compute_stress(
        stress, dstrain, particle.get(), &state_vars, dt);

    // Check state variables
    REQUIRE(state_vars.at("yield_state") == Approx(1.).epsilon(Tolerance));
    REQUIRE(state_vars.at("pressure") ==
            Approx(844.3429257361).epsilon(Tolerance));
    REQUIRE(state_vars.at("rmap_niteration") == Approx(2.).epsilon(Tolerance));
    REQUIRE(state_vars.at("volumetric_strain") ==
            Approx(-0.0018750000).epsilon(Tolerance));
    REQUIRE(state_vars.at("shear_stress_ratio") ==
            Approx(0.0156687852).epsilon(Tolerance));
    REQUIRE(state_vars.at("lambda") == Approx(1.0199831596).epsilon(Tolerance));
    REQUIRE(state_vars.at("pgamma_dot") ==
            Approx(0.0016510182).epsilon(Tolerance));
    REQUIRE(state_vars.at("pdstrain") ==
            Approx(0.0009532158).epsilon(Tolerance));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) == Approx(-960.8544055815).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(-780.0607299593).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(-792.1136416674).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.000e+00).epsilon(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.000e+00).epsilon(Tolerance));

    // Compute consistent tangent matrix
    auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, stress, dstrain, particle.get(), &state_vars, dt);

    // Values of reduced constitutive relations matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check <<  8443780.6226642635, 8447391.5770925805, 8439115.5723252594,       0.0000000000,       0.0000000000,       0.0000000000,
                  8447391.5770925805, 8499429.6202376243, 8383466.5747518986,       0.0000000000,       0.0000000000,       0.0000000000,
                  8439115.5723252594, 8383466.5747518986, 8507705.6250049453,       0.0000000000,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,   60264.5585407447,       0.0000000000,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   60264.5585407447,       0.0000000000,
                        0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,       0.0000000000,   60264.5585407447;

    // clang-format on
    // Check cell stiffness matrix
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
  }
}