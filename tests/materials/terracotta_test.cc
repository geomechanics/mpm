#include <cmath>
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

//! \brief Check Terracotta class
//! Check Terracotta 2D without pore fluid
TEST_CASE("Terracotta is checked in 2D (without pore fluid)",
          "[material][terracotta][2D][dry]") {
  // Tolerance
  const double Tolerance = 1.E-7;
  const unsigned Dim = 2;
  const double dt = 0.0006;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 715.5;
  jmaterial["packing_fraction"] = 0.27;
  jmaterial["bulk_modulus"] = 1.56E+7;
  jmaterial["shear_modulus"] = 5.53E+5;
  jmaterial["lambda"] = 4.5;
  jmaterial["reference_pressure"] = 6.6E+6;
  jmaterial["alpha"] = 0.42;
  jmaterial["beta"] = 10.0;
  jmaterial["gamma"] = 1;
  jmaterial["eta"] = 3.0E+6;
  jmaterial["omega"] = 0.4;
  jmaterial["m"] = 0.1;
  jmaterial["meso_temperature"] = 1.E-15;
  jmaterial["rmap_absolute_tolerance"] = 1.E-10;
  jmaterial["rmap_relative_tolerance"] = 1.E-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Terracotta material id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Terracotta accepts the maximum unsigned material id") {
    const unsigned expected_id = std::numeric_limits<unsigned>::max();
    unsigned id = expected_id;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);

    REQUIRE(material->id() == expected_id);
  }

  //! Read material properties
  SECTION("Terracotta check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("bulk_modulus") ==
            Approx(jmaterial["bulk_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("shear_modulus") ==
        Approx(jmaterial["shear_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("lambda") ==
            Approx(jmaterial["lambda"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("reference_pressure") ==
            Approx(jmaterial["reference_pressure"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("alpha") ==
            Approx(jmaterial["alpha"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("beta") ==
            Approx(jmaterial["beta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("gamma") ==
            Approx(jmaterial["gamma"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("eta") ==
            Approx(jmaterial["eta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("omega") ==
            Approx(jmaterial["omega"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("m") ==
            Approx(jmaterial["m"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("meso_temperature") ==
        Approx(jmaterial["meso_temperature"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<unsigned>("rmap_max_iteration") ==
            jmaterial["rmap_max_iteration"].get<unsigned>());

    // Check if state variable is initialised
    SECTION("Terracotta state variables are initialised correctly") {
      const mpm::dense_map state_variables =
          material->initialise_state_variables();

      REQUIRE(state_variables.size() == 18);
      REQUIRE(state_variables.at("pressure") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("q") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm") ==
              Approx(jmaterial["meso_temperature"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm_prev") == Approx(-1.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("packing_fraction") ==
              Approx(jmaterial["packing_fraction"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("pore_pressure") ==
              Approx(0.0).margin(Tolerance));
      const std::vector<std::string> expected_state_variables = {
          "pressure",
          "q",
          "tm",
          "elastic_strain0",
          "elastic_strain1",
          "elastic_strain2",
          "elastic_strain3",
          "elastic_strain4",
          "elastic_strain5",
          "tm_prev",
          "elastic_strain0_prev",
          "elastic_strain1_prev",
          "elastic_strain2_prev",
          "elastic_strain3_prev",
          "elastic_strain4_prev",
          "elastic_strain5_prev",
          "packing_fraction",
          "pore_pressure"};

      const auto actual_state_variables = material->state_variables();
      REQUIRE(actual_state_variables == expected_state_variables);
    }
  }

  SECTION("Terracotta check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);
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

    const double compression_rate = 0.01;
    node0->assign_velocity_constraint(0, 2.0 * compression_rate);
    node0->assign_velocity_constraint(1, 2.0 * compression_rate);
    node1->assign_velocity_constraint(0, -2.0 * compression_rate);
    node1->assign_velocity_constraint(1, 2.0 * compression_rate);
    node2->assign_velocity_constraint(0, -2.0 * compression_rate);
    node2->assign_velocity_constraint(1, -2.0 * compression_rate);
    node3->assign_velocity_constraint(0, 2.0 * compression_rate);
    node3->assign_velocity_constraint(1, -2.0 * compression_rate);
    node0->apply_velocity_constraints();
    node1->apply_velocity_constraints();
    node2->apply_velocity_constraints();
    node3->apply_velocity_constraints();
    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);
    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);
    REQUIRE(particle->assign_cell(cell));
    REQUIRE(particle->assign_material(material));
    REQUIRE(particle->assign_volume(1.0));
    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE(particle->volume() == Approx(1.0).epsilon(Tolerance));
    REQUIRE(particle->mass_density() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(particle->mass() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE_NOTHROW(particle->compute_shapefn());
    REQUIRE_NOTHROW(particle->compute_strain(dt));
    const auto strain_rate = particle->strain_rate();
    // strain rate check to confirms that the node velocities produced the
    // intended particle loading before Terracotta uses it.
    REQUIRE(strain_rate(0) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(1) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(2) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(5) == Approx(0.0).margin(Tolerance));

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain = mpm::Material<Dim>::Vector6d::Zero();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d previous_stress =
        mpm::Material<Dim>::Vector6d::Zero();
    const auto initial_state_vars = state_vars;
    const auto updated_stress = material->compute_stress(
        previous_stress, dstrain, particle.get(), &state_vars, dt);

    // Check updated Terracotta state variables
    REQUIRE(std::isfinite(state_vars.at("pressure")));
    REQUIRE(std::isfinite(state_vars.at("q")));
    REQUIRE(std::isfinite(state_vars.at("tm")));
    REQUIRE(std::isfinite(state_vars.at("packing_fraction")));
    REQUIRE(std::isfinite(state_vars.at("pore_pressure")));
    REQUIRE(state_vars.at("pressure") ==
            Approx(1.000061694969029E-4).epsilon(Tolerance));
    REQUIRE(state_vars.at("q") == Approx(7.981425924081973E-8).margin(1.E-12));
    REQUIRE(state_vars.at("tm") ==
            Approx(3.672239315180879E-7).epsilon(Tolerance));
    REQUIRE(state_vars.at("tm_prev") ==
            Approx(initial_state_vars.at("tm")).margin(1.E-20));
    REQUIRE(
        state_vars.at("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(state_vars.at("pore_pressure") == Approx(0.0).margin(Tolerance));

    // Previous elastic strains were zero before the first update
    REQUIRE(state_vars.at("elastic_strain0_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain3_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5_prev") ==
            Approx(0.0).margin(Tolerance));

    // Compare updated elastic strains with MATLAB
    REQUIRE(state_vars.at("elastic_strain0") ==
            Approx(-5.999999998922999E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1") ==
            Approx(-5.999999998922999E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2") ==
            Approx(1.077000698367948E-15).margin(1.E-12));
    REQUIRE(state_vars.at("elastic_strain3") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5") == Approx(0.0).margin(Tolerance));

    // Updated elastic strains must be valid finite numbers
    REQUIRE(std::isfinite(state_vars.at("elastic_strain0")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain1")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain2")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain3")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain4")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain5")));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) ==
            Approx(-1.000327742499832E-4).epsilon(Tolerance));
    REQUIRE(updated_stress(1) ==
            Approx(-1.000327742499832E-4).epsilon(Tolerance));
    REQUIRE(updated_stress(2) ==
            Approx(-9.995295999074238E-5).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.0).margin(Tolerance));

    // Newmark parameters used by the MATLAB reference
    const double newmark_beta = 0.25;
    const double newmark_gamma = 0.5;
    const double lin_v = newmark_gamma / (newmark_beta * dt);

    // Compute the consistent tangent matrix
    const auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, previous_stress, dstrain, particle.get(), &state_vars,
        dt, lin_v);

    // Check the tangent matrix size
    REQUIRE(dep.rows() == 6);
    REQUIRE(dep.cols() == 6);

    // Check that the tangent matrix does not contain NaN or infinity
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(std::isfinite(dep(i, j)));

    // Expected consistent tangent matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check << 1.2045732477640873e-01, 9.7280468181890359e-02, 8.2983685563231033e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                 9.7280468181890359e-02, 1.2045732477640872e-01, 8.2983685563231033e-02, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                 8.2983815904201341e-02, 8.2983815904201341e-02, 1.1552591265178340e-01, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00,
                 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.1588428297259184e-02, 0.0000000000000000e+00, 0.0000000000000000e+00,
                 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.1588428297259184e-02, 0.0000000000000000e+00,
                 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 0.0000000000000000e+00, 1.1588428297259184e-02;
    // clang-format on

    // Compare every C++ value with the MATLAB reference value
    for (unsigned i = 0; i < dep.rows(); ++i) {
      for (unsigned j = 0; j < dep.cols(); ++j) {
        INFO("Tangent row = " << i << ", column = " << j);
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
      }
    }
  }
}

//! \brief Check Terracotta class
//! Check Terracotta 2D with pore fluid
TEST_CASE("Terracotta is checked in 2D (with pore fluid)",
          "[material][terracotta][2D][fluid]") {
  // Tolerance
  const double Tolerance = 1.E-7;
  const unsigned Dim = 2;
  const double dt = 0.0006;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1445.4;
  jmaterial["packing_fraction"] = 0.27;
  jmaterial["bulk_modulus"] = 1.56E+7;
  jmaterial["shear_modulus"] = 5.53E+5;
  jmaterial["lambda"] = 4.5;
  jmaterial["reference_pressure"] = 6.6E+6;
  jmaterial["alpha"] = 0.42;
  jmaterial["beta"] = 10.0;
  jmaterial["gamma"] = 1;
  jmaterial["eta"] = 3.0E+6;
  jmaterial["omega"] = 0.4;
  jmaterial["m"] = 0.1;
  jmaterial["meso_temperature"] = 1.E-15;
  jmaterial["pore_fluid_density"] = 1000.0;
  jmaterial["pore_fluid_bulk_modulus"] = 2.2E+9;
  jmaterial["rmap_absolute_tolerance"] = 1.E-10;
  jmaterial["rmap_relative_tolerance"] = 1.E-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Terracotta material id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Terracotta accepts the maximum unsigned material id") {
    const unsigned expected_id = std::numeric_limits<unsigned>::max();
    unsigned id = expected_id;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);

    REQUIRE(material->id() == expected_id);
  }

  //! Read material properties
  SECTION("Terracotta check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("bulk_modulus") ==
            Approx(jmaterial["bulk_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("shear_modulus") ==
        Approx(jmaterial["shear_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("lambda") ==
            Approx(jmaterial["lambda"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("reference_pressure") ==
            Approx(jmaterial["reference_pressure"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("alpha") ==
            Approx(jmaterial["alpha"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("beta") ==
            Approx(jmaterial["beta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("gamma") ==
            Approx(jmaterial["gamma"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("eta") ==
            Approx(jmaterial["eta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("omega") ==
            Approx(jmaterial["omega"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("m") ==
            Approx(jmaterial["m"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("meso_temperature") ==
        Approx(jmaterial["meso_temperature"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("pore_fluid_density") ==
            Approx(jmaterial["pore_fluid_density"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("pore_fluid_bulk_modulus") ==
            Approx(jmaterial["pore_fluid_bulk_modulus"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<unsigned>("rmap_max_iteration") ==
            jmaterial["rmap_max_iteration"].get<unsigned>());

    // Check if state variable is initialised
    SECTION("Terracotta state variables are initialised correctly") {
      const mpm::dense_map state_variables =
          material->initialise_state_variables();

      REQUIRE(state_variables.size() == 18);
      REQUIRE(state_variables.at("pressure") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("q") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm") ==
              Approx(jmaterial["meso_temperature"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm_prev") == Approx(-1.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("packing_fraction") ==
              Approx(jmaterial["packing_fraction"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("pore_pressure") ==
              Approx(0.0).margin(Tolerance));
      const std::vector<std::string> expected_state_variables = {
          "pressure",
          "q",
          "tm",
          "elastic_strain0",
          "elastic_strain1",
          "elastic_strain2",
          "elastic_strain3",
          "elastic_strain4",
          "elastic_strain5",
          "tm_prev",
          "elastic_strain0_prev",
          "elastic_strain1_prev",
          "elastic_strain2_prev",
          "elastic_strain3_prev",
          "elastic_strain4_prev",
          "elastic_strain5_prev",
          "packing_fraction",
          "pore_pressure"};

      const auto actual_state_variables = material->state_variables();
      REQUIRE(actual_state_variables == expected_state_variables);
    }
  }

  SECTION("Terracotta check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta2D", std::move(id), jmaterial);
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

    const double compression_rate = 0.01;
    node0->assign_velocity_constraint(0, 2.0 * compression_rate);
    node0->assign_velocity_constraint(1, 2.0 * compression_rate);
    node1->assign_velocity_constraint(0, -2.0 * compression_rate);
    node1->assign_velocity_constraint(1, 2.0 * compression_rate);
    node2->assign_velocity_constraint(0, -2.0 * compression_rate);
    node2->assign_velocity_constraint(1, -2.0 * compression_rate);
    node3->assign_velocity_constraint(0, 2.0 * compression_rate);
    node3->assign_velocity_constraint(1, -2.0 * compression_rate);
    node0->apply_velocity_constraints();
    node1->apply_velocity_constraints();
    node2->apply_velocity_constraints();
    node3->apply_velocity_constraints();
    auto cell = std::make_shared<mpm::Cell<Dim>>(cell_id, Nnodes, shapefn);
    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);

    // Initialise cell
    REQUIRE(cell->initialise() == true);
    // Check if cell is initialised, after addition of nodes
    REQUIRE(cell->is_initialised() == true);
    REQUIRE(particle->assign_cell(cell));
    REQUIRE(particle->assign_material(material));
    REQUIRE(particle->assign_volume(1.0));
    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE(particle->volume() == Approx(1.0).epsilon(Tolerance));
    REQUIRE(particle->mass_density() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(particle->mass() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE_NOTHROW(particle->compute_shapefn());
    REQUIRE_NOTHROW(particle->compute_strain(dt));
    const auto strain_rate = particle->strain_rate();
    // strain rate check to confirms that the node velocities produced the
    // intended particle loading before Terracotta uses it.
    REQUIRE(strain_rate(0) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(1) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(2) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(5) == Approx(0.0).margin(Tolerance));

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain = mpm::Material<Dim>::Vector6d::Zero();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d previous_stress =
        mpm::Material<Dim>::Vector6d::Zero();
    const auto initial_state_vars = state_vars;
    const auto updated_stress = material->compute_stress(
        previous_stress, dstrain, particle.get(), &state_vars, dt);

    // Check updated Terracotta state variables
    REQUIRE(std::isfinite(state_vars.at("pressure")));
    REQUIRE(std::isfinite(state_vars.at("q")));
    REQUIRE(std::isfinite(state_vars.at("tm")));
    REQUIRE(std::isfinite(state_vars.at("packing_fraction")));
    REQUIRE(std::isfinite(state_vars.at("pore_pressure")));
    // Compression-positive volumetric strain rate
    const double expected_volumetric_strain_rate =
        -(strain_rate(0) + strain_rate(1) + strain_rate(2));
    REQUIRE(expected_volumetric_strain_rate == Approx(0.02).epsilon(Tolerance));
    // Pore-pressure increment produced by compression
    const double expected_pore_pressure =
        jmaterial["pore_fluid_bulk_modulus"].get<double>() /
        (1.0 - jmaterial["packing_fraction"].get<double>()) *
        expected_volumetric_strain_rate * dt;
    // Dry skeleton pressure obtained from the corresponding dry test
    const double expected_dry_pressure = 1.000061694969029E-4;
    // Total pressure includes the pore-fluid contribution
    const double expected_total_pressure =
        expected_dry_pressure + expected_pore_pressure;
    REQUIRE(state_vars.at("pressure") ==
            Approx(expected_total_pressure).epsilon(Tolerance));
    REQUIRE(state_vars.at("q") == Approx(7.981425924081973E-8).margin(1.E-12));
    REQUIRE(state_vars.at("tm") ==
            Approx(3.672239315180879E-7).epsilon(Tolerance));
    REQUIRE(state_vars.at("tm_prev") ==
            Approx(initial_state_vars.at("tm")).margin(1.E-20));
    REQUIRE(
        state_vars.at("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(state_vars.at("pore_pressure") ==
            Approx(expected_pore_pressure).epsilon(Tolerance));

    // Previous elastic strains were zero before the first update
    REQUIRE(state_vars.at("elastic_strain0_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain3_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5_prev") ==
            Approx(0.0).margin(Tolerance));

    // Compare updated elastic strains with MATLAB
    REQUIRE(state_vars.at("elastic_strain0") ==
            Approx(-5.999999998922999E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1") ==
            Approx(-5.999999998922999E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2") ==
            Approx(1.077000698367948E-15).margin(1.E-12));
    REQUIRE(state_vars.at("elastic_strain3") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5") == Approx(0.0).margin(Tolerance));

    // Updated elastic strains must be valid finite numbers
    REQUIRE(std::isfinite(state_vars.at("elastic_strain0")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain1")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain2")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain3")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain4")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain5")));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    // Pore pressure shifts all normal stresses equally
    const double expected_sigma_xx =
        -1.000327742499832E-4 - expected_pore_pressure;
    const double expected_sigma_yy =
        -1.000327742499832E-4 - expected_pore_pressure;
    const double expected_sigma_zz =
        -9.995295999074238E-5 - expected_pore_pressure;
    REQUIRE(updated_stress(0) == Approx(expected_sigma_xx).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(expected_sigma_yy).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(expected_sigma_zz).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.0).margin(Tolerance));
    // Newmark parameters used by the MATLAB reference
    const double newmark_beta = 0.25;
    const double newmark_gamma = 0.5;
    const double lin_v = newmark_gamma / (newmark_beta * dt);

    // Compute the saturated consistent tangent matrix
    const auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, previous_stress, dstrain, particle.get(), &state_vars,
        dt, lin_v);

    // Check tangent matrix dimensions
    REQUIRE(dep.rows() == 6);
    REQUIRE(dep.cols() == 6);

    // Check that the tangent matrix contains finite values
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(std::isfinite(dep(i, j)));

    // Expected saturated consistent tangent matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check << 6.0273972603944302E+9, 6.0273972603712530E+9, 6.0273972603569565E+9, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 6.0273972603712530E+9, 6.0273972603944302E+9, 6.0273972603569565E+9, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 6.0273972603569565E+9, 6.0273972603569565E+9, 6.0273972603894987E+9, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.1588428297259184E-2, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.1588428297259184E-2, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.1588428297259184E-2;
    // clang-format on

    // Compare the C++ tangent matrix with the MATLAB reference
    const double TangentTolerance = 1.E-12;
    for (unsigned i = 0; i < dep.rows(); ++i) {
      for (unsigned j = 0; j < dep.cols(); ++j) {
        INFO("Tangent row = " << i << ", column = " << j);
        INFO("C++ value = " << dep(i, j));
        INFO("MATLAB value = " << dep_check(i, j));
        REQUIRE(
            dep(i, j) ==
            Approx(dep_check(i, j)).epsilon(TangentTolerance).margin(1.E-12));
      }
    }
  }
}

//! \brief Check Terracotta class
//! Check Terracotta 3D without pore fluid
TEST_CASE("Terracotta is checked in 3D (without pore fluid)",
          "[material][terracotta][3D][dry]") {
  // Tolerance
  const double Tolerance = 1.E-7;
  const unsigned Dim = 3;
  const double dt = 0.0006;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 715.5;
  jmaterial["packing_fraction"] = 0.27;
  jmaterial["bulk_modulus"] = 1.56E+7;
  jmaterial["shear_modulus"] = 5.53E+5;
  jmaterial["lambda"] = 4.5;
  jmaterial["reference_pressure"] = 6.6E+6;
  jmaterial["alpha"] = 0.42;
  jmaterial["beta"] = 10.0;
  jmaterial["gamma"] = 1;
  jmaterial["eta"] = 3.0E+6;
  jmaterial["omega"] = 0.4;
  jmaterial["m"] = 0.1;
  jmaterial["meso_temperature"] = 1.E-15;
  jmaterial["rmap_absolute_tolerance"] = 1.E-10;
  jmaterial["rmap_relative_tolerance"] = 1.E-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Terracotta material id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Terracotta accepts the maximum unsigned material id") {
    const unsigned expected_id = std::numeric_limits<unsigned>::max();
    unsigned id = expected_id;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);

    REQUIRE(material->id() == expected_id);
  }

  //! Read material properties
  SECTION("Terracotta check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("bulk_modulus") ==
            Approx(jmaterial["bulk_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("shear_modulus") ==
        Approx(jmaterial["shear_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("lambda") ==
            Approx(jmaterial["lambda"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("reference_pressure") ==
            Approx(jmaterial["reference_pressure"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("alpha") ==
            Approx(jmaterial["alpha"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("beta") ==
            Approx(jmaterial["beta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("gamma") ==
            Approx(jmaterial["gamma"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("eta") ==
            Approx(jmaterial["eta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("omega") ==
            Approx(jmaterial["omega"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("m") ==
            Approx(jmaterial["m"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("meso_temperature") ==
        Approx(jmaterial["meso_temperature"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<unsigned>("rmap_max_iteration") ==
            jmaterial["rmap_max_iteration"].get<unsigned>());

    // Check if state variable is initialised
    SECTION("Terracotta state variables are initialised correctly") {
      const mpm::dense_map state_variables =
          material->initialise_state_variables();

      REQUIRE(state_variables.size() == 18);
      REQUIRE(state_variables.at("pressure") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("q") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm") ==
              Approx(jmaterial["meso_temperature"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm_prev") == Approx(-1.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("packing_fraction") ==
              Approx(jmaterial["packing_fraction"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("pore_pressure") ==
              Approx(0.0).margin(Tolerance));
      const std::vector<std::string> expected_state_variables = {
          "pressure",
          "q",
          "tm",
          "elastic_strain0",
          "elastic_strain1",
          "elastic_strain2",
          "elastic_strain3",
          "elastic_strain4",
          "elastic_strain5",
          "tm_prev",
          "elastic_strain0_prev",
          "elastic_strain1_prev",
          "elastic_strain2_prev",
          "elastic_strain3_prev",
          "elastic_strain4_prev",
          "elastic_strain5_prev",
          "packing_fraction",
          "pore_pressure"};

      const auto actual_state_variables = material->state_variables();
      REQUIRE(actual_state_variables == expected_state_variables);
    }
  }

  SECTION("Terracotta check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);
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

    // Top four nodes
    coords << -2.0, 2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2.0, 2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2.0, 2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2.0, 2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    // Bottom four nodes
    coords << -2.0, -2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);
    coords << 2.0, -2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);
    coords << 2.0, -2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);
    coords << -2.0, -2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);
    // Use a 3D, eight-node hexahedral element
    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

    const double compression_rate = 0.01;
    node0->assign_velocity_constraint(0, 2.0 * compression_rate);
    node0->assign_velocity_constraint(1, -2.0 * compression_rate);
    node0->assign_velocity_constraint(2, 2.0 * compression_rate);

    node1->assign_velocity_constraint(0, -2.0 * compression_rate);
    node1->assign_velocity_constraint(1, -2.0 * compression_rate);
    node1->assign_velocity_constraint(2, 2.0 * compression_rate);

    node2->assign_velocity_constraint(0, -2.0 * compression_rate);
    node2->assign_velocity_constraint(1, -2.0 * compression_rate);
    node2->assign_velocity_constraint(2, -2.0 * compression_rate);

    node3->assign_velocity_constraint(0, 2.0 * compression_rate);
    node3->assign_velocity_constraint(1, -2.0 * compression_rate);
    node3->assign_velocity_constraint(2, -2.0 * compression_rate);

    node4->assign_velocity_constraint(0, 2.0 * compression_rate);
    node4->assign_velocity_constraint(1, 2.0 * compression_rate);
    node4->assign_velocity_constraint(2, 2.0 * compression_rate);

    node5->assign_velocity_constraint(0, -2.0 * compression_rate);
    node5->assign_velocity_constraint(1, 2.0 * compression_rate);
    node5->assign_velocity_constraint(2, 2.0 * compression_rate);

    node6->assign_velocity_constraint(0, -2.0 * compression_rate);
    node6->assign_velocity_constraint(1, 2.0 * compression_rate);
    node6->assign_velocity_constraint(2, -2.0 * compression_rate);

    node7->assign_velocity_constraint(0, 2.0 * compression_rate);
    node7->assign_velocity_constraint(1, 2.0 * compression_rate);
    node7->assign_velocity_constraint(2, -2.0 * compression_rate);
    // Apply all assigned velocity constraints
    node0->apply_velocity_constraints();
    node1->apply_velocity_constraints();
    node2->apply_velocity_constraints();
    node3->apply_velocity_constraints();
    node4->apply_velocity_constraints();
    node5->apply_velocity_constraints();
    node6->apply_velocity_constraints();
    node7->apply_velocity_constraints();

    // Create the cell and add all eight nodes
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
    REQUIRE(particle->assign_cell(cell));
    REQUIRE(particle->assign_material(material));
    REQUIRE(particle->assign_volume(1.0));
    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE(particle->volume() == Approx(1.0).epsilon(Tolerance));
    REQUIRE(particle->mass_density() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(particle->mass() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE_NOTHROW(particle->compute_shapefn());
    REQUIRE_NOTHROW(particle->compute_strain(dt));
    const auto strain_rate = particle->strain_rate();
    // strain rate check to confirms that the node velocities produced the
    // intended particle loading before Terracotta uses it.
    REQUIRE(strain_rate(0) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(1) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(2) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(5) == Approx(0.0).margin(Tolerance));

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain = mpm::Material<Dim>::Vector6d::Zero();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d previous_stress =
        mpm::Material<Dim>::Vector6d::Zero();
    const auto initial_state_vars = state_vars;
    const auto updated_stress = material->compute_stress(
        previous_stress, dstrain, particle.get(), &state_vars, dt);

    // Check updated Terracotta state variables
    REQUIRE(std::isfinite(state_vars.at("pressure")));
    REQUIRE(std::isfinite(state_vars.at("q")));
    REQUIRE(std::isfinite(state_vars.at("tm")));
    REQUIRE(std::isfinite(state_vars.at("packing_fraction")));
    REQUIRE(std::isfinite(state_vars.at("pore_pressure")));
    REQUIRE(state_vars.at("pressure") ==
            Approx(1.000057130800860E-4).epsilon(Tolerance));
    REQUIRE(state_vars.at("q") == Approx(0.0).margin(1.E-12));
    REQUIRE(state_vars.at("tm") ==
            Approx(2.267074876870378E-7).epsilon(Tolerance));
    REQUIRE(state_vars.at("tm_prev") ==
            Approx(initial_state_vars.at("tm")).margin(1.E-20));
    REQUIRE(
        state_vars.at("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(state_vars.at("pore_pressure") == Approx(0.0).margin(Tolerance));

    // Previous elastic strains were zero before the first update
    REQUIRE(state_vars.at("elastic_strain0_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain3_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5_prev") ==
            Approx(0.0).margin(Tolerance));

    // Compare updated elastic strains with MATLAB
    REQUIRE(state_vars.at("elastic_strain0") ==
            Approx(-5.999999999335107E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1") ==
            Approx(-5.999999999335107E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2") ==
            Approx(-5.999999999335107E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain3") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5") == Approx(0.0).margin(Tolerance));

    // Updated elastic strains must be valid finite numbers
    REQUIRE(std::isfinite(state_vars.at("elastic_strain0")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain1")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain2")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain3")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain4")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain5")));

    // Check stressees
    REQUIRE(updated_stress.size() == 6);
    REQUIRE(updated_stress(0) ==
            Approx(-1.000057130800860E-4).epsilon(Tolerance));
    REQUIRE(updated_stress(1) ==
            Approx(-1.000057130800860E-4).epsilon(Tolerance));
    REQUIRE(updated_stress(2) ==
            Approx(-1.000057130800860E-4).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.0).margin(Tolerance));

    // Newmark parameters used by the MATLAB reference
    const double newmark_beta = 0.25;
    const double newmark_gamma = 0.5;
    const double lin_v = newmark_gamma / (newmark_beta * dt);

    // Compute the 3D dry consistent tangent matrix
    const auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, previous_stress, dstrain, particle.get(), &state_vars,
        dt, lin_v);

    // Check tangent matrix dimensions
    REQUIRE(dep.rows() == 6);
    REQUIRE(dep.cols() == 6);

    // Check that every tangent value is finite
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(std::isfinite(dep(i, j)));

    // Expected 3D dry consistent tangent matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check << 1.6010387907370113E-1, 1.3974430127318668E-1, 1.3974430127318668E-1, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 1.3974430127318668E-1, 1.6010387907370113E-1, 1.3974430127318668E-1, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 1.3974430127318671E-1, 1.3974430127318668E-1, 1.6010387907370113E-1, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.0179788900257213E-2, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.0179788900257213E-2, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.0179788900257213E-2;
    // clang-format on

    // Compare the C++ tangent matrix with the MATLAB reference
    for (unsigned i = 0; i < dep.rows(); ++i) {
      for (unsigned j = 0; j < dep.cols(); ++j) {
        INFO("Tangent row = " << i << ", column = " << j);
        INFO("C++ value = " << dep(i, j));
        INFO("MATLAB value = " << dep_check(i, j));
        REQUIRE(dep(i, j) == Approx(dep_check(i, j)).epsilon(Tolerance));
      }
    }
  }
}

//! \brief Check Terracotta class
//! Check Terracotta 3D with pore fluid
TEST_CASE("Terracotta is checked in 3D (with pore fluid)",
          "[material][terracotta][3D][fluid]") {
  // Tolerance
  const double Tolerance = 1.E-7;
  const unsigned Dim = 3;
  const double dt = 0.0006;

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1445.4;
  jmaterial["packing_fraction"] = 0.27;
  jmaterial["bulk_modulus"] = 1.56E+7;
  jmaterial["shear_modulus"] = 5.53E+5;
  jmaterial["lambda"] = 4.5;
  jmaterial["reference_pressure"] = 6.6E+6;
  jmaterial["alpha"] = 0.42;
  jmaterial["beta"] = 10.0;
  jmaterial["gamma"] = 1;
  jmaterial["eta"] = 3.0E+6;
  jmaterial["omega"] = 0.4;
  jmaterial["m"] = 0.1;
  jmaterial["meso_temperature"] = 1.E-15;
  jmaterial["pore_fluid_density"] = 1000.0;
  jmaterial["pore_fluid_bulk_modulus"] = 2.2E+9;
  jmaterial["rmap_absolute_tolerance"] = 1.E-10;
  jmaterial["rmap_relative_tolerance"] = 1.E-8;
  jmaterial["rmap_max_iteration"] = 15;

  //! Check for id = 0
  SECTION("Terracotta material id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("Terracotta accepts the maximum unsigned material id") {
    const unsigned expected_id = std::numeric_limits<unsigned>::max();
    unsigned id = expected_id;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);

    REQUIRE(material->id() == expected_id);
  }

  //! Read material properties
  SECTION("Terracotta check properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("bulk_modulus") ==
            Approx(jmaterial["bulk_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("shear_modulus") ==
        Approx(jmaterial["shear_modulus"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("lambda") ==
            Approx(jmaterial["lambda"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("reference_pressure") ==
            Approx(jmaterial["reference_pressure"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("alpha") ==
            Approx(jmaterial["alpha"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("beta") ==
            Approx(jmaterial["beta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("gamma") ==
            Approx(jmaterial["gamma"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("eta") ==
            Approx(jmaterial["eta"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("omega") ==
            Approx(jmaterial["omega"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("m") ==
            Approx(jmaterial["m"].get<double>()).epsilon(Tolerance));
    REQUIRE(
        material->template property<double>("meso_temperature") ==
        Approx(jmaterial["meso_temperature"].get<double>()).epsilon(Tolerance));
    REQUIRE(material->template property<double>("pore_fluid_density") ==
            Approx(jmaterial["pore_fluid_density"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("pore_fluid_bulk_modulus") ==
            Approx(jmaterial["pore_fluid_bulk_modulus"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_absolute_tolerance") ==
            Approx(jmaterial["rmap_absolute_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<double>("rmap_relative_tolerance") ==
            Approx(jmaterial["rmap_relative_tolerance"].get<double>())
                .epsilon(Tolerance));
    REQUIRE(material->template property<unsigned>("rmap_max_iteration") ==
            jmaterial["rmap_max_iteration"].get<unsigned>());

    // Check if state variable is initialised
    SECTION("Terracotta state variables are initialised correctly") {
      const mpm::dense_map state_variables =
          material->initialise_state_variables();

      REQUIRE(state_variables.size() == 18);
      REQUIRE(state_variables.at("pressure") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("q") == Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm") ==
              Approx(jmaterial["meso_temperature"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("tm_prev") == Approx(-1.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain0_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain1_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain2_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain3_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain4_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("elastic_strain5_prev") ==
              Approx(0.0).margin(Tolerance));
      REQUIRE(state_variables.at("packing_fraction") ==
              Approx(jmaterial["packing_fraction"].get<double>())
                  .epsilon(Tolerance));
      REQUIRE(state_variables.at("pore_pressure") ==
              Approx(0.0).margin(Tolerance));
      const std::vector<std::string> expected_state_variables = {
          "pressure",
          "q",
          "tm",
          "elastic_strain0",
          "elastic_strain1",
          "elastic_strain2",
          "elastic_strain3",
          "elastic_strain4",
          "elastic_strain5",
          "tm_prev",
          "elastic_strain0_prev",
          "elastic_strain1_prev",
          "elastic_strain2_prev",
          "elastic_strain3_prev",
          "elastic_strain4_prev",
          "elastic_strain5_prev",
          "packing_fraction",
          "pore_pressure"};

      const auto actual_state_variables = material->state_variables();
      REQUIRE(actual_state_variables == expected_state_variables);
    }
  }

  SECTION("Terracotta check stresses with strain rate") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Terracotta3D", std::move(id), jmaterial);
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

    // Top four nodes
    coords << -2.0, 2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);
    coords << 2.0, 2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);
    coords << 2.0, 2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    coords << -2.0, 2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    // Bottom four nodes
    coords << -2.0, -2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);
    coords << 2.0, -2.0, -2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);
    coords << 2.0, -2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);
    coords << -2.0, -2.0, 2.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);
    // Use a 3D, eight-node hexahedral element
    std::shared_ptr<mpm::Element<Dim>> shapefn =
        Factory<mpm::Element<Dim>>::instance()->create("ED3H8");

    const double compression_rate = 0.01;
    node0->assign_velocity_constraint(0, 2.0 * compression_rate);
    node0->assign_velocity_constraint(1, -2.0 * compression_rate);
    node0->assign_velocity_constraint(2, 2.0 * compression_rate);

    node1->assign_velocity_constraint(0, -2.0 * compression_rate);
    node1->assign_velocity_constraint(1, -2.0 * compression_rate);
    node1->assign_velocity_constraint(2, 2.0 * compression_rate);

    node2->assign_velocity_constraint(0, -2.0 * compression_rate);
    node2->assign_velocity_constraint(1, -2.0 * compression_rate);
    node2->assign_velocity_constraint(2, -2.0 * compression_rate);

    node3->assign_velocity_constraint(0, 2.0 * compression_rate);
    node3->assign_velocity_constraint(1, -2.0 * compression_rate);
    node3->assign_velocity_constraint(2, -2.0 * compression_rate);

    node4->assign_velocity_constraint(0, 2.0 * compression_rate);
    node4->assign_velocity_constraint(1, 2.0 * compression_rate);
    node4->assign_velocity_constraint(2, 2.0 * compression_rate);

    node5->assign_velocity_constraint(0, -2.0 * compression_rate);
    node5->assign_velocity_constraint(1, 2.0 * compression_rate);
    node5->assign_velocity_constraint(2, 2.0 * compression_rate);

    node6->assign_velocity_constraint(0, -2.0 * compression_rate);
    node6->assign_velocity_constraint(1, 2.0 * compression_rate);
    node6->assign_velocity_constraint(2, -2.0 * compression_rate);

    node7->assign_velocity_constraint(0, 2.0 * compression_rate);
    node7->assign_velocity_constraint(1, 2.0 * compression_rate);
    node7->assign_velocity_constraint(2, -2.0 * compression_rate);
    // Apply all assigned velocity constraints
    node0->apply_velocity_constraints();
    node1->apply_velocity_constraints();
    node2->apply_velocity_constraints();
    node3->apply_velocity_constraints();
    node4->apply_velocity_constraints();
    node5->apply_velocity_constraints();
    node6->apply_velocity_constraints();
    node7->apply_velocity_constraints();

    // Create the cell and add all eight nodes
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
    REQUIRE(particle->assign_cell(cell));
    REQUIRE(particle->assign_material(material));
    REQUIRE(particle->assign_volume(1.0));
    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE(particle->volume() == Approx(1.0).epsilon(Tolerance));
    REQUIRE(particle->mass_density() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE(particle->mass() ==
            Approx(jmaterial["density"].get<double>()).epsilon(Tolerance));
    REQUIRE_NOTHROW(particle->compute_shapefn());
    REQUIRE_NOTHROW(particle->compute_strain(dt));
    const auto strain_rate = particle->strain_rate();
    // strain rate check to confirms that the node velocities produced the
    // intended particle loading before Terracotta uses it.
    REQUIRE(strain_rate(0) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(1) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(2) == Approx(-compression_rate).epsilon(Tolerance));
    REQUIRE(strain_rate(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(strain_rate(5) == Approx(0.0).margin(Tolerance));

    // Initialise dstrain
    mpm::Material<Dim>::Vector6d dstrain = mpm::Material<Dim>::Vector6d::Zero();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    mpm::Material<Dim>::Vector6d previous_stress =
        mpm::Material<Dim>::Vector6d::Zero();
    const auto initial_state_vars = state_vars;
    const auto updated_stress = material->compute_stress(
        previous_stress, dstrain, particle.get(), &state_vars, dt);

    // Check updated Terracotta state variables
    REQUIRE(std::isfinite(state_vars.at("pressure")));
    REQUIRE(std::isfinite(state_vars.at("q")));
    REQUIRE(std::isfinite(state_vars.at("tm")));
    REQUIRE(std::isfinite(state_vars.at("packing_fraction")));
    REQUIRE(std::isfinite(state_vars.at("pore_pressure")));
    // Compression-positive volumetric strain rate
    const double expected_volumetric_strain_rate =
        -(strain_rate(0) + strain_rate(1) + strain_rate(2));
    REQUIRE(expected_volumetric_strain_rate == Approx(0.03).epsilon(Tolerance));
    // Calculate the expected pore-pressure increment
    const double expected_pore_pressure =
        jmaterial["pore_fluid_bulk_modulus"].get<double>() /
        (1.0 - jmaterial["packing_fraction"].get<double>()) *
        expected_volumetric_strain_rate * dt;
    // Dry Terracotta skeleton pressure for this loading condition
    const double expected_dry_pressure = 1.000057130800860E-4;
    // Total pressure is skeleton pressure plus pore pressure
    const double expected_total_pressure =
        expected_dry_pressure + expected_pore_pressure;
    REQUIRE(state_vars.at("pressure") ==
            Approx(expected_total_pressure).epsilon(Tolerance));

    REQUIRE(state_vars.at("q") == Approx(0.0).margin(1.E-12));
    REQUIRE(state_vars.at("tm") ==
            Approx(2.267074876870378E-7).epsilon(Tolerance));

    REQUIRE(state_vars.at("tm_prev") ==
            Approx(initial_state_vars.at("tm")).margin(1.E-20));
    REQUIRE(
        state_vars.at("packing_fraction") ==
        Approx(jmaterial["packing_fraction"].get<double>()).epsilon(Tolerance));
    REQUIRE(state_vars.at("pore_pressure") ==
            Approx(expected_pore_pressure).epsilon(Tolerance));

    // Previous elastic strains were zero before the first update
    REQUIRE(state_vars.at("elastic_strain0_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain3_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4_prev") ==
            Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5_prev") ==
            Approx(0.0).margin(Tolerance));

    // Compare updated elastic strains with MATLAB
    REQUIRE(state_vars.at("elastic_strain0") ==
            Approx(-5.999999999335107E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain1") ==
            Approx(-5.999999999335107E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain2") ==
            Approx(-5.999999999335107E-6).epsilon(Tolerance));
    REQUIRE(state_vars.at("elastic_strain3") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain4") == Approx(0.0).margin(Tolerance));
    REQUIRE(state_vars.at("elastic_strain5") == Approx(0.0).margin(Tolerance));

    // Updated elastic strains must be valid finite numbers
    REQUIRE(std::isfinite(state_vars.at("elastic_strain0")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain1")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain2")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain3")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain4")));
    REQUIRE(std::isfinite(state_vars.at("elastic_strain5")));

    // Check updated total stresses
    REQUIRE(updated_stress.size() == 6);
    // Pore pressure shifts all three normal stresses equally
    const double expected_sigma_xx =
        -1.000057130800860E-4 - expected_pore_pressure;
    const double expected_sigma_yy =
        -1.000057130800860E-4 - expected_pore_pressure;
    const double expected_sigma_zz =
        -1.000057130800860E-4 - expected_pore_pressure;
    REQUIRE(updated_stress(0) == Approx(expected_sigma_xx).epsilon(Tolerance));
    REQUIRE(updated_stress(1) == Approx(expected_sigma_yy).epsilon(Tolerance));
    REQUIRE(updated_stress(2) == Approx(expected_sigma_zz).epsilon(Tolerance));
    REQUIRE(updated_stress(3) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(4) == Approx(0.0).margin(Tolerance));
    REQUIRE(updated_stress(5) == Approx(0.0).margin(Tolerance));

    // Newmark parameters used by the MATLAB reference
    const double newmark_beta = 0.25;
    const double newmark_gamma = 0.5;
    const double lin_v = newmark_gamma / (newmark_beta * dt);

    // Compute the 3D saturated consistent tangent matrix
    const auto dep = material->compute_consistent_tangent_matrix(
        updated_stress, previous_stress, dstrain, particle.get(), &state_vars,
        dt, lin_v);

    // Check tangent matrix dimensions
    REQUIRE(dep.rows() == 6);
    REQUIRE(dep.cols() == 6);

    // Check that every tangent value is finite
    for (unsigned i = 0; i < dep.rows(); ++i)
      for (unsigned j = 0; j < dep.cols(); ++j)
        REQUIRE(std::isfinite(dep(i, j)));

    // Expected 3D saturated consistent tangent matrix
    Eigen::Matrix<double, 6, 6> dep_check;
    // clang-format off
    dep_check << 6.0273972604340773E+9, 6.0273972604137182E+9, 6.0273972604137182E+9, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 6.0273972604137182E+9, 6.0273972604340773E+9, 6.0273972604137182E+9, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 6.0273972604137182E+9, 6.0273972604137182E+9, 6.0273972604340773E+9, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.0179788900257213E-2, 0.0000000000000000E+0, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.0179788900257213E-2, 0.0000000000000000E+0,
                 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 0.0000000000000000E+0, 1.0179788900257213E-2;
    // clang-format on

    // Compare the C++ tangent matrix with the MATLAB reference
    const double TangentTolerance = 1.E-12;
    for (unsigned i = 0; i < dep.rows(); ++i) {
      for (unsigned j = 0; j < dep.cols(); ++j) {
        INFO("Tangent row = " << i << ", column = " << j);
        INFO("C++ value = " << dep(i, j));
        INFO("MATLAB value = " << dep_check(i, j));
        REQUIRE(
            dep(i, j) ==
            Approx(dep_check(i, j)).epsilon(TangentTolerance).margin(1.E-12));
      }
    }
  }
}
