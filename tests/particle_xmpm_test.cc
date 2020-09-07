#include <limits>

#include "catch.hpp"

#include "cell.h"
#include "element.h"
#include "function_base.h"
#include "hdf5_particle.h"
#include "hexahedron_element.h"
#include "linear_function.h"
#include "material.h"
#include "node.h"
#include "particle.h"
#include "particle_xmpm.h"
#include "quadrilateral_element.h"

//! \brief Check particle class for 3D case
TEST_CASE("Particle_XMPM is checked for 3D case", "[particle][3D][XMPM]") {
  // Dimension
  const unsigned Dim = 3;
  // Dimension
  const unsigned Dof = 6;
  // Number of nodes per cell
  const unsigned Nnodes = 8;
  // Number of phases
  const unsigned Nphases = 1;
  // Phase
  const unsigned phase = 0;
  // Tolerance
  const double Tolerance = 1.E-7;
  // Json property
  Json jfunctionproperties;
  jfunctionproperties["id"] = 0;
  std::vector<double> x_values{{0.0, 0.5, 1.0}};
  std::vector<double> fx_values{{0.0, 1.0, 1.0}};
  jfunctionproperties["xvalues"] = x_values;
  jfunctionproperties["fxvalues"] = fx_values;

  // math function
  std::shared_ptr<mpm::FunctionBase> mfunction =
      std::make_shared<mpm::LinearFunction>(0, jfunctionproperties);
  // Current time for traction force
  double current_time = 10.0;

  // Coordinates
  Eigen::Vector3d coords;
  coords.setZero();

  //! Check for id = 0
  SECTION("Particle id is zero") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::ParticleXMPM<Dim>>(id, coords);
    REQUIRE(particle->id() == 0);
    REQUIRE(particle->status() == true);
  }

  SECTION("Particle id is positive") {
    //! Check for id is a positive value
    mpm::Index id = std::numeric_limits<mpm::Index>::max();
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::ParticleXMPM<Dim>>(id, coords);
    REQUIRE(particle->id() == std::numeric_limits<mpm::Index>::max());
    REQUIRE(particle->status() == true);
  }

  //! Construct with id, coordinates and status
  SECTION("Particle with id, coordinates, and status") {
    mpm::Index id = 0;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::ParticleXMPM<Dim>>(id, coords, status);
    REQUIRE(particle->id() == 0);
    REQUIRE(particle->status() == true);
    particle->assign_status(false);
    REQUIRE(particle->status() == false);
  }

  //! Test particle, cell and node functions
  SECTION("Test particle, cell and node functions") {
    // Add particle
    mpm::Index id = 0;
    coords << 1.5, 1.5, 1.5;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::ParticleXMPM<Dim>>(id, coords);

    // Phase
    const unsigned phase = 0;
    // Time-step
    const double dt = 0.1;

    // Check particle coordinates
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    double levelsetphi = 1;
    REQUIRE_NOTHROW(particle->assign_levelsetphi(levelsetphi));

    // Assign hexahedron shape function
    std::shared_ptr<mpm::Element<Dim>> element =
        std::make_shared<mpm::HexahedronElement<Dim, 8>>();

    // Create cell
    auto cell = std::make_shared<mpm::Cell<Dim>>(10, Nnodes, element);
    // Add nodes
    coords << 0, 0, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);

    coords << 2, 0, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);

    coords << 2, 2, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);

    coords << 0, 2, 0;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    coords << 0, 0, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);

    coords << 2, 0, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);

    coords << 2, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node6 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);

    coords << 0, 2, 2;
    std::shared_ptr<mpm::NodeBase<Dim>> node7 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);

    std::vector<std::shared_ptr<mpm::NodeBase<Dim>>> nodes;
    nodes.emplace_back(node0);
    nodes.emplace_back(node1);
    nodes.emplace_back(node2);
    nodes.emplace_back(node3);
    nodes.emplace_back(node4);
    nodes.emplace_back(node5);
    nodes.emplace_back(node6);
    nodes.emplace_back(node7);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    cell->add_node(4, node4);
    cell->add_node(5, node5);
    cell->add_node(6, node6);
    cell->add_node(7, node7);
    REQUIRE(cell->nnodes() == 8);

    auto nodal_properties_ = std::make_shared<mpm::NodalProperties>();
    // Compute number of rows in nodal properties for vector entities
    const unsigned nrows = nodes.size() * Dim;
    // Create pool data for each property in the nodal properties struct
    // object. Properties must be named in the plural form
    nodal_properties_->create_property("mass_enrich", nodes.size(), 1);
    nodal_properties_->create_property("momenta_enrich", nrows, 1);
    nodal_properties_->create_property("internal_force_enrich", nrows, 1);
    nodal_properties_->create_property("external_force_enrich", nrows, 1);
    // Iterate over all nodes to initialise the property handle in each node
    // and assign its node id as the prop id in the nodal property data pool
    for (unsigned i = 0; i < nodes.size(); ++i)
      nodes.at(i)->initialise_discontinuity_property_handle(nodes.at(i)->id(),
                                                            nodal_properties_);
    // Initialise cell properties
    cell->initialise();

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Add cell to particle
    REQUIRE(cell->status() == false);
    // Compute reference location should throw
    REQUIRE(particle->compute_reference_location() == false);

    REQUIRE(particle->assign_cell(cell) == true);
    REQUIRE(cell->status() == true);
    REQUIRE(particle->cell_id() == 10);

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Check compute shape functions of a particle
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Check reference location
    REQUIRE(particle->compute_reference_location() == true);

    // Assign material
    unsigned mid = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(mid), jmaterial);

    // Assign material properties
    REQUIRE(particle->assign_material(material) == true);

    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());

    // Compute mass
    REQUIRE_NOTHROW(particle->compute_mass());

    // Check velocity
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = i;
    REQUIRE(particle->assign_velocity(velocity) == true);

    REQUIRE_NOTHROW(particle->map_mass_momentum_to_nodes());

    // Values of nodal mass
    std::array<double, 8> nodal_mass{125., 375.,  1125., 375.,
                                     375., 1125., 3375., 1125.};

    // Check nodal mass
    for (unsigned i = 0; i < nodes.size(); ++i)
      REQUIRE(nodes.at(i)->mass(phase) ==
              Approx(nodal_mass.at(i)).epsilon(Tolerance));
    // Check nodal enriched mass
    for (unsigned i = 0; i < nodes.size(); ++i)
      REQUIRE(nodes.at(i)->discontinuity_property("mass_enrich", 1)(0, 0) ==
              Approx(nodal_mass.at(i)).epsilon(Tolerance));

    // Values of nodal momentum
    Eigen::Matrix<double, 8, 3> nodal_momentum;
    // clang-format off
    nodal_momentum << 0.,  125.,  250.,
                      0.,  375.,  750.,
                      0., 1125., 2250.,
                      0.,  375.,  750.,
                      0.,  375.,  750.,
                      0., 1125., 2250.,
                      0., 3375., 6750.,
                      0., 1125., 2250.;
    // clang-format on

    // Check nodal momentum
    for (unsigned i = 0; i < nodal_momentum.rows(); ++i)
      for (unsigned j = 0; j < nodal_momentum.cols(); ++j)
        REQUIRE(nodes.at(i)->momentum(phase)[j] ==
                Approx(nodal_momentum(i, j)).epsilon(Tolerance));
    // Check nodal enriched momentum
    for (unsigned i = 0; i < nodal_momentum.rows(); ++i)
      for (unsigned j = 0; j < nodal_momentum.cols(); ++j)
        REQUIRE(nodes.at(i)->discontinuity_property("momenta_enrich", 3)(
                    j, 0) == Approx(nodal_momentum(i, j)).epsilon(Tolerance));

    // // Set momentum to get non-zero strain
    // // clang-format off
    // nodal_momentum << 0.,  125. * 1.,  250. * 1.,
    //                   0.,  375. * 2.,  750. * 2.,
    //                   0., 1125. * 3., 2250. * 3.,
    //                   0.,  375. * 4.,  750. * 4.,
    //                   0.,  375. * 5.,  750. * 5.,
    //                   0., 1125. * 6., 2250. * 6.,
    //                   0., 3375. * 7., 6750. * 7.,
    //                   0., 1125. * 8., 2250. * 8.;
    // // clang-format on
    // for (unsigned i = 0; i < nodes.size(); ++i)
    //   REQUIRE_NOTHROW(
    //       nodes.at(i)->update_momentum(false, phase, nodal_momentum.row(i)));

    // // nodal velocity
    // // clang-format off
    // nodal_velocity << 0., 1.,  2.,
    //                   0., 2.,  4.,
    //                   0., 3.,  6.,
    //                   0., 4.,  8.,
    //                   0., 5., 10.,
    //                   0., 6., 12.,
    //                   0., 7., 14.,
    //                   0., 8., 16.;
    // // clang-format on
    // // Compute nodal velocity
    // for (const auto& node : nodes) node->compute_velocity();
    // // Check nodal velocity
    // for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
    //   for (unsigned j = 0; j < nodal_velocity.cols(); ++j)
    //     REQUIRE(nodes.at(i)->velocity(phase)(j) ==
    //             Approx(nodal_velocity(i, j)).epsilon(Tolerance));

    // // Check pressure
    // REQUIRE(std::isnan(particle->pressure()) == true);

    // // Compute strain
    // particle->compute_strain(dt);
    // // Strain
    // Eigen::Matrix<double, 6, 1> strain;
    // strain << 0.00000, 0.07500, 0.40000, -0.02500, 0.35000, -0.05000;

    // // Check strains
    // for (unsigned i = 0; i < strain.rows(); ++i)
    //   REQUIRE(particle->strain()(i) == Approx(strain(i)).epsilon(Tolerance));

    // // Check volumetric strain at centroid
    // double volumetric_strain = 0.5;
    // REQUIRE(particle->volumetric_strain_centroid() ==
    //         Approx(volumetric_strain).epsilon(Tolerance));

    // // Check updated pressure
    // REQUIRE(std::isnan(particle->pressure()) == true);

    // // Update volume strain rate
    // REQUIRE(particle->volume() == Approx(8.0).epsilon(Tolerance));
    // particle->compute_strain(dt);
    // REQUIRE_NOTHROW(particle->update_volume());
    // REQUIRE(particle->volume() == Approx(12.0).epsilon(Tolerance));

    // // Compute stress
    // REQUIRE_NOTHROW(particle->compute_stress());

    // Eigen::Matrix<double, 6, 1> stress;
    // // clang-format off
    // stress << 2740384.6153846150,
    //           3317307.6923076920,
    //           5817307.6923076920,
    //            -96153.8461538463,
    //           1346153.8461538465,
    //           -192307.6923076927;
    // // clang-format on
    // // Check stress
    // for (unsigned i = 0; i < stress.rows(); ++i)
    //   REQUIRE(particle->stress()(i) == Approx(stress(i)).epsilon(Tolerance));

    // Check body force
    Eigen::Matrix<double, 3, 1> gravity;
    gravity << 0., 0., -9.81;

    particle->map_body_force(gravity);

    // Body force
    Eigen::Matrix<double, 8, 3> body_force;
    // clang-format off
    body_force << 0., 0.,  -1226.25,
                  0., 0.,  -3678.75,
                  0., 0., -11036.25,
                  0., 0.,  -3678.75,
                  0., 0.,  -3678.75,
                  0., 0., -11036.25,
                  0., 0., -33108.75,
                  0., 0., -11036.25;
    // clang-format on

    // Check nodal body force
    for (unsigned i = 0; i < body_force.rows(); ++i)
      for (unsigned j = 0; j < body_force.cols(); ++j)
        REQUIRE(nodes[i]->external_force(phase)[j] ==
                Approx(body_force(i, j)).epsilon(Tolerance));
    // Check nodal enriched body force
    for (unsigned i = 0; i < body_force.rows(); ++i)
      for (unsigned j = 0; j < body_force.cols(); ++j)
        REQUIRE(nodes.at(i)->discontinuity_property("external_force_enrich", 3)(
                    j, 0) == Approx(body_force(i, j)).epsilon(Tolerance));
    // // Check traction force
    // double traction = 7.68;
    // const unsigned direction = 2;
    // // Assign volume
    // REQUIRE(particle->assign_volume(0.0) == false);
    // REQUIRE(particle->assign_volume(-5.0) == false);
    // REQUIRE(particle->assign_volume(2.0) == true);
    // // Assign traction to particle
    // particle->assign_traction(direction,
    //                           mfunction->value(current_time) * traction);
    // // Map traction force
    // particle->map_traction_force();

    // // Traction force
    // Eigen::Matrix<double, 8, 3> traction_force;
    // // shapefn * volume / size_(dir) * traction
    // // clang-format off
    // traction_force << 0., 0., 0.015625 * 1.587401052 * 7.68,
    //                   0., 0., 0.046875 * 1.587401052 * 7.68,
    //                   0., 0., 0.140625 * 1.587401052 * 7.68,
    //                   0., 0., 0.046875 * 1.587401052 * 7.68,
    //                   0., 0., 0.046875 * 1.587401052 * 7.68,
    //                   0., 0., 0.140625 * 1.587401052 * 7.68,
    //                   0., 0., 0.421875 * 1.587401052 * 7.68,
    //                   0., 0., 0.140625 * 1.587401052 * 7.68;
    // // clang-format on
    // // Add previous external body force
    // traction_force += body_force;

    // // Check nodal traction force
    // for (unsigned i = 0; i < traction_force.rows(); ++i)
    //   for (unsigned j = 0; j < traction_force.cols(); ++j)
    //     REQUIRE(nodes[i]->external_force(phase)[j] ==
    //             Approx(traction_force(i, j)).epsilon(Tolerance));
    // // Reset traction
    // particle->assign_traction(direction,
    //                           mfunction->value(current_time) * -traction);
    // // Map traction force
    // particle->map_traction_force();
    // // Check nodal external force
    // for (unsigned i = 0; i < traction_force.rows(); ++i)
    //   for (unsigned j = 0; j < traction_force.cols(); ++j)
    //     REQUIRE(nodes[i]->external_force(phase)[j] ==
    //             Approx(body_force(i, j)).epsilon(Tolerance));

    // Internal force
    // Eigen::Matrix<double, 8, 3> internal_force;
    // // clang-format off
    // internal_force <<  612980.7692307689,  1141826.923076923,
    // 1742788.461538461,
    //                   -901442.3076923079,  3521634.615384615,
    //                   5420673.076923076, -2415865.384615385,
    //                   612980.7692307703,  12223557.69230769,
    //                    1935096.153846153,  108173.0769230771,
    //                    3882211.538461538,
    //                              2031250,  2079326.923076922,
    //                              -588942.3076923075,
    //                   -2127403.846153846,  6526442.307692306,
    //                   -1189903.846153845,
    //                    -5516826.92307692, -10276442.30769231,
    //                    -15685096.15384615, 6382211.538461537,
    //                    -3713942.307692308, -5805288.461538462;
    // // clang-format on

    // // Map particle internal force
    // particle->assign_volume(8.0);
    // particle->map_internal_force();

    // // Check nodal internal force
    // for (unsigned i = 0; i < internal_force.rows(); ++i)
    //   for (unsigned j = 0; j < internal_force.cols(); ++j)
    //     REQUIRE(nodes[i]->internal_force(phase)[j] ==
    //             Approx(internal_force(i, j)).epsilon(Tolerance));
    // // Check nodal enriched internal force
    // for (unsigned i = 0; i < internal_force.rows(); ++i)
    //   for (unsigned j = 0; j < internal_force.cols(); ++j)
    //     REQUIRE(nodes.at(i)->discontinuity_property("internal_force_enrich",
    //     3)(j,0) ==
    //             Approx(internal_force(i, j)).epsilon(Tolerance));
    // // Calculate nodal acceleration and velocity
    // for (const auto& node : nodes)
    //   node->compute_acceleration_velocity(phase, dt);

    // // Check nodal velocity
    // // clang-format off
    // nodal_velocity <<  490.3846153846152,  914.4615384615383,
    // 1395.249769230769,
    //                   -240.3846153846155,  941.1025641025641,
    //                   1448.531820512821, -214.7435897435898,
    //                   57.4871794871796, 1091.557461538462,
    //                    516.0256410256410,   32.8461538461539,
    //                    1042.275410256410, 541.6666666666666,
    //                    559.4871794871794, -148.032282051282,
    //                   -189.1025641025641,  586.1282051282051,
    //                   -94.75023076923067, -163.4615384615384,
    //                   -297.4871794871795, -451.7245897435898,
    //                    567.3076923076923, -322.1282051282053,
    //                    -501.0066410256412;
    // // clang-format on
    // // Check nodal velocity
    // for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
    //   for (unsigned j = 0; j < nodal_velocity.cols(); ++j)
    //     REQUIRE(nodes[i]->velocity(phase)[j] ==
    //             Approx(nodal_velocity(i, j)).epsilon(Tolerance));

    // // Check nodal acceleration
    // Eigen::Matrix<double, 8, 3> nodal_acceleration;
    // // clang-format off
    // nodal_acceleration << 4903.846153846152, 9134.615384615383,
    // 13932.49769230769,
    //                      -2403.846153846155, 9391.025641025641,
    //                      14445.31820512821, -2147.435897435898,
    //                      544.8717948717959, 10855.57461538462,
    //                       5160.256410256409, 288.461538461539,
    //                       10342.7541025641, 5416.666666666666,
    //                       5544.871794871794, -1580.32282051282,
    //                      -1891.025641025641, 5801.282051282051,
    //                      -1067.502307692307, -1634.615384615384,
    //                      -3044.871794871795, -4657.245897435898,
    //                       5673.076923076923, -3301.282051282052,
    //                       -5170.066410256411;
    // // clang-format on
    // // Check nodal acceleration
    // for (unsigned i = 0; i < nodal_acceleration.rows(); ++i)
    //   for (unsigned j = 0; j < nodal_acceleration.cols(); ++j)
    //     REQUIRE(nodes[i]->acceleration(phase)[j] ==
    //             Approx(nodal_acceleration(i, j)).epsilon(Tolerance));
    // // Approx(nodal_velocity(i, j) / dt).epsilon(Tolerance));

    // // Check original particle coordinates
    // coords << 1.5, 1.5, 1.5;
    // coordinates = particle->coordinates();
    // for (unsigned i = 0; i < coordinates.size(); ++i)
    //   REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // // Compute updated particle location
    // REQUIRE_NOTHROW(particle->compute_updated_position(dt));
    // // Check particle velocity
    // velocity << 0., 1., 1.019;
    // for (unsigned i = 0; i < velocity.size(); ++i)
    //   REQUIRE(particle->velocity()(i) ==
    //           Approx(velocity(i)).epsilon(Tolerance));

    // // Check particle displacement
    // Eigen::Vector3d displacement;
    // displacement << 0.0, 0.5875, 1.0769;
    // for (unsigned i = 0; i < displacement.size(); ++i)
    //   REQUIRE(particle->displacement()(i) ==
    //           Approx(displacement(i)).epsilon(Tolerance));

    // // Updated particle coordinate
    // coords << 1.5, 2.0875, 2.5769;
    // // Check particle coordinates
    // coordinates = particle->coordinates();
    // for (unsigned i = 0; i < coordinates.size(); ++i)
    //   REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // // Compute updated particle location based on nodal velocity
    // REQUIRE_NOTHROW(particle->compute_updated_position(dt, true));
    // // Check particle velocity
    // velocity << 0., 5.875, 10.769;
    // for (unsigned i = 0; i < velocity.size(); ++i)
    //   REQUIRE(particle->velocity()(i) ==
    //           Approx(velocity(i)).epsilon(Tolerance));

    // // Check particle displacement
    // displacement << 0.0, 1.175, 2.1538;
    // for (unsigned i = 0; i < displacement.size(); ++i)
    //   REQUIRE(particle->displacement()(i) ==
    //           Approx(displacement(i)).epsilon(Tolerance));

    // // Updated particle coordinate
    // coords << 1.5, 2.675, 3.6538;
    // // Check particle coordinates
    // coordinates = particle->coordinates();
    // for (unsigned i = 0; i < coordinates.size(); ++i)
    //   REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
  }
}
