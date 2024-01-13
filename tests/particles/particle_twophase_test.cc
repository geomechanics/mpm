#include <limits>

#include "catch.hpp"

#include "cell.h"
#include "element.h"
#include "function_base.h"
#include "hexahedron_element.h"
#include "linear_function.h"
#include "material.h"
#include "node.h"
#include "particle.h"
#include "particle_twophase.h"
#include "pod_particle.h"
#include "quadrilateral_element.h"

//! \brief Check twophase particle class for 1D case
TEST_CASE("TwoPhase Particle is checked for 1D case",
          "[particle][1D][2Phase]") {
  // Dimension
  const unsigned Dim = 1;
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

  // Coordinates
  Eigen::Matrix<double, 1, 1> coords;
  coords.setZero();

  //! Check for id = 0
  SECTION("TwoPhase Particle id is zero") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);
    REQUIRE(particle->id() == 0);
    REQUIRE(particle->status() == true);
  }

  SECTION("TwoPhase Particle id is positive") {
    //! Check for id is a positive value
    mpm::Index id = std::numeric_limits<mpm::Index>::max();
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);
    REQUIRE(particle->id() == std::numeric_limits<mpm::Index>::max());
    REQUIRE(particle->status() == true);
  }

  //! Construct with id, coordinates and status
  SECTION("TwoPhase Particle with id, coordinates, and status") {
    mpm::Index id = 0;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    REQUIRE(particle->id() == 0);
    REQUIRE(particle->status() == true);
    particle->assign_status(false);
    REQUIRE(particle->status() == false);
  }

  //! Test coordinates function
  SECTION("coordinates function is checked") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;

    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Check for coordinates being zero
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
    REQUIRE(coordinates.size() == Dim);

    // Check for negative value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = -1. * std::numeric_limits<double>::max();
    particle->assign_coordinates(coords);
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);

    // Check for positive value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = std::numeric_limits<double>::max();
    particle->assign_coordinates(coords);
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);
  }

  //! Test initialise particle stresses
  SECTION("TwoPhase Particle with initial stress") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    //! Test initialise particle stresses
    Eigen::Matrix<double, 6, 1> stress =
        Eigen::Matrix<double, 6, 1>::Constant(5.7);
    particle->initial_stress(stress);
    REQUIRE(particle->stress().size() == stress.size());
    auto pstress = particle->stress();
    for (unsigned i = 0; i < pstress.size(); ++i)
      REQUIRE(pstress[i] == Approx(stress[i]).epsilon(Tolerance));

    auto pstress_data = particle->tensor_data("stresses");
    for (unsigned i = 0; i < pstress_data.size(); ++i)
      REQUIRE(pstress_data[i] == Approx(stress[i]).epsilon(Tolerance));
  }

  //! Test particles velocity constraints
  SECTION("TwoPhase Particle with velocity constraints") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    // Apply constraints for solid phase
    particle->apply_particle_velocity_constraints(0, 10.5);
    particle->apply_particle_velocity_constraints(1, 20.5);

    // Check apply constraints
    REQUIRE(particle->velocity()(0) == Approx(10.5).epsilon(Tolerance));
    REQUIRE(particle->liquid_velocity()(0) == Approx(20.5).epsilon(Tolerance));
  }

  SECTION("Check particle properties") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Check mass
    REQUIRE(particle->mass() == Approx(0.0).epsilon(Tolerance));
    double mass = 100.5;
    particle->assign_mass(mass);
    REQUIRE(particle->mass() == Approx(100.5).epsilon(Tolerance));

    // Check stress
    Eigen::Matrix<double, 6, 1> stress;
    for (unsigned i = 0; i < stress.size(); ++i) stress(i) = 17.51;

    for (unsigned i = 0; i < stress.size(); ++i)
      REQUIRE(particle->stress()(i) == Approx(0.).epsilon(Tolerance));

    // Check velocity
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = 17.51;

    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(0.).epsilon(Tolerance));

    REQUIRE(particle->assign_velocity(velocity) == true);
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(17.51).epsilon(Tolerance));

    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Check volume
    REQUIRE(particle->volume() == Approx(2.0).epsilon(Tolerance));
    // Traction
    double traction = 65.32;
    const unsigned Direction = 0;
    // Check traction
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));
    // Try with a null math fuction ptr
    REQUIRE(particle->assign_traction(Direction, traction) == true);

    for (unsigned i = 0; i < Dim; ++i) {
      if (i == Direction)
        REQUIRE(particle->traction()(i) == Approx(traction).epsilon(Tolerance));
      else
        REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));
    }

    // Check for incorrect direction
    const unsigned wrong_dir = 2;
    REQUIRE(particle->assign_traction(wrong_dir, traction) == false);

    // Check again to ensure value hasn't been updated
    for (unsigned i = 0; i < Dim; ++i) {
      if (i == Direction)
        REQUIRE(particle->traction()(i) == Approx(traction).epsilon(Tolerance));
      else
        REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));
    }
  }

  SECTION("Check initialise particle POD") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    mpm::PODParticleTwoPhase h5_particle;
    h5_particle.id = 13;
    h5_particle.mass = 501.5;

    Eigen::Vector3d coords;
    coords << 1., 0., 0.;
    h5_particle.coord_x = coords[0];
    h5_particle.coord_y = coords[1];
    h5_particle.coord_z = coords[2];

    Eigen::Vector3d displacement;
    displacement << 0.01, 0.0, 0.0;
    h5_particle.displacement_x = displacement[0];
    h5_particle.displacement_y = displacement[1];
    h5_particle.displacement_z = displacement[2];

    Eigen::Vector3d lsize;
    lsize << 0.25, 0.0, 0.0;
    h5_particle.nsize_x = lsize[0];
    h5_particle.nsize_y = lsize[1];
    h5_particle.nsize_z = lsize[2];

    Eigen::Vector3d velocity;
    velocity << 1.5, 0., 0.;
    h5_particle.velocity_x = velocity[0];
    h5_particle.velocity_y = velocity[1];
    h5_particle.velocity_z = velocity[2];

    Eigen::Matrix<double, 6, 1> stress;
    stress << 11.5, -12.5, 13.5, 14.5, -15.5, 16.5;
    h5_particle.stress_xx = stress[0];
    h5_particle.stress_yy = stress[1];
    h5_particle.stress_zz = stress[2];
    h5_particle.tau_xy = stress[3];
    h5_particle.tau_yz = stress[4];
    h5_particle.tau_xz = stress[5];

    Eigen::Matrix<double, 6, 1> strain;
    strain << 0.115, -0.125, 0.135, 0.145, -0.155, 0.165;
    h5_particle.strain_xx = strain[0];
    h5_particle.strain_yy = strain[1];
    h5_particle.strain_zz = strain[2];
    h5_particle.gamma_xy = strain[3];
    h5_particle.gamma_yz = strain[4];
    h5_particle.gamma_xz = strain[5];

    Eigen::Matrix<double, 3, 3> deformation_gradient =
        Eigen::Matrix<double, 3, 3>::Identity();
    h5_particle.defgrad_00 = deformation_gradient(0, 0);
    h5_particle.defgrad_01 = deformation_gradient(0, 1);
    h5_particle.defgrad_02 = deformation_gradient(0, 2);
    h5_particle.defgrad_10 = deformation_gradient(1, 0);
    h5_particle.defgrad_11 = deformation_gradient(1, 1);
    h5_particle.defgrad_12 = deformation_gradient(1, 2);
    h5_particle.defgrad_20 = deformation_gradient(2, 0);
    h5_particle.defgrad_21 = deformation_gradient(2, 1);
    h5_particle.defgrad_22 = deformation_gradient(2, 2);

    Eigen::Matrix<double, 3, 3> mapping_matrix;
    mapping_matrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    h5_particle.mapping_matrix_00 = mapping_matrix(0, 0);
    h5_particle.mapping_matrix_01 = mapping_matrix(0, 1);
    h5_particle.mapping_matrix_02 = mapping_matrix(0, 2);
    h5_particle.mapping_matrix_10 = mapping_matrix(1, 0);
    h5_particle.mapping_matrix_11 = mapping_matrix(1, 1);
    h5_particle.mapping_matrix_12 = mapping_matrix(1, 2);
    h5_particle.mapping_matrix_20 = mapping_matrix(2, 0);
    h5_particle.mapping_matrix_21 = mapping_matrix(2, 1);
    h5_particle.mapping_matrix_22 = mapping_matrix(2, 2);
    h5_particle.initialise_mapping_matrix = true;

    h5_particle.status = true;

    h5_particle.cell_id = 1;

    h5_particle.volume = 2.;

    h5_particle.material_id = 1;

    h5_particle.liquid_mass = 100.1;

    Eigen::Vector3d liquid_velocity;
    liquid_velocity << 5.5, 0., 0.;
    h5_particle.liquid_velocity_x = liquid_velocity[0];
    h5_particle.liquid_velocity_y = liquid_velocity[1];
    h5_particle.liquid_velocity_z = liquid_velocity[2];

    h5_particle.porosity = 0.33;

    h5_particle.liquid_saturation = 1.;

    h5_particle.liquid_material_id = 2;

    // Reinitialise particle from POD data
    REQUIRE(particle->initialise_particle(h5_particle) == true);

    // Check particle id
    REQUIRE(particle->id() == h5_particle.id);
    // Check particle mass
    REQUIRE(particle->mass() == h5_particle.mass);
    // Check particle volume
    REQUIRE(particle->volume() == h5_particle.volume);
    // Check particle mass density
    REQUIRE(particle->mass_density() == h5_particle.mass / h5_particle.volume);
    // Check particle status
    REQUIRE(particle->status() == h5_particle.status);

    // Check for coordinates
    auto coordinates = particle->coordinates();
    REQUIRE(coordinates.size() == Dim);
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Check for displacement
    auto pdisplacement = particle->displacement();
    REQUIRE(pdisplacement.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pdisplacement(i) == Approx(displacement(i)).epsilon(Tolerance));

    // Check for size
    auto size = particle->natural_size();
    REQUIRE(size.size() == Dim);
    for (unsigned i = 0; i < size.size(); ++i)
      REQUIRE(size(i) == Approx(lsize(i)).epsilon(Tolerance));

    // Check velocity
    auto pvelocity = particle->velocity();
    REQUIRE(pvelocity.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pvelocity(i) == Approx(velocity(i)).epsilon(Tolerance));

    // Check stress
    auto pstress = particle->stress();
    REQUIRE(pstress.size() == stress.size());
    for (unsigned i = 0; i < stress.size(); ++i)
      REQUIRE(pstress(i) == Approx(stress(i)).epsilon(Tolerance));

    // Check strain
    auto pstrain = particle->strain();
    REQUIRE(pstrain.size() == strain.size());
    for (unsigned i = 0; i < strain.size(); ++i)
      REQUIRE(pstrain(i) == Approx(strain(i)).epsilon(Tolerance));

    // Check deformation gradient
    auto pdef_grad = particle->deformation_gradient();
    REQUIRE(pdef_grad.rows() == deformation_gradient.rows());
    REQUIRE(pdef_grad.cols() == deformation_gradient.cols());
    for (unsigned i = 0; i < deformation_gradient.rows(); ++i)
      for (unsigned j = 0; j < deformation_gradient.cols(); ++j)
        REQUIRE(pdef_grad(i, j) ==
                Approx(deformation_gradient(i, j)).epsilon(Tolerance));

    // Check mapping matrix
    auto map = particle->mapping_matrix();
    REQUIRE(Dim == map.rows());
    REQUIRE(Dim == map.cols());
    for (unsigned i = 0; i < map.rows(); ++i)
      for (unsigned j = 0; j < map.cols(); ++j)
        REQUIRE(mapping_matrix(i, j) == Approx(map(i, j)).epsilon(Tolerance));

    // Check cell id
    REQUIRE(particle->cell_id() == h5_particle.cell_id);

    // Check material id
    REQUIRE(particle->material_id() == h5_particle.material_id);

    // Check liquid mass
    REQUIRE(particle->liquid_mass() == h5_particle.liquid_mass);

    // Check liquid velocity
    auto pliquid_velocity = particle->liquid_velocity();
    REQUIRE(pliquid_velocity.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pliquid_velocity(i) ==
              Approx(liquid_velocity(i)).epsilon(Tolerance));

    // Check porosity
    REQUIRE(particle->porosity() == h5_particle.porosity);

    // Check liquid material id
    REQUIRE(particle->material_id(mpm::ParticlePhase::Liquid) ==
            h5_particle.liquid_material_id);

    // Write Particle POD data
    auto pod_test =
        std::static_pointer_cast<mpm::PODParticleTwoPhase>(particle->pod());

    REQUIRE(h5_particle.id == pod_test->id);
    REQUIRE(h5_particle.mass == pod_test->mass);

    REQUIRE(h5_particle.coord_x ==
            Approx(pod_test->coord_x).epsilon(Tolerance));
    REQUIRE(h5_particle.coord_y ==
            Approx(pod_test->coord_y).epsilon(Tolerance));
    REQUIRE(h5_particle.coord_z ==
            Approx(pod_test->coord_z).epsilon(Tolerance));

    REQUIRE(h5_particle.displacement_x ==
            Approx(pod_test->displacement_x).epsilon(Tolerance));
    REQUIRE(h5_particle.displacement_y ==
            Approx(pod_test->displacement_y).epsilon(Tolerance));
    REQUIRE(h5_particle.displacement_z ==
            Approx(pod_test->displacement_z).epsilon(Tolerance));

    REQUIRE(h5_particle.nsize_x == pod_test->nsize_x);
    REQUIRE(h5_particle.nsize_y == pod_test->nsize_y);
    REQUIRE(h5_particle.nsize_z == pod_test->nsize_z);

    REQUIRE(h5_particle.velocity_x ==
            Approx(pod_test->velocity_x).epsilon(Tolerance));
    REQUIRE(h5_particle.velocity_y ==
            Approx(pod_test->velocity_y).epsilon(Tolerance));
    REQUIRE(h5_particle.velocity_z ==
            Approx(pod_test->velocity_z).epsilon(Tolerance));

    REQUIRE(h5_particle.stress_xx ==
            Approx(pod_test->stress_xx).epsilon(Tolerance));
    REQUIRE(h5_particle.stress_yy ==
            Approx(pod_test->stress_yy).epsilon(Tolerance));
    REQUIRE(h5_particle.stress_zz ==
            Approx(pod_test->stress_zz).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_xy == Approx(pod_test->tau_xy).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_yz == Approx(pod_test->tau_yz).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_xz == Approx(pod_test->tau_xz).epsilon(Tolerance));

    REQUIRE(h5_particle.strain_xx ==
            Approx(pod_test->strain_xx).epsilon(Tolerance));
    REQUIRE(h5_particle.strain_yy ==
            Approx(pod_test->strain_yy).epsilon(Tolerance));
    REQUIRE(h5_particle.strain_zz ==
            Approx(pod_test->strain_zz).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_xy ==
            Approx(pod_test->gamma_xy).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_yz ==
            Approx(pod_test->gamma_yz).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_xz ==
            Approx(pod_test->gamma_xz).epsilon(Tolerance));

    REQUIRE(h5_particle.defgrad_00 ==
            Approx(pod_test->defgrad_00).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_01 ==
            Approx(pod_test->defgrad_01).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_02 ==
            Approx(pod_test->defgrad_02).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_10 ==
            Approx(pod_test->defgrad_10).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_11 ==
            Approx(pod_test->defgrad_11).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_12 ==
            Approx(pod_test->defgrad_12).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_20 ==
            Approx(pod_test->defgrad_20).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_21 ==
            Approx(pod_test->defgrad_21).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_22 ==
            Approx(pod_test->defgrad_22).epsilon(Tolerance));

    REQUIRE(h5_particle.initialise_mapping_matrix ==
            Approx(pod_test->initialise_mapping_matrix).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_00 ==
            Approx(pod_test->mapping_matrix_00).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_01 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_02 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_10 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_11 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_12 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_20 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_21 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_22 == Approx(0.0).epsilon(Tolerance));

    REQUIRE(h5_particle.status == pod_test->status);
    REQUIRE(h5_particle.cell_id == pod_test->cell_id);
    REQUIRE(h5_particle.material_id == pod_test->material_id);

    REQUIRE(h5_particle.liquid_mass ==
            Approx(pod_test->liquid_mass).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_x ==
            Approx(pod_test->liquid_velocity_x).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_y ==
            Approx(pod_test->liquid_velocity_y).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_z ==
            Approx(pod_test->liquid_velocity_z).epsilon(Tolerance));
    REQUIRE(h5_particle.porosity ==
            Approx(pod_test->porosity).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_saturation ==
            Approx(pod_test->liquid_saturation).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_material_id ==
            Approx(pod_test->liquid_material_id).epsilon(Tolerance));
  }
}

//! \brief Check twophase particle class for 2D case
TEST_CASE("TwoPhase Particle is checked for 2D case",
          "[particle][2D][2Phase]") {
  // Dimension
  const unsigned Dim = 2;
  // Degree of freedom
  const unsigned Dof = 2;
  // Number of nodes per cell
  const unsigned Nnodes = 4;
  // Number of phases
  const unsigned Nphases = 2;
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
  // Coordinates
  Eigen::Vector2d coords;
  coords.setZero();

  //! Check for id = 0
  SECTION("TwoPhase Particle id is zero") {
    mpm::Index id = 0;
    auto particle = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);
    REQUIRE(particle->id() == 0);
  }

  SECTION("TwoPhase Particle id is positive") {
    //! Check for id is a positive value
    mpm::Index id = std::numeric_limits<mpm::Index>::max();
    auto particle = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);
    REQUIRE(particle->id() == std::numeric_limits<mpm::Index>::max());
  }

  //! Test coordinates function
  SECTION("coordinates function is checked") {
    mpm::Index id = 0;
    auto particle = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Particle type
    REQUIRE(particle->type() == "P2D2PHASE");

    //! Check for coordinates being zero
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
    REQUIRE(coordinates.size() == Dim);

    //! Check for negative value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = -1. * std::numeric_limits<double>::max();
    particle->assign_coordinates(coords);
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);

    //! Check for positive value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = std::numeric_limits<double>::max();
    particle->assign_coordinates(coords);
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);
  }

  // Test assign cell to a particle
  SECTION("Add a pointer to a cell to particle") {
    // Add particle
    mpm::Index id = 0;
    coords << 0.75, 0.75;
    auto particle = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Check particle coordinates
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Element
    std::shared_ptr<mpm::Element<Dim>> element =
        std::make_shared<mpm::QuadrilateralElement<Dim, 4>>();

    // Create cell
    auto cell = std::make_shared<mpm::Cell<Dim>>(10, Nnodes, element);
    // Add nodes to cell
    coords << 0.5, 0.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);

    coords << 1.5, 0.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);

    coords << 0.5, 1.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    coords << 1.5, 1.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);

    coords << 0.5, 3.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node4 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    coords << 1.5, 3.0;
    std::shared_ptr<mpm::NodeBase<Dim>> node5 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node3);
    cell->add_node(3, node2);
    REQUIRE(cell->nnodes() == 4);

    // Initialise cell properties
    cell->initialise();

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Add cell to particle
    REQUIRE(cell->status() == false);
    // Check particle cell status
    REQUIRE(particle->cell_ptr() == false);
    // Assign cell id
    REQUIRE(particle->assign_cell_id(10) == true);
    // Require cell id
    REQUIRE(particle->cell_id() == 10);
    // Assign a very large cell id
    REQUIRE(particle->assign_cell_id(std::numeric_limits<mpm::Index>::max()) ==
            false);
    // Require cell id
    REQUIRE(particle->cell_id() == 10);
    // Assign particle to cell
    REQUIRE(particle->assign_cell(cell) == true);
    // Local coordinates
    Eigen::Vector2d xi;
    xi.fill(std::numeric_limits<double>::max());
    // Assign particle to cell
    REQUIRE(particle->assign_cell_xi(cell, xi) == false);
    // Compute reference location
    cell->is_point_in_cell(particle->coordinates(), &xi);
    // Assign particle to cell
    REQUIRE(particle->assign_cell_xi(cell, xi) == true);

    // Assign cell id again
    REQUIRE(particle->assign_cell_id(10) == false);
    // Check particle cell status
    REQUIRE(particle->cell_ptr() == true);
    // Check cell status on addition of particle
    REQUIRE(cell->status() == true);

    // Create cell
    auto cell2 = std::make_shared<mpm::Cell<Dim>>(20, Nnodes, element);

    cell2->add_node(0, node2);
    cell2->add_node(1, node3);
    cell2->add_node(2, node5);
    cell2->add_node(3, node4);
    REQUIRE(cell2->nnodes() == 4);

    // Initialise cell2 properties
    cell2->initialise();

    // Check if cell2 is initialised
    REQUIRE(cell2->is_initialised() == true);

    // Add cell2 to particle
    REQUIRE(cell2->status() == false);
    // Assign particle to cell2
    REQUIRE(particle->assign_cell(cell2) == false);
    // Check cell2 status for failed addition of particle
    REQUIRE(cell2->status() == false);
    // Check cell status because this should not have removed the particle
    REQUIRE(cell->status() == true);

    // Remove assigned cell
    particle->remove_cell();
    REQUIRE(particle->assign_cell(cell) == true);

    // Clear all particle ids
    REQUIRE(cell->nparticles() == 1);
    cell->clear_particle_ids();
    REQUIRE(cell->nparticles() == 0);
  }

  //! Test initialise particle stresses
  SECTION("TwoPhase Particle with initial stress") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    //! Test initialise particle stresses
    Eigen::Matrix<double, 6, 1> stress =
        Eigen::Matrix<double, 6, 1>::Constant(5.7);
    particle->initial_stress(stress);
    REQUIRE(particle->stress().size() == stress.size());
    auto pstress = particle->stress();
    for (unsigned i = 0; i < pstress.size(); ++i)
      REQUIRE(pstress[i] == Approx(stress[i]).epsilon(Tolerance));
  }

  // !Test initialise particle pore pressure
  SECTION("TwoPhase Particle with initial pore pressure") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    coords << 0.1, 0.2;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    // Assign liquid material
    unsigned liquid_mid = 0;
    // Initialise material
    Json jmaterial_liquid;
    jmaterial_liquid["density"] = 1000.;
    jmaterial_liquid["bulk_modulus"] = 1.0E+9;
    jmaterial_liquid["mu"] = 0.3;
    jmaterial_liquid["dynamic_viscosity"] = 0.;

    auto liquid_material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(liquid_mid), jmaterial_liquid);

    REQUIRE(particle->assign_material(liquid_material,
                                      mpm::ParticlePhase::Liquid) == true);

    Eigen::Matrix<double, Dim, 1> gravity;
    gravity << 0., -9.81;
    // Test only lefe boundary
    std::map<double, double> reference_points;
    reference_points.insert(std::make_pair<double, double>(
        static_cast<double>(0.), static_cast<double>(0.5)));
    //! Test initialise pore pressure by water table
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 0, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(2943).epsilon(Tolerance));

    // Test only right boundary
    reference_points.erase(0.);
    reference_points.insert(std::make_pair<double, double>(
        static_cast<double>(1.), static_cast<double>(0.7)));
    //! Test initialise pore pressure by water table
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 0, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(4905).epsilon(Tolerance));

    // Test both left and right boundaries
    reference_points.insert(std::make_pair<double, double>(
        static_cast<double>(0.), static_cast<double>(0.5)));
    //! Test initialise pore pressure by water table
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 0, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(3139.2).epsilon(Tolerance));
  }

  //! Test particles velocity constraints
  SECTION("TwoPhase Particle with velocity constraints") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);

    // Apply constraints
    particle->apply_particle_velocity_constraints(0, 10.5);
    particle->apply_particle_velocity_constraints(1, -12.5);
    particle->apply_particle_velocity_constraints(2, 20.5);
    particle->apply_particle_velocity_constraints(3, -22.5);

    // Check apply constraints
    REQUIRE(particle->velocity()(0) == Approx(10.5).epsilon(Tolerance));
    REQUIRE(particle->velocity()(1) == Approx(-12.5).epsilon(Tolerance));
    REQUIRE(particle->liquid_velocity()(0) == Approx(20.5).epsilon(Tolerance));
    REQUIRE(particle->liquid_velocity()(1) == Approx(-22.5).epsilon(Tolerance));
  }

  //! Test particle, cell and node functions
  SECTION("Test twophase particle, cell and node functions") {
    // Add particle
    mpm::Index id = 0;
    coords << 0.75, 0.75;
    auto particle = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Time-step
    const double dt = 0.1;

    // Check particle coordinates
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Shape function
    std::shared_ptr<mpm::Element<Dim>> element =
        std::make_shared<mpm::QuadrilateralElement<Dim, 4>>();

    // Create cell
    auto cell = std::make_shared<mpm::Cell<Dim>>(10, Nnodes, element);
    // Add nodes to cell
    coords << 0.5, 0.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);

    coords << 1.5, 0.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);

    coords << 1.5, 1.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);

    coords << 0.5, 1.5;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);
    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    REQUIRE(cell->nnodes() == 4);

    std::vector<std::shared_ptr<mpm::NodeBase<Dim>>> nodes;
    nodes.emplace_back(node0);
    nodes.emplace_back(node1);
    nodes.emplace_back(node2);
    nodes.emplace_back(node3);

    // Initialise cell properties
    cell->initialise();

    // Add cell to particle
    REQUIRE(cell->status() == false);
    // Check compute shape functions of a particle
    // TODO Assert: REQUIRE_NOTHROW(particle->compute_shapefn());
    // Compute reference location should throw
    REQUIRE(particle->compute_reference_location() == false);
    // Compute updated particle location should fail
    // TODO Assert:
    // REQUIRE_NOTHROW(particle->compute_updated_position(dt) == false);
    // Compute updated particle location from nodal velocity should fail
    // TODO Assert: REQUIRE_NOTHROW(particle->compute_updated_position(dt,
    // true)); Compute volume
    // TODO Assert: REQUIRE(particle->compute_volume() == false);
    // Update volume should fail
    // TODO Assert: REQUIRE(particle->update_volume() == false);

    REQUIRE(particle->assign_cell(cell) == true);
    REQUIRE(cell->status() == true);
    REQUIRE(particle->cell_id() == 10);

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Check compute shape functions of a particle
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Check volume
    REQUIRE(particle->volume() == Approx(2.0).epsilon(Tolerance));
    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());
    // Check volume
    REQUIRE(particle->volume() == Approx(1.0).epsilon(Tolerance));

    // Check reference location
    coords << -0.5, -0.5;
    REQUIRE(particle->compute_reference_location() == true);
    auto ref_coordinates = particle->reference_location();
    for (unsigned i = 0; i < ref_coordinates.size(); ++i)
      REQUIRE(ref_coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Assign material
    unsigned mid = 1;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["porosity"] = 0.3;
    jmaterial["k_x"] = 0.001;
    jmaterial["k_y"] = 0.001;
    jmaterial["intrinsic_permeability"] = false;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(mid), jmaterial);

    // Assign liquid material
    unsigned liquid_mid = 2;
    // Initialise material
    Json jmaterial_liquid;
    jmaterial_liquid["density"] = 1000.;
    jmaterial_liquid["bulk_modulus"] = 1.0E+9;
    jmaterial_liquid["mu"] = 0.3;
    jmaterial_liquid["dynamic_viscosity"] = 0.;

    auto liquid_material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian2D", std::move(liquid_mid), jmaterial_liquid);

    // Check compute mass before material and volume
    // TODO Assert: REQUIRE(particle->compute_mass() == false);

    // Test compute stress before material assignment
    // TODO Assert: REQUIRE(particle->compute_stress() == false);

    // Initialise nodal variables
    for (unsigned i = 0; i < nodes.size(); ++i)
      REQUIRE_NOTHROW(nodes.at(i)->initialise_twophase());

    // Assign material properties
    REQUIRE(particle->assign_material(material) == true);
    REQUIRE(particle->assign_material(liquid_material,
                                      mpm::ParticlePhase::Liquid) == true);

    // Assign porosity
    REQUIRE(particle->assign_porosity() == true);

    // Assign permeability
    REQUIRE(particle->assign_permeability() == true);

    // Check material id
    REQUIRE(particle->material_id() == 1);
    REQUIRE(particle->material_id(mpm::ParticlePhase::Liquid) == 2);

    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());

    // Compute mass
    REQUIRE_NOTHROW(particle->compute_mass());
    // Mass
    REQUIRE(particle->mass() == Approx(700.).epsilon(Tolerance));
    REQUIRE(particle->liquid_mass() == Approx(300.).epsilon(Tolerance));

    // Map particle mass to nodes
    particle->assign_mass(std::numeric_limits<double>::max());
    // TODO Assert: REQUIRE_NOTHROW(particle->map_mass_momentum_to_nodes());

    // Map particle pressure to nodes
    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);

    // Assign mass to nodes
    REQUIRE(particle->compute_reference_location() == true);
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Check velocity
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = i;
    REQUIRE(particle->assign_velocity(velocity) == true);
    REQUIRE(particle->assign_liquid_velocity(velocity) == true);
    for (unsigned i = 0; i < velocity.size(); ++i) {
      REQUIRE(particle->velocity()(i) == Approx(i).epsilon(Tolerance));
      REQUIRE(particle->liquid_velocity()(i) == Approx(i).epsilon(Tolerance));
    }

    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE_NOTHROW(particle->map_mass_momentum_to_nodes());

    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);
    REQUIRE(particle->compute_pressure_smoothing() == false);

    // Values of nodal mass
    std::array<double, 4> nodal_mass{562.5, 187.5, 62.5, 187.5};
    // Check nodal mass
    for (unsigned i = 0; i < nodes.size(); ++i) {
      // Solid phase
      REQUIRE(nodes.at(i)->mass(mpm::NodePhase::NSolid) ==
              Approx(nodal_mass.at(i) * (1 - particle->porosity()))
                  .epsilon(Tolerance));
      // Liquid phase
      REQUIRE(
          nodes.at(i)->mass(mpm::NodePhase::NLiquid) ==
          Approx(nodal_mass.at(i) * particle->porosity()).epsilon(Tolerance));
    }

    // Compute nodal velocity
    for (const auto& node : nodes) node->compute_velocity();

    // Values of nodal momentum
    Eigen::Matrix<double, 4, 2> nodal_momentum;
    // clang-format off
    nodal_momentum << 0., 562.5,
                      0., 187.5,
                      0., 62.5,
                      0., 187.5;
    // clang-format on
    // Check nodal momentum
    for (unsigned i = 0; i < nodal_momentum.rows(); ++i)
      for (unsigned j = 0; j < nodal_momentum.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes.at(i)->momentum(mpm::NodePhase::NSolid)(j) ==
                Approx(nodal_momentum(i, j) * (1 - particle->porosity()))
                    .epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes.at(i)->momentum(mpm::NodePhase::NLiquid)(j) ==
                Approx(nodal_momentum(i, j) * particle->porosity())
                    .epsilon(Tolerance));
      }
    // Values of nodal velocity
    Eigen::Matrix<double, 4, 2> nodal_velocity;
    // clang-format off
    nodal_velocity << 0., 1.,
                      0., 1.,
                      0., 1.,
                      0., 1.;
    // clang-format on
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NSolid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NLiquid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
      }

    // Set momentum to get non-zero strain
    // clang-format off
    nodal_momentum << 0., 562.5 * 1.,
                      0., 187.5 * 2.,
                      0.,  62.5 * 3.,
                      0., 187.5 * 4.;
    // clang-format on
    for (unsigned i = 0; i < nodes.size(); ++i) {
      // Solid phase
      REQUIRE_NOTHROW(nodes.at(i)->update_momentum(
          false, mpm::NodePhase::NSolid,
          nodal_momentum.row(i) * (1 - particle->porosity())));
      // Liquid phase
      REQUIRE_NOTHROW(nodes.at(i)->update_momentum(
          false, mpm::NodePhase::NLiquid,
          nodal_momentum.row(i) * particle->porosity()));
    }

    // nodal velocity
    // clang-format off
    nodal_velocity << 0., 1.,
                      0., 2.,
                      0., 3.,
                      0., 4.;
    // clang-format on
    // Compute nodal velocity
    for (const auto& node : nodes) node->compute_velocity();
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NSolid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NLiquid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
      }

    // Check pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Compute strain
    particle->compute_strain(dt);
    // Strain
    Eigen::Matrix<double, 6, 1> strain;
    strain << 0., 0.25, 0., 0.050, 0., 0.;
    // Check strains
    for (unsigned i = 0; i < strain.rows(); ++i)
      REQUIRE(particle->strain()(i) == Approx(strain(i)).epsilon(Tolerance));

    // Check updated pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Update volume strain rate
    REQUIRE(particle->volume() == Approx(1.0).epsilon(Tolerance));
    particle->compute_strain(dt);
    REQUIRE_NOTHROW(particle->update_volume());
    REQUIRE(particle->volume() == Approx(1.2).epsilon(Tolerance));

    // Compute stress
    REQUIRE_NOTHROW(particle->compute_stress(dt));

    Eigen::Matrix<double, 6, 1> stress;
    // clang-format off
    stress <<  721153.8461538460 * 2.,
              1682692.3076923075 * 2.,
               721153.8461538460 * 2.,
                96153.8461538462 * 2.,
                    0.0000000000 * 2.,
                    0.0000000000 * 2.;
    // clang-format on
    // Check stress
    for (unsigned i = 0; i < stress.rows(); ++i)
      REQUIRE(particle->stress()(i) == Approx(stress(i)).epsilon(Tolerance));

    // Compute pore_pressure
    REQUIRE_NOTHROW(particle->compute_pore_pressure(dt));
    // Check pore pressure
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(-666666666.6666667461).epsilon(Tolerance));

    // Check body force
    Eigen::Matrix<double, 2, 1> gravity;
    gravity << 0., -9.81;

    particle->map_body_force(gravity);

    // Body force
    Eigen::Matrix<double, 4, 2> body_force;
    // clang-format off
    body_force << 0., -5518.125,
                  0., -1839.375,
                  0.,  -613.125,
                  0., -1839.375;
    // clang-format on

    // Check nodal body force
    for (unsigned i = 0; i < body_force.rows(); ++i)
      for (unsigned j = 0; j < body_force.cols(); ++j)
        REQUIRE(nodes[i]->external_force(mpm::NodePhase::NSolid)[j] ==
                Approx(body_force(i, j)).epsilon(Tolerance));

    // Check traction force
    double traction = 7.68;
    const unsigned direction = 1;
    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Map traction force
    double current_time = 5.0;
    // Assign traction to particle
    particle->assign_traction(direction,
                              mfunction->value(current_time) * traction);
    particle->map_traction_force();

    // Traction force
    Eigen::Matrix<double, 4, 2> traction_force;
    // shapefn * volume / size_(dir) * traction
    // clang-format off
    traction_force << 0., 0.5625 * 1.414213562 * 7.68,
                      0., 0.1875 * 1.414213562 * 7.68,
                      0., 0.0625 * 1.414213562 * 7.68,
                      0., 0.1875 * 1.414213562 * 7.68;
    // clang-format on
    // Add previous external body force
    traction_force.noalias() += body_force;

    // Check nodal traction force
    for (unsigned i = 0; i < traction_force.rows(); ++i)
      for (unsigned j = 0; j < traction_force.cols(); ++j)
        REQUIRE(nodes[i]->external_force(mpm::NodePhase::NSolid)[j] ==
                Approx(traction_force(i, j)).epsilon(Tolerance));
    // Reset traction
    particle->assign_traction(direction,
                              -traction * mfunction->value(current_time));
    // Map traction force
    particle->map_traction_force();
    // Check nodal external force
    for (unsigned i = 0; i < traction_force.rows(); ++i)
      for (unsigned j = 0; j < traction_force.cols(); ++j)
        REQUIRE(nodes[i]->external_force(mpm::NodePhase::NSolid)[j] ==
                Approx(body_force(i, j)).epsilon(Tolerance));

    // Internal force
    Eigen::Matrix<double, 4, 2> internal_force;
    // clang-format off
    internal_force << 501225961.538461626, 502668269.230769277,
        -501033653.846153915, 167363782.051282048, -167075320.512820542,
        -167556089.743589759, 166883012.820512831, -502475961.538461566;
    // clang-format on

    // Map particle internal force
    particle->assign_volume(1.0);
    particle->map_internal_force();

    // Check nodal internal force
    for (unsigned i = 0; i < internal_force.rows(); ++i)
      for (unsigned j = 0; j < internal_force.cols(); ++j)
        REQUIRE(nodes[i]->internal_force(mpm::NodePhase::NMixture)[j] ==
                Approx(internal_force(i, j)).epsilon(Tolerance));

    // Internal force
    Eigen::Matrix<double, 4, 2> drag_force_coefficient;
    // clang-format off
    drag_force_coefficient << 496631.25,  496631.25,
                              165543.75,  165543.75,
                              55181.25,   55181.25,
                              165543.75,  165543.75;

    // Map drag force coefficient
    particle->map_drag_force_coefficient();

    // Check nodal drag force coefficient
    for (unsigned i = 0; i < drag_force_coefficient.rows(); ++i)
      for (unsigned j = 0; j < drag_force_coefficient.cols(); ++j)
        REQUIRE(nodes[i]->drag_force_coefficient()[j] ==
                Approx(drag_force_coefficient(i, j)).epsilon(Tolerance));

    // Calculate nodal acceleration and velocity
    for (const auto& node : nodes)
      node->compute_acceleration_velocity_twophase_explicit(dt);

    // Check nodal velocity
    Eigen::Matrix<double, 4, 2> nodal_liquid_velocity;
    // clang-format off
    nodal_velocity << 89200.2442002442258, 89566.563566544588,
        -267454.212454212538, 89421.0434200244199, -267600.732600732648,
        -268697.614699633734, 89053.7240537240723, -268550.094553113624;
    nodal_liquid_velocity << 88888.8888888888905, 88888.9078888889053,
        -266666.666666666686, 88889.9078888889053, -266666.666666666686,
        -266664.647666666715, 88888.8888888889051, -266663.647666666657;
    // clang-format on
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes[i]->velocity(mpm::NodePhase::NSolid)[j] ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes[i]->velocity(mpm::NodePhase::NLiquid)[j] ==
                Approx(nodal_liquid_velocity(i, j)).epsilon(Tolerance));
      }

    // Check nodal acceleration
    Eigen::Matrix<double, 4, 2> nodal_acceleration;
    Eigen::Matrix<double, 4, 2> nodal_liquid_acceleration;
    // clang-format off
    nodal_acceleration << 892002.4420024422, 895655.635665445821,
        -2674542.12454212504, 894190.434200244141, -2676007.3260073266,
        -2687006.14699633745, 890537.240537240636, -2685540.94553113589;
    nodal_liquid_acceleration << 888888.888888888876, 888879.078888888936,
        -2666666.66666666651, 888879.078888889053, -2666666.66666666698,
        -2666676.47666666703, 888888.888888888992, -2666676.47666666657;
    // clang-format on
    // Check nodal acceleration
    for (unsigned i = 0; i < nodal_acceleration.rows(); ++i)
      for (unsigned j = 0; j < nodal_acceleration.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes[i]->acceleration(mpm::NodePhase::NSolid)[j] ==
                Approx(nodal_acceleration(i, j)).epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes[i]->acceleration(mpm::NodePhase::NLiquid)[j] ==
                Approx(nodal_liquid_acceleration(i, j)).epsilon(Tolerance));
      }
    // Approx(nodal_velocity(i, j) / dt).epsilon(Tolerance));

    // Check original particle coordinates
    coords << 0.75, 0.75;
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Compute updated particle location
    REQUIRE_NOTHROW(particle->compute_updated_position(dt));
    // Check particle velocity
    velocity << 0., 0.019;
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) ==
              Approx(velocity(i)).epsilon(Tolerance));

    // Check particle displacement
    Eigen::Vector2d displacement;
    displacement << 0., 0.0894;
    for (unsigned i = 0; i < displacement.size(); ++i)
      REQUIRE(particle->displacement()(i) ==
              Approx(displacement(i)).epsilon(Tolerance));

    // Updated particle coordinate
    coords << 0.75, .8394;
    // Check particle coordinates
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Compute updated particle location from nodal velocity
    REQUIRE_NOTHROW(
        particle->compute_updated_position(dt, mpm::VelocityUpdate::PIC));
    // Check particle velocity
    velocity << 0., 0.894;
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) ==
              Approx(velocity(i)).epsilon(Tolerance));

    // Check particle displacement
    displacement << 0., 0.1788;
    for (unsigned i = 0; i < displacement.size(); ++i)
      REQUIRE(particle->displacement()(i) ==
              Approx(displacement(i)).epsilon(Tolerance));

    // Updated particle coordinate
    coords << 0.75, .9288;
    // Check particle coordinates
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Update porosity
    REQUIRE_NOTHROW(particle->update_porosity(dt));

    // Check porosity
    REQUIRE(particle->porosity() == Approx(0.44).epsilon(Tolerance));

    SECTION("TwoPhase Particle assign state variables") {
      SECTION("Assign state variable fail") {
        mid = 0;
        Json jmaterial;
        jmaterial["density"] = 1000.;
        jmaterial["youngs_modulus"] = 1.0E+7;
        jmaterial["poisson_ratio"] = 0.3;
        jmaterial["softening"] = false;
        jmaterial["friction"] = 0.;
        jmaterial["dilation"] = 0.;
        jmaterial["cohesion"] = 2000.;
        jmaterial["residual_friction"] = 0.;
        jmaterial["residual_dilation"] = 0.;
        jmaterial["residual_cohesion"] = 1000.;
        jmaterial["peak_pdstrain"] = 0.;
        jmaterial["residual_pdstrain"] = 0.;
        jmaterial["tension_cutoff"] = 0.;

        auto mc_material =
            Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
                ->create("MohrCoulomb2D", std::move(id), jmaterial);
        REQUIRE(mc_material->id() == 0);

        mpm::dense_map state_variables =
            mc_material->initialise_state_variables();
        REQUIRE(state_variables.at("phi") ==
                Approx(jmaterial["friction"]).epsilon(Tolerance));
        REQUIRE(state_variables.at("psi") ==
                Approx(jmaterial["dilation"]).epsilon(Tolerance));
        REQUIRE(state_variables.at("cohesion") ==
                Approx(jmaterial["cohesion"]).epsilon(Tolerance));
        REQUIRE(state_variables.at("epsilon") == Approx(0.).epsilon(Tolerance));
        REQUIRE(state_variables.at("rho") == Approx(0.).epsilon(Tolerance));
        REQUIRE(state_variables.at("theta") == Approx(0.).epsilon(Tolerance));
        REQUIRE(state_variables.at("pdstrain") ==
                Approx(0.).epsilon(Tolerance));

        SECTION("Assign state variables") {
          // Assign material properties
          REQUIRE(particle->assign_material(mc_material) == true);
          // Assign state variables
          REQUIRE(particle->assign_material_state_vars(state_variables,
                                                       mc_material) == true);
          // Assign and read a state variable
          REQUIRE_NOTHROW(particle->assign_state_variable("phi", 30.));
          REQUIRE(particle->state_variable("phi") == 30.);
          // Assign and read pressure though MC does not contain pressure
          REQUIRE_NOTHROW(
              particle->assign_pressure(30., mpm::ParticlePhase::Liquid));
          REQUIRE(std::isnan(particle->pressure()) == true);
        }

        SECTION("Assign state variables fail on state variables size") {
          // Assign material
          unsigned mid1 = 0;
          // Initialise material
          Json jmaterial1;
          jmaterial1["density"] = 1000.;
          jmaterial1["bulk_modulus"] = 8333333.333333333;
          jmaterial1["dynamic_viscosity"] = 8.9E-4;

          auto newtonian_material =
              Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
                  ->create("Newtonian2D", std::move(mid1), jmaterial1);

          // Assign material properties
          REQUIRE(particle->assign_material(newtonian_material) == true);
          // Assign state variables
          REQUIRE(particle->assign_material_state_vars(state_variables,
                                                       mc_material) == false);
        }

        SECTION("Assign state variables fail on material id") {
          // Assign material
          unsigned mid1 = 1;
          // Initialise material
          Json jmaterial1;
          jmaterial1["density"] = 1000.;
          jmaterial1["bulk_modulus"] = 8333333.333333333;
          jmaterial1["dynamic_viscosity"] = 8.9E-4;

          auto newtonian_material =
              Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
                  ->create("Newtonian2D", std::move(mid1), jmaterial1);

          // Assign material properties
          REQUIRE(particle->assign_material(newtonian_material) == true);
          // Assign state variables
          REQUIRE(particle->assign_material_state_vars(state_variables,
                                                       mc_material) == false);
        }
      }
    }
  }

  SECTION("Check assign material to particle") {
    // Add particle
    mpm::Index id = 0;
    coords << 0.75, 0.75;
    auto particle = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    unsigned mid = 1;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(mid), jmaterial);
    REQUIRE(material->id() == 1);

    // Check if particle can be assigned a material is null
    REQUIRE(particle->assign_material(nullptr) == false);

    // Check material id
    REQUIRE(particle->material_id() == std::numeric_limits<unsigned>::max());

    // Assign material to particle
    REQUIRE(particle->assign_material(material) == true);

    // Check material id
    REQUIRE(particle->material_id() == 1);
  }

  SECTION("Check twophase particle properties") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Check mass
    REQUIRE(particle->mass() == Approx(0.0).epsilon(Tolerance));
    double mass = 100.5;
    particle->assign_mass(mass);
    REQUIRE(particle->mass() == Approx(100.5).epsilon(Tolerance));

    // Check stress
    Eigen::Matrix<double, 6, 1> stress;
    for (unsigned i = 0; i < stress.size(); ++i) stress(i) = 17.52;

    for (unsigned i = 0; i < stress.size(); ++i)
      REQUIRE(particle->stress()(i) == Approx(0.).epsilon(Tolerance));

    // Check velocity
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = 19.745;

    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(0.).epsilon(Tolerance));

    REQUIRE(particle->assign_velocity(velocity) == true);
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(19.745).epsilon(Tolerance));

    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Check volume
    REQUIRE(particle->volume() == Approx(2.0).epsilon(Tolerance));
    // Traction
    double traction = 65.32;
    const unsigned Direction = 1;
    // Check traction
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));

    REQUIRE(particle->assign_traction(Direction, traction) == true);

    // Calculate traction force = traction * volume / spacing
    traction *= 2.0 / (std::pow(2.0, 1. / Dim));

    for (unsigned i = 0; i < Dim; ++i) {
      if (i == Direction)
        REQUIRE(particle->traction()(i) == Approx(traction).epsilon(Tolerance));
      else
        REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));
    }

    // Check for incorrect direction
    const unsigned wrong_dir = 4;
    REQUIRE(particle->assign_traction(wrong_dir, traction) == false);

    // Check again to ensure value hasn't been updated
    for (unsigned i = 0; i < Dim; ++i) {
      if (i == Direction)
        REQUIRE(particle->traction()(i) == Approx(traction).epsilon(Tolerance));
      else
        REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));
    }
  }

  // Check initialise particle from POD file
  SECTION("Check initialise particle POD") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    mpm::PODParticleTwoPhase h5_particle;
    h5_particle.id = 13;
    h5_particle.mass = 501.5;

    Eigen::Vector3d coords;
    coords << 1., 2., 0.;
    h5_particle.coord_x = coords[0];
    h5_particle.coord_y = coords[1];
    h5_particle.coord_z = coords[2];

    Eigen::Vector3d displacement;
    displacement << 0.01, 0.02, 0.0;
    h5_particle.displacement_x = displacement[0];
    h5_particle.displacement_y = displacement[1];
    h5_particle.displacement_z = displacement[2];

    Eigen::Vector3d lsize;
    lsize << 0.25, 0.5, 0.;
    h5_particle.nsize_x = lsize[0];
    h5_particle.nsize_y = lsize[1];
    h5_particle.nsize_z = lsize[2];

    Eigen::Vector3d velocity;
    velocity << 1.5, 2.5, 0.0;
    h5_particle.velocity_x = velocity[0];
    h5_particle.velocity_y = velocity[1];
    h5_particle.velocity_z = velocity[2];

    Eigen::Matrix<double, 6, 1> stress;
    stress << 11.5, -12.5, 13.5, 14.5, -15.5, 16.5;
    h5_particle.stress_xx = stress[0];
    h5_particle.stress_yy = stress[1];
    h5_particle.stress_zz = stress[2];
    h5_particle.tau_xy = stress[3];
    h5_particle.tau_yz = stress[4];
    h5_particle.tau_xz = stress[5];

    Eigen::Matrix<double, 6, 1> strain;
    strain << 0.115, -0.125, 0.135, 0.145, -0.155, 0.165;
    h5_particle.strain_xx = strain[0];
    h5_particle.strain_yy = strain[1];
    h5_particle.strain_zz = strain[2];
    h5_particle.gamma_xy = strain[3];
    h5_particle.gamma_yz = strain[4];
    h5_particle.gamma_xz = strain[5];

    Eigen::Matrix<double, 3, 3> deformation_gradient =
        Eigen::Matrix<double, 3, 3>::Identity();
    h5_particle.defgrad_00 = deformation_gradient(0, 0);
    h5_particle.defgrad_01 = deformation_gradient(0, 1);
    h5_particle.defgrad_02 = deformation_gradient(0, 2);
    h5_particle.defgrad_10 = deformation_gradient(1, 0);
    h5_particle.defgrad_11 = deformation_gradient(1, 1);
    h5_particle.defgrad_12 = deformation_gradient(1, 2);
    h5_particle.defgrad_20 = deformation_gradient(2, 0);
    h5_particle.defgrad_21 = deformation_gradient(2, 1);
    h5_particle.defgrad_22 = deformation_gradient(2, 2);

    Eigen::Matrix<double, 3, 3> mapping_matrix;
    mapping_matrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0;
    h5_particle.mapping_matrix_00 = mapping_matrix(0, 0);
    h5_particle.mapping_matrix_01 = mapping_matrix(0, 1);
    h5_particle.mapping_matrix_02 = mapping_matrix(0, 2);
    h5_particle.mapping_matrix_10 = mapping_matrix(1, 0);
    h5_particle.mapping_matrix_11 = mapping_matrix(1, 1);
    h5_particle.mapping_matrix_12 = mapping_matrix(1, 2);
    h5_particle.mapping_matrix_20 = mapping_matrix(2, 0);
    h5_particle.mapping_matrix_21 = mapping_matrix(2, 1);
    h5_particle.mapping_matrix_22 = mapping_matrix(2, 2);
    h5_particle.initialise_mapping_matrix = true;

    h5_particle.status = true;

    h5_particle.cell_id = 1;

    h5_particle.volume = 2.;

    h5_particle.material_id = 1;

    h5_particle.liquid_mass = 100.1;

    Eigen::Vector3d liquid_velocity;
    liquid_velocity << 5.5, 2.1, 0.;
    h5_particle.liquid_velocity_x = liquid_velocity[0];
    h5_particle.liquid_velocity_y = liquid_velocity[1];
    h5_particle.liquid_velocity_z = liquid_velocity[2];

    h5_particle.porosity = 0.33;

    h5_particle.liquid_saturation = 1.;

    h5_particle.liquid_material_id = 2;

    // Reinitialise particle from POD data
    REQUIRE(particle->initialise_particle(h5_particle) == true);

    // Check particle id
    REQUIRE(particle->id() == h5_particle.id);
    // Check particle mass
    REQUIRE(particle->mass() == h5_particle.mass);
    // Check particle volume
    REQUIRE(particle->volume() == h5_particle.volume);
    // Check particle mass density
    REQUIRE(particle->mass_density() == h5_particle.mass / h5_particle.volume);
    // Check particle status
    REQUIRE(particle->status() == h5_particle.status);

    // Check for coordinates
    auto coordinates = particle->coordinates();
    REQUIRE(coordinates.size() == Dim);
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Check for displacement
    auto pdisplacement = particle->displacement();
    REQUIRE(pdisplacement.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pdisplacement(i) == Approx(displacement(i)).epsilon(Tolerance));

    // Check for size
    auto size = particle->natural_size();
    REQUIRE(size.size() == Dim);
    for (unsigned i = 0; i < size.size(); ++i)
      REQUIRE(size(i) == Approx(lsize(i)).epsilon(Tolerance));

    // Check velocity
    auto pvelocity = particle->velocity();
    REQUIRE(pvelocity.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pvelocity(i) == Approx(velocity(i)).epsilon(Tolerance));

    // Check stress
    auto pstress = particle->stress();
    REQUIRE(pstress.size() == stress.size());
    for (unsigned i = 0; i < stress.size(); ++i)
      REQUIRE(pstress(i) == Approx(stress(i)).epsilon(Tolerance));

    // Check strain
    auto pstrain = particle->strain();
    REQUIRE(pstrain.size() == strain.size());
    for (unsigned i = 0; i < strain.size(); ++i)
      REQUIRE(pstrain(i) == Approx(strain(i)).epsilon(Tolerance));

    // Check deformation gradient
    auto pdef_grad = particle->deformation_gradient();
    REQUIRE(pdef_grad.rows() == deformation_gradient.rows());
    REQUIRE(pdef_grad.cols() == deformation_gradient.cols());
    for (unsigned i = 0; i < deformation_gradient.rows(); ++i)
      for (unsigned j = 0; j < deformation_gradient.cols(); ++j)
        REQUIRE(pdef_grad(i, j) ==
                Approx(deformation_gradient(i, j)).epsilon(Tolerance));

    // Check mapping matrix
    auto map = particle->mapping_matrix();
    REQUIRE(Dim == map.rows());
    REQUIRE(Dim == map.cols());
    for (unsigned i = 0; i < map.rows(); ++i)
      for (unsigned j = 0; j < map.cols(); ++j)
        REQUIRE(mapping_matrix(i, j) == Approx(map(i, j)).epsilon(Tolerance));

    // Check cell id
    REQUIRE(particle->cell_id() == h5_particle.cell_id);

    // Check material id
    REQUIRE(particle->material_id() == h5_particle.material_id);

    // Check liquid mass
    REQUIRE(particle->liquid_mass() == h5_particle.liquid_mass);

    // Check liquid velocity
    auto pliquid_velocity = particle->liquid_velocity();
    REQUIRE(pliquid_velocity.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pliquid_velocity(i) ==
              Approx(liquid_velocity(i)).epsilon(Tolerance));

    // Check porosity
    REQUIRE(particle->porosity() == h5_particle.porosity);

    // Check liquid material id
    REQUIRE(particle->material_id(mpm::ParticlePhase::Liquid) ==
            h5_particle.liquid_material_id);

    // Write Particle POD data
    auto pod_test =
        std::static_pointer_cast<mpm::PODParticleTwoPhase>(particle->pod());

    REQUIRE(h5_particle.id == pod_test->id);
    REQUIRE(h5_particle.mass == pod_test->mass);

    REQUIRE(h5_particle.coord_x ==
            Approx(pod_test->coord_x).epsilon(Tolerance));
    REQUIRE(h5_particle.coord_y ==
            Approx(pod_test->coord_y).epsilon(Tolerance));
    REQUIRE(h5_particle.coord_z ==
            Approx(pod_test->coord_z).epsilon(Tolerance));

    REQUIRE(h5_particle.displacement_x ==
            Approx(pod_test->displacement_x).epsilon(Tolerance));
    REQUIRE(h5_particle.displacement_y ==
            Approx(pod_test->displacement_y).epsilon(Tolerance));
    REQUIRE(h5_particle.displacement_z ==
            Approx(pod_test->displacement_z).epsilon(Tolerance));

    REQUIRE(h5_particle.nsize_x == pod_test->nsize_x);
    REQUIRE(h5_particle.nsize_y == pod_test->nsize_y);
    REQUIRE(h5_particle.nsize_z == pod_test->nsize_z);

    REQUIRE(h5_particle.velocity_x ==
            Approx(pod_test->velocity_x).epsilon(Tolerance));
    REQUIRE(h5_particle.velocity_y ==
            Approx(pod_test->velocity_y).epsilon(Tolerance));
    REQUIRE(h5_particle.velocity_z ==
            Approx(pod_test->velocity_z).epsilon(Tolerance));

    REQUIRE(h5_particle.stress_xx ==
            Approx(pod_test->stress_xx).epsilon(Tolerance));
    REQUIRE(h5_particle.stress_yy ==
            Approx(pod_test->stress_yy).epsilon(Tolerance));
    REQUIRE(h5_particle.stress_zz ==
            Approx(pod_test->stress_zz).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_xy == Approx(pod_test->tau_xy).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_yz == Approx(pod_test->tau_yz).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_xz == Approx(pod_test->tau_xz).epsilon(Tolerance));

    REQUIRE(h5_particle.strain_xx ==
            Approx(pod_test->strain_xx).epsilon(Tolerance));
    REQUIRE(h5_particle.strain_yy ==
            Approx(pod_test->strain_yy).epsilon(Tolerance));
    REQUIRE(h5_particle.strain_zz ==
            Approx(pod_test->strain_zz).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_xy ==
            Approx(pod_test->gamma_xy).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_yz ==
            Approx(pod_test->gamma_yz).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_xz ==
            Approx(pod_test->gamma_xz).epsilon(Tolerance));

    REQUIRE(h5_particle.defgrad_00 ==
            Approx(pod_test->defgrad_00).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_01 ==
            Approx(pod_test->defgrad_01).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_02 ==
            Approx(pod_test->defgrad_02).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_10 ==
            Approx(pod_test->defgrad_10).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_11 ==
            Approx(pod_test->defgrad_11).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_12 ==
            Approx(pod_test->defgrad_12).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_20 ==
            Approx(pod_test->defgrad_20).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_21 ==
            Approx(pod_test->defgrad_21).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_22 ==
            Approx(pod_test->defgrad_22).epsilon(Tolerance));

    REQUIRE(h5_particle.initialise_mapping_matrix ==
            Approx(pod_test->initialise_mapping_matrix).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_00 ==
            Approx(pod_test->mapping_matrix_00).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_01 ==
            Approx(pod_test->mapping_matrix_01).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_10 ==
            Approx(pod_test->mapping_matrix_10).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_11 ==
            Approx(pod_test->mapping_matrix_11).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_02 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_12 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_20 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_21 == Approx(0.0).epsilon(Tolerance));
    REQUIRE(pod_test->mapping_matrix_22 == Approx(0.0).epsilon(Tolerance));

    REQUIRE(h5_particle.status == pod_test->status);
    REQUIRE(h5_particle.cell_id == pod_test->cell_id);
    REQUIRE(h5_particle.material_id == pod_test->material_id);

    REQUIRE(h5_particle.liquid_mass ==
            Approx(pod_test->liquid_mass).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_x ==
            Approx(pod_test->liquid_velocity_x).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_y ==
            Approx(pod_test->liquid_velocity_y).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_z ==
            Approx(pod_test->liquid_velocity_z).epsilon(Tolerance));
    REQUIRE(h5_particle.porosity ==
            Approx(pod_test->porosity).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_saturation ==
            Approx(pod_test->liquid_saturation).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_material_id ==
            Approx(pod_test->liquid_material_id).epsilon(Tolerance));
  }

  // Check twophase particle's material id maping to nodes
  SECTION("Check twophase particle's material id maping to nodes") {
    // Add particle
    mpm::Index id1 = 0;
    coords << 0.75, 0.75;
    auto particle1 = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id1, coords);

    // Add particle
    mpm::Index id2 = 1;
    coords << 0.25, 0.25;
    auto particle2 = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id2, coords);

    // Element
    std::shared_ptr<mpm::Element<Dim>> element =
        std::make_shared<mpm::QuadrilateralElement<Dim, 4>>();

    // Create cell
    auto cell = std::make_shared<mpm::Cell<Dim>>(10, Nnodes, element);
    // Create vector of nodes and add them to cell
    coords << 0., 0.;
    std::shared_ptr<mpm::NodeBase<Dim>> node0 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(0, coords);

    coords << 1., 0.;
    std::shared_ptr<mpm::NodeBase<Dim>> node1 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(1, coords);

    coords << 1., 1.;
    std::shared_ptr<mpm::NodeBase<Dim>> node2 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(3, coords);

    coords << 0., 1.;
    std::shared_ptr<mpm::NodeBase<Dim>> node3 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(2, coords);
    std::vector<std::shared_ptr<mpm::NodeBase<Dim>>> nodes = {node0, node1,
                                                              node2, node3};

    for (int j = 0; j < nodes.size(); ++j) cell->add_node(j, nodes[j]);

    // Initialise cell properties and assign cell to particle
    cell->initialise();
    particle1->assign_cell(cell);
    particle2->assign_cell(cell);

    // Assign material 1
    unsigned mid1 = 0;
    // Initialise material 1
    Json jmaterial1;
    jmaterial1["density"] = 1000.;
    jmaterial1["youngs_modulus"] = 1.0E+7;
    jmaterial1["poisson_ratio"] = 0.3;

    auto material1 =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(mid1), jmaterial1);

    particle1->assign_material(material1);

    // Assign material 2
    unsigned mid2 = 1;
    // Initialise material 2
    Json jmaterial2;
    jmaterial2["density"] = 2000.;
    jmaterial2["youngs_modulus"] = 2.0E+7;
    jmaterial2["poisson_ratio"] = 0.25;

    auto material2 =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(mid2), jmaterial2);

    particle2->assign_material(material2);

    // Append particle's material id to nodes in cell
    particle1->append_material_id_to_nodes();
    particle2->append_material_id_to_nodes();

    // check if the correct amount of material ids were added to node and if
    // their indexes are correct
    std::vector<unsigned> material_ids = {0, 1};
    for (const auto& node : nodes) {
      REQUIRE(node->material_ids().size() == 2);
      auto mat_ids = node->material_ids();
      unsigned i = 0;
      for (auto mitr = mat_ids.begin(); mitr != mat_ids.end(); ++mitr, ++i)
        REQUIRE(*mitr == material_ids.at(i));
    }
  }
}

//! \brief Check twophase particle class for 3D case
TEST_CASE("TwoPhase Particle is checked for 3D case",
          "[particle][3D][2Phase]") {
  // Dimension
  const unsigned Dim = 3;
  // Dimension
  const unsigned Dof = 6;
  // Number of nodes per cell
  const unsigned Nnodes = 8;
  // Number of phases
  const unsigned Nphases = 2;
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
  SECTION("TwoPhase Particle id is zero") {
    mpm::Index id = 0;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);
    REQUIRE(particle->id() == 0);
    REQUIRE(particle->status() == true);
  }

  SECTION("TwoPhase Particle id is positive") {
    //! Check for id is a positive value
    mpm::Index id = std::numeric_limits<mpm::Index>::max();
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);
    REQUIRE(particle->id() == std::numeric_limits<mpm::Index>::max());
    REQUIRE(particle->status() == true);
  }

  //! Construct with id, coordinates and status
  SECTION("TwoPhase Particle with id, coordinates, and status") {
    mpm::Index id = 0;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    REQUIRE(particle->id() == 0);
    REQUIRE(particle->status() == true);
    particle->assign_status(false);
    REQUIRE(particle->status() == false);
  }

  //! Test coordinates function
  SECTION("coordinates function is checked") {
    mpm::Index id = 0;
    // Create particle
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Particle type
    REQUIRE(particle->type() == "P3D2PHASE");

    //! Check for coordinates being zero
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
    REQUIRE(coordinates.size() == Dim);

    //! Check for negative value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = -1. * std::numeric_limits<double>::max();
    particle->assign_coordinates(coords);
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);

    //! Check for positive value of coordinates
    for (unsigned i = 0; i < coordinates.size(); ++i)
      coords(i) = std::numeric_limits<double>::max();
    particle->assign_coordinates(coords);
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    REQUIRE(coordinates.size() == Dim);
  }

  //! Test assign cell pointer to particle
  SECTION("Add a pointer to a cell to particle") {
    // Add particle
    mpm::Index id = 0;
    coords << 1.5, 1.5, 1.5;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Check particle coordinates
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

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

    coords << 0, 0, 4;
    std::shared_ptr<mpm::NodeBase<Dim>> node8 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(4, coords);

    coords << 2, 0, 4;
    std::shared_ptr<mpm::NodeBase<Dim>> node9 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(5, coords);

    coords << 2, 2, 4;
    std::shared_ptr<mpm::NodeBase<Dim>> node10 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(6, coords);

    coords << 0, 2, 4;
    std::shared_ptr<mpm::NodeBase<Dim>> node11 =
        std::make_shared<mpm::Node<Dim, Dof, Nphases>>(7, coords);

    cell->add_node(0, node0);
    cell->add_node(1, node1);
    cell->add_node(2, node2);
    cell->add_node(3, node3);
    cell->add_node(4, node4);
    cell->add_node(5, node5);
    cell->add_node(6, node6);
    cell->add_node(7, node7);
    REQUIRE(cell->nnodes() == 8);

    // Initialise cell properties
    cell->initialise();

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Add cell to particle
    REQUIRE(cell->status() == false);
    // Check particle cell status
    REQUIRE(particle->cell_ptr() == false);
    // Assign cell id
    REQUIRE(particle->assign_cell_id(10) == true);
    // Require cell id
    REQUIRE(particle->cell_id() == 10);
    // Assign a very large cell id
    REQUIRE(particle->assign_cell_id(std::numeric_limits<mpm::Index>::max()) ==
            false);
    // Require cell id
    REQUIRE(particle->cell_id() == 10);
    // Assign particle to cell
    REQUIRE(particle->assign_cell(cell) == true);
    // Local coordinates
    Eigen::Vector3d xi;
    xi.fill(std::numeric_limits<double>::max());
    // Assign particle to cell
    REQUIRE(particle->assign_cell_xi(cell, xi) == false);
    // Compute reference location
    cell->is_point_in_cell(particle->coordinates(), &xi);
    // Assign particle to cell
    REQUIRE(particle->assign_cell_xi(cell, xi) == true);

    // Assign cell id again
    REQUIRE(particle->assign_cell_id(10) == false);
    // Check particle cell status
    REQUIRE(particle->cell_ptr() == true);
    // Check cell status on addition of particle
    REQUIRE(cell->status() == true);

    // Create cell
    auto cell2 = std::make_shared<mpm::Cell<Dim>>(20, Nnodes, element);

    cell2->add_node(0, node4);
    cell2->add_node(1, node5);
    cell2->add_node(2, node6);
    cell2->add_node(3, node7);
    cell2->add_node(4, node8);
    cell2->add_node(5, node9);
    cell2->add_node(6, node10);
    cell2->add_node(7, node11);
    REQUIRE(cell2->nnodes() == 8);

    // Initialise cell2 properties
    cell2->initialise();

    // Check if cell2 is initialised
    REQUIRE(cell2->is_initialised() == true);

    // Add cell2 to particle
    REQUIRE(cell2->status() == false);
    // Assign particle to cell2
    REQUIRE(particle->assign_cell(cell2) == false);
    // Check cell2 status for failed addition of particle
    REQUIRE(cell2->status() == false);
    // Check cell status because this should not have removed the particle
    REQUIRE(cell->status() == true);

    // Remove assigned cell
    particle->remove_cell();
    REQUIRE(particle->assign_cell(cell) == true);

    // Clear all particle ids
    REQUIRE(cell->nparticles() == 1);
    cell->clear_particle_ids();
    REQUIRE(cell->nparticles() == 0);
  }

  //! Test initialise particle stresses
  SECTION("TwoPhase Particle with initial stress") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    //! Test initialise particle stresses
    Eigen::Matrix<double, 6, 1> stress =
        Eigen::Matrix<double, 6, 1>::Constant(5.7);
    particle->initial_stress(stress);
    REQUIRE(particle->stress().size() == stress.size());
    auto pstress = particle->stress();
    for (unsigned i = 0; i < pstress.size(); ++i)
      REQUIRE(pstress[i] == Approx(stress[i]).epsilon(Tolerance));
  }

  // !Test initialise particle pore pressure
  SECTION("TwoPhase Particle with initial pore pressure") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    coords << 0.1, 0.2, 0.3;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    // Assign liquid material
    unsigned liquid_mid = 0;
    // Initialise material
    Json jmaterial_liquid;
    jmaterial_liquid["density"] = 1000.;
    jmaterial_liquid["bulk_modulus"] = 1.0E+9;
    jmaterial_liquid["mu"] = 0.3;
    jmaterial_liquid["dynamic_viscosity"] = 0.;

    auto liquid_material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(liquid_mid), jmaterial_liquid);

    REQUIRE(particle->assign_material(liquid_material,
                                      mpm::ParticlePhase::Liquid) == true);

    Eigen::Matrix<double, Dim, 1> gravity;
    gravity << 0., -9.81, 0;
    // Test only lefe boundary
    std::map<double, double> reference_points;
    reference_points.insert(std::make_pair<double, double>(
        static_cast<double>(0.), static_cast<double>(0.5)));
    //! Test initialise pore pressure by water table with x-interpolation
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 0, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(2943).epsilon(Tolerance));
    //! Test initialise pore pressure by water table with z-interpolation
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 2, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(2943).epsilon(Tolerance));

    // Test only right boundary
    reference_points.erase(0.);
    reference_points.insert(std::make_pair<double, double>(
        static_cast<double>(1.), static_cast<double>(0.7)));
    //! Test initialise pore pressure by water table x-interpolation
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 0, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(4905).epsilon(Tolerance));
    //! Test initialise pore pressure by water table with z-interpolation
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 2, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(4905).epsilon(Tolerance));

    // Test both left and right boundaries
    reference_points.insert(std::make_pair<double, double>(
        static_cast<double>(0.), static_cast<double>(0.5)));
    //! Test initialise pore pressure by water table x-interpolation
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 0, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(3139.2).epsilon(Tolerance));
    //! Test initialise pore pressure by water table with z-interpolation
    REQUIRE(particle->initialise_pore_pressure_watertable(1, 2, gravity,
                                                          reference_points));
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(3531.6).epsilon(Tolerance));
  }

  //! Test twophase particles velocity constraints
  SECTION("TwoPhase Particle with velocity constraints") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    bool status = true;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords, status);
    // Apply constraints
    particle->apply_particle_velocity_constraints(0, 10.5);
    particle->apply_particle_velocity_constraints(1, -12.5);
    particle->apply_particle_velocity_constraints(2, 14.5);
    particle->apply_particle_velocity_constraints(3, 20.5);
    particle->apply_particle_velocity_constraints(4, -22.5);
    particle->apply_particle_velocity_constraints(5, 24.5);

    // Check apply constraints
    REQUIRE(particle->velocity()(0) == Approx(10.5).epsilon(Tolerance));
    REQUIRE(particle->velocity()(1) == Approx(-12.5).epsilon(Tolerance));
    REQUIRE(particle->velocity()(2) == Approx(14.5).epsilon(Tolerance));
    REQUIRE(particle->liquid_velocity()(0) == Approx(20.5).epsilon(Tolerance));
    REQUIRE(particle->liquid_velocity()(1) == Approx(-22.5).epsilon(Tolerance));
    REQUIRE(particle->liquid_velocity()(2) == Approx(24.5).epsilon(Tolerance));
  }

  //! Test particle, cell and node functions
  SECTION("Test particle, cell and node functions") {
    // Add particle
    mpm::Index id = 0;
    coords << 1.5, 1.5, 1.5;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Time-step
    const double dt = 0.1;

    // Check particle coordinates
    auto coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

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

    // Initialise cell properties
    cell->initialise();

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Add cell to particle
    REQUIRE(cell->status() == false);
    // Check compute shape functions of a particle
    // TODO Assert: REQUIRE(particle->compute_shapefn() == false);
    // Compute reference location should throw
    REQUIRE(particle->compute_reference_location() == false);
    // Compute updated particle location should fail
    // TODO Assert: REQUIRE(particle->compute_updated_position(dt) == false);
    // Compute updated particle location from nodal velocity should fail
    // TODO Assert: REQUIRE_NOTHROW(particle->compute_updated_position(dt,
    // true));

    // Compute volume
    // TODO Assert: REQUIRE(particle->compute_volume() == false);
    // Update volume should fail
    // TODO Assert: REQUIRE(particle->update_volume() == false);

    REQUIRE(particle->assign_cell(cell) == true);
    REQUIRE(cell->status() == true);
    REQUIRE(particle->cell_id() == 10);

    // Check if cell is initialised
    REQUIRE(cell->is_initialised() == true);

    // Check compute shape functions of a particle
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Check volume
    REQUIRE(particle->volume() == Approx(2.0).epsilon(Tolerance));
    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());
    // Check volume
    REQUIRE(particle->volume() == Approx(8.0).epsilon(Tolerance));

    // Check reference location
    coords << 0.5, 0.5, 0.5;
    REQUIRE(particle->compute_reference_location() == true);
    auto ref_coordinates = particle->reference_location();
    for (unsigned i = 0; i < ref_coordinates.size(); ++i)
      REQUIRE(ref_coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Assign material
    unsigned mid = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;
    jmaterial["porosity"] = 0.3;
    jmaterial["k_x"] = 0.001;
    jmaterial["k_y"] = 0.001;
    jmaterial["k_z"] = 0.001;
    jmaterial["intrinsic_permeability"] = false;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(mid), jmaterial);

    // Assign liquid material
    unsigned liquid_mid = 1;
    // Initialise material
    Json jmaterial_liquid;
    jmaterial_liquid["density"] = 1000.;
    jmaterial_liquid["bulk_modulus"] = 1.0E+9;
    jmaterial_liquid["mu"] = 0.3;
    jmaterial_liquid["dynamic_viscosity"] = 0.;

    auto liquid_material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(liquid_mid), jmaterial_liquid);

    // Check compute mass before material and volume
    // TODO Assert: REQUIRE(particle->compute_mass() == false);

    // Test compute stress before material assignment
    // TODO Assert: REQUIRE(particle->compute_stress() == false);

    // Assign material properties
    REQUIRE(particle->assign_material(material) == true);
    REQUIRE(particle->assign_material(liquid_material,
                                      mpm::ParticlePhase::Liquid) == true);

    // Check material id from particle
    REQUIRE(particle->material_id() == 0);
    REQUIRE(particle->material_id(mpm::ParticlePhase::Liquid) == 1);

    // Assign porosity
    REQUIRE(particle->assign_porosity() == true);

    // Assign permeability
    REQUIRE(particle->assign_permeability() == true);

    // Compute volume
    REQUIRE_NOTHROW(particle->compute_volume());

    // Compute mass
    REQUIRE_NOTHROW(particle->compute_mass());
    // Mass
    REQUIRE(particle->mass() == Approx(5600.).epsilon(Tolerance));
    REQUIRE(particle->liquid_mass() == Approx(2400.).epsilon(Tolerance));

    // Map particle mass to nodes
    particle->assign_mass(std::numeric_limits<double>::max());
    // TODO Assert: REQUIRE(particle->map_mass_momentum_to_nodes() == false);

    // Map particle pressure to nodes
    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);

    // Assign mass to nodes
    REQUIRE(particle->compute_reference_location() == true);
    REQUIRE_NOTHROW(particle->compute_shapefn());

    // Check velocity
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = i;
    REQUIRE(particle->assign_velocity(velocity) == true);
    REQUIRE(particle->assign_liquid_velocity(velocity) == true);
    for (unsigned i = 0; i < velocity.size(); ++i) {
      REQUIRE(particle->velocity()(i) == Approx(i).epsilon(Tolerance));
      REQUIRE(particle->liquid_velocity()(i) == Approx(i).epsilon(Tolerance));
    }

    REQUIRE_NOTHROW(particle->compute_mass());
    REQUIRE_NOTHROW(particle->map_mass_momentum_to_nodes());

    // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);
    REQUIRE(particle->compute_pressure_smoothing() == false);

    // Values of nodal mass
    std::array<double, 8> nodal_mass{125., 375.,  1125., 375.,
                                     375., 1125., 3375., 1125.};
    // Check nodal mass
    for (unsigned i = 0; i < nodes.size(); ++i) {
      // Solid phase
      REQUIRE(nodes.at(i)->mass(mpm::NodePhase::NSolid) ==
              Approx(nodal_mass.at(i) * (1 - particle->porosity())));
      // Liquid phase
      REQUIRE(
          nodes.at(i)->mass(mpm::NodePhase::NLiquid) ==
          Approx(nodal_mass.at(i) * particle->porosity()).epsilon(Tolerance));
    }

    // Compute nodal velocity
    for (const auto& node : nodes) node->compute_velocity();

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
      for (unsigned j = 0; j < nodal_momentum.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes.at(i)->momentum(mpm::NodePhase::NSolid)(j) ==
                Approx(nodal_momentum(i, j) * (1 - particle->porosity()))
                    .epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes.at(i)->momentum(mpm::NodePhase::NLiquid)(j) ==
                Approx(nodal_momentum(i, j) * particle->porosity())
                    .epsilon(Tolerance));
      }

    // Values of nodal velocity
    Eigen::Matrix<double, 8, 3> nodal_velocity;
    // clang-format off
    nodal_velocity << 0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.,
                      0., 1., 2.;
    // clang-format on
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NSolid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NLiquid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
      }

    // Set momentum to get non-zero strain
    // clang-format off
    nodal_momentum << 0.,  125. * 1.,  250. * 1.,
                      0.,  375. * 2.,  750. * 2.,
                      0., 1125. * 3., 2250. * 3.,
                      0.,  375. * 4.,  750. * 4.,
                      0.,  375. * 5.,  750. * 5.,
                      0., 1125. * 6., 2250. * 6.,
                      0., 3375. * 7., 6750. * 7.,
                      0., 1125. * 8., 2250. * 8.;
    // clang-format on
    for (unsigned i = 0; i < nodes.size(); ++i) {
      // Solid phase
      REQUIRE_NOTHROW(nodes.at(i)->update_momentum(
          false, mpm::NodePhase::NSolid,
          nodal_momentum.row(i) * (1 - particle->porosity())));
      // Liquid phase
      REQUIRE_NOTHROW(nodes.at(i)->update_momentum(
          false, mpm::NodePhase::NLiquid,
          nodal_momentum.row(i) * particle->porosity()));
    }

    // nodal velocity
    // clang-format off
    nodal_velocity << 0., 1.,  2.,
                      0., 2.,  4.,
                      0., 3.,  6.,
                      0., 4.,  8.,
                      0., 5., 10.,
                      0., 6., 12.,
                      0., 7., 14.,
                      0., 8., 16.;
    // clang-format on
    // Compute nodal velocity
    for (const auto& node : nodes) node->compute_velocity();
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j) {
        // Solid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NSolid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
        // Liquid phase
        REQUIRE(nodes.at(i)->velocity(mpm::NodePhase::NLiquid)(j) ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));
      }

    // Check pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Compute strain
    particle->compute_strain(dt);
    // Strain
    Eigen::Matrix<double, 6, 1> strain;
    strain << 0.00000, 0.07500, 0.40000, -0.02500, 0.35000, -0.05000;

    // Check strains
    for (unsigned i = 0; i < strain.rows(); ++i)
      REQUIRE(particle->strain()(i) == Approx(strain(i)).epsilon(Tolerance));

    // Check updated pressure
    REQUIRE(std::isnan(particle->pressure()) == true);

    // Update volume strain rate
    REQUIRE(particle->volume() == Approx(8.0).epsilon(Tolerance));
    particle->compute_strain(dt);
    REQUIRE_NOTHROW(particle->update_volume());
    REQUIRE(particle->volume() == Approx(12.0).epsilon(Tolerance));

    // Compute stress
    REQUIRE_NOTHROW(particle->compute_stress(dt));

    Eigen::Matrix<double, 6, 1> stress;
    // clang-format off
    stress << 2740384.6153846150,
              3317307.6923076920,
              5817307.6923076920,
               -96153.8461538463,
              1346153.8461538465,
              -192307.6923076927;
    // clang-format on
    // Check stress
    for (unsigned i = 0; i < stress.rows(); ++i)
      REQUIRE(particle->stress()(i) == Approx(stress(i)).epsilon(Tolerance));

    // Compute pore_pressure
    REQUIRE_NOTHROW(particle->compute_pore_pressure(dt));
    // Check pore pressure
    REQUIRE(particle->pressure(mpm::ParticlePhase::Liquid) ==
            Approx(-1666666666.6666669846).epsilon(Tolerance));

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
        REQUIRE(nodes[i]->external_force(mpm::NodePhase::NSolid)[j] ==
                Approx(body_force(i, j)).epsilon(Tolerance));

    // Check traction force
    double traction = 7.68;
    const unsigned direction = 2;
    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Assign traction to particle
    particle->assign_traction(direction,
                              mfunction->value(current_time) * traction);
    // Map traction force
    particle->map_traction_force();

    // Traction force
    Eigen::Matrix<double, 8, 3> traction_force;
    // shapefn * volume / size_(dir) * traction
    // clang-format off
    traction_force << 0., 0., 0.015625 * 1.587401052 * 7.68,
                      0., 0., 0.046875 * 1.587401052 * 7.68,
                      0., 0., 0.140625 * 1.587401052 * 7.68,
                      0., 0., 0.046875 * 1.587401052 * 7.68,
                      0., 0., 0.046875 * 1.587401052 * 7.68,
                      0., 0., 0.140625 * 1.587401052 * 7.68,
                      0., 0., 0.421875 * 1.587401052 * 7.68,
                      0., 0., 0.140625 * 1.587401052 * 7.68;
    // clang-format on
    // Add previous external body force
    traction_force.noalias() += body_force;

    // Check nodal traction force
    for (unsigned i = 0; i < traction_force.rows(); ++i)
      for (unsigned j = 0; j < traction_force.cols(); ++j)
        REQUIRE(nodes[i]->external_force(mpm::NodePhase::NSolid)[j] ==
                Approx(traction_force(i, j)).epsilon(Tolerance));
    // Reset traction
    particle->assign_traction(direction,
                              mfunction->value(current_time) * -traction);
    // Map traction force
    particle->map_traction_force();
    // Check nodal external force
    for (unsigned i = 0; i < traction_force.rows(); ++i)
      for (unsigned j = 0; j < traction_force.cols(); ++j)
        REQUIRE(nodes[i]->external_force(mpm::NodePhase::NSolid)[j] ==
                Approx(body_force(i, j)).epsilon(Tolerance));

    // Internal force
    Eigen::Matrix<double, 8, 3> internal_force;
    // clang-format off
    internal_force << 417279647.435897529, 417808493.589743674,
        418409455.12820518, -417568108.974359035, 1253521634.61538506,
        1255420673.07692337, -1252415865.38461566, -1249387019.2307694,
        3762223557.69230843, 1251935096.1538465, -416558493.589743674,
        1253882211.53846169, 1252031250.00000024, 1252079326.92307711,
        -417255608.974359095, -1252127403.84615421, 3756526442.307693,
        -1251189903.84615397, -3755516826.92307711, -3760276442.307693,
        -3765685096.15384674, 3756382211.53846216, -1253713942.30769277,
        -1255805288.46153879;
    // clang-format on

    // Map particle internal force
    particle->assign_volume(8.0);
    particle->map_internal_force();

    // Check nodal internal force
    for (unsigned i = 0; i < internal_force.rows(); ++i)
      for (unsigned j = 0; j < internal_force.cols(); ++j)
        REQUIRE(nodes[i]->internal_force(mpm::NodePhase::NMixture)[j] ==
                Approx(internal_force(i, j)).epsilon(Tolerance));

    // Calculate nodal acceleration and velocity
    for (const auto& node : nodes)
      node->compute_acceleration_velocity(mpm::NodePhase::NSolid, dt);

    // Check nodal velocity
    // clang-format off
    nodal_velocity << 476891.025641025801, 477496.421245421399,
        478182.833003663225, -159073.565323565388, 477534.051282051601,
        478258.093076923338, -159036.935286935361, -158649.319902319956,
        477747.272564102721, 476927.655677655945, -158684.949938950012,
        477676.012490842724, 476964.285714285914, 476987.600732600898,
        -158945.919133089221, -159000.305250305333, 477025.230769230926,
        -158870.659059829108, -158963.675213675248, -159158.140415140486,
        -159381.479572649638, 477000.915750915941, -159193.770451770542,
        -159452.739645909722;
    // clang-format on
    // Check nodal velocity
    for (unsigned i = 0; i < nodal_velocity.rows(); ++i)
      for (unsigned j = 0; j < nodal_velocity.cols(); ++j)
        REQUIRE(nodes[i]->velocity(mpm::NodePhase::NSolid)[j] ==
                Approx(nodal_velocity(i, j)).epsilon(Tolerance));

    // Check nodal acceleration
    Eigen::Matrix<double, 8, 3> nodal_acceleration;
    // clang-format off
    nodal_acceleration << 4768910.25641025789, 4774954.21245421376,
        4781808.33003663179, -1590735.65323565388, 4775320.51282051578,
        4782540.93076923303, -1590369.35286935349, -1586523.1990231995,
        4777412.72564102709, 4769276.55677655898, -1586889.49938950012,
        4776680.12490842678, 4769642.85714285914, 4769826.00732600875,
        -1589559.19133089203, -1590003.0525030531, 4770192.30769230891,
        -1588826.59059829102, -1589636.75213675248, -1591651.40415140474,
        -1593954.79572649626, 4770009.1575091593, -1592017.70451770537,
        -1594687.39645909704;
    // clang-format on
    // Check nodal acceleration
    for (unsigned i = 0; i < nodal_acceleration.rows(); ++i)
      for (unsigned j = 0; j < nodal_acceleration.cols(); ++j)
        REQUIRE(nodes[i]->acceleration(mpm::NodePhase::NSolid)[j] ==
                Approx(nodal_acceleration(i, j)).epsilon(Tolerance));

    // Approx(nodal_velocity(i, j) / dt).epsilon(Tolerance));

    // Check original particle coordinates
    coords << 1.5, 1.5, 1.5;
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    SECTION("Particle pressure smoothing") {
      // Assign material
      unsigned mid1 = 0;
      // Initialise material
      Json jmaterial1;
      jmaterial1["density"] = 1000.;
      jmaterial1["bulk_modulus"] = 8333333.333333333;
      jmaterial1["dynamic_viscosity"] = 8.9E-4;

      auto material1 =
          Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
              ->create("Newtonian3D", std::move(mid1), jmaterial1);

      // Assign material properties
      REQUIRE(particle->assign_material(material1) == true);

      // Compute volume
      REQUIRE_NOTHROW(particle->compute_volume());

      // Compute mass
      REQUIRE_NOTHROW(particle->compute_mass());
      // Mass
      REQUIRE(particle->mass() == Approx(5600.).epsilon(Tolerance));
      REQUIRE(particle->liquid_mass() == Approx(2400.).epsilon(Tolerance));

      // Map particle mass to nodes
      particle->assign_mass(std::numeric_limits<double>::max());
      // TODO Assert: REQUIRE(particle->map_mass_momentum_to_nodes() ==
      // false);

      // Map particle pressure to nodes
      // TODO Assert: REQUIRE(particle->map_pressure_to_nodes() == false);

      // Assign mass to nodes
      REQUIRE(particle->compute_reference_location() == true);
      REQUIRE_NOTHROW(particle->compute_shapefn());

      // Check velocity
      velocity.resize(Dim);
      for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = i;
      REQUIRE(particle->assign_velocity(velocity) == true);
      for (unsigned i = 0; i < velocity.size(); ++i)
        REQUIRE(particle->velocity()(i) == Approx(i).epsilon(Tolerance));

      REQUIRE_NOTHROW(particle->compute_mass());
      REQUIRE_NOTHROW(particle->map_mass_momentum_to_nodes());

      // Check volumetric strain at centroid
      double volumetric_strain = 0.5;
      REQUIRE(particle->dvolumetric_strain() ==
              Approx(volumetric_strain).epsilon(Tolerance));

      // Compute stress
      REQUIRE_NOTHROW(particle->compute_stress(dt));

      REQUIRE(
          particle->pressure() ==
          Approx(-8333333.333333333 * volumetric_strain).epsilon(Tolerance));

      REQUIRE_NOTHROW(particle->map_pressure_to_nodes());
      REQUIRE(particle->compute_pressure_smoothing() == true);
    }

    SECTION("Particle assign state variables") {
      SECTION("Assign state variable fail") {
        mid = 0;
        Json jmaterial;
        jmaterial["density"] = 1000.;
        jmaterial["youngs_modulus"] = 1.0E+7;
        jmaterial["poisson_ratio"] = 0.3;
        jmaterial["softening"] = false;
        jmaterial["friction"] = 0.;
        jmaterial["dilation"] = 0.;
        jmaterial["cohesion"] = 2000.;
        jmaterial["residual_friction"] = 0.;
        jmaterial["residual_dilation"] = 0.;
        jmaterial["residual_cohesion"] = 1000.;
        jmaterial["peak_pdstrain"] = 0.;
        jmaterial["residual_pdstrain"] = 0.;
        jmaterial["tension_cutoff"] = 0.;

        auto mc_material =
            Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
                ->create("MohrCoulomb3D", std::move(id), jmaterial);
        REQUIRE(mc_material->id() == 0);

        mpm::dense_map state_variables =
            mc_material->initialise_state_variables();
        REQUIRE(state_variables.at("phi") ==
                Approx(jmaterial["friction"]).epsilon(Tolerance));
        REQUIRE(state_variables.at("psi") ==
                Approx(jmaterial["dilation"]).epsilon(Tolerance));
        REQUIRE(state_variables.at("cohesion") ==
                Approx(jmaterial["cohesion"]).epsilon(Tolerance));
        REQUIRE(state_variables.at("epsilon") == Approx(0.).epsilon(Tolerance));
        REQUIRE(state_variables.at("rho") == Approx(0.).epsilon(Tolerance));
        REQUIRE(state_variables.at("theta") == Approx(0.).epsilon(Tolerance));
        REQUIRE(state_variables.at("pdstrain") ==
                Approx(0.).epsilon(Tolerance));

        SECTION("Assign state variables") {
          // Assign material properties
          REQUIRE(particle->assign_material(mc_material) == true);
          // Assign state variables
          REQUIRE(particle->assign_material_state_vars(state_variables,
                                                       mc_material) == true);
          // Assign and read a state variable
          REQUIRE_NOTHROW(particle->assign_state_variable("phi", 30.));
          REQUIRE(particle->state_variable("phi") == 30.);
          // Assign and read pressure though MC does not contain pressure
          REQUIRE_NOTHROW(
              particle->assign_pressure(30., mpm::ParticlePhase::Liquid));
          REQUIRE(std::isnan(particle->pressure()) == true);
        }

        SECTION("Assign state variables fail on state variables size") {
          // Assign material
          unsigned mid1 = 0;
          // Initialise material
          Json jmaterial1;
          jmaterial1["density"] = 1000.;
          jmaterial1["bulk_modulus"] = 8333333.333333333;
          jmaterial1["dynamic_viscosity"] = 8.9E-4;

          auto newtonian_material =
              Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
                  ->create("Newtonian3D", std::move(mid1), jmaterial1);

          // Assign material properties
          REQUIRE(particle->assign_material(newtonian_material) == true);
          // Assign state variables
          REQUIRE(particle->assign_material_state_vars(state_variables,
                                                       mc_material) == false);
        }

        SECTION("Assign state variables fail on material id") {
          // Assign material
          unsigned mid1 = 1;
          // Initialise material
          Json jmaterial1;
          jmaterial1["density"] = 1000.;
          jmaterial1["bulk_modulus"] = 8333333.333333333;
          jmaterial1["dynamic_viscosity"] = 8.9E-4;

          auto newtonian_material =
              Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()
                  ->create("Newtonian3D", std::move(mid1), jmaterial1);

          // Assign material properties
          REQUIRE(particle->assign_material(newtonian_material) == true);
          // Assign state variables
          REQUIRE(particle->assign_material_state_vars(state_variables,
                                                       mc_material) == false);
        }
      }
    }

    // Compute updated particle location
    REQUIRE_NOTHROW(particle->compute_updated_position(dt));
    // Check particle velocity
    velocity << 0., 1., 0.5985714286;
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) ==
              Approx(velocity(i)).epsilon(Tolerance));

    // Check particle displacement
    Eigen::Vector3d displacement;
    displacement << 0.0, 0.5875, 1.0348571429;
    for (unsigned i = 0; i < displacement.size(); ++i)
      REQUIRE(particle->displacement()(i) ==
              Approx(displacement(i)).epsilon(Tolerance));

    // Updated particle coordinate
    coords << 1.5, 2.0875, 2.5348571429;
    // Check particle coordinates
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Compute updated particle location based on nodal velocity
    REQUIRE_NOTHROW(
        particle->compute_updated_position(dt, mpm::VelocityUpdate::PIC));
    // Check particle velocity
    velocity << 0., 5.875, 10.3485714286;
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) ==
              Approx(velocity(i)).epsilon(Tolerance));

    // Check particle displacement
    displacement << 0.0, 1.175, 2.0697142857;
    for (unsigned i = 0; i < displacement.size(); ++i)
      REQUIRE(particle->displacement()(i) ==
              Approx(displacement(i)).epsilon(Tolerance));

    // Updated particle coordinate
    coords << 1.5, 2.675, 3.5697142857;
    // Check particle coordinates
    coordinates = particle->coordinates();
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
  }

  SECTION("Check assign material to particle") {
    // Add particle
    mpm::Index id = 0;
    coords << 0.75, 0.75, 0.75;
    auto particle = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    unsigned mid = 1;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["youngs_modulus"] = 1.0E+7;
    jmaterial["poisson_ratio"] = 0.3;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(mid), jmaterial);
    REQUIRE(material->id() == 1);

    // Check if particle can be assigned a null material
    REQUIRE(particle->assign_material(nullptr) == false);
    // Check material id
    REQUIRE(particle->material_id() == std::numeric_limits<unsigned>::max());

    // Assign material to particle
    REQUIRE(particle->assign_material(material) == true);
    // Check material id
    REQUIRE(particle->material_id() == 1);
  }

  SECTION("Check particle properties") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    // Check mass
    REQUIRE(particle->mass() == Approx(0.0).epsilon(Tolerance));
    double mass = 100.5;
    particle->assign_mass(mass);
    REQUIRE(particle->mass() == Approx(100.5).epsilon(Tolerance));

    // Check stress
    Eigen::Matrix<double, 6, 1> stress;
    for (unsigned i = 0; i < stress.size(); ++i) stress(i) = 1.;

    for (unsigned i = 0; i < stress.size(); ++i)
      REQUIRE(particle->stress()(i) == Approx(0.).epsilon(Tolerance));

    // Check velocity
    Eigen::VectorXd velocity;
    velocity.resize(Dim);
    for (unsigned i = 0; i < velocity.size(); ++i) velocity(i) = 17.51;

    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(0.).epsilon(Tolerance));

    REQUIRE(particle->assign_velocity(velocity) == true);
    for (unsigned i = 0; i < velocity.size(); ++i)
      REQUIRE(particle->velocity()(i) == Approx(17.51).epsilon(Tolerance));

    // Assign volume
    REQUIRE(particle->assign_volume(0.0) == false);
    REQUIRE(particle->assign_volume(-5.0) == false);
    REQUIRE(particle->assign_volume(2.0) == true);
    // Check volume
    REQUIRE(particle->volume() == Approx(2.0).epsilon(Tolerance));
    // Traction
    double traction = 65.32;
    const unsigned Direction = 1;
    // Check traction
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));

    REQUIRE(particle->assign_traction(Direction, traction) == true);

    // Calculate traction force = traction * volume / spacing
    traction *= 2.0 / (std::pow(2.0, 1. / Dim));

    for (unsigned i = 0; i < Dim; ++i) {
      if (i == Direction)
        REQUIRE(particle->traction()(i) == Approx(traction).epsilon(Tolerance));
      else
        REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));
    }

    // Check for incorrect direction
    const unsigned wrong_dir = 6;
    REQUIRE(particle->assign_traction(wrong_dir, traction) == false);

    // Check again to ensure value hasn't been updated
    for (unsigned i = 0; i < Dim; ++i) {
      if (i == Direction)
        REQUIRE(particle->traction()(i) == Approx(traction).epsilon(Tolerance));
      else
        REQUIRE(particle->traction()(i) == Approx(0.).epsilon(Tolerance));
    }
  }

  // Check initialise particle from POD file
  SECTION("Check initialise particle POD") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::TwoPhaseParticle<Dim>>(id, coords);

    mpm::PODParticleTwoPhase h5_particle;
    h5_particle.id = 13;
    h5_particle.mass = 501.5;

    Eigen::Vector3d coords;
    coords << 1., 2., 3.;
    h5_particle.coord_x = coords[0];
    h5_particle.coord_y = coords[1];
    h5_particle.coord_z = coords[2];

    Eigen::Vector3d displacement;
    displacement << 0.01, 0.02, 0.03;
    h5_particle.displacement_x = displacement[0];
    h5_particle.displacement_y = displacement[1];
    h5_particle.displacement_z = displacement[2];

    Eigen::Vector3d lsize;
    lsize << 0.25, 0.5, 0.75;
    h5_particle.nsize_x = lsize[0];
    h5_particle.nsize_y = lsize[1];
    h5_particle.nsize_z = lsize[2];

    Eigen::Vector3d velocity;
    velocity << 1.5, 2.5, 3.5;
    h5_particle.velocity_x = velocity[0];
    h5_particle.velocity_y = velocity[1];
    h5_particle.velocity_z = velocity[2];

    Eigen::Matrix<double, 6, 1> stress;
    stress << 11.5, -12.5, 13.5, 14.5, -15.5, 16.5;
    h5_particle.stress_xx = stress[0];
    h5_particle.stress_yy = stress[1];
    h5_particle.stress_zz = stress[2];
    h5_particle.tau_xy = stress[3];
    h5_particle.tau_yz = stress[4];
    h5_particle.tau_xz = stress[5];

    Eigen::Matrix<double, 6, 1> strain;
    strain << 0.115, -0.125, 0.135, 0.145, -0.155, 0.165;
    h5_particle.strain_xx = strain[0];
    h5_particle.strain_yy = strain[1];
    h5_particle.strain_zz = strain[2];
    h5_particle.gamma_xy = strain[3];
    h5_particle.gamma_yz = strain[4];
    h5_particle.gamma_xz = strain[5];

    Eigen::Matrix<double, 3, 3> deformation_gradient =
        Eigen::Matrix<double, 3, 3>::Identity();
    h5_particle.defgrad_00 = deformation_gradient(0, 0);
    h5_particle.defgrad_01 = deformation_gradient(0, 1);
    h5_particle.defgrad_02 = deformation_gradient(0, 2);
    h5_particle.defgrad_10 = deformation_gradient(1, 0);
    h5_particle.defgrad_11 = deformation_gradient(1, 1);
    h5_particle.defgrad_12 = deformation_gradient(1, 2);
    h5_particle.defgrad_20 = deformation_gradient(2, 0);
    h5_particle.defgrad_21 = deformation_gradient(2, 1);
    h5_particle.defgrad_22 = deformation_gradient(2, 2);

    Eigen::Matrix<double, 3, 3> mapping_matrix;
    mapping_matrix << 0.115, -0.125, 0.135, 0.145, -0.155, 0.165, 0.145, -0.155,
        0.165;
    h5_particle.mapping_matrix_00 = mapping_matrix(0, 0);
    h5_particle.mapping_matrix_01 = mapping_matrix(0, 1);
    h5_particle.mapping_matrix_02 = mapping_matrix(0, 2);
    h5_particle.mapping_matrix_10 = mapping_matrix(1, 0);
    h5_particle.mapping_matrix_11 = mapping_matrix(1, 1);
    h5_particle.mapping_matrix_12 = mapping_matrix(1, 2);
    h5_particle.mapping_matrix_20 = mapping_matrix(2, 0);
    h5_particle.mapping_matrix_21 = mapping_matrix(2, 1);
    h5_particle.mapping_matrix_22 = mapping_matrix(2, 2);
    h5_particle.initialise_mapping_matrix = true;

    h5_particle.status = true;

    h5_particle.cell_id = 1;

    h5_particle.volume = 2.;

    h5_particle.material_id = 1;

    h5_particle.liquid_mass = 100.1;

    Eigen::Vector3d liquid_velocity;
    liquid_velocity << 5.5, 3.12, 2.1;
    h5_particle.liquid_velocity_x = liquid_velocity[0];
    h5_particle.liquid_velocity_y = liquid_velocity[1];
    h5_particle.liquid_velocity_z = liquid_velocity[2];

    h5_particle.porosity = 0.33;

    h5_particle.liquid_saturation = 1.;

    h5_particle.liquid_material_id = 2;

    // Reinitialise particle from POD data
    REQUIRE(particle->initialise_particle(h5_particle) == true);

    // Check particle id
    REQUIRE(particle->id() == h5_particle.id);
    // Check particle mass
    REQUIRE(particle->mass() == h5_particle.mass);
    // Check particle volume
    REQUIRE(particle->volume() == h5_particle.volume);
    // Check particle mass density
    REQUIRE(particle->mass_density() == h5_particle.mass / h5_particle.volume);
    // Check particle status
    REQUIRE(particle->status() == h5_particle.status);

    // Check for coordinates
    auto coordinates = particle->coordinates();
    REQUIRE(coordinates.size() == Dim);
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));
    REQUIRE(coordinates.size() == Dim);

    // Check for displacement
    auto pdisplacement = particle->displacement();
    REQUIRE(pdisplacement.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pdisplacement(i) == Approx(displacement(i)).epsilon(Tolerance));

    // Check for size
    auto size = particle->natural_size();
    REQUIRE(size.size() == Dim);
    for (unsigned i = 0; i < size.size(); ++i)
      REQUIRE(size(i) == Approx(lsize(i)).epsilon(Tolerance));

    // Check velocity
    auto pvelocity = particle->velocity();
    REQUIRE(pvelocity.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pvelocity(i) == Approx(velocity(i)).epsilon(Tolerance));

    // Check stress
    auto pstress = particle->stress();
    REQUIRE(pstress.size() == stress.size());
    for (unsigned i = 0; i < stress.size(); ++i)
      REQUIRE(pstress(i) == Approx(stress(i)).epsilon(Tolerance));

    // Check strain
    auto pstrain = particle->strain();
    REQUIRE(pstrain.size() == strain.size());
    for (unsigned i = 0; i < strain.size(); ++i)
      REQUIRE(pstrain(i) == Approx(strain(i)).epsilon(Tolerance));

    // Check deformation gradient
    auto pdef_grad = particle->deformation_gradient();
    REQUIRE(pdef_grad.rows() == deformation_gradient.rows());
    REQUIRE(pdef_grad.cols() == deformation_gradient.cols());
    for (unsigned i = 0; i < deformation_gradient.rows(); ++i)
      for (unsigned j = 0; j < deformation_gradient.cols(); ++j)
        REQUIRE(pdef_grad(i, j) ==
                Approx(deformation_gradient(i, j)).epsilon(Tolerance));

    // Check mapping matrix
    auto map = particle->mapping_matrix();
    REQUIRE(Dim == map.rows());
    REQUIRE(Dim == map.cols());
    for (unsigned i = 0; i < map.rows(); ++i)
      for (unsigned j = 0; j < map.cols(); ++j)
        REQUIRE(mapping_matrix(i, j) == Approx(map(i, j)).epsilon(Tolerance));

    // Check cell id
    REQUIRE(particle->cell_id() == h5_particle.cell_id);

    // Check material id
    REQUIRE(particle->material_id() == h5_particle.material_id);

    // Check liquid mass
    REQUIRE(particle->liquid_mass() == h5_particle.liquid_mass);

    // Check liquid velocity
    auto pliquid_velocity = particle->liquid_velocity();
    REQUIRE(pliquid_velocity.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pliquid_velocity(i) ==
              Approx(liquid_velocity(i)).epsilon(Tolerance));

    // Check porosity
    REQUIRE(particle->porosity() == h5_particle.porosity);

    // Check liquid material id
    REQUIRE(particle->material_id(mpm::ParticlePhase::Liquid) ==
            h5_particle.liquid_material_id);

    // Write Particle POD data
    auto pod_test =
        std::static_pointer_cast<mpm::PODParticleTwoPhase>(particle->pod());

    REQUIRE(h5_particle.id == pod_test->id);
    REQUIRE(h5_particle.mass == pod_test->mass);

    REQUIRE(h5_particle.coord_x ==
            Approx(pod_test->coord_x).epsilon(Tolerance));
    REQUIRE(h5_particle.coord_y ==
            Approx(pod_test->coord_y).epsilon(Tolerance));
    REQUIRE(h5_particle.coord_z ==
            Approx(pod_test->coord_z).epsilon(Tolerance));

    REQUIRE(h5_particle.displacement_x ==
            Approx(pod_test->displacement_x).epsilon(Tolerance));
    REQUIRE(h5_particle.displacement_y ==
            Approx(pod_test->displacement_y).epsilon(Tolerance));
    REQUIRE(h5_particle.displacement_z ==
            Approx(pod_test->displacement_z).epsilon(Tolerance));

    REQUIRE(h5_particle.nsize_x == pod_test->nsize_x);
    REQUIRE(h5_particle.nsize_y == pod_test->nsize_y);
    REQUIRE(h5_particle.nsize_z == pod_test->nsize_z);

    REQUIRE(h5_particle.velocity_x ==
            Approx(pod_test->velocity_x).epsilon(Tolerance));
    REQUIRE(h5_particle.velocity_y ==
            Approx(pod_test->velocity_y).epsilon(Tolerance));
    REQUIRE(h5_particle.velocity_z ==
            Approx(pod_test->velocity_z).epsilon(Tolerance));

    REQUIRE(h5_particle.stress_xx ==
            Approx(pod_test->stress_xx).epsilon(Tolerance));
    REQUIRE(h5_particle.stress_yy ==
            Approx(pod_test->stress_yy).epsilon(Tolerance));
    REQUIRE(h5_particle.stress_zz ==
            Approx(pod_test->stress_zz).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_xy == Approx(pod_test->tau_xy).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_yz == Approx(pod_test->tau_yz).epsilon(Tolerance));
    REQUIRE(h5_particle.tau_xz == Approx(pod_test->tau_xz).epsilon(Tolerance));

    REQUIRE(h5_particle.strain_xx ==
            Approx(pod_test->strain_xx).epsilon(Tolerance));
    REQUIRE(h5_particle.strain_yy ==
            Approx(pod_test->strain_yy).epsilon(Tolerance));
    REQUIRE(h5_particle.strain_zz ==
            Approx(pod_test->strain_zz).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_xy ==
            Approx(pod_test->gamma_xy).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_yz ==
            Approx(pod_test->gamma_yz).epsilon(Tolerance));
    REQUIRE(h5_particle.gamma_xz ==
            Approx(pod_test->gamma_xz).epsilon(Tolerance));

    REQUIRE(h5_particle.defgrad_00 ==
            Approx(pod_test->defgrad_00).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_01 ==
            Approx(pod_test->defgrad_01).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_02 ==
            Approx(pod_test->defgrad_02).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_10 ==
            Approx(pod_test->defgrad_10).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_11 ==
            Approx(pod_test->defgrad_11).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_12 ==
            Approx(pod_test->defgrad_12).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_20 ==
            Approx(pod_test->defgrad_20).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_21 ==
            Approx(pod_test->defgrad_21).epsilon(Tolerance));
    REQUIRE(h5_particle.defgrad_22 ==
            Approx(pod_test->defgrad_22).epsilon(Tolerance));

    REQUIRE(h5_particle.initialise_mapping_matrix ==
            Approx(pod_test->initialise_mapping_matrix).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_00 ==
            Approx(pod_test->mapping_matrix_00).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_01 ==
            Approx(pod_test->mapping_matrix_01).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_02 ==
            Approx(pod_test->mapping_matrix_02).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_10 ==
            Approx(pod_test->mapping_matrix_10).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_11 ==
            Approx(pod_test->mapping_matrix_11).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_12 ==
            Approx(pod_test->mapping_matrix_12).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_20 ==
            Approx(pod_test->mapping_matrix_20).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_21 ==
            Approx(pod_test->mapping_matrix_21).epsilon(Tolerance));
    REQUIRE(h5_particle.mapping_matrix_22 ==
            Approx(pod_test->mapping_matrix_22).epsilon(Tolerance));

    REQUIRE(h5_particle.status == pod_test->status);
    REQUIRE(h5_particle.cell_id == pod_test->cell_id);
    REQUIRE(h5_particle.material_id == pod_test->material_id);

    REQUIRE(h5_particle.liquid_mass ==
            Approx(pod_test->liquid_mass).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_x ==
            Approx(pod_test->liquid_velocity_x).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_y ==
            Approx(pod_test->liquid_velocity_y).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_velocity_z ==
            Approx(pod_test->liquid_velocity_z).epsilon(Tolerance));
    REQUIRE(h5_particle.porosity ==
            Approx(pod_test->porosity).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_saturation ==
            Approx(pod_test->liquid_saturation).epsilon(Tolerance));
    REQUIRE(h5_particle.liquid_material_id ==
            Approx(pod_test->liquid_material_id).epsilon(Tolerance));
  }

  // Check particle's material id maping to nodes
  SECTION("Check particle's material id maping to nodes") {
    // Add particle
    mpm::Index id1 = 0;
    coords << 1.5, 1.5, 1.5;
    auto particle1 = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id1, coords);

    // Add particle
    mpm::Index id2 = 1;
    coords << 0.5, 0.5, 0.5;
    auto particle2 = std::make_shared<mpm::TwoPhaseParticle<Dim>>(id2, coords);

    // Element
    std::shared_ptr<mpm::Element<Dim>> element =
        std::make_shared<mpm::HexahedronElement<Dim, 8>>();

    // Create cell
    auto cell = std::make_shared<mpm::Cell<Dim>>(10, Nnodes, element);
    // Create vector of nodes and add them to cell
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
    std::vector<std::shared_ptr<mpm::NodeBase<Dim>>> nodes = {
        node0, node1, node2, node3, node4, node5, node6, node7};

    for (int j = 0; j < nodes.size(); ++j) cell->add_node(j, nodes[j]);

    // Initialise cell properties and assign cell to particle
    cell->initialise();
    particle1->assign_cell(cell);
    particle2->assign_cell(cell);

    // Assign material 1
    unsigned mid1 = 0;
    // Initialise material 1
    Json jmaterial1;
    jmaterial1["density"] = 1000.;
    jmaterial1["youngs_modulus"] = 1.0E+7;
    jmaterial1["poisson_ratio"] = 0.3;

    auto material1 =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(mid1), jmaterial1);

    particle1->assign_material(material1);

    // Assign material 2
    unsigned mid2 = 1;
    // Initialise material 2
    Json jmaterial2;
    jmaterial2["density"] = 2000.;
    jmaterial2["youngs_modulus"] = 2.0E+7;
    jmaterial2["poisson_ratio"] = 0.25;

    auto material2 =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(mid2), jmaterial2);

    particle2->assign_material(material2);

    // Append particle's material id to nodes in cell
    particle1->append_material_id_to_nodes();
    particle2->append_material_id_to_nodes();

    // check if the correct amount of material ids were added to node and if
    // their indexes are correct
    std::vector<unsigned> material_ids = {0, 1};
    for (const auto& node : nodes) {
      REQUIRE(node->material_ids().size() == 2);
      auto mat_ids = node->material_ids();
      unsigned i = 0;
      for (auto mitr = mat_ids.begin(); mitr != mat_ids.end(); ++mitr, ++i)
        REQUIRE(*mitr == material_ids.at(i));
    }
  }
}
