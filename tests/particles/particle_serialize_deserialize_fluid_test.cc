#include <chrono>
#include <limits>

#include "catch.hpp"

#include "data_types.h"
#include "logger.h"
#include "material.h"
#include "particle.h"
#include "particle_fluid.h"
#include "pod_particle.h"

//! \brief Check particle class for serialization and deserialization
TEST_CASE("Fluid particle is checked for serialization and deserialization",
          "[particle][3D][serialize][fluid]") {
  // Dimension
  const unsigned Dim = 3;

  // Logger
  std::unique_ptr<spdlog::logger> console_ = std::make_unique<spdlog::logger>(
      "particle_serialize_deserialize_fluid_test", mpm::stdout_sink);

  // Check initialise particle from POD file
  SECTION("Check initialise particle POD") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    // Coordinates
    Eigen::Matrix<double, Dim, 1> pcoords;
    pcoords.setZero();

    std::shared_ptr<mpm::ParticleBase<Dim>> particle =
        std::make_shared<mpm::FluidParticle<Dim>>(id, pcoords);

    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["bulk_modulus"] = 2.E9;
    jmaterial["dynamic_viscosity"] = 8.9E-4;
    jmaterial["incompressible"] = true;
    unsigned mid = 1;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "Newtonian3D", std::move(mid), jmaterial);
    std::vector<std::shared_ptr<mpm::Material<Dim>>> materials;
    materials.emplace_back(material);

    mpm::PODParticle h5_particle;
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

    Eigen::Vector3d acceleration;
    acceleration << 15, 25, 0.0;
    h5_particle.acceleration_x = acceleration[0];
    h5_particle.acceleration_y = acceleration[1];
    h5_particle.acceleration_z = acceleration[2];

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

    h5_particle.nstate_vars = 1;

    h5_particle.svars[0] = 1000.0;

    // Reinitialise particle from HDF5 data
    REQUIRE(particle->initialise_particle(h5_particle, materials) == true);

    // Assign projection parameter
    particle->assign_projection_parameter(1.0);

    // Serialize particle
    auto buffer = particle->serialize();
    REQUIRE(buffer.size() > 0);

    // Deserialize particle
    std::shared_ptr<mpm::ParticleBase<Dim>> rparticle =
        std::make_shared<mpm::FluidParticle<Dim>>(id, pcoords);

    REQUIRE_NOTHROW(rparticle->deserialize(buffer, materials));

    // Check particle id
    REQUIRE(particle->id() == particle->id());
    // Check particle mass
    REQUIRE(particle->mass() == rparticle->mass());
    // Check particle volume
    REQUIRE(particle->volume() == rparticle->volume());
    // Check particle mass density
    REQUIRE(particle->mass_density() == rparticle->mass_density());
    // Check particle pressure
    REQUIRE(particle->pressure() == rparticle->pressure());
    // Check particle status
    REQUIRE(particle->status() == rparticle->status());
    // Check particle projection parameter
    REQUIRE(particle->projection_parameter() ==
            rparticle->projection_parameter());

    // Check for coordinates
    auto coordinates = rparticle->coordinates();
    REQUIRE(coordinates.size() == Dim);
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Check for displacement
    auto pdisplacement = rparticle->displacement();
    REQUIRE(pdisplacement.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pdisplacement(i) == Approx(displacement(i)).epsilon(Tolerance));

    // Check for size
    auto size = rparticle->natural_size();
    REQUIRE(size.size() == Dim);
    for (unsigned i = 0; i < size.size(); ++i)
      REQUIRE(size(i) == Approx(lsize(i)).epsilon(Tolerance));

    // Check velocity
    auto pvelocity = rparticle->velocity();
    REQUIRE(pvelocity.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pvelocity(i) == Approx(velocity(i)).epsilon(Tolerance));

    // Check acceleration
    auto pacceleration = rparticle->acceleration();
    REQUIRE(pacceleration.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pacceleration(i) == Approx(acceleration(i)).epsilon(Tolerance));

    // Check stress
    auto pstress = rparticle->stress();
    REQUIRE(pstress.size() == stress.size());
    for (unsigned i = 0; i < stress.size(); ++i)
      REQUIRE(pstress(i) == Approx(stress(i)).epsilon(Tolerance));

    // Check strain
    auto pstrain = rparticle->strain();
    REQUIRE(pstrain.size() == strain.size());
    for (unsigned i = 0; i < strain.size(); ++i)
      REQUIRE(pstrain(i) == Approx(strain(i)).epsilon(Tolerance));

    // Check deformation gradient
    auto pdef_grad = rparticle->deformation_gradient();
    REQUIRE(pdef_grad.rows() == deformation_gradient.rows());
    REQUIRE(pdef_grad.cols() == deformation_gradient.cols());
    for (unsigned i = 0; i < deformation_gradient.rows(); ++i)
      for (unsigned j = 0; j < deformation_gradient.cols(); ++j)
        REQUIRE(pdef_grad(i, j) ==
                Approx(deformation_gradient(i, j)).epsilon(Tolerance));

    // Check mapping matrix
    auto map = particle->mapping_matrix();
    auto rmap = rparticle->mapping_matrix();
    REQUIRE(map.rows() == rmap.rows());
    REQUIRE(map.cols() == rmap.cols());
    for (unsigned i = 0; i < rmap.rows(); ++i)
      for (unsigned j = 0; j < rmap.cols(); ++j)
        REQUIRE(map(i, j) == Approx(rmap(i, j)).epsilon(Tolerance));

    // Check cell id
    REQUIRE(particle->cell_id() == rparticle->cell_id());

    // Check material id
    REQUIRE(particle->material_id() == rparticle->material_id());

    // Check state variable size
    REQUIRE(particle->state_variables().size() ==
            rparticle->state_variables().size());

    // Check state variables
    auto state_variables = material->state_variables();
    for (const auto& state_var : state_variables) {
      REQUIRE(particle->state_variable(state_var) ==
              rparticle->state_variable(state_var));
    }

    SECTION("Performance benchmarks") {
      // Number of iterations
      unsigned niterations = 1000;

      // Serialization benchmarks
      auto serialize_start = std::chrono::steady_clock::now();
      for (unsigned i = 0; i < niterations; ++i) {
        // Serialize particle
        auto buffer = particle->serialize();
        // Deserialize particle
        std::shared_ptr<mpm::ParticleBase<Dim>> rparticle =
            std::make_shared<mpm::FluidParticle<Dim>>(id, pcoords);

        REQUIRE_NOTHROW(rparticle->deserialize(buffer, materials));
      }
      auto serialize_end = std::chrono::steady_clock::now();

      console_->info("Performance benchmarks: {} ms",
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         serialize_end - serialize_start)
                         .count());
    }
  }
}
