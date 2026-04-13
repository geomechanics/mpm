#include <chrono>
#include <limits>

#include "catch.hpp"

#include "cell.h"
#include "data_types.h"
#include "element.h"
#include "function_base.h"
#include "hexahedron_element.h"
#include "linear_function.h"
#include "logger.h"
#include "node.h"
#include "pod_point.h"
#include "point_base.h"
#include "point_dirichlet_penalty.h"
#include "quadrilateral_element.h"

//! \brief Check point class for serialization and deserialization
TEST_CASE("Point is checked for serialization and deserialization",
          "[point][3D][serialize]") {
  // Dimension
  const unsigned Dim = 3;

  // Logger
  std::unique_ptr<spdlog::logger> console_ = std::make_unique<spdlog::logger>(
      "point_serialize_deserialize_test", mpm::stdout_sink);

  // Check initialise point from POD file
  SECTION("Check initialise point POD") {
    mpm::Index id = 0;
    const double Tolerance = 1.E-7;
    // Coordinates
    Eigen::Matrix<double, Dim, 1> pcoords;
    pcoords.setZero();

    std::shared_ptr<mpm::PointBase<Dim>> point =
        std::make_shared<mpm::PointDirichletPenalty<Dim>>(id, pcoords);

    mpm::PODPoint h5_point;
    h5_point.id = 13;
    h5_point.area = 0.25;

    Eigen::Vector3d coords;
    coords << 1., 2., 0.;
    h5_point.coord_x = coords[0];
    h5_point.coord_y = coords[1];
    h5_point.coord_z = coords[2];

    Eigen::Vector3d displacement;
    displacement << 0.01, 0.02, 0.0;
    h5_point.displacement_x = displacement[0];
    h5_point.displacement_y = displacement[1];
    h5_point.displacement_z = displacement[2];

    h5_point.status = true;

    h5_point.cell_id = 1;

    // Reinitialise point from HDF5 data
    REQUIRE(point->initialise_point(h5_point) == true);

    // Serialize point
    auto buffer = point->serialize();
    REQUIRE(buffer.size() > 0);

    // Deserialize point
    std::shared_ptr<mpm::PointBase<Dim>> rpoint =
        std::make_shared<mpm::PointDirichletPenalty<Dim>>(id, pcoords);

    REQUIRE_NOTHROW(rpoint->deserialize(buffer));

    // Check point id
    REQUIRE(point->id() == point->id());
    // Check point area
    REQUIRE(point->area() == rpoint->area());
    // Check point status
    REQUIRE(point->status() == rpoint->status());

    // Check for coordinates
    auto coordinates = rpoint->coordinates();
    REQUIRE(coordinates.size() == Dim);
    for (unsigned i = 0; i < coordinates.size(); ++i)
      REQUIRE(coordinates(i) == Approx(coords(i)).epsilon(Tolerance));

    // Check for displacement
    auto pdisplacement = rpoint->displacement();
    REQUIRE(pdisplacement.size() == Dim);
    for (unsigned i = 0; i < Dim; ++i)
      REQUIRE(pdisplacement(i) == Approx(displacement(i)).epsilon(Tolerance));

    // Check cell id
    REQUIRE(point->cell_id() == rpoint->cell_id());

    SECTION("Performance benchmarks") {
      // Number of iterations
      unsigned niterations = 1000;

      // Serialization benchmarks
      auto serialize_start = std::chrono::steady_clock::now();
      for (unsigned i = 0; i < niterations; ++i) {
        // Serialize point
        auto buffer = point->serialize();
        // Deserialize point
        std::shared_ptr<mpm::PointBase<Dim>> rpoint =
            std::make_shared<mpm::PointDirichletPenalty<Dim>>(id, pcoords);

        REQUIRE_NOTHROW(rpoint->deserialize(buffer));
      }
      auto serialize_end = std::chrono::steady_clock::now();

      console_->info("Performance benchmarks: {} ms",
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         serialize_end - serialize_start)
                         .count());
    }
  }
}
