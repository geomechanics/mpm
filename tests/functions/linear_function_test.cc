#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "catch.hpp"
#include "json.hpp"
#include "logger.h"

#include "function_base.h"
#include "linear_function.h"

//! \brief Check Functions class
TEST_CASE("Linear function is checked", "[linearfn]") {

  // Tolerance
  const double Tolerance = 1.E-9;

  // Linear function properties
  unsigned id = 0;
  std::vector<double> x_values{{0.0, 0.5, 1.0, 1.5}};
  std::vector<double> fx_values{{0.0, 1.0, 1.0, 0.0}};

  // Logger
  std::unique_ptr<spdlog::logger> console_ = std::make_unique<spdlog::logger>(
      "linear_function_test", mpm::stdout_sink);

  // Json property
  Json jfunctionproperties;
  jfunctionproperties["id"] = id;
  jfunctionproperties["xvalues"] = x_values;
  jfunctionproperties["fxvalues"] = fx_values;

  SECTION("Check incorrect linear function initialisation") {
    bool status = true;
    fx_values.emplace_back(0.0);
    jfunctionproperties["fxvalues"] = fx_values;
    try {
      std::shared_ptr<mpm::FunctionBase> linearfn =
          std::make_shared<mpm::LinearFunction>(id, jfunctionproperties);
    } catch (std::exception& exception) {
      console_->error("Exception caught: {}", exception.what());
      status = false;
    }
    REQUIRE(status == false);
  }

  SECTION("Check correct linear function initialisation") {
    bool status = true;
    REQUIRE_NOTHROW(
        std::make_shared<mpm::LinearFunction>(id, jfunctionproperties));

    REQUIRE(status == true);
  }

  SECTION("Check linear function") {
    std::shared_ptr<mpm::FunctionBase> linearfn =
        std::make_shared<mpm::LinearFunction>(id, jfunctionproperties);
    // check id
    REQUIRE(linearfn->id() == id);

    // check values for different x values
    double x = 0.0;
    REQUIRE(linearfn->value(x) == Approx(0.0).epsilon(Tolerance));
    x = 0.0000001;
    REQUIRE(linearfn->value(x) == Approx(0.0000002).epsilon(Tolerance));
    x = 0.4999999;
    REQUIRE(linearfn->value(x) == Approx(0.9999998).epsilon(Tolerance));
    x = 0.5;
    REQUIRE(linearfn->value(x) == Approx(1.0).epsilon(Tolerance));
    x = 0.9999999;
    REQUIRE(linearfn->value(x) == Approx(1.0).epsilon(Tolerance));
    x = 1;
    REQUIRE(linearfn->value(x) == Approx(1.0).epsilon(Tolerance));
    x = 1.2;
    REQUIRE(linearfn->value(x) == Approx(0.6).epsilon(Tolerance));
    x = 1.5;
    REQUIRE(linearfn->value(x) == Approx(0.0).epsilon(Tolerance));
    x = 1.50000001;
    REQUIRE(linearfn->value(x) == Approx(0.0).epsilon(Tolerance));
    x = 10.0;
    REQUIRE(linearfn->value(x) == Approx(0.0).epsilon(Tolerance));
  }
}
