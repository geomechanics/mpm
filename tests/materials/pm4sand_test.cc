#include <limits>
#include <cmath>
#include <fstream>   // CSV出力用に追加
#include "Eigen/Dense"
#include "catch.hpp"
#include "json.hpp"

#include "cell.h"
#include "element.h"
#include "factory.h"
#include "hexahedron_element.h"
#include "mesh.h"
#include "material.h"
#include "node.h"
#include "particle.h"

//! Check PM4Sand class in 2D model with a cyclic undrained test
TEST_CASE("PM4Sand undrained cyclic test in 2D", "[material][PM4Sand][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7; // Tolerance for checks if needed
  const unsigned Dim = 2;         // Test is for 2D

  // Add a dummy particle (needed for compute_stress signature)
  mpm::Index pid = 0;
  Eigen::Matrix<double, Dim, 1> coords;
  coords.setZero();
  // Use Particle for 2D
  auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

  // --- Initial Conditions & Parameters ---
  double Dr = 0.55;      // Relative density (example value)
  // Use parameter names consistent with PM4Sand::initialise_parameters
  double initial_confining_pressure = 100000.0; // Pa (use positive for calculation, apply sign later)
  double static_tau = 0.0;                       // Initial static shear stress (Pa)

  // Json for PM4Sand material properties (using default values from initialise_parameters)
  Json jmaterial;
  jmaterial["density"] = 2000.0; // Example density
  // PM4Sand specific parameters (add more as needed or read from file)
  jmaterial["Dr"] = Dr;
  jmaterial["G0"] = 677.0; // Example shear modulus coefficient
  jmaterial["hpo"] = 0.4; // Example contraction rate parameter

  // Add other parameters (h0, emax, emin, nb, nd etc.) or rely on defaults in initialise_parameters

  // --- Material Creation ---
  unsigned id = 0;
  // Use the factory name registered for PM4Sand (e.g., "PM4Sand2D")
  auto material = Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
      "PM4Sand2D", std::move(id), jmaterial); // Use Dim=2
  REQUIRE(material != nullptr); // Check if material creation was successful
  REQUIRE(material->id() == 0);

  // --- Initial Stress State ---
  // Use compression negative convention for framework interaction
  double c_pin = -initial_confining_pressure; // Compressive pressure is negative
  Eigen::Matrix3d stres_tensor = Eigen::Matrix3d::Zero(); // Use 3D tensor internally
  stres_tensor(0,0) = c_pin; // sigma_xx
  stres_tensor(1,1) = c_pin; // sigma_yy
  // Estimate initial sigma_zz for K0 consolidation (approximate for testing)
  double nu = jmaterial.value("nu", 0.3); // Get Poisson's ratio
  stres_tensor(2,2) = c_pin ; // sigma_zz approx K0*sigma_yy (often needed for p calculation)
  stres_tensor(0,1) = -static_tau; // Apply static shear (negative to match convention if needed)
  stres_tensor(1,0) = stres_tensor(0,1); // Symmetry

  // Convert to Voigt vector [xx, yy, zz, xy, yz, zx]
  mpm::Material<Dim>::Vector6d stress_initial;
  stress_initial << stres_tensor(0,0), stres_tensor(1,1), stres_tensor(2,2),
                    stres_tensor(0,1), stres_tensor(1,2), stres_tensor(2,0); // yz=0, zx=0 for 2D

  // --- Strain Increment ---
  // Apply simple shear strain increment gamma_xy
  double dstrain_shear_val = -2e-6; // Small engineering shear strain increment gamma_xy
  mpm::Material<Dim>::Vector6d dstrain = mpm::Material<Dim>::Vector6d::Zero();
  dstrain(3) = dstrain_shear_val; // Apply gamma_xy to the xy component (index 3)

  // --- State Variables ---
  mpm::dense_map state_vars = material->initialise_state_variables();
  // Initialise may perform checks or set flags - call it after getting initial state_vars
  material->initialise(&state_vars);

  // --- Simulation Loop Setup ---
  std::vector<std::array<double,3>> result; // [eps_xy (%), p (kPa), tau_xy (kPa)]
  // Store initial state
  result.push_back({0.0,                     // Initial shear strain
                    -stress_initial.head(3).sum() / 3.0 / 1000.0, // Initial p (kPa, compression positive)
                    -stress_initial(3) / 1000.0}); // Initial tau_xy (kPa)

  int nreverse = 6; // Number of cycles (reversals / 2)
  int ireverse = 0;
  int istep = 0;
  double max_tau_abs = 25000.0; // Target shear stress for reversal (Pa)
  double cumulative_strain_xy = 0.0; // Cumulative *tensor* shear strain epsilon_xy

  mpm::Material<Dim>::Vector6d current_stress = stress_initial;

  // --- Cyclic Loading Loop ---
  const int max_steps = 500000; // Safety break for the loop
  while (ireverse < nreverse * 2 && istep < max_steps) { // Loop over reversals

    // Compute stress for the current strain increment
    auto updated_stress_voigt = material->compute_stress(current_stress, dstrain, particle.get(), &state_vars);
    // Print the full Voigt stress vector at each step
    // std::cout << "Step " << istep
    //           << " | updated_stress_voigt = ["
    //           << updated_stress_voigt(0) << ", "
    //           << updated_stress_voigt(1) << ", "
    //           << updated_stress_voigt(2) << ", "
    //           << updated_stress_voigt(3) << ", "
    //           << updated_stress_voigt(4) << ", "
    //           << updated_stress_voigt(5) << "] Pa"
    //           << std::endl;
    // Update state for next step
    current_stress = updated_stress_voigt;
    cumulative_strain_xy += -dstrain(3) * 0.5; // Accumulate tensor strain epsilon_xy = gamma_xy / 2

    // Calculate results (convert back to desired units/sign convention)
    double p_now_kPa = -current_stress.head(3).sum() / 3.0 / 1000.0; // Mean stress (kPa, compression positive)
    double tau_xy_kPa = -current_stress(3) / 1000.0;                // Shear stress (kPa)
    double eps_xy_percent = cumulative_strain_xy * 2.0 * 100.0;     // Engineering shear strain (%) gamma_xy = 2 * epsilon_xy

    result.push_back({eps_xy_percent, p_now_kPa, tau_xy_kPa});

    // Check for reversal (using absolute shear stress in Pa)
    if (std::abs(current_stress(3)) > max_tau_abs) {
        // Reverse strain increment direction
        dstrain(3) *= -1.0;
        ireverse++;
        std::cout << "Reversal #: " << ireverse << " at step " << istep
                  << ", tau_xy = " << current_stress.head(3) << " kPa" << std::endl;
        // Optional: Add check here if needed
        // REQUIRE(std::fabs(current_stress(3)) >= max_tau_abs * (1.0 - Tolerance));
    }

    istep++;
  }

  REQUIRE(istep < max_steps); // Check if loop finished correctly

  // --- NaN Check (Optional but recommended) ---
  for (const auto& row : result) {
    REQUIRE(std::isfinite(row[0])); // eps_xy
    REQUIRE(std::isfinite(row[1])); // p
    REQUIRE(std::isfinite(row[2])); // tau_xy
  }

  // --- CSV Output ---
  {
    std::ofstream ofs("test_pm4sand_result.csv");
    REQUIRE(ofs.is_open());
    ofs << "gamma_xy(%),p(kPa),tau_xy(kPa)\n"; // Header matching results vector order
    for (const auto &row : result) {
      ofs << row[0] << "," << row[1] << "," << row[2] << "\n";
    }
    std::cout << "Results saved to test_pm4sand_result.csv" << std::endl;
  }
}