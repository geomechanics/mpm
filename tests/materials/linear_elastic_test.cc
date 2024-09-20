#include <limits>

#include "Eigen/Dense"
#include "catch.hpp"
#include "json.hpp"

#include "cell.h"
#include "material.h"
#include "node.h"
#include "particle.h"

//! Check linearelastic class in 2D
TEST_CASE("LinearElastic is checked in 2D", "[material][linear_elastic][2D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 2;

  // Add particle
  mpm::Index pid = 0;
  Eigen::Matrix<double, Dim, 1> coords;
  coords.setZero();
  auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;

  //! Check for id = 0
  SECTION("LinearElastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("LinearElastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  //! Check failed initialisation
  SECTION("LinearElastic failed initialisation") {
    unsigned id = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
  }

  //! Check material properties
  SECTION("LinearElastic check material properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));

    // Get material properties fail
    REQUIRE_THROWS(material->property<double>("shear_modulus"));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.empty() == true);
      const std::vector<std::string> state_vars = {};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("LinearElastic check stresses") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();

    // Initialise strain
    mpm::Material<Dim>::Vector6d strain;
    strain.setZero();
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0000000;
    strain(3) = 0.0000000;
    strain(4) = 0.0000000;
    strain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.63461538461538e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.25000000000000e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.86538461538462e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Initialise strain
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0000000;
    strain(3) = 0.0000100;
    strain(4) = 0.0000000;
    strain(5) = 0.0000000;

    // Reset stress
    stress.setZero();

    // Compute updated stress
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.63461538461538e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.25000000000000e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.86538461538462e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(3.84615384615385e+01).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.00000000000000e+00).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.00000000000000e+00).epsilon(Tolerance));
  }

  SECTION("LinearElastic check properties earthquake") {
    unsigned id = 0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);

    // Get P-Wave and S-Wave Velocities
    REQUIRE(material->template property<double>("pwave_velocity") ==
            Approx(116.023870223).epsilon(Tolerance));
    REQUIRE(material->template property<double>("swave_velocity") ==
            Approx(62.0173672946).epsilon(Tolerance));
  }

  SECTION(
      "LinearElastic check objective stress-rate - Jaumann rate, simple "
      "shear") {
    unsigned id = 0;
    jmaterial["stress_rate"] = "jaumann";
    jmaterial["youngs_modulus"] = 210.E9;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
    mpm::dense_map state_vars = material->initialise_state_variables();
    const double G = 8.076923076923077e+10;

    // Shear displacement
    double shear_disp = 10.0;
    int step = 20;
    double shear_inc = shear_disp / double(step);

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();

    // Initialise strain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain.setZero();
    dstrain(3) = shear_inc;

    // Assign deformation gradient increment
    Eigen::Matrix<double, 3, 3> F_inc = Eigen::Matrix<double, 3, 3>::Identity();
    F_inc(0, 1) = shear_inc;

    // Expected solution for step 20
    Eigen::VectorXd exp_normalized_sigma_xy(step);
    exp_normalized_sigma_xy << 0.484456210855322, 0.850300645292233,
        1.00796182648987, 0.918838798665121, 0.604751987303751,
        0.142600797987520, -0.354464040052753, -0.764743718722533,
        -0.987787463679428, -0.968986387235440, -0.712943648614519,
        -0.282347440033760, 0.217377269078489, 0.663880441423026,
        0.947842528067372, 0.999739706676998, 0.806865737950888,
        0.416442896148014, -0.0759396905856465, -0.549729592574642;

    for (unsigned i = 0; i < step; ++i) {
      particle->assign_deformation_gradient_increment(F_inc);

      // Compute updated stress
      stress = material->compute_stress(stress, dstrain, particle.get(),
                                        &state_vars);

      // Check stresses
      REQUIRE((stress(3) / G) ==
              Approx(exp_normalized_sigma_xy(i)).epsilon(Tolerance));

      // Update deformation gradient
      particle->update_deformation_gradient();
    }

    // Now increase number of step
    step = 1000;
    shear_inc = shear_disp / double(step);

    // Initialise stress, strain, and deformation gradient
    stress.setZero();
    dstrain.setZero();
    dstrain(3) = shear_inc;
    F_inc.setIdentity();
    F_inc(0, 1) = shear_inc;
    particle->assign_deformation_gradient(
        Eigen::Matrix<double, 3, 3>::Identity());

    // Expected solution for step 1000: only check every 10 indices
    Eigen::VectorXd exp_normalized_sigma_xy_fine(step / 10);
    exp_normalized_sigma_xy_fine << 0.00999987500026, 0.109778758248096,
        0.208460768431549, 0.305059907524803, 0.398610988861467,
        0.488179280960702, 0.572869847055193, 0.651836487003503,
        0.72429019224217, 0.789507029298459, 0.846835373094324,
        0.895702417768789, 0.935619899964762, 0.966188977395106,
        0.987104213942932, 0.998156631478312, 0.999235797898674,
        0.990330930529874, 0.971531003863101, 0.94302386055118,
        0.905094334546875, 0.858121405136189, 0.802574410302604,
        0.739008357257104, 0.668058376989576, 0.590433378249908,
        0.506908964366092, 0.418319683672156, 0.325550690977072,
        0.229528903390496, 0.13121373887341, 0.031587530050997,
        -0.068354290930237, -0.16761313843142, -0.265197250858933,
        -0.360131600018419, -0.451467633274202, -0.538292751173666,
        -0.61973942583933, -0.694993869020624, -0.763304163197094,
        -0.823987774489708, -0.876438372313767, -0.920131887633675,
        -0.95463174928751, -0.979593246061767, -0.994766970931849,
        -1.00000131305462, -0.995243972613841, -0.980542483382619,
        -0.956043737781689, -0.921992519178889, -0.878729056094659,
        -0.826685622751138, -0.766382219931024, -0.698421379301639,
        -0.623482143117686, -0.542313279455496, -0.455725800769924,
        -0.36458486052598, -0.269801108871392, -0.17232159372135,
        -0.07312029816879, 0.026811591232716, 0.126475588077357,
        0.224875882653201, 0.321029291743658, 0.413975082269188,
        0.502784570614652, 0.586570401729016, 0.664495415283701,
        0.735781010301852, 0.799714924681932, 0.855658351885198,
        0.903052323679473, 0.941423295164953, 0.970387876278312,
        0.989656662499484, 0.999037126485959, 0.998435541742354,
        0.98785791910457, 0.967409946681506, 0.937295933854413,
        0.897816769885089, 0.849366917529841, 0.792430471698009,
        0.727576322535657, 0.655452471263287, 0.576779555561834,
        0.49234364919918, 0.402988407840901, 0.309606639521733,
        0.213131384002812, 0.114526590146804, 0.014777484459507,
        -0.085119272967723, -0.184165546756149, -0.281371699278424,
        -0.375766478790627, -0.466406723858637;

    for (unsigned i = 0; i < step; ++i) {
      particle->assign_deformation_gradient_increment(F_inc);

      // Compute updated stress
      stress = material->compute_stress(stress, dstrain, particle.get(),
                                        &state_vars);

      // Check stresses
      if (i % 10 == 0)
        REQUIRE(
            (stress(3) / G) ==
            Approx(exp_normalized_sigma_xy_fine(i / 10)).epsilon(Tolerance));

      // Update deformation gradient
      particle->update_deformation_gradient();
    }
  }

  SECTION(
      "LinearElastic check objective stress-rate - Green-Naghdi rate, simple "
      "shear") {
    unsigned id = 0;
    jmaterial["stress_rate"] = "green_naghdi";
    jmaterial["youngs_modulus"] = 210.E9;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic2D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
    mpm::dense_map state_vars = material->initialise_state_variables();
    const double G = 8.076923076923077e+10;

    // Shear displacement
    double shear_disp = 10.0;
    int step = 20;
    double shear_inc = shear_disp / double(step);

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();

    // Initialise strain
    mpm::Material<Dim>::Vector6d dstrain;
    dstrain.setZero();
    dstrain(3) = shear_inc;

    // Assign deformation gradient increment
    Eigen::Matrix<double, 3, 3> F_inc = Eigen::Matrix<double, 3, 3>::Identity();
    F_inc(0, 1) = shear_inc;

    // Expected solution for step 20
    Eigen::VectorXd exp_normalized_sigma_xy(step);
    exp_normalized_sigma_xy << 0.48552036199095, 0.87827186512118,
        1.16775103302116, 1.39685746985072, 1.60835973904628, 1.82756842788834,
        2.06575280936477, 2.32637506750973, 2.6092330040069, 2.91263582342706,
        3.23443486486929, 3.57247710486471, 3.92478509438122, 4.28961227911718,
        4.66544422418151, 5.05097846829518, 5.44509785010117, 5.84684375648296,
        6.2553918397289, 6.67003097047487;

    for (unsigned i = 0; i < step; ++i) {
      particle->assign_deformation_gradient_increment(F_inc);

      // Compute updated stress
      stress = material->compute_stress(stress, dstrain, particle.get(),
                                        &state_vars);

      // Check stresses
      REQUIRE((stress(3) / G) ==
              Approx(exp_normalized_sigma_xy(i)).epsilon(Tolerance));

      // Update deformation gradient
      particle->update_deformation_gradient();
    }

    // Now increase number of step
    step = 1000;
    shear_inc = shear_disp / double(step);

    // Initialise stress, strain, and deformation gradient
    stress.setZero();
    dstrain.setZero();
    dstrain(3) = shear_inc;
    F_inc.setIdentity();
    F_inc(0, 1) = shear_inc;
    particle->assign_deformation_gradient(
        Eigen::Matrix<double, 3, 3>::Identity());

    // Expected solution for step 1000: only check every 10 indices
    Eigen::VectorXd exp_normalized_sigma_xy_fine(step / 10);
    exp_normalized_sigma_xy_fine << 0.009999875003906, 0.109779359488205,
        0.208475874860095, 0.305164153952265, 0.399023851388499,
        0.489375868855871, 0.575705526394802, 0.657671814381597,
        0.735103768552344, 0.807986352711252, 0.876438962474606,
        0.940689794803682, 1.00104899501321, 1.05788287932988, 1.11159081467984,
        1.16258565444373, 1.21127806200763, 1.25806463677755, 1.30331948809478,
        1.34738875872586, 1.39058755056039, 1.4331987201633, 1.47547306483942,
        1.51763049109748, 1.55986183324388, 1.60233106186695, 1.64517768567019,
        1.68851920364951, 1.73245350785913, 1.77706117080364, 1.82240757711706,
        1.86854487805478, 1.91551376079174, 1.96334503379247, 2.01206103561488,
        2.06167687825608, 2.11220153819863, 2.16363880917289, 2.21598813069713,
        2.26924530597226, 2.32340312189808, 2.37845188298821, 2.43437986989092,
        2.4911737321452, 2.54881882375351, 2.60729948916874, 2.66659930638196,
        2.7267012929712, 2.78758808022636, 2.84924205980253, 2.91164550676746,
        2.9747806823916, 3.03862991957787, 3.10317569343178, 3.16840067913013,
        3.23428779894764, 3.30082026004228, 3.36798158437678, 3.43575563195917,
        3.50412661842158, 3.57307912780887, 3.6425981213273, 3.7126689426966,
        3.78327732065553, 3.85440936909348, 3.92605158521188, 3.99819084605982,
        4.07081440374096, 4.14390987954141, 4.21746525719487, 4.29146887546615,
        4.3659094202095, 4.44077591603238, 4.51605771767683, 4.59174450121103,
        4.6678262551117, 4.74429327130143, 4.82113613619732, 4.89834572181707,
        4.97591317697823, 5.05382991862331, 5.13208762329426, 5.21067821877664,
        5.28959387592898, 5.36882700070913, 5.44837022640639, 5.5282164060853,
        5.60835860524563, 5.68879009469977, 5.7695043436696, 5.85049501309993,
        5.93175594918846, 6.01328117712991, 6.09506489506739, 6.17710146825274,
        6.25938542340572, 6.34191144327149, 6.42467436136877, 6.50766915692403,
        6.59089094998825;

    for (unsigned i = 0; i < step; ++i) {
      particle->assign_deformation_gradient_increment(F_inc);

      // Compute updated stress
      stress = material->compute_stress(stress, dstrain, particle.get(),
                                        &state_vars);

      // Check stresses
      if (i % 10 == 0)
        REQUIRE(
            (stress(3) / G) ==
            Approx(exp_normalized_sigma_xy_fine(i / 10)).epsilon(Tolerance));

      // Update deformation gradient
      particle->update_deformation_gradient();
    }
  }
}

//! Check linearelastic class in 3D
TEST_CASE("LinearElastic is checked in 3D", "[material][linear_elastic][3D]") {
  // Tolerance
  const double Tolerance = 1.E-7;

  const unsigned Dim = 3;

  // Add particle
  mpm::Index pid = 0;
  Eigen::Matrix<double, Dim, 1> coords;
  coords.setZero();
  auto particle = std::make_shared<mpm::Particle<Dim>>(pid, coords);

  // Initialise material
  Json jmaterial;
  jmaterial["density"] = 1000.;
  jmaterial["youngs_modulus"] = 1.0E+7;
  jmaterial["poisson_ratio"] = 0.3;

  //! Check for id = 0
  SECTION("LinearElastic id is zero") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);
  }

  SECTION("LinearElastic id is positive") {
    //! Check for id is a positive value
    unsigned id = std::numeric_limits<unsigned>::max();
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == std::numeric_limits<unsigned>::max());
  }

  //! Check failed initialisation
  SECTION("LinearElastic failed initialisation") {
    unsigned id = 0;
    // Initialise material
    Json jmaterial;
    jmaterial["density"] = 1000.;
    jmaterial["poisson_ratio"] = 0.3;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
  }

  //! Check material properties
  SECTION("LinearElastic check material properties") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Get material properties
    REQUIRE(material->template property<double>("density") ==
            Approx(jmaterial["density"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("youngs_modulus") ==
            Approx(jmaterial["youngs_modulus"]).epsilon(Tolerance));
    REQUIRE(material->template property<double>("poisson_ratio") ==
            Approx(jmaterial["poisson_ratio"]).epsilon(Tolerance));

    // Check if state variable is initialised
    SECTION("State variable is initialised") {
      mpm::dense_map state_variables = material->initialise_state_variables();
      REQUIRE(state_variables.empty() == true);
      const std::vector<std::string> state_vars = {};
      auto state_vars_test = material->state_variables();
      REQUIRE(state_vars == state_vars_test);
    }
  }

  SECTION("LinearElastic check stresses") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    //    mpm::Material<Dim>::Matrix6x6 de = material->elastic_tensor();

    // Initialise stress
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    REQUIRE(stress(0) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.).epsilon(Tolerance));

    // Initialise strain
    mpm::Material<Dim>::Vector6d strain;
    strain.setZero();
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0005000;
    strain(3) = 0.0000000;
    strain(4) = 0.0000000;
    strain(5) = 0.0000000;

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.92307692307333e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(0.000000e+00).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(0.000000e+00).epsilon(Tolerance));

    // Initialise strain
    strain(0) = 0.0010000;
    strain(1) = 0.0005000;
    strain(2) = 0.0005000;
    strain(3) = 0.0000100;
    strain(4) = 0.0000200;
    strain(5) = 0.0000300;

    // Reset stress
    stress.setZero();

    // Compute updated stress
    stress =
        material->compute_stress(stress, strain, particle.get(), &state_vars);

    // Check stressees
    REQUIRE(stress(0) == Approx(1.92307692307333e+04).epsilon(Tolerance));
    REQUIRE(stress(1) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(2) == Approx(1.53846153845333e+04).epsilon(Tolerance));
    REQUIRE(stress(3) == Approx(3.84615384615385e+01).epsilon(Tolerance));
    REQUIRE(stress(4) == Approx(7.69230769230769e+01).epsilon(Tolerance));
    REQUIRE(stress(5) == Approx(1.15384615384615e+02).epsilon(Tolerance));
  }

  SECTION("LinearElastic check stresses fail") {
    unsigned id = 0;
    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);
    REQUIRE(material->id() == 0);

    // Initialise stress and strain
    mpm::Material<Dim>::Vector6d stress;
    stress.setZero();
    mpm::Material<Dim>::Vector6d updated_stress;
    updated_stress.setZero();

    // Initialise deformation gradient
    Eigen::Matrix<double, 3, 3> deformation_gradient;
    deformation_gradient.setIdentity();

    // Initialise deformation gradient increment
    Eigen::Matrix<double, 3, 3> deformation_gradient_increment;
    deformation_gradient_increment.setIdentity();

    // Compute updated stress
    mpm::dense_map state_vars = material->initialise_state_variables();
    REQUIRE_THROWS(material->compute_stress(stress, deformation_gradient,
                                            deformation_gradient_increment,
                                            particle.get(), &state_vars));

    REQUIRE_THROWS(material->compute_consistent_tangent_matrix(
        updated_stress, stress, deformation_gradient,
        deformation_gradient_increment, particle.get(), &state_vars));
  }

  SECTION("LinearElastic check properties earthquake") {
    unsigned id = 0;

    auto material =
        Factory<mpm::Material<Dim>, unsigned, const Json&>::instance()->create(
            "LinearElastic3D", std::move(id), jmaterial);

    // Get P-Wave and S-Wave Velocities
    REQUIRE(material->template property<double>("pwave_velocity") ==
            Approx(116.023870223).epsilon(Tolerance));
    REQUIRE(material->template property<double>("swave_velocity") ==
            Approx(62.0173672946).epsilon(Tolerance));
  }
}
