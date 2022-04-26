#include "write_mesh_particles.h"

namespace mpm_test {

// Write JSON Configuration file
bool write_json(unsigned dim, bool resume, const std::string& analysis,
                const std::string& mpm_scheme, const std::string& file_name) {
  // Make json object with input files
  // 2D
  std::string dimension = "2d";
  auto particle_type = "P2D";
  auto node_type = "N2D";
  auto cell_type = "ED2Q4";
  auto io_type = "Ascii2D";
  std::string material = "LinearElastic2D";
  std::vector<double> gravity{{0., -9.81}};
  unsigned material_id = 0;
  std::vector<double> xvalues{{0.0, 0.5, 1.0}};
  std::vector<double> fxvalues{{0.0, 1.0, 1.0}};

  // 3D
  if (dim == 3) {
    dimension = "3d";
    particle_type = "P3D";
    node_type = "N3D";
    cell_type = "ED3H8";
    io_type = "Ascii3D";
    material = "LinearElastic3D";
    gravity.clear();
    gravity = {0., 0., -9.81};
  }

  Json json_file = {
      {"title", "Example JSON Input for MPM"},
      {"mesh",
       {{"mesh", "mesh-" + dimension + ".txt"},
        {"entity_sets", "entity_sets_0.json"},
        {"io_type", io_type},
        {"check_duplicates", true},
        {"isoparametric", false},
        {"node_type", node_type},
        {"boundary_conditions",
         {{"velocity_constraints", {{"file", "velocity-constraints.txt"}}},
          {"friction_constraints", {{"file", "friction-constraints.txt"}}}}},
        {"cell_type", cell_type}}},
      {"particles",
       {{{"group_id", 0},
         {"generator",
          {{"type", "file"},
           {"material_id", material_id},
           {"pset_id", 0},
           {"io_type", io_type},
           {"particle_type", particle_type},
           {"check_duplicates", true},
           {"location", "particles-" + dimension + ".txt"}}}}}},
      {"materials",
       {{{"id", 0},
         {"type", material},
         {"density", 1000.},
         {"youngs_modulus", 1.0E+8},
         {"poisson_ratio", 0.495}},
        {{"id", 1},
         {"type", material},
         {"density", 2300.},
         {"youngs_modulus", 1.5E+6},
         {"poisson_ratio", 0.25}}}},
      {"material_sets",
       {{{"material_id", 1}, {"phase_id", 0}, {"pset_id", 2}}}},
      {"external_loading_conditions",
       {{"gravity", gravity},
        {"particle_surface_traction",
         {{{"math_function_id", 0},
           {"pset_id", -1},
           {"dir", 1},
           {"traction", 10.5}}}},
        {"concentrated_nodal_forces",
         {{{"math_function_id", 0},
           {"nset_id", -1},
           {"dir", 1},
           {"force", 10.5}}}}}},
      {"math_functions",
       {{{"id", 0},
         {"type", "Linear"},
         {"xvalues", xvalues},
         {"fxvalues", fxvalues}}}},
      {"analysis",
       {{"type", analysis},
        {"mpm_scheme", mpm_scheme},
        {"locate_particles", true},
        {"dt", 0.001},
        {"uuid", file_name + "-" + dimension},
        {"nsteps", 10},
        {"boundary_friction", 0.5},
        {"resume",
         {{"resume", resume},
          {"uuid", file_name + "-" + dimension},
          {"step", 5}}},
        {"damping", {{"type", "Cundall"}, {"damping_factor", 0.02}}},
        {"newmark", {{"beta", 0.25}, {"gamma", 0.5}}}}},
      {"post_processing",
       {{"path", "results/"},
        {"vtk", {"stresses", "strains", "velocities"}},
        {"vtk_statevars", {{{"phase_id", 0}, {"statevars", {"pdstrain"}}}}},
        {"output_steps", 5}}}};

  // Dump JSON as an input file to be read
  std::string fname = (file_name + "-" + dimension + ".json").c_str();
  std::ofstream file;
  file.open(fname, std::ios_base::out);
  file << json_file.dump(2);
  file.close();

  return true;
}

// Write JSON Configuration file for absorbing boundary
bool write_json_absorbing(unsigned dim, bool resume,
                          const std::string& analysis,
                          const std::string& mpm_scheme,
                          const std::string& file_name,
                          const std::string& position, const double delta) {
  // Make json object with input files
  // 2D
  std::string dimension = "2d";
  auto particle_type = "P2D";
  auto node_type = "N2D";
  auto cell_type = "ED2Q4";
  auto io_type = "Ascii2D";
  std::string material = "LinearElastic2D";
  std::vector<double> gravity{{0., 0.}};
  unsigned material_id = 0;
  std::vector<double> xvalues{{0.0, 0.5, 1.0}};
  std::vector<double> fxvalues{{0.0, 1.0, 1.0}};

  // 3D
  if (dim == 3) {
    dimension = "3d";
    particle_type = "P3D";
    node_type = "N3D";
    cell_type = "ED3H8";
    io_type = "Ascii3D";
    material = "LinearElastic3D";
    gravity.clear();
    gravity = {0., 0., 0.};
  }

  Json json_file = {
      {"title", "Example JSON Input for MPM"},
      {"mesh",
       {{"mesh", "mesh-" + dimension + ".txt"},
        {"entity_sets", "entity_sets_2.json"},
        {"io_type", io_type},
        {"check_duplicates", true},
        {"isoparametric", false},
        {"node_type", node_type},
        {"boundary_conditions",
         {{"absorbing_constraints",
           {{{"nset_id", 97},
             {"dir", 1},
             {"delta", 100.0},
             {"h_min", 0.5},
             {"a", 1},
             {"b", 1},
             {"position", "corner"}},
            {{"nset_id", 98},
             {"dir", 1},
             {"delta", 100.0},
             {"h_min", 0.5},
             {"a", 1},
             {"b", 1},
             {"position", "edge"}},
            {{"nset_id", 99},
             {"dir", 1},
             {"delta", delta},
             {"h_min", 0.5},
             {"a", 1},
             {"b", 1},
             {"position", position}}}}}},
        {"cell_type", cell_type}}},
      {"particles",
       {{{"group_id", 0},
         {"generator",
          {{"type", "file"},
           {"material_id", material_id},
           {"pset_id", 0},
           {"io_type", io_type},
           {"particle_type", particle_type},
           {"check_duplicates", true},
           {"location", "particles-" + dimension + ".txt"}}}}}},
      {"materials",
       {
           {{"id", 0},
            {"type", material},
            {"density", 1000.},
            {"youngs_modulus", 1.0E+6},
            {"poisson_ratio", 0.0}},
       }},
      {"material_sets",
       {{{"material_id", 0}, {"phase_id", 0}, {"pset_id", 2}}}},
      {"external_loading_conditions",
       {{"gravity", gravity},
        {"particle_surface_traction",
         {{{"math_function_id", 0},
           {"pset_id", -1},
           {"dir", 1},
           {"traction", 10.5}}}}}},
      {"math_functions",
       {{{"id", 0},
         {"type", "Linear"},
         {"xvalues", xvalues},
         {"fxvalues", fxvalues}},
        {{"id", 1}, {"type", "Linear"}, {"file", "math_function.csv"}}}},
      {"analysis",
       {{"type", analysis},
        {"mpm_scheme", mpm_scheme},
        {"locate_particles", true},
        {"dt", 0.001},
        {"uuid", file_name + "-" + dimension},
        {"nsteps", 10},
        {"boundary_friction", 0.5},
        {"resume",
         {{"resume", resume},
          {"uuid", file_name + "-" + dimension},
          {"step", 5}}},
        {"damping", {{"type", "Cundall"}, {"damping_factor", 0.02}}},
        {"newmark", {{"beta", 0.25}, {"gamma", 0.5}}}}},
      {"post_processing",
       {{"path", "results/"},
        {"vtk", {"stresses", "strains", "velocities"}},
        {"vtk_statevars", {{{"phase_id", 0}, {"statevars", {"pdstrain"}}}}},
        {"output_steps", 5}}}};

  // Dump JSON as an input file to be read
  std::string fname = (file_name + "-" + dimension + ".json").c_str();
  std::ofstream file;
  file.open(fname, std::ios_base::out);
  file << json_file.dump(2);
  file.close();

  return true;
}

// Write JSON Configuration file for finite strain
bool write_json_finite_strain(unsigned dim, bool resume,
                              const std::string& analysis,
                              const std::string& mpm_scheme,
                              const std::string& file_name) {
  // Make json object with input files
  // 2D
  std::string dimension = "2d";
  auto particle_type = "P2DFS";
  auto node_type = "N2D";
  auto cell_type = "ED2Q4";
  auto io_type = "Ascii2D";
  std::string material = "HenckyHyperElastic2D";
  std::vector<double> gravity{{0., -9.81}};
  unsigned material_id = 0;
  std::vector<double> xvalues{{0.0, 0.5, 1.0}};
  std::vector<double> fxvalues{{0.0, 1.0, 1.0}};

  // 3D
  if (dim == 3) {
    dimension = "3d";
    particle_type = "P3DFS";
    node_type = "N3D";
    cell_type = "ED3H8";
    io_type = "Ascii3D";
    material = "HenckyHyperElastic3D";
    gravity.clear();
    gravity = {0., 0., -9.81};
  }

  Json json_file = {
      {"title", "Example JSON Input for MPM"},
      {"mesh",
       {{"mesh", "mesh-" + dimension + ".txt"},
        {"entity_sets", "entity_sets_0.json"},
        {"io_type", io_type},
        {"check_duplicates", true},
        {"isoparametric", false},
        {"node_type", node_type},
        {"boundary_conditions",
         {{"velocity_constraints", {{"file", "velocity-constraints.txt"}}},
          {"friction_constraints", {{"file", "friction-constraints.txt"}}}}},
        {"cell_type", cell_type}}},
      {"particles",
       {{{"group_id", 0},
         {"generator",
          {{"type", "file"},
           {"material_id", material_id},
           {"pset_id", 0},
           {"io_type", io_type},
           {"particle_type", particle_type},
           {"check_duplicates", true},
           {"location", "particles-" + dimension + ".txt"}}}}}},
      {"materials",
       {{{"id", 0},
         {"type", material},
         {"density", 1000.},
         {"youngs_modulus", 1.0E+8},
         {"poisson_ratio", 0.495}},
        {{"id", 1},
         {"type", material},
         {"density", 2300.},
         {"youngs_modulus", 1.5E+6},
         {"poisson_ratio", 0.25}}}},
      {"material_sets",
       {{{"material_id", 1}, {"phase_id", 0}, {"pset_id", 2}}}},
      {"external_loading_conditions",
       {{"gravity", gravity},
        {"particle_surface_traction",
         {{{"math_function_id", 0},
           {"pset_id", -1},
           {"dir", 1},
           {"traction", 10.5}}}},
        {"concentrated_nodal_forces",
         {{{"math_function_id", 0},
           {"nset_id", -1},
           {"dir", 1},
           {"force", 10.5}}}}}},
      {"math_functions",
       {{{"id", 0},
         {"type", "Linear"},
         {"xvalues", xvalues},
         {"fxvalues", fxvalues}}}},
      {"analysis",
       {{"type", analysis},
        {"mpm_scheme", mpm_scheme},
        {"locate_particles", true},
        {"dt", 0.001},
        {"uuid", file_name + "-" + dimension},
        {"nsteps", 10},
        {"boundary_friction", 0.5},
        {"resume",
         {{"resume", resume},
          {"uuid", file_name + "-" + dimension},
          {"step", 5}}},
        {"damping", {{"type", "Cundall"}, {"damping_factor", 0.02}}},
        {"newmark", {{"beta", 0.25}, {"gamma", 0.5}}}}},
      {"post_processing",
       {{"path", "results/"},
        {"vtk", {"stresses", "strains", "velocities"}},
        {"vtk_statevars", {{{"phase_id", 0}, {"statevars", {"pdstrain"}}}}},
        {"output_steps", 5}}}};

  // Dump JSON as an input file to be read
  std::string fname = (file_name + "-" + dimension + ".json").c_str();
  std::ofstream file;
  file.open(fname, std::ios_base::out);
  file << json_file.dump(2);
  file.close();

  return true;
}

// Write JSON Configuration file for implicit
bool write_json_implicit(unsigned dim, bool resume, const std::string& analysis,
                         const std::string& mpm_scheme, bool nonlinear,
                         bool quasi_static, const std::string& file_name,
                         const std::string& linear_solver_type) {
  // Make json object with input files
  // 2D
  std::string dimension = "2d";
  auto particle_type = "P2D";
  auto node_type = "N2D";
  auto cell_type = "ED2Q4P2B";
  auto io_type = "Ascii2D";
  auto assembler_type = "EigenImplicit2D";
  std::string entity_set_name = "entity_sets_0";
  std::string material = "LinearElastic2D";
  std::vector<double> gravity{{0., -9.81}};
  std::vector<unsigned> material_id{{0}};
  std::vector<double> xvalues{{0.0, 0.5, 1.0}};
  std::vector<double> fxvalues{{0.0, 1.0, 1.0}};

  // 3D
  if (dim == 3) {
    dimension = "3d";
    particle_type = "P3D";
    node_type = "N3D";
    cell_type = "ED3H8";
    assembler_type = "EigenImplicit3D";
    io_type = "Ascii3D";
    material = "LinearElastic3D";
    gravity.clear();
    gravity = {0., 0., -9.81};
    entity_set_name = "entity_sets_1";
  }

  Json json_file = {
      {"title", "Example JSON Input for MPM"},
      {"mesh",
       {{"mesh", "mesh-" + dimension + ".txt"},
        {"entity_sets", entity_set_name + ".json"},
        {"io_type", io_type},
        {"check_duplicates", true},
        {"isoparametric", false},
        {"node_type", node_type},
        {"boundary_conditions",
         {{"displacement_constraints",
           {{"file", "displacement-constraints.txt"}}}}},
        {"cell_type", cell_type},
        {"nonlocal_mesh_properties",
         {{"type", "BSPLINE"},
          {"node_types", {{{"nset_id", 1}, {"dir", 0}, {"type", 1}}}}}}}},
      {"particles",
       {{{"group_id", 0},
         {"generator",
          {{"type", "file"},
           {"material_id", material_id},
           {"pset_id", 0},
           {"io_type", io_type},
           {"particle_type", particle_type},
           {"check_duplicates", true},
           {"location", "particles-" + dimension + ".txt"}}}}}},
      {"materials",
       {{{"id", 0},
         {"type", material},
         {"density", 1000.},
         {"youngs_modulus", 1.0E+8},
         {"poisson_ratio", 0.495}},
        {{"id", 1},
         {"type", material},
         {"density", 2300.},
         {"youngs_modulus", 1.5E+6},
         {"poisson_ratio", 0.25}}}},
      {"material_sets",
       {{{"material_id", 1}, {"phase_id", 0}, {"pset_id", 2}}}},
      {"external_loading_conditions",
       {{"gravity", gravity},
        {"particle_surface_traction",
         {{{"math_function_id", 0},
           {"pset_id", -1},
           {"dir", 1},
           {"traction", 10.5}}}},
        {"concentrated_nodal_forces",
         {{{"math_function_id", 0},
           {"nset_id", -1},
           {"dir", 1},
           {"force", 10.5}}}}}},
      {"math_functions",
       {{{"id", 0},
         {"type", "Linear"},
         {"xvalues", xvalues},
         {"fxvalues", fxvalues}}}},
      {"analysis",
       {{"type", analysis},
        {"mpm_scheme", mpm_scheme},
        {"scheme_settings",
         {{"nonlinear", nonlinear},
          {"quasi_static", quasi_static},
          {"beta", 0.25},
          {"gamma", 0.50},
          {"max_iteration", 20},
          {"displacement_tolerance", 1.0e-10},
          {"residual_tolerance", 1.0e-10},
          {"relative_residual_tolerance", 1.0e-6},
          {"verbosity", 0}}},
        {"locate_particles", true},
        {"pressure_smoothing", true},
        {"dt", 0.0001},
        {"uuid", file_name + "-" + dimension},
        {"nsteps", 10},
        {"resume",
         {{"resume", resume},
          {"uuid", file_name + "-" + dimension},
          {"step", 5}}},
        {"linear_solver",
         {{"assembler_type", assembler_type},
          {"solver_settings",
           {{{"dof", "displacement"},
             {"solver_type", linear_solver_type},
             {"sub_solver_type", "cg"},
             {"preconditioner_type", "none"},
             {"max_iter", 100},
             {"tolerance", 1E-5},
             {"verbosity", 0}}}}}},
        {"damping", {{"type", "Cundall"}, {"damping_factor", 0.0}}},
        {"newmark", {{"beta", 0.25}, {"gamma", 0.5}}}}},
      {"post_processing",
       {{"path", "results/"},
        {"vtk", {"stresses", "strains", "velocity"}},
        {"vtk_statevars", {{{"phase_id", 0}, {"statevars", {"pdstrain"}}}}},
        {"output_steps", 5}}}};

  // Dump JSON as an input file to be read
  std::ofstream file;
  file.open((file_name + "-" + dimension + ".json").c_str());
  file << json_file.dump(2);
  file.close();

  return true;
}

// Write JSON Configuration file for implicit finite strain
bool write_json_implicit_finite_strain(unsigned dim, bool resume,
                                       const std::string& analysis,
                                       const std::string& mpm_scheme,
                                       bool nonlinear, bool quasi_static,
                                       const std::string& file_name,
                                       const std::string& linear_solver_type) {
  // Make json object with input files
  // 2D
  std::string dimension = "2d";
  auto particle_type = "P2DFS";
  auto node_type = "N2D";
  auto cell_type = "ED2Q4P2B";
  auto io_type = "Ascii2D";
  auto assembler_type = "EigenImplicit2D";
  std::string entity_set_name = "entity_sets_0";
  std::string material = "HenckyHyperElastic2D";
  std::vector<double> gravity{{0., -9.81}};
  std::vector<unsigned> material_id{{0}};
  std::vector<double> xvalues{{0.0, 0.5, 1.0}};
  std::vector<double> fxvalues{{0.0, 1.0, 1.0}};

  // 3D
  if (dim == 3) {
    dimension = "3d";
    particle_type = "P3DFS";
    node_type = "N3D";
    cell_type = "ED3H8";
    assembler_type = "EigenImplicit3D";
    io_type = "Ascii3D";
    material = "HenckyHyperElastic3D";
    gravity.clear();
    gravity = {0., 0., -9.81};
    entity_set_name = "entity_sets_1";
  }

  Json json_file = {
      {"title", "Example JSON Input for MPM"},
      {"mesh",
       {{"mesh", "mesh-" + dimension + ".txt"},
        {"entity_sets", entity_set_name + ".json"},
        {"io_type", io_type},
        {"check_duplicates", true},
        {"isoparametric", false},
        {"node_type", node_type},
        {"boundary_conditions",
         {{"displacement_constraints",
           {{"file", "displacement-constraints.txt"}}}}},
        {"cell_type", cell_type},
        {"nonlocal_mesh_properties",
         {{"type", "BSPLINE"},
          {"node_types", {{{"nset_id", 1}, {"dir", 0}, {"type", 1}}}}}}}},
      {"particles",
       {{{"group_id", 0},
         {"generator",
          {{"type", "file"},
           {"material_id", material_id},
           {"pset_id", 0},
           {"io_type", io_type},
           {"particle_type", particle_type},
           {"check_duplicates", true},
           {"location", "particles-" + dimension + ".txt"}}}}}},
      {"materials",
       {{{"id", 0},
         {"type", material},
         {"density", 1000.},
         {"youngs_modulus", 1.0E+8},
         {"poisson_ratio", 0.495}},
        {{"id", 1},
         {"type", material},
         {"density", 2300.},
         {"youngs_modulus", 1.5E+6},
         {"poisson_ratio", 0.25}}}},
      {"material_sets",
       {{{"material_id", 1}, {"phase_id", 0}, {"pset_id", 2}}}},
      {"external_loading_conditions",
       {{"gravity", gravity},
        {"particle_surface_traction",
         {{{"math_function_id", 0},
           {"pset_id", -1},
           {"dir", 1},
           {"traction", 10.5}}}},
        {"concentrated_nodal_forces",
         {{{"math_function_id", 0},
           {"nset_id", -1},
           {"dir", 1},
           {"force", 10.5}}}}}},
      {"math_functions",
       {{{"id", 0},
         {"type", "Linear"},
         {"xvalues", xvalues},
         {"fxvalues", fxvalues}}}},
      {"analysis",
       {{"type", analysis},
        {"mpm_scheme", mpm_scheme},
        {"scheme_settings",
         {{"nonlinear", nonlinear},
          {"quasi_static", quasi_static},
          {"beta", 0.25},
          {"gamma", 0.50},
          {"max_iteration", 20},
          {"displacement_tolerance", 1.0e-10},
          {"residual_tolerance", 1.0e-10},
          {"relative_residual_tolerance", 1.0e-6},
          {"verbosity", 0}}},
        {"locate_particles", true},
        {"pressure_smoothing", true},
        {"dt", 0.0001},
        {"uuid", file_name + "-" + dimension},
        {"nsteps", 10},
        {"resume",
         {{"resume", resume},
          {"uuid", file_name + "-" + dimension},
          {"step", 5}}},
        {"linear_solver",
         {{"assembler_type", assembler_type},
          {"solver_settings",
           {{{"dof", "displacement"},
             {"solver_type", linear_solver_type},
             {"sub_solver_type", "cg"},
             {"preconditioner_type", "none"},
             {"max_iter", 100},
             {"tolerance", 1E-5},
             {"verbosity", 0}}}}}},
        {"damping", {{"type", "Cundall"}, {"damping_factor", 0.0}}},
        {"newmark", {{"beta", 0.25}, {"gamma", 0.5}}}}},
      {"post_processing",
       {{"path", "results/"},
        {"vtk", {"stresses", "strains", "velocity"}},
        {"vtk_statevars", {{{"phase_id", 0}, {"statevars", {"pdstrain"}}}}},
        {"output_steps", 5}}}};

  // Dump JSON as an input file to be read
  std::ofstream file;
  file.open((file_name + "-" + dimension + ".json").c_str());
  file << json_file.dump(2);
  file.close();

  return true;
}

// Write JSON Configuration file for navierstokes
bool write_json_navierstokes(unsigned dim, bool resume,
                             const std::string& analysis,
                             const std::string& mpm_scheme,
                             const std::string& file_name,
                             const std::string& free_surface_type,
                             const std::string& linear_solver_type) {
  // Make json object with input files
  // 2D
  std::string dimension = "2d";
  auto particle_type = "P2DFLUID";
  auto node_type = "N2D";
  auto cell_type = "ED2Q4";
  auto io_type = "Ascii2D";
  auto assembler_type = "EigenSemiImplicitNavierStokes2D";
  std::string entity_set_name = "entity_sets_0";
  std::string material = "Newtonian2D";
  std::vector<double> gravity{{0., -9.81}};
  std::vector<unsigned> material_id{{2}};
  std::vector<double> xvalues{{0.0, 0.5, 1.0}};
  std::vector<double> fxvalues{{0.0, 1.0, 1.0}};

  // 3D
  if (dim == 3) {
    dimension = "3d";
    particle_type = "P3DFLUID";
    node_type = "N3D";
    cell_type = "ED3H8";
    assembler_type = "EigenSemiImplicitNavierStokes3D";
    io_type = "Ascii3D";
    material = "Newtonian3D";
    gravity.clear();
    gravity = {0., 0., -9.81};
    entity_set_name = "entity_sets_1";
  }

  Json json_file = {
      {"title", "Example JSON Input for MPM"},
      {"mesh",
       {{"mesh", "mesh-" + dimension + ".txt"},
        {"entity_sets", entity_set_name + ".json"},
        {"io_type", io_type},
        {"check_duplicates", true},
        {"isoparametric", false},
        {"node_type", node_type},
        {"boundary_conditions",
         {{"velocity_constraints", {{"file", "velocity-constraints.txt"}}},
          {"pressure_constraints",
           {{{"phase_id", 0}, {"nset_id", 1}, {"pressure", 0.0}}}},
          {"friction_constraints", {{"file", "friction-constraints.txt"}}}}},
        {"cell_type", cell_type}}},
      {"particles",
       {{{"group_id", 0},
         {"generator",
          {{"type", "file"},
           {"material_id", material_id},
           {"pset_id", 0},
           {"io_type", io_type},
           {"particle_type", particle_type},
           {"check_duplicates", true},
           {"location", "particles-" + dimension + ".txt"}}}}}},
      {"materials",
       {{{"id", 2},
         {"type", material},
         {"density", 1000.},
         {"bulk_modulus", 1.E+9},
         {"mu", 0.3},
         {"dynamic_viscosity", 0.}}}},
      {"material_sets",
       {{{"material_id", 1}, {"phase_id", 0}, {"pset_id", 2}}}},
      {"external_loading_conditions",
       {{"gravity", gravity},
        {"particle_surface_traction",
         {{{"math_function_id", 0},
           {"pset_id", -1},
           {"dir", 1},
           {"traction", 10.5}}}},
        {"concentrated_nodal_forces",
         {{{"math_function_id", 0},
           {"nset_id", -1},
           {"dir", 1},
           {"force", 10.5}}}}}},
      {"math_functions",
       {{{"id", 0},
         {"type", "Linear"},
         {"xvalues", xvalues},
         {"fxvalues", fxvalues}}}},
      {"analysis",
       {{"type", analysis},
        {"mpm_scheme", mpm_scheme},
        {"locate_particles", true},
        {"pressure_smoothing", true},
        {"pore_pressure_smoothing", true},
        {"free_surface_detection",
         {{"type", "density"}, {"volume_tolerance", 0.25}}},
        {"dt", 0.0001},
        {"uuid", file_name + "-" + dimension},
        {"nsteps", 10},
        {"boundary_friction", 0.5},
        {"resume",
         {{"resume", resume},
          {"uuid", file_name + "-" + dimension},
          {"step", 5}}},
        {"scheme_settings", {{"beta", 1}, {"integration", "mp"}}},
        {"free_surface_detection",
         {{"type", free_surface_type}, {"volume_tolerance", 0.25}}},
        {"linear_solver",
         {{"assembler_type", assembler_type},
          {"solver_settings",
           {{{"dof", "pressure"},
             {"solver_type", linear_solver_type},
             {"sub_solver_type", "cg"},
             {"preconditioner_type", "none"},
             {"max_iter", 100},
             {"tolerance", 1E-5},
             {"verbosity", 0}}}}}},
        {"damping", {{"type", "Cundall"}, {"damping_factor", 0.02}}},
        {"newmark", {{"beta", 0.25}, {"gamma", 0.5}}}}},
      {"post_processing",
       {{"path", "results/"},
        {"vtk", {"stresses", "strains", "velocity"}},
        {"vtk_statevars", {{{"phase_id", 0}, {"statevars", {"pdstrain"}}}}},
        {"output_steps", 5}}}};

  // Dump JSON as an input file to be read
  std::ofstream file;
  file.open((file_name + "-" + dimension + ".json").c_str());
  file << json_file.dump(2);
  file.close();

  return true;
}

// Write JSON Configuration file for twophase
bool write_json_twophase(unsigned dim, bool resume, const std::string& analysis,
                         const std::string& mpm_scheme,
                         const std::string& file_name,
                         const std::string& free_surface_type,
                         const std::string& linear_solver_type) {
  // Make json object with input files
  // 2D
  std::string dimension = "2d";
  auto particle_type = "P2D2PHASE";
  auto node_type = "N2D2P";
  auto cell_type = "ED2Q4";
  auto io_type = "Ascii2D";
  auto assembler_type = "EigenSemiImplicitTwoPhase2D";
  std::string entity_set_name = "entity_sets_0";
  std::string material = "LinearElastic2D";
  std::string liquid_material = "Newtonian2D";
  std::vector<double> gravity{{0., -9.81}};
  std::vector<unsigned> material_id{{0, 2}};
  std::vector<double> xvalues{{0.0, 0.5, 1.0}};
  std::vector<double> fxvalues{{0.0, 1.0, 1.0}};

  // 3D
  if (dim == 3) {
    dimension = "3d";
    particle_type = "P3D2PHASE";
    node_type = "N3D2P";
    cell_type = "ED3H8";
    assembler_type = "EigenSemiImplicitTwoPhase3D";
    io_type = "Ascii3D";
    material = "LinearElastic3D";
    liquid_material = "Newtonian3D";
    gravity.clear();
    gravity = {0., 0., -9.81};
    entity_set_name = "entity_sets_1";
  }

  Json json_file = {
      {"title", "Example JSON Input for MPM"},
      {"mesh",
       {{"mesh", "mesh-" + dimension + ".txt"},
        {"entity_sets", entity_set_name + ".json"},
        {"io_type", io_type},
        {"check_duplicates", true},
        {"isoparametric", false},
        {"node_type", node_type},
        {"boundary_conditions",
         {{"velocity_constraints", {{"file", "velocity-constraints.txt"}}},
          {"pressure_constraints",
           {{{"phase_id", 1}, {"nset_id", 1}, {"pressure", 0.0}}}},
          {"friction_constraints", {{"file", "friction-constraints.txt"}}}}},
        {"cell_type", cell_type},
        {"particles_pore_pressures",
         {{"type", "isotropic"}, {"values", 0.0}}}}},
      {"particles",
       {{{"group_id", 0},
         {"generator",
          {{"type", "file"},
           {"material_id", material_id},
           {"pset_id", 0},
           {"io_type", io_type},
           {"particle_type", particle_type},
           {"check_duplicates", true},
           {"location", "particles-" + dimension + ".txt"}}}}}},
      {"materials",
       {{{"id", 0},
         {"type", material},
         {"density", 1000.},
         {"youngs_modulus", 1.0E+8},
         {"poisson_ratio", 0.495},
         {"porosity", 0.3},
         {"k_x", 0.001},
         {"k_y", 0.001},
         {"k_z", 0.001},
         {"intrinsic_permeability", false}},
        {{"id", 1},
         {"type", material},
         {"density", 2300.},
         {"youngs_modulus", 1.5E+6},
         {"poisson_ratio", 0.25},
         {"porosity", 0.3},
         {"k_x", 0.001},
         {"k_y", 0.001},
         {"k_z", 0.001},
         {"intrinsic_permeability", true}},
        {{"id", 2},
         {"type", liquid_material},
         {"density", 1000.},
         {"bulk_modulus", 1.E+9},
         {"mu", 0.3},
         {"dynamic_viscosity", 8.9E-4}}}},
      {"material_sets",
       {{{"material_id", 1}, {"phase_id", 0}, {"pset_id", 2}}}},
      {"external_loading_conditions",
       {{"gravity", gravity},
        {"particle_surface_traction",
         {{{"math_function_id", 0},
           {"pset_id", -1},
           {"dir", 1},
           {"traction", 10.5}}}},
        {"concentrated_nodal_forces",
         {{{"math_function_id", 0},
           {"nset_id", -1},
           {"dir", 1},
           {"force", 10.5}}}}}},
      {"math_functions",
       {{{"id", 0},
         {"type", "Linear"},
         {"xvalues", xvalues},
         {"fxvalues", fxvalues}}}},
      {"analysis",
       {{"type", analysis},
        {"mpm_scheme", mpm_scheme},
        {"locate_particles", true},
        {"pressure_smoothing", true},
        {"pore_pressure_smoothing", true},
        {"free_surface_detection",
         {{"type", "density"}, {"volume_tolerance", 0.25}}},
        {"dt", 0.0001},
        {"uuid", file_name + "-" + dimension},
        {"nsteps", 10},
        {"boundary_friction", 0.5},
        {"resume",
         {{"resume", resume},
          {"uuid", file_name + "-" + dimension},
          {"step", 5}}},
        {"scheme_settings", {{"beta", 1}, {"integration", "mp"}}},
        {"free_surface_detection",
         {{"type", free_surface_type}, {"volume_tolerance", 0.25}}},
        {"linear_solver",
         {{"assembler_type", assembler_type},
          {"solver_settings",
           {{{"dof", "pressure"},
             {"solver_type", linear_solver_type},
             {"sub_solver_type", "cg"},
             {"preconditioner_type", "none"},
             {"max_iter", 100},
             {"tolerance", 1E-5},
             {"verbosity", 0}},
            {{"dof", "acceleration"},
             {"solver_type", linear_solver_type},
             {"sub_solver_type", "lscg"},
             {"preconditioner_type", "none"},
             {"max_iter", 100},
             {"tolerance", 1E-5},
             {"verbosity", 0}}}}}},
        {"damping", {{"type", "Cundall"}, {"damping_factor", 0.02}}},
        {"newmark", {{"beta", 0.25}, {"gamma", 0.5}}}}},
      {"post_processing",
       {{"path", "results/"},
        {"vtk", {"stresses", "strains", "velocity"}},
        {"vtk_statevars", {{{"phase_id", 0}, {"statevars", {"pdstrain"}}}}},
        {"output_steps", 5}}}};

  // Dump JSON as an input file to be read
  std::ofstream file;
  file.open((file_name + "-" + dimension + ".json").c_str());
  file << json_file.dump(2);
  file.close();

  return true;
}

// Write JSON Entity Set
bool write_entity_set() {
  // JSON Entity Sets
  Json json_file0 = {
      {"particle_sets",
       {{{"id", 2},
         {"set", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}}}},
      {"node_sets", {{{"id", 1}, {"set", {4, 5}}}}}};

  Json json_file1 = {
      {"particle_sets",
       {{{"id", 2},
         {"set", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}}}},
      {"node_sets", {{{"id", 1}, {"set", {8, 9, 10, 11}}}}}};

  Json json_file2 = {
      {"particle_sets",
       {{{"id", 2},
         {"set", {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}}}},
      {"node_sets",
       {{{"id", 1}, {"set", {4, 5}}},
        {{"id", 97}, {"set", {0}}},
        {{"id", 98}, {"set", {1}}},
        {{"id", 99}, {"set", {4}}}}}};

  // Dump JSON as an input file to be read
  std::ofstream file;
  file.open("entity_sets_0.json");
  file << json_file0.dump(2);
  file.close();

  file.open("entity_sets_1.json");
  file << json_file1.dump(2);
  file.close();

  file.open("entity_sets_2.json");
  file << json_file2.dump(2);
  file.close();

  return true;
}

// Write Mesh file in 2D
bool write_mesh_2d() {
  // Dimension
  const unsigned dim = 2;

  // Vector of nodal coordinates
  std::vector<Eigen::Matrix<double, dim, 1>> coordinates;

  // Nodal coordinates
  Eigen::Matrix<double, dim, 1> node;

  // Cell 0
  // Node 0
  node << 0., 0.;
  coordinates.emplace_back(node);
  // Node 1
  node << 0.5, 0.;
  coordinates.emplace_back(node);
  // Node 2
  node << 0.5, 0.5;
  coordinates.emplace_back(node);
  // Node 3
  node << 0., 0.5;
  coordinates.emplace_back(node);

  // Cell 1
  // Node 4
  node << 1.0, 0.;
  coordinates.emplace_back(node);
  // Node 5
  node << 1.0, 0.5;
  coordinates.emplace_back(node);

  // Cell with node ids
  std::vector<std::vector<unsigned>> cells{// cell #0
                                           {0, 1, 2, 3},
                                           // cell #1
                                           {1, 4, 5, 2}};

  // Dump mesh file as an input file to be read
  std::ofstream file;
  file.open("mesh-2d.txt");
  file << "! elementShape hexahedron\n";
  file << "! elementNumPoints 8\n";
  file << coordinates.size() << "\t" << cells.size() << "\n";

  // Write nodal coordinates
  for (const auto& coord : coordinates) {
    for (unsigned i = 0; i < coord.size(); ++i) file << coord[i] << "\t";
    file << "\n";
  }

  // Write cell node ids
  for (const auto& cell : cells) {
    for (auto nid : cell) file << nid << "\t";
    file << "\n";
  }
  file.close();

  // Dump mesh velocity constraints
  std::ofstream file_constraints;
  file_constraints.open("velocity-constraints.txt");
  file_constraints << 0 << "\t" << 1 << "\t" << 0 << "\n";
  file_constraints << 1 << "\t" << 1 << "\t" << 0 << "\n";
  file_constraints << 4 << "\t" << 1 << "\t" << 0 << "\n";
  file_constraints.close();

  // Dump mesh displacement constraints
  file_constraints.open("displacement-constraints.txt");
  file_constraints << 0 << "\t" << 1 << "\t" << 0 << "\n";
  file_constraints << 1 << "\t" << 1 << "\t" << 0 << "\n";
  file_constraints << 4 << "\t" << 1 << "\t" << 0 << "\n";
  file_constraints.close();

  return true;
}

// Write particles file in 2D
bool write_particles_2d() {
  const unsigned dim = 2;
  // Vector of particle coordinates
  std::vector<Eigen::Matrix<double, dim, 1>> coordinates;
  coordinates.clear();

  // Particle coordinates
  Eigen::Matrix<double, dim, 1> particle;

  // Cell 0
  // Particle 0
  particle << 0.125, 0.125;
  coordinates.emplace_back(particle);
  // Particle 1
  particle << 0.25, 0.125;
  coordinates.emplace_back(particle);
  // Particle 2
  particle << 0.25, 0.25;
  coordinates.emplace_back(particle);
  // Particle 3
  particle << 0.125, 0.25;
  coordinates.emplace_back(particle);

  // Cell 1
  // Particle 4
  particle << 0.675, 0.125;
  coordinates.emplace_back(particle);
  // Particle 5
  particle << 0.85, 0.125;
  coordinates.emplace_back(particle);
  // Particle 6
  particle << 0.85, 0.25;
  coordinates.emplace_back(particle);
  // Particle 7
  particle << 0.675, 0.25;
  coordinates.emplace_back(particle);

  // Dump particles coordinates as an input file to be read
  std::ofstream file;
  file.open("particles-2d.txt");
  file << coordinates.size() << "\n";
  // Write particle coordinates
  for (const auto& coord : coordinates) {
    for (unsigned i = 0; i < coord.size(); ++i) {
      file << coord[i] << "\t";
    }
    file << "\n";
  }

  file.close();
  return true;
}

// Write mesh file in 3D
bool write_mesh_3d() {

  // Dimension
  const unsigned dim = 3;

  // Vector of nodal coordinates
  std::vector<Eigen::Matrix<double, dim, 1>> coordinates;

  // Nodal coordinates
  Eigen::Matrix<double, dim, 1> node;

  // Cell 0
  // Node 0
  node << 0., 0., 0.;
  coordinates.emplace_back(node);
  // Node 1
  node << 0.5, 0., 0.;
  coordinates.emplace_back(node);
  // Node 2
  node << 0.5, 0.5, 0.;
  coordinates.emplace_back(node);
  // Node 3
  node << 0., 0.5, 0.;
  coordinates.emplace_back(node);
  // Node 4
  node << 0., 0., 0.5;
  coordinates.emplace_back(node);
  // Node 5
  node << 0.5, 0., 0.5;
  coordinates.emplace_back(node);
  // Node 6
  node << 0.5, 0.5, 0.5;
  coordinates.emplace_back(node);
  // Node 7
  node << 0., 0.5, 0.5;
  coordinates.emplace_back(node);

  // Cell 1
  // Node 8
  node << 1.0, 0., 0.;
  coordinates.emplace_back(node);
  // Node 9
  node << 1.0, 0.5, 0.;
  coordinates.emplace_back(node);
  // Node 10
  node << 1.0, 0., 0.5;
  coordinates.emplace_back(node);
  // Node 11
  node << 1.0, 0.5, 0.5;
  coordinates.emplace_back(node);

  // Cell with node ids
  std::vector<std::vector<unsigned>> cells{// cell #0
                                           {0, 1, 2, 3, 4, 5, 6, 7},
                                           // cell #1
                                           {1, 8, 9, 2, 5, 10, 11, 6}};

  // Dump mesh file as an input file to be read
  std::ofstream file;
  file.open("mesh-3d.txt");
  file << "! elementShape hexahedron\n";
  file << "! elementNumPoints 8\n";
  file << coordinates.size() << "\t" << cells.size() << "\n";

  // Write nodal coordinates
  for (const auto& coord : coordinates) {
    for (unsigned i = 0; i < coord.size(); ++i) file << coord[i] << "\t";
    file << "\n";
  }

  // Write cell node ids
  for (const auto& cell : cells) {
    for (auto nid : cell) file << nid << "\t";
    file << "\n";
  }

  file.close();

  // Dump mesh velocity constraints
  std::ofstream file_constraints;
  file_constraints.open("velocity-constraints.txt");
  file_constraints << 0 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 1 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 2 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 3 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 8 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 9 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints.close();

  // Dump mesh displacement constraints
  file_constraints.open("displacement-constraints.txt");
  file_constraints << 0 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 1 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 2 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 3 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 8 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints << 9 << "\t" << 2 << "\t" << 0 << "\n";
  file_constraints.close();

  return true;
}

// Write particles file in 3D
bool write_particles_3d() {
  const unsigned dim = 3;
  // Vector of particle coordinates
  std::vector<Eigen::Matrix<double, dim, 1>> coordinates;

  // Particle coordinates
  Eigen::Matrix<double, dim, 1> particle;

  // Cell 0
  // Particle 0
  particle << 0.125, 0.125, 0.125;
  coordinates.emplace_back(particle);
  // Particle 1
  particle << 0.25, 0.125, 0.125;
  coordinates.emplace_back(particle);
  // Particle 2
  particle << 0.25, 0.25, 0.125;
  coordinates.emplace_back(particle);
  // Particle 3
  particle << 0.125, 0.25, 0.125;
  coordinates.emplace_back(particle);
  // Particle 4
  particle << 0.125, 0.125, 0.25;
  coordinates.emplace_back(particle);
  // Particle 5
  particle << 0.25, 0.125, 0.25;
  coordinates.emplace_back(particle);
  // Particle 6
  particle << 0.25, 0.25, 0.25;
  coordinates.emplace_back(particle);
  // Particle 7
  particle << 0.125, 0.25, 0.25;
  coordinates.emplace_back(particle);

  // Cell 1
  // Particle 8
  particle << 0.675, 0.125, 0.125;
  coordinates.emplace_back(particle);
  // Particle 9
  particle << 0.85, 0.125, 0.125;
  coordinates.emplace_back(particle);
  // Particle 10
  particle << 0.85, 0.25, 0.125;
  coordinates.emplace_back(particle);
  // Particle 11
  particle << 0.675, 0.25, 0.125;
  coordinates.emplace_back(particle);
  // Particle 12
  particle << 0.675, 0.125, 0.25;
  coordinates.emplace_back(particle);
  // Particle 13
  particle << 0.85, 0.125, 0.25;
  coordinates.emplace_back(particle);
  // Particle 14
  particle << 0.85, 0.25, 0.25;
  coordinates.emplace_back(particle);
  // Particle 15
  particle << 0.675, 0.25, 0.25;
  coordinates.emplace_back(particle);

  // Dump particles coordinates as an input file to be read
  std::ofstream file;
  file.open("particles-3d.txt");
  file << coordinates.size() << "\n";
  // Write particle coordinates
  for (const auto& coord : coordinates) {
    for (unsigned i = 0; i < coord.size(); ++i) {
      file << coord[i] << "\t";
    }
    file << "\n";
  }

  file.close();
  return true;
}

// Write JSON Entity Set
bool write_math_function() {
  // Vector of math function
  std::vector<std::tuple<double, double>> math_function;
  math_function.emplace_back(std::make_tuple(0.0, 0.0));
  math_function.emplace_back(std::make_tuple(0.5, 1.0));
  math_function.emplace_back(std::make_tuple(1.0, 1.0));

  // Dump math function as an input file to be read
  std::ofstream file;
  file.open("math-function.csv");
  // Write math function file
  for (const auto& math_fn : math_function) {
    file << std::get<0>(math_fn) << "\t";
    file << std::get<1>(math_fn) << "\t";

    file << "\n";
  }

  file.close();
  return true;
}

}  // namespace mpm_test
