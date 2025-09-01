#!/bin/bash

# Path to your JSON file
JSON_FILE="../../AcademicPython/MPM/2dRicker/dynamic_pml_riker.json"
# Command to run simulation
SIM_COMMAND="./mpm -f ../../AcademicPython/MPM/2dRicker/ -i dynamic_pml_riker.json"

# Constants for P-wave velocity calculation
E=2000000000    # Young's modulus in Pa
RHO=2000        # Density in kg/m³
NU=0.25         # Poisson's ratio
L=1000          # L value for damping calculation

# R values as specified, increasing by increments of 2
R_VALUES=(5 7 9 11 13 15)

# Function to calculate P-wave velocity
calculate_vp() {
    # V_p = sqrt(E(1-ν)/((1+ν)(1-2ν)ρ))
    local numerator=$(echo "$E * (1 - $NU)" | bc -l)
    local denominator=$(echo "(1 + $NU) * (1 - 2 * $NU) * $RHO" | bc -l)
    local vp=$(echo "sqrt($numerator / $denominator)" | bc -l)
    echo $vp
}

# Function to calculate damping based on R, V_p, and L
calculate_damping() {
    local r_value=$1
    local vp=$(calculate_vp)
    
    # damping = (v_p/L) * ln(10^R)
    # Using natural logarithm (ln) and power function
    local power=$(echo "10^$r_value" | bc -l)
    local ln_power=$(echo "l($power)" | bc -l)  # l() is natural log in bc
    local damping=$(echo "($vp / $L) * $ln_power" | bc -l)
    
    echo $damping
}

# Function to update the JSON file
update_json() {
    local r_value=$1
    local damping_ratio=$(calculate_damping $r_value)
    local new_uuid="simulation_R_${r_value}"
    
    # Use jq to update the JSON file
    # 1. Update the second material's maximum_damping_ratio
    # 2. Update the UUID in analysis section to use R value
    jq --arg damping "$damping_ratio" \
       --arg uuid "$new_uuid" \
       '.materials[1].maximum_damping_ratio = ($damping|tonumber) | .analysis.uuid = $uuid' \
       "$JSON_FILE" > temp.json
    
    # Replace the original file with the modified one
    mv temp.json "$JSON_FILE"
    
    echo "Updated JSON file with R value: $r_value"
    echo "Calculated damping ratio: $damping_ratio"
    echo "New UUID: $new_uuid"
}

# Main execution
echo "Starting sequential simulations..."
VP=$(calculate_vp)
echo "Calculated P-wave velocity: $VP m/s"

for r in "${R_VALUES[@]}"; do
    echo "=========================================="
    echo "Running simulation with R value: $r"
    
    # Update the JSON file
    update_json "$r"
    
    # Run the simulation
    $SIM_COMMAND
    
    echo "Simulation with R value $r completed"
    echo "=========================================="
    echo ""
done

echo "All simulations completed!"
