#!/usr/bin/env python3
"""
Script to run assignHouseholds.py for all 17 area codes,
performing hyperparameter tuning for household assignment models across all Oxford areas.
"""

import os
import sys
import subprocess
import time

# Define the area codes to process (all 17 Oxford areas)
# AREA_CODES = [
#     'E02005940', 'E02005941', 'E02005942', 'E02005943',
#     'E02005944', 'E02005945', 'E02005946', 'E02005947',
#     'E02005948', 'E02005949', 'E02005950', 'E02005951',
#     'E02005953', 'E02005954', 'E02005955', 'E02005956',
#     'E02005957'
# ]

AREA_CODES = [
    'E02005941', 'E02005942', 'E02005943',
    'E02005944', 'E02005945', 'E02005946', 'E02005947',
    'E02005948', 'E02005949', 'E02005950',
    'E02005953', 'E02005954', 'E02005955', 'E02005956'
]

def run_command(cmd, description, working_dir=None):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    if working_dir:
        print(f"Working directory: {working_dir}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=working_dir)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ Completed successfully in {duration:.1f} seconds")
            return True
        else:
            print(f"\n❌ Failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

def check_prerequisites(area_code):
    """Check if required input files exist for the area code"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for required input files
    person_nodes_path = os.path.join(current_dir, f"../outputs/individuals_{area_code}/person_nodes.pt")
    household_nodes_path = os.path.join(current_dir, f"../outputs/households_{area_code}/household_nodes.pt")
    
    missing_files = []
    if not os.path.exists(person_nodes_path):
        missing_files.append(f"individuals_{area_code}/person_nodes.pt")
    if not os.path.exists(household_nodes_path):
        missing_files.append(f"households_{area_code}/household_nodes.pt")
    
    return missing_files

def main():
    """Main function to run assignment HP tuning for all areas"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Change to parent directory (code/) where the main scripts are located
    code_dir = os.path.dirname(current_dir)
    
    print("="*80)
    print("RUNNING HOUSEHOLD ASSIGNMENT HYPERPARAMETER TUNING FOR ALL AREAS")
    print("="*80)
    print(f"Processing area codes: {', '.join(AREA_CODES)}")
    print(f"utils directory: {current_dir}")
    print(f"Code directory: {code_dir}")
    print("\nThis script will:")
    print("  • Run hyperparameter tuning for household assignment models")
    print("  • Process all 17 Oxford area codes")
    print("  • Generate assignment results for each area")
    print("  • Create performance plots and analysis")
    
    # Track results
    assignment_results = {}
    areas_with_missing_files = {}
    
    # Check prerequisites for all areas first
    print(f"\n{'='*80}")
    print("CHECKING PREREQUISITES")
    print(f"{'='*80}")
    
    for area_code in AREA_CODES:
        missing_files = check_prerequisites(area_code)
        if missing_files:
            areas_with_missing_files[area_code] = missing_files
            print(f"⚠️  {area_code}: Missing files - {', '.join(missing_files)}")
        else:
            print(f"✅ {area_code}: All required files found")
    
    if areas_with_missing_files:
        print(f"\n⚠️  WARNING: {len(areas_with_missing_files)} areas have missing input files.")
        print("These areas will be skipped. Run individual/household generation first.")
        print("\nMissing files by area:")
        for area_code, missing_files in areas_with_missing_files.items():
            print(f"  {area_code}: {', '.join(missing_files)}")
    
    # Process each area code
    successful_areas = [area for area in AREA_CODES if area not in areas_with_missing_files]
    
    if not successful_areas:
        print(f"\n❌ No areas have complete input files. Exiting.")
        return
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {len(successful_areas)} AREAS WITH COMPLETE INPUT FILES")
    print(f"{'='*80}")
    
    total_start_time = time.time()
    
    for i, area_code in enumerate(successful_areas, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING AREA {i}/{len(successful_areas)}: {area_code}")
        print(f"{'='*80}")
        
        # Run assignment hyperparameter tuning from code directory
        assignment_cmd = [sys.executable, 'assignHouseholds.py', '--area_code', area_code]
        assignment_success = run_command(
            assignment_cmd, 
            f"Assignment HP Tuning for {area_code}",
            working_dir=code_dir
        )
        assignment_results[area_code] = assignment_success
        
        # Show progress
        completed = sum(1 for success in assignment_results.values() if success)
        failed = len(assignment_results) - completed
        remaining = len(successful_areas) - len(assignment_results)
        
        print(f"\n📊 Progress Summary:")
        print(f"  Completed: {completed}/{len(successful_areas)}")
        print(f"  Failed: {failed}")
        print(f"  Remaining: {remaining}")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    print(f"Total execution time: {total_duration/60:.1f} minutes")
    
    print("\nAssignment HP Tuning Results:")
    successful_count = 0
    for area_code in AREA_CODES:
        if area_code in areas_with_missing_files:
            print(f"  {area_code}: ⏭️  SKIPPED (missing input files)")
        elif area_code in assignment_results:
            success = assignment_results[area_code]
            status = "✅ SUCCESS" if success else "❌ FAILED"
            if success:
                successful_count += 1
            print(f"  {area_code}: {status}")
        else:
            print(f"  {area_code}: ❓ NOT PROCESSED")
    
    # Overall status
    total_processed = len(assignment_results)
    total_successful = successful_count
    total_failed = total_processed - total_successful
    total_skipped = len(areas_with_missing_files)
    
    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total areas: {len(AREA_CODES)}")
    print(f"Processed: {total_processed}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")
    
    if total_successful == total_processed and total_processed > 0:
        print("\n🎉 ALL PROCESSED AREAS COMPLETED SUCCESSFULLY!")
        print("\nGenerated files can be found in:")
        print("  - code/outputs/assignment_hp_tuning_{area_code}/")
        print("  - Each area will have its own directory with:")
        print("    • Hyperparameter tuning results")
        print("    • Assignment performance plots")
        print("    • Model accuracy metrics")
        print("    • Edge index files")
    elif total_successful > 0:
        print(f"\n✅ {total_successful} AREAS COMPLETED SUCCESSFULLY")
        if total_failed > 0:
            print(f"⚠️  {total_failed} AREAS FAILED - CHECK LOGS ABOVE")
        if total_skipped > 0:
            print(f"⏭️  {total_skipped} AREAS SKIPPED - MISSING INPUT FILES")
    else:
        print("\n❌ NO AREAS COMPLETED SUCCESSFULLY")
        if total_skipped > 0:
            print("Make sure to run individual and household generation first")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 