import subprocess
from pathlib import Path

def check_files_exist(directory, pattern):
    """Check if any files matching the pattern exist in the directory."""
    path = directory / pattern
    return any(path.parent.glob(path.name))

def run_script(script_path):
    """Run a Python script using subprocess."""
    print(f"Running script: {script_path}")
    subprocess.run(["python", str(script_path)], check=True)

def main():
    # Get project root relative to this script
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    # First run the map_info_gen.py script as requested
    map_info_script = project_root / "data_qa_generate" / "data_traj_generate" / "map_info_gen.py"
    if map_info_script.exists():
        print("\nRunning map_info_gen.py first as requested")
        run_script(map_info_script)
    else:
        print(f"\nWarning: map_info_gen.py not found at {map_info_script}")

    datasets = [ "kitti", "once","argoverse","idd","waymo","map"]
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Path configurations
        qa_results_dir = project_root / "data_qa_results" / dataset
        qa_generate_dir = project_root / "data_qa_generate"
        traj_results_dir = qa_generate_dir / "data_traj_generate" / "data_traj_results" / dataset
        
        # Check condition 1: if no q3 files in qa_results_dir
        if not check_files_exist(qa_results_dir, "q3*.json*"):
            script_path = qa_generate_dir / f"{dataset}_qa.py"
            if script_path.exists():
                run_script(script_path)
            else:
                print(f"Script not found: {script_path}")
        
        # Check condition 2: if no json/jsonl files in traj_results_dir
        if not (check_files_exist(traj_results_dir, "traj*.json") or 
                check_files_exist(traj_results_dir, "traj*.jsonl")):
            traj_script_path = qa_generate_dir / "data_traj_generate" / f"pipeline_{dataset}_traj.py"
            if traj_script_path.exists():
                run_script(traj_script_path)
                
                # After running traj script, check for q7 files
                if not check_files_exist(qa_results_dir, "q7*.json*"):
                    q7_script_path = qa_generate_dir / "data_traj_generate" / f"q7_{dataset}.py"
                    if q7_script_path.exists():
                        run_script(q7_script_path)
                    else:
                        print(f"Script not found: {q7_script_path}")
            else:
                print(f"Script not found: {traj_script_path}")


main()