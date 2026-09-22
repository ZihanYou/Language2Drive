import json
import re
import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
root_dir = f"{project_root}/data_qa_results"
output_disabled = f"{project_root}/data_qa_generate/data_traj_generate/data_traj_results/q7_filter_disabled.jsonl"
    
def parse_trajectory(content: str) -> List[Tuple[float, float]]:
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, content)
    traj = []
    
    for match in matches:
        if any(keyword in match.lower() for keyword in ['displacement', 'acceleration', 'velocity', 'theta']):
            continue
            
        coords = match.split(',')
        if len(coords) != 2:
            continue
        
        try:
            x = float(coords[0].strip())
            y = float(coords[1].strip())
            traj.append((x, y))
        except ValueError:
            continue
    return traj[:10]

def process_json_file(file_path: str, output_disabled: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {file_path}")
            return 0
    
    if not isinstance(data, list):
        print(f"Skipping non-list JSON file: {file_path}")
        return 0
    
    filtered_data = []
    disabled_records = []
    
    for record in data:
        try:
            assistant_content = next(
                msg['content'] for msg in record['messages'] 
                if msg['role'] == 'assistant'
            )
            trajectory = parse_trajectory(assistant_content)
            
            # Check if any x coordinate is < -1
            if any(x < -1 for x, y in trajectory):
                disabled_records.append(record)
            else:
                filtered_data.append(record)
        except (KeyError, StopIteration):
            # Keep records that don't have the expected structure
            filtered_data.append(record)
    
    # Write disabled records to output file
    if disabled_records:
        with open(output_disabled, 'a', encoding='utf-8') as f:  # Changed to 'a' for append mode
            for record in disabled_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Save filtered data back to original file if changes were made
    if len(disabled_records) > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
    
    return len(disabled_records)

def main():

    # Initialize the output file (create empty or clear if exists)
    open(output_disabled, 'w', encoding='utf-8').close()
    
    total_disabled = 0
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith('q7') and file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    disabled_count = process_json_file(file_path, output_disabled)
                    total_disabled += disabled_count
                    if disabled_count > 0:
                        print(f"Processed {file_path}: filtered {disabled_count} records")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Processing complete. Total disabled records: {total_disabled}")

if __name__ == "__main__":
    main()