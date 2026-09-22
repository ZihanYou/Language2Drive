
import glob
import shutil
import pathlib


dataset_path = 'database/simlingo/data'
all_data_folders = glob.glob(f'{dataset_path}/**/Town*', recursive=True)
delete = False

# if multiple foulder only differ by the time and date (last part of the path) we delete all but the newest one
already_checked_root = []
num_deleted = 0
for data_folder in all_data_folders:
    data_folder = pathlib.Path(data_folder)
    data_folder_name = data_folder.name
    data_folder_parts = data_folder_name.split('route')[-1]
    # remove everything before first _
    data_folder_date_time = '_'.join(data_folder_parts.split('_')[1:])
    path_without_date_time = str(data_folder).split(data_folder_date_time)[0]
    if path_without_date_time in already_checked_root:
        continue
    already_checked_root.append(path_without_date_time)
    
    all_data_folders_without_date_time = glob.glob(f'{path_without_date_time}*')
    if len(all_data_folders_without_date_time) > 1:
        all_data_folders_without_date_time.sort()
        for folder in all_data_folders_without_date_time[:-1]:
            print(f"Deleting {folder}")
            num_deleted += 1
            if delete:
                shutil.rmtree(folder) # uncomment to delete the folder

print(f"Deleted {num_deleted} folders out of {len(all_data_folders)}")