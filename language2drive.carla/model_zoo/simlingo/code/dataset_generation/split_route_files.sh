sh ~/software/carla0915/CarlaUE4.sh -RenderOffScreen -nosound -graphicsadapter=0 --world-port=6222 &
sleep 180

export CARLA_ROOT=~/software/carla0915
export WORK_DIR=~/coding/wayve_carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.8-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

# Custom routes
echo "Generating custom routes"
python dataset_generation/split_route_files.py --seed 10 --path-in leaderboard/data/parking_lane/Town12_short.xml --save-path data/simlingo/training_parking_lane --max-scenarios 0 --routes-per-file 1
python dataset_generation/balance_scenarios.py --seed 10 --num-samples 150 --path-in data/simlingo/training_parking_lane/Town12_short/random_weather_seed_10 --routes-per-file 1

echo "Generating custom routes"
python dataset_generation/split_route_files.py --seed 11 --path-in leaderboard/data/parking_lane/Town13_short.xml --save-path data/simlingo/validation_parking_lane --max-scenarios 0 --routes-per-file 1
python dataset_generation/balance_scenarios.py --seed 11 --num-samples 150 --path-in data/simlingo/validation_parking_lane/Town13_short/random_weather_seed_11 --routes-per-file 1

# Generate training routes with 1 scenarios and upsample
echo "Generating training routes with 1 scenario"
python dataset_generation/split_route_files.py --seed 1 --path-in leaderboard/data/routes_training.xml --save-path data/simlingo/training_1_scenario --max-scenarios 1 --routes-per-file 1
python dataset_generation/balance_scenarios.py --seed 1 --num-samples 150 --path-in data/simlingo/training_1_scenario/routes_training/random_weather_seed_1 --routes-per-file 1

# Generate validation routes with 1 scenarios and upsample
echo "Generating validation routes with 1 scenario"
python dataset_generation/split_route_files.py --seed 2 --path-in leaderboard/data/routes_validation.xml --save-path data/simlingo/validation_1_scenario --max-scenarios 1 --routes-per-file 1
python dataset_generation/balance_scenarios.py --seed 2 --num-samples 150 --path-in data/simlingo/validation_1_scenario/routes_validation/random_weather_seed_2 --routes-per-file 1

# Generate training routes with 3 scenarios and upsample
echo "Generating training routes with 3 scenarios"
python dataset_generation/split_route_files.py --seed 3 --path-in leaderboard/data/routes_training.xml --save-path data/simlingo/training_3_scenarios --max-scenarios 3 --routes-per-file 1
python dataset_generation/balance_scenarios.py --seed 3 --num-samples 100 --path-in data/simlingo/training_3_scenarios/routes_training/random_weather_seed_3 --routes-per-file 1

# Generate validation routes with 3 scenarios and upsample
echo "Generating validation routes with 3 scenarios"
python dataset_generation/split_route_files.py --seed 4 --path-in leaderboard/data/routes_validation.xml --save-path data/simlingo/validation_3_scenarios --max-scenarios 3 --routes-per-file 1
python dataset_generation/balance_scenarios.py --seed 4 --num-samples 100 --path-in data/simlingo/validation_3_scenarios/routes_validation/random_weather_seed_4 --routes-per-file 1


pkill -9 -f CarlaUE4