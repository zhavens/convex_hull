#1 /bin/bash

algos=("dc" "gw" "grahams" "chans")
inputs=$(find inputs -maxdepth 1 -type f | sed -e "s|inputs/||g")
# inputs="clustered500K_100c_box"

for algo in ${algos[@]}; do
	for input in ${inputs[@]}; do
		echo "Running $algo against $input"
		python3 ./src/convex_hull_main.py \
			--algo=$algo \
			--infile=inputs/$input \
			--hull_dir=/tmp/hulls \
			--stats_outfile=results/results.csv \
			--validate_hull=false \
			--chans_eliminate_points \
			--chans_subset_algo=dc \
			--alsologtostderr 2> /tmp/${algo}_${input}
		if [ $? -ne 0 ]; then
			echo "Error running $algo against $input."
		else
			echo "Complete."
		fi
	done
done
