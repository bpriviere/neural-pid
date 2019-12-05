./discretePlanning.sh $1
cd multi-robot-trajectory-planning/smoothener
matlab -nosplash -nodesktop -r "path_setup,smoothener,quit"
cd ../..
./export.sh $1 $2