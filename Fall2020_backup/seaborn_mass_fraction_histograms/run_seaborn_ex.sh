#!/bin/bash

# bash script to run seaborn_ex.py with different  _*.cfit files
# run with a list cfit_files.lis that has the different _*.cfit files you want to run seaborn_ex.py code with

file_list="$1"

# expand all files in file_list, run a cat command to put the output into another shell variable:

files=`cat $file_list` # ` -- command line argument 


for file in $files; 
do
	echo "Running seaborn_ex.py with $file"
	 python seaborn_ex.py $file &
done
