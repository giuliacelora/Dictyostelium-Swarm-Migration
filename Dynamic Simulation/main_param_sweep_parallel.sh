#!/bin/bash 
for ((j=0;j<=10;j++)); do
	for ((i=0;i<=10;i++)); do
		python3 odel_thin_film_migration.py $i $j &
	done
done

