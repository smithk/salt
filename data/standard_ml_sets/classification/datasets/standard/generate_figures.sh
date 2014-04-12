GENPLOTSSCRIPT=/home/roger/salt/code/salt/proof_of_concept/generate_plots.py
mkdir -p figures
rm -rf figures/*

for DATASET in *.arff
do
	DATASETNAME=`echo $DATASET | grep -o '[^\.]*' | head -n 1`
	python $GENPLOTSSCRIPT $DATASETNAME $DATASETNAME
done
