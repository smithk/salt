#echo base file $1
PATHTOSUGGEST=/home/roger/salt/code/salt/proof_of_concept/suggest.py 
LEARNERS=(PassiveAggressiveClassifier RadiusNeighborsClassifier GaussianNBClassifier ExtraTreeEnsembleClassifier SVMClassifier LinearDiscriminantClassifier KNNClassifier RandomForestClassifier SGDClassifier LogisticRegressionClassifier NearestCentroidClassifier LinearSVMClassifier NuSVMClassifier DecisionTreeClassifier RidgeClassifier QuadraticDiscriminantClassifier GradientBoostingClassifier)

for DATASET in *.arff
do
	DATASETNAME=`echo $DATASET | grep -o '[^\.]*' | head -n 1`

	for LEARNER in ${LEARNERS[*]}
	do
		cd $DATASETNAME
		python $PATHTOSUGGEST data $LEARNER getcandidates
		# or send jobs
		salt -i $DATASET -o list -l $LEARNER
		python $PATHTOSUGGEST data $LEARNER score
		cd ..
	done
done
