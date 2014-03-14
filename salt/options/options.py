# from datetime import timedelta
from configobj import ConfigObj
import os
import re
from ..learn import classifiers
from collections import OrderedDict


class Settings(object):
    '''
    def __init__(self, config=None):
        # Build a dictionary with settings
        if config is None:
            config = Settings.load_or_create_config()

        # Load options for Logistic Regression classifier
        learner_config = config['Classifiers']['LogisticRegressionClassifier']
        _penalty = learner_config['penalty']
        _dual = learner_config['dual']

        log_reg_classif = {'l1_prior': 0.05,
                           'l2_prior': 0.95,
                           'dual_false_prior': 0.75,
                           'dual_true_prior': 0.25,
                           'C_prior_lower': 0.0,
                           'C_prior_upper': 2 ** 10,
                           'fit_intercept_false_prior': 0.5,
                           'fit_intercept_true_prior': 0.5,
                           'intercept_scaling_lower_prior': 0.0,
                           'intercept_scaling_upper_prior': 5.0,
                           'class_weight_none_prior': 0.5,
                           'class_weight_auto_prior': 0.5,
                           'tolerance_prior_mean': 1e-4,
                           'tolerance_prior_stdev': 1
                           }
        self.learner_settings = {'log_reg_classif': log_reg_classif, }
        self.timeout = timedelta(seconds=1)  # weeks, days, hours, minutes, seconds, milliseconds, microseconds
        self.random_state = 283
    '''

    @classmethod
    def create_default_config(self):
        config = ConfigObj()
        config.initial_comment = ['This file has been automatically generated by SALT.', '']
        config.indent_type = '    '
        config.list_values = False
        config['Global'] = {'timeout': 10,
                            'randomstate': 283,
                            'maxprocesses': 10,
                            'localcores': 0,
                            'nodes': ['127.0.0.1'],
                            'ip_addr': '127.0.0.1',
                            'crossvalidation': '15x2',
                            'holdout': 0.3}
        config['GUI'] = {'maxresults': 10}

        log_reg_classif_options = classifiers.LogisticRegressionClassifier.get_default_cfg()
        sgd_classif_options = classifiers.SGDClassifier.get_default_cfg()
        pac_classif_options = classifiers.PassiveAggressiveClassifier.get_default_cfg()
        ridge_classif_options = classifiers.RidgeClassifier.get_default_cfg()
        gaussian_classif_options = classifiers.GaussianNBClassifier.get_default_cfg()
        knn_classif_options = classifiers.KNNClassifier.get_default_cfg()
        radius_classif_options = classifiers.RadiusNeighborsClassifier.get_default_cfg()
        centroid_classif_options = classifiers.NearestCentroidClassifier.get_default_cfg()
        tree_classif_options = classifiers.DecisionTreeClassifier.get_default_cfg()
        random_forest_classif_options = classifiers.RandomForestClassifier.get_default_cfg()
        extra_trees_classif_options = classifiers.ExtraTreeEnsembleClassifier.get_default_cfg()
        grad_boost_classif_options = classifiers.GradientBoostingClassifier.get_default_cfg()
        gaussian_proc_classif_options = classifiers.GaussianProcessClassifier.get_default_cfg()
        svm_classif_options = classifiers.SVMClassifier.get_default_cfg()
        linearsvm_classif_options = classifiers.LinearSVMClassifier.get_default_cfg()
        nu_svm_classif_options = classifiers.NuSVMClassifier.get_default_cfg()
        linear_disc_classif_options = classifiers.LinearDiscriminantClassifier.get_default_cfg()
        quadratic_disc_classif_options = classifiers.QuadraticDiscriminantClassifier.get_default_cfg()

        classifier_options = OrderedDict()

        classifier_options['LogisticRegressionClassifier'] = log_reg_classif_options
        classifier_options['SGDClassifier'] = sgd_classif_options
        classifier_options['PassiveAggressiveClassifier'] = pac_classif_options
        classifier_options['RidgeClassifier'] = ridge_classif_options
        classifier_options['GaussianNBClassifier'] = gaussian_classif_options
        classifier_options['KNNClassifier'] = knn_classif_options
        classifier_options['RadiusNeighborsClassifier'] = radius_classif_options
        classifier_options['NearestCentroidClassifier'] = centroid_classif_options
        classifier_options['DecisionTreeClassifier'] = tree_classif_options
        classifier_options['RandomForestClassifier'] = random_forest_classif_options
        classifier_options['ExtraTreeEnsembleClassifier'] = extra_trees_classif_options
        classifier_options['GradientBoostingClassifier'] = grad_boost_classif_options
        '''
        classifier_options['GaussianProcessClassifier'] = gaussian_proc_classif_options
        '''
        classifier_options['SVMClassifier'] = svm_classif_options
        classifier_options['LinearSVMClassifier'] = linearsvm_classif_options
        classifier_options['NuSVMClassifier'] = nu_svm_classif_options
        classifier_options['LinearDiscriminantClassifier'] = linear_disc_classif_options
        classifier_options['QuadraticDiscriminantClassifier'] = quadratic_disc_classif_options

        config['Classifiers'] = classifier_options
        for section in config['Classifiers'].sections:
            if config['Classifiers'][section] == {}:
                config['Classifiers'].inline_comments[section] = 'This classifier does not accept any parameters.'
        return config

    @classmethod
    def validate_config(self, config):
        # TODO Implement settings validation
        return True

    @classmethod
    def read_config(self, path):
        config = ConfigObj(path)
        return config if Settings.validate_config(config) else None

    @classmethod
    def load_or_create_config(self, path='salt.ini'):
        if os.path.exists(path):
            config = Settings.read_config(path)
            if config is None:
                # TODO Report invalid configuration file
                config = Settings.create_default_config()
        else:
            config = self.create_default_config()  # TODO Report default config file creation?
            config.filename = 'salt.ini'
            config.write()
        return config

    @classmethod
    def read_nodes(self, nodedef):
        nodes = []
        if type(nodedef) is not list:
            nodedef = [nodedef]
        readfile_regex = 'read\((\'(?P<fname1>.+)\'|\"(?P<fname2>.+)\"|(?P<fname3>.+))\)'
        for node in nodedef:
            parsed = re.match(readfile_regex, node)
            if parsed is None:
                nodes.append(node)
            else:
                fname_values = parsed.groupdict()
                filename = fname_values.get('fname1') or fname_values.get('fname2') or fname_values.get('fname3')
                try:
                    if filename is not None:
                        with open(filename) as nodefile:
                            nodes.extend([node[:-1] for node in nodefile.readlines() if node[:-1] != ''])
                except IOError as exc:
                    print("Hosts file {0}: {1}".format(exc.filename, exc.strerror))
        return nodes

    @classmethod
    def get_crossvalidation(self, crossvalidation):
        """Parses a cross-validation specification.
        Examples of valid cross-validation specifications:
            10: 10-fold cross-validation.
            2x15: 2 repetitions of 15-fold cross-validation.
        """
        xval_parser = re.match("((?P<xval_rep>\d+)x)?(?P<xval_folds>\d+)", crossvalidation)
        xval_values = xval_parser.groupdict()
        xval_folds = int(xval_values['xval_folds'])
        xval_rep = int(xval_values['xval_rep'] or 1)
        return xval_rep, xval_folds
