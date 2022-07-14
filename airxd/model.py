import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

class ARIXD:
    def __init__(self, algorithm, parameters, features):
        self.algorithm = algorithm
        self.parameters = self.set_parameters(algorithm, parameters)
        self.model = self.get_model(algorithm)
        self.features = features
        if self.features['locations']:
            self.xloc, self.yloc = self.get_locations()
        else:
            self.xloc, self.yloc = None, None

        self.no_of_features = 0
        for k, v in self.features.items():
            if v:
                self.no_of_features += 1
                if k == 'locations':
                    self.no_of_features += 1

    def set_parameters(self, algorithm, parameters):
        if algorithm == 'gradient_boosting':
            params = {'n_estimators': 100,
                      'eta': 0.3,
                      'tree_method': 'hist',
                      'gamma': 0.,
                      'max_bin': 10000,
                      'predictor': 'cpu_predictor',
                      'max_depth': 10,
                      'reg_lambda': 1.,
                      'eval_metric': 'rmsle',
                      'n_jobs': 1,
                      'base_score': 0.5,
                      'colsample_bylevel': 1,
                      'colsample_bytree': 1,
                      'min_child_weight': 1,
                      'missing': None,
                      'n_estimators': 100,
                      'nthread': -1,
                      'objective': 'binary:logistic',
                      'reg_alpha': 0,
                      'scale_pos_weight': 1,
                      'seed': 0,
                      'subsample': 1,
                      'max_delta_step': 0}

        elif algorithm in ['random_forest', 'extra_trees']:
            params = {'n_estimators': 100,
                      'criterion': 'gini',
                      'max_depth': None,
                      'min_samples_split': 2,
                      'min_samples_leaf': 1,
                      'min_weight_fraction_leaf': 0.0,
                      'max_features': 'sqrt',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'bootstrap': True,
                      'oob_score': False,
                      'n_jobs': None,
                      'random_state': None,
                      'verbose': 0,
                      'warm_start': False,
                      'class_weight': None,
                      'ccp_alpha': 0.0,
                      'max_samples': None}

        elif algorithm == 'knn':
            params = {'n_neighbors': 5,
                      'weights': 'uniform',
                      'algorithm': 'auto',
                      'leaf_size': 30,
                      'p': 2,
                      'metric': 'minkowski',
                      'metric_params': None,
                      'n_jobs': None}

        params.update(parameters)
        return params

    def get_model(self, algorithm):
        if algorithm == 'gradient_boosting':
            model = XGBClassifier(n_estimators=self.parameters['n_estimators'],
                                       eta=self.parameters['eta'],
                                       tree_method=self.parameters['tree_method'],
                                       gamma=self.parameters['gamma'],
                                       max_bin=self.parameters['max_bin'],
                                       predictor=self.parameters['predictor'],
                                       max_depth=self.parameters['max_depth'],
                                       reg_lambda=self.parameters['reg_lambda'],
                                       eval_metric=self.parameters['eval_metric'],
                                       n_jobs=self.parameters['n_jobs'],
                                       #base_score=self.parameters['base_score'],
                                       #colsample_bylevel=self.parameters['colsample_bylevel'],
                                       #colsample_bytree=self.parameters['colsample_bytree'],
                                       #min_child_weight=self.parameters['min_child_weight'],
                                       #missing=self.parameters['missing'],
                                       #nthread=self.parameters['nthread'],
                                       #objective=self.parameters['objective'],
                                       #reg_alpha=self.parameters['reg_alpha'],
                                       #scale_pos_weight=self.parameters['scale_pos_weight'],
                                       #seed=self.parameters['seed'],
                                       #subsample=self.parameters['subsample'],
                                       #max_delta_step=self.parameters['max_delta_step'])
                                       )

        elif algorithm == 'random_forest':
            model = RandomForestClassifier(n_estimators=self.parameters['n_estimators'],
                                                criterion=self.parameters['gini'],
                                                max_depth=self.parameters['max_depth'],
                                                min_samples_split=self.parameters['min_samples_split'],
                                                min_samples_leaf=self.parameters['min_samples_leaf'],
                                                min_weight_fraction_leaf=self.parameters['min_weight_fraction_leaf'],
                                                max_features=self.parameters['max_features'],
                                                max_leaf_node=self.parameters['max_leaf_node'],
                                                min_impurity_decrease=self.parameters['min_impurity_decrease'],
                                                bootstrap=self.parameters['bootstrap'],
                                                oob_score=self.parameters['oob_score'],
                                                n_jobs=self.parameters['n_jobs'],
                                                random_state=self.parameters['random_state'],
                                                verbose=self.parameters['verbose'],
                                                warm_start=self.parameters['warm_start'],
                                                class_weight=self.parameters['class_weight'],
                                                ccp_alpha=self.parameters['ccp_alpha'],
                                                max_samples=self.parameters['max_samples'])

        elif algorithm == 'extra_trees':
            model = ExtraTreesClassifier(n_estimators=self.parameters['n_estimators'],
                                              criterion=self.parameters['gini'],
                                              max_depth=self.parameters['max_depth'],
                                              min_samples_split=self.parameters['min_samples_split'],
                                              min_samples_leaf=self.parameters['min_samples_leaf'],
                                              min_weight_fraction_leaf=self.parameters['min_weight_fraction_leaf'],
                                              max_features=self.parameters['max_features'],
                                              max_leaf_node=self.parameters['max_leaf_node'],
                                              min_impurity_decrease=self.parameters['min_impurity_decrease'],
                                              bootstrap=self.parameters['bootstrap'],
                                              oob_score=self.parameters['oob_score'],
                                              n_jobs=self.parameters['n_jobs'],
                                              random_state=self.parameters['random_state'],
                                              verbose=self.parameters['verbose'],
                                              warm_start=self.parameters['warm_start'],
                                              class_weight=self.parameters['class_weight'],
                                              ccp_alpha=self.parameters['ccp_alpha'],
                                              max_samples=self.parameters['max_samples'])

        elif algorithm == 'knn':
            model = KNeighborsClassifier(n_neighbors=self.parameters['n_neighbors'],
                                              weights=self.parameters['weights'],
                                              algorithm=self.parameters['algorithm'],
                                              leaf_size=self.parameters['leaf_size'],
                                              p=self.parameters['p'],
                                              metric=self.parameters['metric'],
                                              metric_params=self.parameters['metric_params'],
                                              n_jobs=self.parameters['n_jobs'])

        return model

    def train(self, dataset, include_data='random', training_images=3):
        include = {}
        if include_data == 'random':
            print("Data included in training: ")
            for i in range(dataset.n):
                #n = int(training_split/100*len(images))
                include[i] = random.sample(range(0, len(dataset.images[i])), training_images)
                print(i, ": ", include[i])
        else:
            include = include_data
            for k, v in include.items():
                print(k, ": ", v)
    
        self.X = self.get_feature(dataset, include)
        self.y = self.get_label(dataset, include)
        self.model.fit(self.X, self.y)
        return

    def predict(self, image):
        shp = image.shape
        X = np.zeros((shp[0]*shp[1], self.no_of_features))

        f = 0
        X[:, f] += image.ravel()
        f += 1
        if self.TA is not None:
            X[:, f] += self.TA.ravel()
            f += 1
        if self.xloc is not None and self.yloc is not None:
            X[:, f] += self.xloc.ravel()
            X[:, f+1] += self.yloc.ravel()
            f += 2
        if 'savgol' in self.features:
            for window in self.features['savgol']:
                X[:, f] += self.get_savgol_filter(image, window).ravel()
                f += 1

        X = self.normalizer.transform(X)
        y_pred = self.model.predict(X)

        return y_pred.reshape(shp[0], shp[1])

    def save(self):
        return

    def load(self):
        return

    def get_feature(self, dataset, include_data):
        shp = [dataset.shape[0], dataset.shape[1]]
        n = sum([len(v) for v in include_data.values()])
        X = np.zeros((n*shp[0]*shp[1], self.no_of_features))
        
        c = 0
        for i, images in dataset.images.items():
            for j in include_data[i]:
                f = 0
                X[c:c+shp[0]*shp[1], f] += images[i][j].ravel()
                f += 1
                
                if self.features['angle']:
                    X[c:c+shp[0]*shp[1], f] +=  dataset.TAs[i].ravel()
                    f += 1

                if self.xloc is not None and self.yloc is not None:
                    X[c:c+shp[0]*shp[1], f] += self.xloc.ravel()
                    X[c:c+shp[0]*shp[1], f+1] += self.yloc.ravel()
                    f += 2

                if 'savgol' in self.features:
                    for window in self.features['savgol']:
                        X[c:c+shp[0]*shp[1], f] += self.get_savgol_filter(images[i][j], window).ravel()
                        f += 1
                c += shp[0]*shp[1]
                    
        self.normalizer = StandardScaler()
        X = self.normalizer.fit_transform(X)

        return X

    def get_label(self, dataset, include_data):
        shp = [dataset.shape[0], dataset.shape[1]]
        n = sum([len(v) for v in include_data.values()])
        y = np.zeros((n*dataset.shape[0]*dataset.shape[1], 1))

        c = 0
        for i, labels in dataset.labels.items():
            for j in include_data[i]:
                y[c:c+shp[0]*shp[1], 0] += labels[i][j].ravel()
                c += shp[1]*shp[2]
        
        return y

    def get_savgol_filter(self, image, window, poly=3, order=1):
        xs, ys = image.shape
        dx = np.zeros((xs, ys))
        dy = np.zeros((xs, ys))
        for i in range(0, xs):
            dx[i,:] = savgol_filter(image[i,:], window, poly, deriv=order)
            dy[:,i] = savgol_filter(image[:,i], window, poly, deriv=order)
        return dx+dy

    def get_locations(self):
        i_mesh, j_mesh = np.meshgrid(range(2880), range(2880), indexing='ij')
        return i_mesh, j_mesh
