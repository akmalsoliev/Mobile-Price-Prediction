from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import clone


def main():
    pass

def metrics_display(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g', square=True, cbar=False)
    plt.ylabel('Predicted Label'), plt.xlabel('True Label')
    plt.show()


def decision_plotted(model, X, y, feature_list):
    def color_mapping(x):
        colors = ['blue', 'red', 'purple', 'brown', 'yellow', 'green', 'darkblue', 'magenta']
        return colors[x]

    X_sample = X[feature_list].sample(300)
    y_sample = y[X_sample.index]

    model_plot = clone(model)
    model_plot.fit(X_sample, y_sample)
    y_color = y_sample.map(color_mapping)

    x_grid, y_grid = np.arange(0, 1.01, 0.1), np.arange(0, 1.01, 0.1)
    xx_mesh, yy_mesh = np.meshgrid(x_grid, y_grid)
    xx, yy = xx_mesh.ravel(), yy_mesh.ravel()
    X_grid = pd.DataFrame([xx, yy]).T

    zz = model_plot.predict(X_grid)
    zz = zz.reshape(xx_mesh.shape)

    plt.figure(figsize=(10, 10))
    plt.scatter(X_sample.iloc[:, 0], X_sample.iloc[:, 1], color=y_color)
    plt.contourf(xx_mesh, yy_mesh, zz, alpha=0.3)
    plt.xlabel(feature_list[0]), plt.ylabel(feature_list[1]), plt.title('Decision boundary')
    plt.show()


def trees_oob_error(model, X, y, max_trees):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    obb_error_list = list()

    n_trees = np.linspace(15, max_trees, 10).astype(int)

    for n_trees in n_trees:
        model.set_params(n_estimators=n_trees)
        model.fit(X_train, y_train)
        obb_error = 1 - model.oob_score_
        obb_error_list.append(pd.Series(
            {'n_trees': n_trees, 'obb_error': obb_error}
        ))

    obb_error_df = pd.concat(obb_error_list, axis=1).T.set_index('n_trees')

    return obb_error_df


def trees_accuracy_error(model, X, y, max_trees):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    obb_error_list = list()

    n_trees = np.linspace(15, max_trees, 10).astype(int)

    for n_trees in n_trees:
        model.set_params(n_estimators=n_trees)
        model.fit(X_train, y_train)
        obb_error = 1 - accuracy_score(y_test, model.predict(X_test))
        obb_error_list.append(pd.Series(
            {'n_trees': n_trees, 'obb_error': obb_error}
        ))

    error_df = pd.concat(obb_error_list, axis=1).T.set_index('n_trees')

    return error_df


if __name__ == '__main__':
    main()
