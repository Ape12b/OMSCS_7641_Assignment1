
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score, make_scorer
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

def grid_search_func(estimator, params, param_names,
                     cv=5, scoring='accuracy', 
                     X_train=None, X_test=None, 
                     y_train=None, y_test=None,
                     verbose=0, pos_label = None):
    """
    Perform grid search with cross-validation to find the best hyperparameters for the given estimator.

    Parameters:
    estimator (object): The estimator object to be tuned.
    params (list of dicts): List of dictionaries containing the hyperparameters to tune.
    param_names (list of str): List of hyperparameter names.
    cv (int, cross-validation generator, iterable, default=5): Determines the cross-validation splitting strategy.
    scoring (str or callable, default='accuracy'): Scoring metric to use for evaluation.
    X_train (array-like, optional): Training data.
    X_test (array-like, optional): Testing data.
    y_train (array-like, optional): Training labels.
    y_test (array-like, optional): Testing labels.
    verbose (int, default=0): Controls the verbosity of the grid search.
    pos_label (int, default=0): Controls the verbosity of the grid search.

    Returns:
    dict: Best parameters found during grid search.
    """
    # Define the hyperparameters grid
    param_grid = {param_name: param for param_name, param in zip(param_names, params)}

    scorers = {"f1": f1_score,
    "precision": precision_score,
    "recall": recall_score}

    if scoring in ["f1", "recall", "precision"]:
      scorer = make_scorer(scorers[scoring], pos_label = pos_label)
    else:
      scorer = 'accuracy'
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scorer, verbose=verbose)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Train the classifier with the best parameters
    clf = estimator.set_params(**best_params)
    clf.fit(X_train, y_train)

    # Predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate score
    score = accuracy_score(y_test, y_pred)

    print("Best parameters:", best_params)
    print("Test accuracy:", score)
    return (best_params)

def plot_train_test_metrics(train_scores, test_scores, params, param_names, scoring = 'Accuracy'):
    """
    Plot training and test metrics (e.g., accuracy) across different hyperparameter values.

    Parameters:
    train_scores (list of arrays): Training scores for each hyperparameter configuration.
    test_scores (list of arrays): Test scores for each hyperparameter configuration.
    params (list of arrays): Hyperparameter values.
    param_names (list of str): Names of hyperparameters.
    scoring (str): Scoring metric used

    Returns:
    None
    """
    num_params = len(param_names)
    fig, axs = plt.subplots(1, num_params, figsize=(num_params * 5, 5))
    contains_strings = any(isinstance(element, str) for element in params)
    
    for i in range(num_params):
        if not contains_strings:
            axs[i].plot(params[i], train_scores[i], marker="o", drawstyle="steps-post", label='Training Score')
            axs[i].plot(params[i], test_scores[i], marker="o", drawstyle="steps-post", label='Validation Score')
            axs[i].legend(loc='best')
            axs[i].set_title(f"Training and Test {scoring}: {param_names[i]}")
            axs[i].set_xlabel(param_names[i])
            axs[i].set_ylabel(scoring)
        else:
            axs[i].bar(params[i], train_scores[i], label='Training Score')
            axs[i].bar(params[i], test_scores[i],  label='Validation Score')
            axs[i].legend(loc='best')
            axs[i].set_title(f"Training and Test {scoring}: {param_names[i]}")
            axs[i].set_xlabel(param_names[i])
            axs[i].set_ylabel(scoring)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

def plot_learning_curves(X_train, X_test, y_train, y_test, clfs, plot_train=True, cv=5, names=None):
    """
    Plot learning curves for multiple classifiers.

    Parameters:
    X_train (array-like): Training data.
    X_test (array-like): Testing data.
    y_train (array-like): Training labels.
    y_test (array-like): Testing labels.
    clfs (list of classifiers): List of classifier instances.
    plot_train (bool, default=True): Whether to plot training scores.
    cv (int, cross-validation generator, iterable, optional, default=5): Determines the cross-validation splitting strategy.
    names (list of str, optional): List of names for each classifier.

    Returns:
    None
    """
    i = 0
    j = 0
    num_colors = len(clfs) * 2
    cmap = plt.get_cmap('viridis', num_colors)
    rgb_values = [cmap(i)[:3] for i in np.linspace(0, 1, num_colors)]

    plt.figure(figsize=(15, 9))
    for clf in clfs:
        start_time = time.time()
        if len(set(y_train)) > 2:
            train_sizes, train_scores, val_scores = learning_curve(clf, X_train, y_train, cv=cv,
                                                                    scoring='neg_log_loss', train_sizes=np.linspace(0.1, 1.0, 10))
            train_scores = -train_scores
            val_scores = -val_scores
        else:
            train_sizes, train_scores, val_scores = learning_curve(clf, X_train, y_train, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_sem = np.std(train_scores, axis=1) / np.sqrt(len(train_scores))
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_sem = np.std(val_scores, axis=1) / np.sqrt(len(val_scores))

        if names is None:
            clf_name = set([clf.__class__.__name__ for clf in clfs])
            title = " vs. ".join([name for name in clf_name])
            label_train = f'Training Score {clf.__class__.__name__}'
            label_val = f'Validation Score {clf.__class__.__name__}'
        else:
            title = " vs. ".join([name for name in names])
            label_train = f'Training Score: {names[j]}'
            label_val = f'Validation Score: {names[j]}'
            j += 1

        if plot_train:
            plt.plot(train_sizes, train_scores_mean, label=label_train, color=rgb_values[i], marker='o')
            plt.fill_between(train_sizes, train_scores_mean - train_scores_sem, train_scores_mean + train_scores_sem, alpha=0.2, color=rgb_values[i])

        plt.plot(train_sizes, val_scores_mean, label=label_val, color=rgb_values[i + 1], marker='o')
        plt.fill_between(train_sizes, val_scores_mean - val_scores_sem, val_scores_mean + val_scores_sem, alpha=0.2, color=rgb_values[i + 1])
        i += 2

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{clf.__class__.__name__} time: {elapsed_time:.4f} seconds")

    plt.title(title)
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy' if len(set(y_train)) == 2 else 'Negative Log Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_score(y_true, y_pred, score_name, pos_label = 'M'):
    """
    Calculate and return different scores based on the passed string name of the score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    score_name (str): Name of the score ('accuracy', 'precision', 'recall', 'f1').

    Returns:
    float: Calculated score.
    """
    if score_name == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif score_name == 'precision':
        return precision_score(y_true, y_pred, pos_label = 'M')
    elif score_name == 'recall':
        return recall_score(y_true, y_pred, pos_label = 'M')
    elif score_name == 'f1':
        return f1_score(y_true, y_pred, pos_label = 'M')
    else:
        raise ValueError("Invalid score name. Please choose from 'accuracy', 'precision', 'recall', or 'f1'.")

def train_clfs_with_hyperparameters(model, param_name, param_values, X_train, y_train):
    """
    Train classifiers with different values of a hyperparameter.

    Parameters:
    model (class): The classifier model to be tuned.
    param_name (str): Name of the hyperparameter.
    param_values (list): List of values for the hyperparameter.
    X_train (array-like): Training data.
    y_train (array-like): Training labels.

    Returns:
    list: List of trained classifiers with different hyperparameter values.
    """
    clfs = []
  
    # Loop through values of the hyperparameter
    for value in param_values:
        params = {param_name: value}
        clf = model(**params)
        clf.fit(X_train, y_train)
        clfs.append(clf)
        
    return clfs