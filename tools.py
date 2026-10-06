# ---------------------------------- #
#         HELPER FUNCTIONS           #
# ---------------------------------- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Data Generating Process

def simulate_data(n, p, true_ate, seed):
    # set seed
    rng = np.random.default_rng(seed)

    # covariates
    X = rng.normal(size=(n, p))

    # Treatment assignment mechanism (nuisanse function)
    log_odds_prop = (
        0.8 * X[:, 0]
        - 0.7 * X[:, 1]**2
        + 0.6 * np.abs(X[:, 2])
        - 0.5 * X[:, 3]
        + 0.5 * X[:, 3]*X[:, 0]
    )
    e = 1 / (1 + np.exp(-log_odds_prop))
    D = rng.binomial(1, e)

    # Covariates impact on Y
    g = (
        1.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 1.0 * np.sin(X[:, 2])
        - .05 * np.exp(X[:, 3])
        + 1.0 * X[:, 4]*X[:,1]
    )

    u = rng.normal(size=n)

    Y = g + true_ate * D + u

    return X, D, Y

# Bias visualuzation

def bias_plot(Y, D, X, ATE):
    mean_treated = Y[D == 1].mean()
    mean_untreated = Y[D != 1].mean()
    biased_estimation =  mean_treated - mean_untreated


    fig, axs = plt.subplots(1,2, figsize=(12,6))
    axs = axs.flatten()

    D_plot = pd.Series(D).map({0:"Untreated", 1:"Treated"})
    palette = {"main": "orange", "secondary": "ligthgray", "alternative": "black"}

    sns.violinplot(x=D_plot, y=Y,
                hue=D_plot,
                #hue_order=['Treated', 'Untreated'],
                palette={"Treated": "orange", "Untreated": "lightgray"},
                width=.4,
                dodge=False,
                ax=axs[0])
    axs[0].set_xlabel("")

    sns.scatterplot(x=['Untreated', 'Treated'],
                    hue=['Untreated', 'Treated'],
                    style=['Untreated', 'Treated'],
                    palette={"Treated": "orange", "Untreated": "lightgray"},
                    y=[mean_untreated, mean_treated],
                    s=100,
                    zorder=10,
                    ax=axs[0])

    sns.barplot(x=['Biased Estimation', 'ATE'],
                hue=['Biased Estimation', 'ATE'],
                y=[biased_estimation, ATE],
                palette={"ATE": "orange", "Biased Estimation": "lightgray"},
                width=.025,
                ax=axs[1])
    sns.scatterplot(x=['Biased Estimation', 'ATE'],
                    hue=['Biased Estimation', 'ATE'],
                    style=['Biased Estimation', 'ATE'],
                    palette={"ATE": "orange", "Biased Estimation": "lightgray"},
                    y=[biased_estimation, ATE],
                    s=300,
                    ax=axs[1])

    axs[0].set_title("E[Y|D=1] vs E[Y|D=0]", fontstyle='italic')
    axs[1].set_title("E[Y|D=1] - E[Y|D=0]", fontstyle='italic')
    plt.suptitle(f"Naive Estimation Bias (%): {(biased_estimation - ATE)/biased_estimation:.2%}", fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

# S-learner

def estimate_s_learner(X, D, Y):

    XD = np.column_stack([X, D])

    model = make_pipeline(
        StandardScaler(),
        LassoCV(alphas=np.logspace(-3, 3, 25), max_iter=20_000)
    )

    model.fit(XD, Y)

    X_treated = np.column_stack([X, np.ones(len(X))])
    X_control = np.column_stack([X, np.zeros(len(X))])

    mu1_hat = model.predict(X_treated)
    mu0_hat = model.predict(X_control)

    ate_hat = np.mean(mu1_hat - mu0_hat)

    return ate_hat

# DML-AIPW

def estimate_dml_aipw(X, D, Y, seed, n_splits=5, components=False):
    n = len(Y)

    # estimations
    mu1_hat = np.zeros(n)
    mu0_hat = np.zeros(n)
    propensity_hat = np.zeros(n)

    # cross fit
    folds = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    # ridge mdl for outcome 
    outcome_model = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-3, 3, 25)),
    )

    # logistic reg for propensity model
    propensity_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=5_000,
        ),
    )

    # cross fit to get the propencity scores, and mu_0 and mu_1 estimations
    for train_idx, test_idx in folds.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        D_train = D[train_idx]
        Y_train = Y[train_idx]

        # propensity score
        propensity_fold = clone(propensity_model)
        propensity_fold.fit(X_train, D_train)
        propensity_hat[test_idx] = propensity_fold.predict_proba(X_test)[:, 1]

        # E[Y | D=1, X]
        treated_model = clone(outcome_model)
        treated_model.fit(
            X_train[D_train == 1],
            Y_train[D_train == 1],
        )
        mu1_hat[test_idx] = treated_model.predict(X_test)

        # E[Y | D=0, X]
        control_model = clone(outcome_model)
        control_model.fit(
            X_train[D_train == 0],
            Y_train[D_train == 0],
        )

        mu0_hat[test_idx] = control_model.predict(X_test)

    # for score stability
    propensity_hat = np.clip(propensity_hat, 0.02, 0.98)

    plugin_score = mu1_hat - mu0_hat
    treatment_correction = D * (Y - mu1_hat) / propensity_hat
    control_correction = - (1 - D) * (Y - mu0_hat) / (1 - propensity_hat)
    
    aipw_score = plugin_score + treatment_correction + control_correction

    ate_hat = aipw_score.mean()
    standard_error = aipw_score.std(ddof=1) / np.sqrt(n)

    if components:
        components_df = pd.DataFrame({
            "plugin_score":plugin_score,
            "treated_correction":treatment_correction,
            "control_correction":control_correction,
            "aipw_score": aipw_score
        })

        return components_df
    else:
        return ate_hat, standard_error

# DML-AIPW decomposition

def plot_aipw_decomposition(components, true_ate):
    # Components
    plugin = components["plugin_score"]
    treated = components["treated_correction"]
    control = components["control_correction"]
    aipw = components["aipw_score"]

    # Rebase relative to the plug-in estimate
    total_correction = aipw - plugin
    true_ate_change = true_ate - plugin

    # Cumulative positions
    after_treated = treated
    after_control = treated + control

    labels = [
        "Plug-in",
        "Treated",
        "Control",
        "AIPW",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plug-in baseline
    ax.scatter(
        0,
        0,
        color="gray",
        s=60,
        zorder=3,
    )

    # Residual corrections
    ax.bar(
        1,
        treated,
        bottom=0,
        width=0.55,
        color="red",
        alpha=.7
    )

    ax.bar(
        2,
        control,
        bottom=after_treated,
        width=0.55,
        color="steelblue",
    )

    # Total AIPW correction
    ax.bar(
        3,
        total_correction,
        width=0.55,
        color="orange",
    )

    # Waterfall connectors
    ax.hlines(0, 0.28, 0.72, color="gray", linewidth=1)
    ax.hlines(after_treated, 1.28, 1.72, color="gray", linewidth=1)
    ax.hlines(after_control, 2.28, 2.72, color="gray", linewidth=1)

    # References
    ax.axhline(0, color="black", linewidth=0.8)

    ax.axhline(
        true_ate_change,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"True ATE = {true_ate:.3f}",
    )

    # Value labels
    positions = [
        (0, 0, f"{plugin:.3f}"),
        (1, after_treated, f"{treated:+.3f}"),
        (2, after_control, f"{control:+.3f}"),
        (3, total_correction, f"{aipw:.3f}"),
    ]

    plotted_values = np.array([
        0,
        after_treated,
        after_control,
        total_correction,
        true_ate_change,
    ])

    offset = max(np.ptp(plotted_values) * 0.04, 0.002)

    for x, y, text in positions:
        ax.text(
            x,
            y + offset,
            text,
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Formatting
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)

    ax.set_ylabel("Change relative to plug-in")
    plt.suptitle("AIPW correction", fontsize=16, fontweight='bold')

    # ax.grid(axis="y", alpha=0.25)
    # ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)

    sns.despine()

    plt.tight_layout()
    plt.show()

# Simulations by sample size

def simulations_sample_size(
    sample_sizes,
    true_ate,
    n_simulations=100,
    p=50,
    n_splits=5,
):
    results = pd.DataFrame()

    for n_index, n in enumerate(sample_sizes):
        print(f"Running n = {n:,}")

        for simulation in range(n_simulations):
            X, D, Y = simulate_data(
                n=n,
                p=p,
                true_ate=true_ate,
                seed=simulation,
            )

            s_learner_ate = estimate_s_learner(X, D, Y)
            s_learner_result = pd.DataFrame({
                                "n": [n],
                                "Simulation": [simulation],
                                "Method": ["S-learner"],
                                "Estimate": [s_learner_ate],
                                "Bias": [s_learner_ate - true_ate],
                            })
            
            dml_ate, dml_se = estimate_dml_aipw(X, D, Y, n_splits=n_splits, seed=simulation)
            dml_result = pd.DataFrame({
                            "n": [n],
                            "Simulation": [simulation],
                            "Method": ["DML/AIPW"],
                            "Estimate": [dml_ate],
                            "Bias": [dml_ate - true_ate],
                        })

            result = pd.concat([s_learner_result, dml_result])
            results = pd.concat([results, result])


    return results
