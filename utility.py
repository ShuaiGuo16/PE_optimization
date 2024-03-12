### Utility functions
import pickle
from scipy.stats import qmc, norm
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, max_error, brier_score_loss


def generate_candidates(lb, ub, num=10000):
    """
    Efficiently generate candidate samples considering dependencies between features.

    Args:
        lb (array): Lower bounds for each dimension.
        ub (array): Upper bounds for each dimension.
        num (int): Number of candidate samples to generate.

    Returns:
        np.ndarray: Array of generated candidate samples with valid positions.
    """

    # Constants
    c_module = 61.4e-3
    d_module = 106e-3

    # Generate scaled samples using Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=len(lb))
    X = sampler.random(n=num) * (ub - lb) + lb
    X[:, -1] = np.round(X[:, -1]).astype(int)  # Assuming rounding is needed for the last column

    # Storage for results
    result_design = []

    # Vectorized calculations for t_min, t_max (avoiding divide by zero)
    t_min = np.ones(num) * 1e-3
    t_max = X[:, 1] / X[:, 5] - 1e-3
    valid_t_mask = t_min < t_max

    # Pre-calculate ranges for position sampling
    Xc_min, Xc_max = c_module / 2, X[:, 1] - c_module / 2
    Yc_min, Yc_max = d_module / 2, X[:, 2] - d_module / 2

    # Generate positions once outside the loop for all samples and filter valid ones later
    position_sampler = qmc.LatinHypercube(d=4)
    all_positions = position_sampler.random(n=20 * num).reshape(num, 20, 4)

    for i, (sample, is_valid_t) in enumerate(zip(X, valid_t_mask)):
        if not is_valid_t:
            continue

        t = np.random.uniform(t_min[i], t_max[i])

        # Scale positions for the current sample
        positions = all_positions[i]
        positions[:, 0] = positions[:, 0] * (Xc_max[i] - Xc_min) + Xc_min
        positions[:, 1] = positions[:, 1] * (Yc_max[i] - Yc_min) + Yc_min
        positions[:, 2] = positions[:, 2] * (Xc_max[i] - Xc_min) + Xc_min
        positions[:, 3] = positions[:, 3] * (Yc_max[i] - Yc_min) + Yc_min

        # Check for non-overlapping positions
        xc1, yc1, xc2, yc2 = positions.T
        non_overlapping = (np.abs(xc1 - xc2) > c_module) | (np.abs(yc1 - yc2) > d_module)

        valid_positions = positions[non_overlapping]
        if len(valid_positions) > 0:
            random_index = np.random.randint(len(valid_positions))
            selected_position = valid_positions[random_index]
            result_design.append(np.concatenate([sample, [t], selected_position]))

    return np.array(result_design)



def create_samples(df, train_num):

    # Create dataset
    X = df.iloc[:, :-3].to_numpy()
    y = df.iloc[:, -2].to_numpy()

    # Train-test split
    if train_num < len(df):
        test_size = 1-train_num/len(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    return X_train, X_test, y_train, y_test



def evaluate_model(y_true, y_pred):
    """This function is used for evaluating the ML models performance."""

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    max_e = max_error(y_true, y_pred)

    percentage = np.abs(y_true-y_pred)/y_true
    max_percentage = np.max(percentage)*100
    max_percentage_loc = np.argmax(percentage)
    mean_percentage = np.mean(percentage)*100

    return rmse, max_e, max_percentage, max_percentage_loc, mean_percentage



def init_length_scales(dim, n_restarts, initial_guess=None):

    # Random initial params
    lb, ub = -2, 2
    lhd = qmc.LatinHypercube(d=dim, seed=42).random(n_restarts)
    lhd = (ub-lb)*lhd + lb
    length_scales = 10**lhd

    # Informed initial guess
    if initial_guess is not None:
        length_scales = np.vstack((length_scales, initial_guess))

    return length_scales



def fit(X, y, n_restarts=20, init_lengthscales=None, init_variance=None, trainable=True, verbose=True):
    models = []
    log_likelihoods = []

    # Generate initial guesses for length scale
    length_scales = init_length_scales(X.shape[1], n_restarts, init_lengthscales)
    if init_variance is None:
        variance=np.var(y)
    else:
        variance=init_variance

    if not trainable:
        model = gpflow.models.GPR(
            (X, y.reshape(-1, 1)),
            kernel=gpflow.kernels.SquaredExponential(variance=variance, lengthscales=init_lengthscales),
            # mean_function=gpflow.functions.Polynomial(0),
        )

        return model

    else:
        with tf.device("CPU:0"):

            for i, init in enumerate(length_scales):

                if verbose:
                    print(f"Performing {i+1}-th optimization:")

                # Set up the model
                kernel = gpflow.kernels.SquaredExponential(variance=variance, lengthscales=init)
                model = gpflow.models.GPR(
                    (X, y.reshape(-1, 1)),
                    kernel=kernel,
                    # mean_function=gpflow.functions.Polynomial(0),
                )

                opt = gpflow.optimizers.Scipy()
                opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))

                models.append(model)
                log_likelihoods.append(model.log_marginal_likelihood().numpy())

        # Select the model with the highest log-marginal likelihood
        best_model_index = np.argmax(log_likelihoods)
        best_model = models[best_model_index]

        return best_model



def load_GP_model(GP_params, X, y):

    # Configure GP model
    kernel = gpflow.kernels.SquaredExponential(lengthscales=qmc.LatinHypercube(d=X.shape[1]).random(1).flatten())
    GP = gpflow.models.GPR(
        (X, y.reshape(-1, 1)),
        kernel=kernel,
        # mean_function=gpflow.functions.Polynomial(0),
    )

    # Assign parameters
    gpflow.utilities.multiple_assign(GP, GP_params)

    return GP



def evaluate_weight(X):
    # Properties
    density_Al = 2700
    Fan_height = 40e-3
    Fan_Weight = 50.8e-3
    N_fan = np.ceil(X[:, 3] / Fan_height)

    # Weight calculation
    w = density_Al*(X[:, 3]*X[:, 2]*X[:, 4]+X[:, 7]*(X[:, 5]*X[:, 8]*X[:, 4]))+ Fan_Weight*N_fan

    return w


def select_diverse_batch(samples, acq, batch_size=5, pre_filter=False):

    if pre_filter:
        thred = np.quantile(acq, pre_filter)
        filtered_indices = np.arange(len(samples))[acq>thred]
        samples = samples[acq>thred]
        acq = acq[acq>thred]

    else:
        filtered_indices = np.arange(len(samples))

    # Perform weighted K-means clustering on the samples
    kmeans = KMeans(n_clusters=batch_size, n_init=10, random_state=0).fit(samples, sample_weight=acq)
    cluster_labels = kmeans.labels_

    # Find the highest acquisition value sample in each cluster
    selected_indices = []
    for cluster_idx in range(batch_size):
        cluster_indices = np.where(cluster_labels == cluster_idx)[0]
        cluster_acquisition_values = acq[cluster_indices]
        best_index_in_cluster = cluster_indices[np.argmax(cluster_acquisition_values)]
        selected_indices.append(best_index_in_cluster)

    return filtered_indices[selected_indices]



def predict_in_batches(model, X, batch_size=5000):
    """
    Predicts the output of a GPflow model in batches, returning flat numpy arrays for
    mean and variance predictions.

    Parameters:
    - model: The trained GPflow model.
    - X: The input features to predict on, expected to be a numpy array with shape (N, D),
         where N is the number of samples and D is the number of features.
    - batch_size: The size of each batch.

    Returns:
    - A tuple of flat numpy arrays (mean predictions, variance predictions).
    """

    with tf.device("CPU:0"):

        # Convert X to a TensorFlow dataset for efficient batching
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(batch_size)

        # Initialize lists to store predictions
        mean_predictions = []
        variance_predictions = []

        for X_batch in dataset:
            # Predict on the batch
            mean, variance = model.predict_f(X_batch, full_cov=False)

            # Store predictions
            mean_predictions.append(mean)
            variance_predictions.append(variance)

    # Concatenate all batch predictions and convert to numpy arrays
    mean_predictions = tf.concat(mean_predictions, axis=0).numpy().flatten()
    variance_predictions = tf.concat(variance_predictions, axis=0).numpy().flatten()

    return mean_predictions, variance_predictions



def GP_predict_candidates(model, X_candidates, scaler):
    X_candidates_scaled = scaler.transform(X_candidates)
    # f_mean, f_var = predict_in_batches(model, X_candidates_scaled)
    f_mean, f_var = model.predict_f(X_candidates_scaled, full_cov=False)
    f_mean = f_mean.numpy().flatten()
    f_var = f_var.numpy().flatten()

    return f_mean, f_var



def acquisition(model, f_mean, f_var, X_candidates, scaler, Tjmax, batch_mode=False, batch_size=None):

    # Evaluate weights
    w = evaluate_weight(X_candidates)

    # Calculate U values
    if model is None:
        f_mean, f_var = f_mean, f_var

    else:
        X_candidates_scaled = scaler.transform(X_candidates)
        # f_mean, f_var = predict_in_batches(model, X_candidates_scaled)
        f_mean, f_var = GP_predict_candidates(model, X_candidates, scaler)

    likelihood = norm.cdf(Tjmax, loc=f_mean, scale=np.sqrt(f_var))
    acq = likelihood*1/w

    print("Select candidates==>")
    # Sample selection
    X_candidates_scaled = scaler.transform(X_candidates)
    if batch_mode:
        # Batch selection mode
        acq_normalied = MinMaxScaler().fit_transform(acq.reshape(-1, 1))
        indices = select_diverse_batch(X_candidates_scaled, acq_normalied.flatten(), batch_size=batch_size)

    else:
        # Single point selection mode
        indices = np.argmax(acq)
    print("==> Complete candidates selection.")

    return acq, indices
