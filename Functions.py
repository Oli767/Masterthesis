# Import of Packages for Functions
import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools
import itertools

# Import of Packages for Optimization
import scipy.optimize as opt


def Generate_scenarios(Param):
    """
    This function is calculating a denfined number of scenarios (Forecasts)
    with a defined (length) number of years (Fth)

    Parameters:
        Param (dict): Parameter Dictionary

    Returns:
        Demand (ndarray): Demand Matrix

    To call the function use following syntax:
        Scenario_creation(Param)
    """
    # Parameters
    mu = Param["mu"]
    sigma = Param["sigma"]
    Dt0 = Param["Dt0"]
    dt = Param["dt"]
    Fth = Param["Fth"] + 1
    Forecasts = Param["No_Forecasts"]

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    # Create arrays for indices
    scenarios = np.arange(0, Forecasts)

    # Random values for spread of the scenario
    random_values = np.random.normal(0, 1, size=(len(scenarios), Fth))

    # Calculation of the Demand Matrix
    D = Dt0 * np.exp((mu * dt + sigma * np.sqrt(dt) * random_values).cumsum(axis=1))

    return np.round(D, 0)


def Scenario_plot(
    Param,
    Scenarios,
    NoStep=True,
    Title="Demand Scenarios",
    label="Passenger Numbers",
):
    """
    This function plots any data vector or matrix against the forecast time
    horizon vector Fth, allowing visualization of a selected number (n) of plots.

    Parameters:
        Param (dict): Dictionary
        Scenarios (ndarray): Scenario (Plotting) Data.
        NoStep (bool): If True, uses a line plot; otherwise, a step plot.
        Title (str): Title for the plot.
        label (str): Y-Axis description.

    Returns:
        None: Displays a plot.

    To call the function use following syntax:
        Scenario_plot(Param, Scenarios, NoStep=True, Title="Title", label="Title")
    """
    Fth = Param["Fth"] + 1
    n = Param["No_Forecasts_plot"]
    colors = ["blue", "green", "red"]  # Define colors for the last dimension

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    if isinstance(Scenarios, tuple):
        raise TypeError(
            "Scenarios is a Tuple! Please provide a NumPy array. At the function call Shock_generation the value of display should be set to false!"
        )

    # Ensure Scenarios is at least 2D
    if Scenarios.ndim == 1:
        Scenarios = Scenarios.reshape(1, -1)

    indices = np.random.choice(Scenarios.shape[0], size=n)
    Small_Scenario = Scenarios[indices]
    plotvector = np.arange(Fth)

    # Check if the input is 3D and handle accordingly
    if Small_Scenario.ndim == 3 and Small_Scenario.shape[2] == 3:
        for i in range(3):
            for scenario in Small_Scenario[:, :, i]:
                if NoStep:
                    plt.plot(plotvector, scenario, color=colors[i], alpha=0.5)
                else:
                    plt.step(
                        plotvector, scenario, where="post", color=colors[i], alpha=0.5
                    )
    else:
        for scenario in Small_Scenario:
            if NoStep:
                plt.plot(plotvector, scenario, label="Scenario")
            else:
                plt.step(plotvector, scenario, where="post", label="Scenario")

    plt.grid(True)
    plt.xlabel("Years")
    plt.ylabel(label)
    plt.title(Title)
    plt.show()


def Shock_generation(Param, Forecast_input, num_shocks=None, display=False):
    """
    This function simulates a demand shock and recovery process over a forecast period for multiple vectors
    by calling the Shock_vector function multiple times

    Parameters:
        Param (dict): Parameter Dictionary
        Forecast (np.array): Time series forecast data (vector or matrix)
        num_shocks (int): Number of shocks (optional)
        display (bool): Display number of shocks

    Returns:
        np.array: Updated forecast with shock and recovery adjustments.

    To call the function use following syntax:
        Shock_generation(Param, Forecast, num_shocks)
    """

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    if (
        len(Forecast_input.shape) == 1
    ):  # If Forecast is a single vector, apply shock normally
        return Shock_vector(Param, Forecast_input)

    num_vectors = Forecast_input.shape[0]

    # Determine number of vectors to shock (default: exponential distribution)
    if num_shocks is None:
        num_shocks = min(
            num_vectors, max(1, int(np.random.exponential(scale=Param["num_shocks"])))
        )

    # Randomly select `num_shocks` unique vectors to apply the shock
    chosen_vectors = np.random.choice(num_vectors, size=num_shocks, replace=False)

    for vector_index in chosen_vectors:
        Forecast_input[vector_index] = Shock_vector(Param, Forecast_input[vector_index])

    if display == True:
        return Forecast_input, num_shocks
    else:
        return Forecast_input


def Shock_vector(Param, Forecast_vector):
    """
    This Function applies a demand shock and recovery process to a single forecast vector

    Parameters:
        Param (dict): Parameter Dictionary
        Forecast_vector (np.array): Vector to apply a shock to

    Returns:
        np.array: Updated forecast vector with shock and recovery adjustments.

    To call the function use following syntax:
        Shock_vector(Param, Forecast_vector)
    """
    # Parameters
    shock_time_scale = Param["shock_scale"]
    recovery_time_scale = Param["recovery_scale"]
    recovery_time_sigma = Param["recovery_sigma"]
    shock_drop_scale = Param["shock_drop_scale"]
    recovery_steepness = Param["recovery_steepness"]

    # --- Independent Randomness for Each Vector ---
    rng = np.random.default_rng()  # Using independent RNG for randomness per vector

    # --- Duration Calculations ---
    duration_shock = min(
        max(int(rng.exponential(scale=shock_time_scale) + 1), 2), int(Param["Fth"] / 2)
    )  # Randomized shock duration
    duration_recovery = min(
        max(
            int(
                np.round(
                    rng.lognormal(mean=recovery_time_scale, sigma=recovery_time_sigma)
                )
            ),
            2,
        ),
        Param["Fth"] - duration_shock,
    )
    # Randomized recovery
    duration_combined = duration_shock + duration_recovery
    duration = min(Param["Fth"], duration_combined)

    # Determine start index for the shock event
    max_start_index = len(Forecast_vector) - duration
    start_index = rng.integers(0, max_start_index)

    # --- Shock and Recovery Parameters ---
    D0 = Forecast_vector[start_index]  # Initial demand value before shock
    target = Forecast_vector[start_index + duration]  # Target demand after recovery

    # --- Shock Calculation ---
    shock_drop = Param["Dt0"] * (shock_drop_scale / 100)
    delta_demand = max(
        -rng.exponential(scale=shock_drop), -D0
    )  # Random shock intensity
    raw_splits = np.sort(rng.uniform(0, 1, duration_shock - 1))
    raw_splits = np.insert(raw_splits, 0, 0)
    raw_splits = np.append(raw_splits, 1)
    shock_vector = D0 + delta_demand * raw_splits

    # --- Recovery Calculation ---
    k = abs(rng.normal(loc=recovery_steepness, scale=0.1))  # Random recovery rate
    t = np.arange(1, duration_recovery)
    recovery_vector = target - (target - D0) * np.exp(-k * t)

    # --- Combine Shock and Recovery ---
    combined_vector = np.round(np.concatenate((shock_vector, recovery_vector)), 2)
    Forecast_vector[start_index : start_index + duration] = combined_vector

    return Forecast_vector


def DHL_Calculation(Param, Scenario):
    """
    The DHL function calculates the Demand Hour Load (DHL) based on the given Secnario the Demand Hour Factors with its corresponding limits.

    Parameters:
        Param (dict): Parameter Dictionary
        Scenario (ndarray): Passenger Demand Scenario Matrix

    Returns:
        DHL (ndarray): Demand Hour Load Passenger Matrix

    To call the function use following syntax:
        DHL(Param, Scenario, DHL_Limits, DHL_Factors)
    """
    # Parameters
    DHL_L = Param["DHL_Limits"] * 1000000
    DHL_F = Param["DHL_Factors"] / 100

    # Check if the scenario values are within the limits
    Limit = (Scenario < DHL_L[0]).sum()
    if Limit > 0:
        raise ValueError(
            f"Scenario Pax values below the minimum limit of DHL_Factor: {DHL_L[0]} Pax exist."
        )

    # Create the index matrix with 1s where the condition is met, else 0
    index_1 = ((Scenario >= DHL_L[0]) & (Scenario < DHL_L[1])).astype(int) * DHL_F[0]
    index_2 = ((Scenario >= DHL_L[1]) & (Scenario < DHL_L[2])).astype(int) * DHL_F[1]
    index_3 = ((Scenario >= DHL_L[2]) & (Scenario < DHL_L[3])).astype(int) * DHL_F[2]
    index_4 = (Scenario >= DHL_L[3]).astype(int) * DHL_F[3]

    # Sum the index matrices to get the final index matrix
    index_matrix = sum([index_1, index_2, index_3, index_4])

    # Calculate the DHL by multiplying the Scenarios with the index matrix
    DHL = Scenario * index_matrix

    return DHL


def ATM_yearly(Param, Scenario):
    """
    This function calculates the Demand per year based on the Demand Hour Load (DHL) given the Demand Hour Factors with its corresponding limits.
    Lowerlimit set to 0 ATMs, accoridng to the litaerature it should start at approximately 2 ATMs in the DHL which corresponds to 0.05 Mil Pax.
    To adjust it, change first index lower boundary from limit_0 to limit_00 or adjust the first DHL_Limit from 0 to 0.5

    Parameters:
        Param (dict): Parameter Dictionary
        Scenario (ndarray): DHL ATM Demand Scenario Matrix

    Returns:
        DHL_yearly (ndarray): Yearly ATMs

    To call the function use following syntax:
        ATM_yearly(Param, Scenario)
    """
    # Parameters
    Mix = Param["Mix"]
    Pax_capacity = Param["Pax_capacity"]
    Average_Pax_capacity = np.round(np.sum(Mix * Pax_capacity), 0)
    DHL_L = Param["DHL_Limits"] * 1000000
    DHL_F = Param["DHL_Factors"] / 100

    if Scenario.ndim == 2:
        Scenario = Scenario
    elif Scenario.ndim == 3:
        Scenario = np.sum(Scenario, axis=2)
    else:
        raise ValueError("Scenario must be a 2D or 3D array.")

    # 0.05 Mil Pax ~ 2-4  ATM
    limit_0 = np.round((DHL_L[0] * DHL_F[0]) / Average_Pax_capacity, 0)
    limit_00 = np.round((0.5 / 100 * DHL_F[0]) / Average_Pax_capacity, 0)

    # 1 Mil Pax ~ 3-4 ATM
    limit_1 = np.round((DHL_L[1] * DHL_F[0]) / Average_Pax_capacity, 0)
    limit_01 = np.round((DHL_L[1] * DHL_F[1]) / Average_Pax_capacity, 0)

    # 10 Mil Pax ~27-31 ATM
    limit_10 = np.round((DHL_L[2] * DHL_F[1]) / Average_Pax_capacity, 0)
    limit_010 = np.round((DHL_L[2] * DHL_F[2]) / Average_Pax_capacity, 0)

    # 20 Mil Pax ~ 47-55ATM
    limit_20 = np.round((DHL_L[3] * DHL_F[2]) / Average_Pax_capacity, 0)
    limit_020 = np.round((DHL_L[3] * DHL_F[3]) / Average_Pax_capacity, 0)

    # Create the index matrix with 1s where the condition is met, else 0
    index_1 = ((Scenario >= limit_0) & (Scenario < limit_1)).astype(int) / DHL_F[0]
    index_2 = ((Scenario >= limit_01) & (Scenario < limit_10)).astype(int) / DHL_F[1]
    index_3 = ((Scenario >= limit_010) & (Scenario < limit_20)).astype(int) / DHL_F[2]
    index_4 = (Scenario >= limit_020).astype(int) / DHL_F[3]

    # Sum the index matrices to get the final index matrix
    index_matrix = sum([index_1, index_2, index_3, index_4])

    # Calculate the DHL by multiplying the Scenarios with the index matrix
    DHL_yearly = Scenario * index_matrix

    DHL_yearly_mix = DHL_yearly[:, :, np.newaxis] * Mix

    return np.round(DHL_yearly, 0), np.round(DHL_yearly_mix, 0)


def Model(t, D0, mu, sigma):
    """
    In this function the model for the approximation of the demand, Load factor and more is defined and returned.
    The current model is an exponential function.

    Parameters:
        t (int): Time step in years
        D0 (float: Initial demand at t=0
        mu (float): Growth rate of demand per year
        sigma (float): Volatility of demand development per year

    Returns:
        Model: Approximation Model for the demand, load factor and more

    To call the function use following syntax:
        Model(t, D0, mu, sigma)
    """
    return D0 * np.exp(mu * t + sigma * np.sqrt(t))


def exponential_fit(x, y):
    """
    Fits the defined approxiamtion function to the given data and returns the fitted parameters.

    Parameters:
        x (array-like): Independent variable data (time).
        y (array-like): Dependent variable data (demand, load factor or any other dependent value).

    Returns:
        trend: Approximation vector in shape (Fth,) if the input is a vector, else the shape is (n_rows, Fth)

    To call the function use following syntax:
        exponential_fit(x, y)
    """

    # Call the defined Model function to fit the data
    def model_call(t, a, mu, sigma):
        return Model(t, a, mu, sigma)

    # Fit the model to the data using curve_fit, starting with an initial guess
    initial_guess = [y[0], 0.1, 0.1]
    params, _ = opt.curve_fit(Model, x, y, p0=initial_guess)

    return params


def interpolate_exponential(data):
    """
    This function fits exponential parameters for a vector or matrix of inputdata and returns the fitted parameters in a vector or matrix format.
    The parameters contain the following values:
    - a: Initial demand at t=0
    - mu: Growth rate of demand per year
    - sigma: Volatility of demand development per year

    The shape of the output depends on the shape of the input data:
    - For a 1D input: returns a single parameter vector [a, mu, sigma]
    - For a 2D input: returns a parameter matrix (n_rows, 3)

    Parameters:
        data (array-like): 1D or 2D array of inputdata to fit.

    Returns:
        params (array): Fitted parameters [a, mu, sigma] for each row of data.
                        For 1D input, returns a single parameter vector.

    To call the function is use following syntax:
        interpolate_exponential(data)
    """
    # For 1D data, fit a single exponential function
    if data.ndim == 1:
        x = np.linspace(0, len(data) - 1, len(data))
        return exponential_fit(x, data)

    # For 2D data, fit an exponential function to each row
    elif data.ndim == 2:
        _, n_cols = data.shape
        x = np.linspace(0, n_cols - 1, n_cols)

        def fit_row(y):
            return exponential_fit(x, y)

        param_matrix = np.apply_along_axis(fit_row, axis=1, arr=data)
        return param_matrix

    # If the input data is neither 1D nor 2D array, raise an error
    else:
        raise ValueError("Input data must be 1D or 2D array.")


def trend_approximation(Param, CurveParameters, Adjust_mu=False):
    """
    This function computes trend approximation for a single vector or matrix of curve parameters by calling the previously defined model function.

    Parameters:
        Param (dict): Dictionary
        CurveParameters (ndarray): Inital value a, growth rate mu, and volatility sigma
        Adjust_mu (Bool): Condition whether to subtract Adjust_mu_factor from mu
        Adjust_mu_factor (float): adjustment amount for mu if Adjust_mu is enabled (True)

    Returns:
        trend (ndarray): approximation trend for the demand, load factor or any other dependent value accoridngt to the model function.

    To call the function use following syntax:
        trend_approximation(Param, CurveParameters, Adjust_mu=True, Adjust_mu_factor=0.002)
    """
    # Parameters
    time = Param["time"]
    Adjust_mu_factor = Param["adjusted_mu"]

    # Adjust the shape to (1, Fth)
    time_expanded = time[np.newaxis, :]

    # Ensure the input is at least of 2D shape
    CurveParameters = np.atleast_2d(CurveParameters)

    # Extract parameters from CurveParameters
    initial_value = CurveParameters[:, [0]]  # (n, 1)
    mu = CurveParameters[:, [1]]  # (n, 1)
    sigma = CurveParameters[:, [2]]  # (n, 1)

    # Adjust mu if condition applies
    if Adjust_mu:
        mu = mu - Adjust_mu_factor

    # Compute the trend using the model function
    trend = Model(time_expanded, initial_value, mu, sigma)

    # Readjust the shape of the output if the input was a 1D vector
    return trend[0] if CurveParameters.shape[0] == 1 else trend


def Load_Factor_matrix(Param, Scenario):
    """
    This function calcuates the Loadfactor according to the forecasts and a calculated trendline.
    The function uses the exponential fit model function to calculate the trendline.

    Parameters:
        Param (dict): Paremter dictionary
        Scenario (ndarray): Passenger Demand Scenario Matrix
        LF (float): Initial Loadfactor

    Returns:
        smoothed_Load_Factor (ndarray): Smoothed Load Factor matrix

    To call the function use following syntax:
        Load_Factor_matrix(Param, Scenario, LF)
    """
    # Initial Loadfactor
    LF = Param["LF"]
    # Interpolation/ Approximation of Demand-Scenario
    params_list = interpolate_exponential(Scenario)

    # Demand-Scenario-Trend Approximation
    trend = trend_approximation(Param, params_list)

    # Smoothing of the Demand-Scenario-Trend
    smoothing_factor_demand = Param["smoothing_factor_demand"]
    smoothed_scenario = trend + (smoothing_factor_demand * (Scenario - trend))

    # Interpolation of the smoothed Demand-Scenario
    params_scenario = interpolate_exponential(smoothed_scenario)
    trend_smoothed_shifted = trend_approximation(Param, params_scenario, True) / LF

    # Calculation of Load Factor
    Load_Factor = smoothed_scenario / trend_smoothed_shifted

    # # Smoothing of Load Factor
    # params_Load_Factor = interpolate_exponential(Load_Factor)
    # trend_Load_Factor = trend_approximation(Param, params_Load_Factor, False)

    # smoothing_factor_Load_Factor = Param["smoothing_factor_Load_Factor"]
    # smoothed_Load_Factor = trend_Load_Factor + (
    #     smoothing_factor_Load_Factor * (Load_Factor - trend_Load_Factor)
    # )
    smoothed_Load_Factor = Load_Factor
    # Adjust Loadfactor for initial setting
    delta = LF - smoothed_Load_Factor[:, 0]
    smoothed_Load_Factor = smoothed_Load_Factor + delta.reshape(-1, 1)
    smoothed_Load_Factor = np.clip(smoothed_Load_Factor, 0.01, 1)

    return smoothed_Load_Factor


def ATM(Param, Demand, Load_Factor):
    """
    This function calculates the Air Traffic Movements (ATM) based on the demand, load factor, aircraft mix and passenger capacity.

    Parameters:
        Param (dict): Parameter dictionary.
        Demand (ndarray): Demand scenario matrix.
        Load_Factor (ndarray): Load factor values.

    Returns:
        ATM (ndarray): Total air traffic movements.

    To call the function use following syntax:
        ATM(Param, Demand, Load_Factor)
    """
    # Parameters
    Pax_capacity = Param["Pax_capacity"]
    Mix = Param["Mix"]

    # Calculation of the Transported Pax per Aircraft category
    Pax_s = Load_Factor * Pax_capacity[0]
    Pax_m = Load_Factor * Pax_capacity[1]
    Pax_l = Load_Factor * Pax_capacity[2]

    # Calculation of the Air Traffic Movements (ATM)
    ATM = Demand / (Mix[0] * Pax_s + Mix[1] * Pax_m + Mix[2] * Pax_l)
    ATM = np.round(ATM, 0)

    # Prepare an ATM-Matrix for plotting
    ATM_plotting = ATM[:, :, np.newaxis] * Mix

    return ATM, ATM_plotting


def ATM_plot(ATM_Fleet, Param):
    """
    This function is plotting the Air Traffic Mix Forecast data vector or matrix against the forecast time
    horizon vector Fth, it allows to shows only a selected number (No_Forecasts_plot) of plots

    Parameters:
        ATM_Fleet: ATM Szenario
        Param (dict): Parameter Dictionary

    Returns:
        Plot of "No_Forecasts_plot" Air Traffic Mix Forecasts Against the Forecast Time Horizon

    To call the function use following syntax:
        ATM_plot(ATM_Fleet, Param)
    """
    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    time = Param["time"]
    # Randomly select indices without replacement

    No_Forecasts_plot = min(Param["No_Forecasts_plot"], Param["No_Forecasts"])
    selected_indices = np.random.choice(
        Param["No_Forecasts"], No_Forecasts_plot, replace=False
    )

    plt.figure(figsize=(12, 6))

    # Loop through selected simulation runs
    for i in selected_indices:
        Shorthaul_Fleet = ATM_Fleet[i, :, 0]  # Short-haul for this run
        Mediumhaul_Fleet = ATM_Fleet[i, :, 1]  # Medium-haul for this run
        Longhaul_Fleet = ATM_Fleet[i, :, 2]  # Long-haul for this run

        # Plot each run as a thin line to see individual trends
        plt.plot(time, Shorthaul_Fleet, color="blue", alpha=0.3)
        plt.plot(time, Mediumhaul_Fleet, color="green", alpha=0.3)
        plt.plot(time, Longhaul_Fleet, color="red", alpha=0.3)

    # Add labels and legend
    plt.xlabel("Years")
    plt.ylabel("Number of Aircraft Movements")
    plt.title("Total Air Traffic Movements per Aircarft Category")

    # Add a single bold line to represent one of the runs for visibility in legend
    plt.plot(
        time,
        ATM_Fleet[selected_indices[0], :, 0],
        color="blue",
        label="Short-haul Fleet",
        linewidth=2,
    )
    plt.plot(
        time,
        ATM_Fleet[selected_indices[0], :, 1],
        color="green",
        label="Medium-haul Fleet",
        linewidth=2,
    )
    plt.plot(
        time,
        ATM_Fleet[selected_indices[0], :, 2],
        color="red",
        label="Long-haul Fleet",
        linewidth=2,
    )

    plt.legend()
    plt.grid()
    plt.show()


def exponential_matrix(Param):
    """
    This function generates a matrix where each row is a vector of length Fth following an exponential distribution
    with occasional negative steps accroding to probability p_down

    Parameters:
        Param (dict): Parameter Dictionary

    Returns:
        matrix (ndarray): step matrix

    To call the function use following syntax:
        exponential_matrix(Param)

    """
    # Parameters
    Fth = Param["Fth"] + 1
    No_Forecasts = Param["No_Forecasts"]
    Average_step = Param["dt"]
    p_down = Param["p_down"]
    scale_down = Param["scale_down"]
    offset_scale = Param["Fth"] * Param["S_curve_offset"] / 100

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    # Generate positive and negative steps
    pos_steps = np.random.exponential(scale=Average_step, size=(No_Forecasts, Fth - 1))
    neg_steps = np.random.exponential(scale=scale_down, size=(No_Forecasts, Fth - 1))

    # Create mask for negative steps
    is_negative = np.random.rand(No_Forecasts, Fth - 1) < p_down
    steps = np.where(is_negative, -neg_steps, pos_steps)

    # Generate random offsets per forecast
    t_offset = np.random.exponential(scale=offset_scale, size=(No_Forecasts, 1))

    # Initialize matrix with zeros
    matrix = np.zeros((No_Forecasts, Fth))

    # Compute cumulative sum with offset
    matrix[:, 1:] = np.cumsum(steps, axis=1) - t_offset

    return np.clip(matrix, 0, Fth)


def S_curve(Param):
    """
    This function calculates S-Curve values according to the set Parameters and the
    matrix from the exponential_matrix function

    Parameters:
        Param (dict): Parameter Dictionary

    Returns:
        S_values_matrix (ndarray): S-Curve values matrix

    To call the function use following syntax:
        S_curve(Param)

    """
    # Parameters
    t_max = Param["Fth"] + 1
    t0 = Param["t0_factor"] * t_max
    L = Param["L"]
    k = Param["k"]

    # Generate the exponential matrix
    matrix = exponential_matrix(Param)

    # Compute S-Curve values with zero offset period
    S_values_matrix = np.zeros_like(matrix)
    S_values_matrix[:, :] = L / (1 + np.exp(-k * (matrix[:, :] - t0)))

    # Correcting the S-Curve values to eliminate the non-zero baseline at time = 0
    S_values_correction = L / (1 + np.exp(-k * (0 - t0)))
    S_values_matrix = S_values_matrix - S_values_correction

    return S_values_matrix


def S_curve_plot(Param, S_Values):
    """
    This function Plots the set S-Curve with a set of randomly choosen S-Cruve values from the
    exponential_matrix and S_curve functions.

    Parameters:
        Param (dict): Parameter Dictionary
        S_Values (ndarray): S-Curve values matrix

    Returns:
        Plot of the S-Curve and a random set of the S-Curve values

    To call the function use following syntax:
        S_curve_plot(Param, S_Values)

    """
    # Parameters
    Fth = Param["Fth"] + 1
    time = Param["time"]
    S = Param["L"] / (1 + np.exp(-Param["k"] * (time - (Param["t0_factor"] * Fth))))

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    plt.figure(figsize=(10, 6))
    plt.plot(time, S, label="S-Curve", color="blue")

    # Randomly select indices to plot
    No_Forecasts_plot = min(Param["No_Forecasts_plot"], Param["No_Forecasts"])
    selected_indices = np.random.choice(
        S_Values.shape[0], No_Forecasts_plot, replace=False
    )

    # Plot selected forecasts
    plt.plot(time, S_Values[selected_indices].T, alpha=0.7)

    plt.xlabel("Time [yrs]")
    plt.ylabel("Percentage [%]")
    plt.legend()
    plt.grid()
    plt.title("Technology Adoption with Standard S-Curve")
    plt.show()


def LH2_technology_adoption(Param, S_values, ATM_matrix):
    """
    This function calculates the LH2 technology adoption based on the S-curve adaption Scenarios
    and ATM matrix as well as the ATM mix

    Parameters:
        Param (dict): Parameter dictionary
        S_values (ndarray): S-curve values for technology adoption
        ATM_matrix (ndarray): Air Traffic Movement matrix

    Returns:
        LH2_adoption (ndarray): LH2 technology adoption values
        LH2_mix_adoption (ndarray): LH2 mix adoption values

    To call the function use following syntax:
        LH2_technology_adoption(Param, S_values, ATM_matrix)
    """
    # Parameters
    Mix = Param["Mix"]  # Aircraft Mix

    if ATM_matrix.ndim == 2:
        ATM_matrix = ATM_matrix
    elif ATM_matrix.ndim == 3:
        ATM_matrix = np.sum(ATM_matrix, axis=2)
    else:
        raise ValueError("ATM_matrix must be either 2D or 3D.")

    # Calcualtion of the LH2 technology adoption
    LH2_adoption = np.round(ATM_matrix * S_values, 0)

    # Calculation of the LH2 mix adoption
    LH2_mix_adoption = np.round(LH2_adoption[:, :, np.newaxis] * Mix, 0)

    # Alternative calculation of the LH2 mix adoption
    # LH2_mix_adoption2 = np.round(ATM_matrix[1] * S_values[:, :, np.newaxis], 0)

    return LH2_adoption, LH2_mix_adoption


def Capacity_2D(Param, delta_K, adjustK0=False):
    """
    This function calculates the capacity based on the initial capacity (K0) and the cumulative sum of delta_K.
    It returns the capacity either as a vector or as a matrix depending on the Matrix parameter.
    The capacity is calculated for each forecast time step, and if Matrix is True, it stacks the vector K vertically

    Args:
        Param (dict): Parameter Dictionary
        delta_K (np.array): Capacity Change Vector

    Returns:
        K (np.array): Capacity vectors in 2D shape

    To call the function use following syntax:
        Capacity(Param, delta_K)
    """
    # Parameters
    K0 = Param["K0"]  # Initial Capacity of Jet A1 aircraft stands
    K0_LH = Param["K0_LH"]  # Initial Capacity of LH2 aircraft stands
    No_Forecasts = Param["No_Forecasts"]
    TTF = Param["TTF"][1]

    if adjustK0 == True:
        K0 = K0_LH
    else:
        K0 = K0

    # If condition to adjust delta_K shape
    if delta_K.ndim == 1:
        delta_K_new = np.tile(delta_K, (No_Forecasts, 1))
    elif delta_K.ndim == 2:
        delta_K_new = delta_K
    elif delta_K.ndim == 3:
        delta_K_new = np.sum(delta_K, axis=2)
    else:
        raise ValueError("delta_K data has wrong shape.")

    K = np.round(K0 + np.cumsum(delta_K_new, axis=1) * TTF, 0)
    return K


def Capacity_3D(Param, delta_K, adjustK0=False):
    """
    This function calculates the capacity based on the initial capacity (K0) and the cumulative sum of delta_K.
    It returns the capacity either as a vector or as a matrix depending on the Matrix parameter.
    The capacity is calculated for each forecast time step, and if Matrix is True, it stacks the vector K vertically

    Args:
        Param (dict): Parameter Dictionary
        delta_K (np.array): Capacity Change Vector

    Returns:
        K (np.array): Capacity vector or matrix depending on the Matrix parameter.

    To call the function use following syntax:
        Capacity(Param, delta_K)
    """
    # Parameters
    K0 = Param["K0"] * Param["Mix"]  # Initial Capacity of Jet A1 aircraft stands
    K0_LH = Param["K0_LH"] * Param["Mix"]  # Initial Capacity of LH2 aircraft stands
    Mix = Param["Mix"]  # Aircraft Mix
    TTF = Param["TTF"]  # Turnaround time factor

    if adjustK0 == True:
        K0 = K0_LH
    else:
        K0 = K0

    # If condition to adjust delta_K shape
    if delta_K.ndim == 2:
        delta_K_new = np.tile(delta_K[:, :, np.newaxis], (1, 1, len(Mix)))
    elif delta_K.ndim == 3:
        delta_K_new = delta_K
    else:
        raise ValueError(
            "delta_K data must be at least 2D or 3D, otherwise use Capacity_2D Function."
        )

    K = np.round(K0 + np.cumsum(delta_K_new, axis=1) * TTF, 0)
    return K


def Capex_Jet(Param, delta_K_Jet):
    """
    This function calculates the capital expenditure (Capex) for Jet A1 aircraft stands based on the change in capacity (delta_K).

    Args:
        Param (dict): Parameter Dictionary
        delta_K_Jet (np.array): Capacity Change Vector

    Returns:
        CI_Jet (np.array): Installation Cost for Jet A1 aircraft stands in USD

    To call the function use following syntax:
        Capex_Jet(Param, delta_K_Jet)
    """
    # Parameters
    alpha = Param["alpha"]  # Cost elasticity parameter
    p_Dock = Param["p_Dock"]
    CC_Dock = Param["CC_Dock_Jet"]  # Cost of docking
    CC_Open = Param["CC_Open_Jet"]  # Cost of standing

    # If condition to adjust delta_K shape
    if delta_K_Jet.ndim == 2:
        delta_K_Jet = delta_K_Jet
    elif delta_K_Jet.ndim == 3:
        delta_K_Jet = np.sum(delta_K_Jet, axis=2)
    else:
        raise ValueError("delta_K has neiter shape 2D or 3D")

    CI_Jet = np.round(
        ((delta_K_Jet) ** alpha) * (p_Dock * CC_Dock + (1 - p_Dock) * CC_Open),
        2,
    )

    return CI_Jet


def Capex_LH(Param, delta_K_LH, D_LH):
    """
    This function calculates the capital expenditure (Capex) for LH2 aircraft stands based on the change in capacity (delta_K).

    Args:
        Param (dict): Parameter Dictionary
        delta_K_LH (np.array): Capacity Change Vector
        D_LH (np.array): LH2 Aircraft Movement Demand

    Returns:
        CI_LH (np.array): Installation Cost for LH2 aircraft stands in USD

    To call the function use following syntax:
        Capex_Jet(Param, delta_K, D_LH)
    """
    # Parameters
    TS_LH = Param["TS_LH"]  # LH2 Demand Threshold for Pipeline construction
    DHL_Factor = Param["DHL_Factors"][0]  # Design hour load factor
    LH_Fuel_ATM = Param["LH_Fuel_ATM"]  # LH2 Demand per Aircraft Movementt
    alpha = Param["alpha"]  # Cost elasticity parameter for LH2
    p_Dock = Param["p_Dock"]  # Share of Aircrafts using Dockstands
    CC_Dock = Param[
        "CC_Dock_LH2"
    ]  # Dockstand Construction Cost for LH2 Aircraft stands
    CC_Open = Param[
        "CC_Open_LH2"
    ]  # Openstand Construction Cost for LH2 Aircraft stands
    CC_Truck = Param["CC_Truck_LH2"]  # Fueltruck Cost for LH2 Aircraft stands
    CC_Pipeline = Param[
        "CC_Pipeline_LH2"
    ]  # Pipeline Construction Cost for LH2 Aircraft stands
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust delta_K shape
    if delta_K_LH.ndim == 2:
        delta_K_LH_Matrix = delta_K_LH
    elif delta_K_LH.ndim == 3:
        delta_K_LH_Matrix = np.sum(delta_K_LH, axis=2)
    else:
        raise ValueError("delta_K has neiter shape 2D or 3D")

    # If condition to adjust D_LH shape
    if D_LH.ndim == 2:
        D_LH = D_LH
    elif D_LH.ndim == 3:
        D_LH = np.sum(D_LH, axis=2)
    else:
        raise ValueError("D_LH has neiter shape 2D or 3D")

    # Adjust the threshold for desing hour and calculate the LH2 Fuel via ATMs
    TS_LH = TS_LH * DHL_Factor  # Threshold for Pipeline Demand, corrected for DHL
    D_LH_Fuel = D_LH * LH_Fuel_ATM  # Demand for LH2 Fuel based on ATMs

    # Calculating the difference matrix
    diff = TS_LH - D_LH_Fuel

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculating the Installation Cost for LH2 stands based on the conditions
    # TS_LH > D_LH
    CI_LH_greater = (
        greater
        * (delta_K_LH_Matrix) ** alpha
        * (p_Dock * (CC_Dock + CC_Truck) + (1 - p_Dock) * CC_Open)
    )
    # TS_LH <= D_LH
    CI_LH_less_equal = (
        less_equal
        * (delta_K_LH_Matrix) ** alpha
        * (p_Dock * (CC_Dock + CC_Pipeline) + (1 - p_Dock) * CC_Open)
    )
    # Combine the two conditions to get the final Installation Cost for LH2 stands
    CI_LH = np.round(CI_LH_greater + CI_LH_less_equal, 2)

    return CI_LH


def Opex_Jet(Param, K_Jet, D_Jet):
    """
    This function calculates the operating expenditure (Opex) for Jet A1 aircraft stands based on the stand capacity K and stand demand D.

    Args:
        Param (dict): Parameter Dictionary
        K_Jet (np.array): Stand Capacity Vector
        D_Jet (np.array): Stand Demand Vector

    Returns:
        CO_Jet (np.array): Operating Cost for Jet A1 aircraft stands in USD

    To call the function use following syntax:
        Opex_Jet(Param, K_Jet, D_Jet)
    """
    # Parameters
    p_Dock = Param["p_Dock"]
    CE_Dock_Jet = Param["CE_Dock_Jet"]  # Cost of energy for Jet A1
    CE_Open_Jet = Param["CE_Open_Jet"]  # Cost of energy for open Jet A1
    CM_Over_Jet = Param["CM_Over_Jet"]  # Cost of maintenance for over capacity
    CM_Under_Jet = Param["CM_Under_Jet"]  # Cost of maintenance for under capacity
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust K_Jet shape
    if K_Jet.ndim == 2:
        K_Jet = K_Jet
    elif K_Jet.ndim == 3:
        K_Jet = np.sum(K_Jet, axis=2)
    else:
        raise ValueError("K_Jet has neiter shape 2D or 3D")

    # If condition to adjust D_Jet shape
    if D_Jet.ndim == 1:
        D_Jet = np.tile(D_Jet[:, :, np.newaxis], (1, 1, len(Param["Mix"])))
    elif D_Jet.ndim == 2:
        D_Jet = D_Jet
    elif D_Jet.ndim == 3:
        D_Jet = np.sum(D_Jet, axis=2)
    else:
        raise ValueError("D_Jet has neiter shape 1D, 2D or 3D")

    # Calculating the difference matrix capacity minus demand
    diff = K_Jet - D_Jet

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculate the operational expenditure based on the demand and capacity
    # K_Jet > D_Jet
    CO_Jet_greater = greater * (
        D_Jet * (p_Dock * CE_Dock_Jet + (1 - p_Dock) * CE_Open_Jet)
        + (K_Jet - D_Jet) * CM_Over_Jet
    )
    # K_Jet <= D_Jet
    CO_Jet_less_equal = less_equal * (
        K_Jet * (p_Dock * CE_Dock_Jet + (1 - p_Dock) * CE_Open_Jet)
        + (D_Jet - K_Jet) * CM_Under_Jet
    )

    # Combine the two conditions to get the final operational expenditure
    CO_Jet = np.round(CO_Jet_greater + CO_Jet_less_equal, 2)

    return CO_Jet


def Opex_LH(Param, K_LH, D_LH):
    """
    This function calculates the operating expenditure (Opex) for LH2 aircraft stands based on the stand capacity K and stand demand D.

    Args:
        Param (dict): Parameter Dictionary
        K_LH (np.array): Stand Capacity Vector
        D_LH (np.array): Stand Demand Vector

    Returns:
        CO_LH (np.array): Operating Cost for LH2 aircraft stands in USD

    To call the function use following syntax:
        Opex_Jet(Param, K_LH, D_LH)
    """
    # Parameters
    p_Dock = Param["p_Dock"]
    CE_Dock_LH = Param["CE_Dock_LH"]  # Cost of energy for LH2
    CE_Open_LH = Param["CE_Open_LH"]  # Cost of energy for open LH2
    CM_Over_LH = Param["CM_Over_LH"]  # Cost of maintenance for over capacity
    CM_Under_LH = Param["CM_Under_LH"]  # Cost of maintenance for under capacity
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust K_LH shape
    if K_LH.ndim == 2:
        K_LH = K_LH
    elif K_LH.ndim == 3:
        K_LH = np.sum(K_LH, axis=2)
    else:
        raise ValueError("K_LH has neiter shape 2D or 3D")

    # If condition to adjust D_LH shape
    if D_LH.ndim == 1:
        D_LH = np.tile(D_LH[:, :, np.newaxis], (1, 1, len(Param["Mix"])))
    elif D_LH.ndim == 2:
        D_LH = D_LH
    elif D_LH.ndim == 3:
        D_LH = np.sum(D_LH, axis=2)
    else:
        raise ValueError("D_LH has neiter shape 1D, 2D or 3D")

    # Calculating the difference matrix capacity minus demand
    diff = K_LH - D_LH

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculate the operational expenditure based on the demand and capacity
    # K_LH > D_LH
    CO_LH_greater = greater * (
        D_LH * (p_Dock * CE_Dock_LH + (1 - p_Dock) * CE_Open_LH)
        + (K_LH - D_LH) * CM_Over_LH
    )
    # K_LH <= D_LH
    CO_LH_less_equal = less_equal * (
        K_LH * (p_Dock * CE_Dock_LH + (1 - p_Dock) * CE_Open_LH)
        + (D_LH - K_LH) * CM_Under_LH
    )
    # Combine the two conditions to get the final operational expenditure
    CO_LH = np.round(CO_LH_greater + CO_LH_less_equal, 2)

    return CO_LH


def Opex_Terminal(Param, PAX):
    """
    This function calculates the operating expenditure (Opex) for terminal based on Passenger numbers.

    Args:
        Param (dict): Parameter Dictionary
        PAX (np.array): Number of Passengers per Year

    Returns:
        CO_Terminal (np.array): Operating for Jet A1 aircraft stands in USD

    To call the function use following syntax:
        Opex_Terminal(Param, PAX)
    """
    # Parameters
    CE_Terminal = Param[
        "CE_Terminal"
    ]  # Cost of terminal operations in USD per passenger

    CO_Terminal = np.round(PAX * CE_Terminal, 2)

    return CO_Terminal


def Total_Cost_calculation(
    Param, delta_K_Jet, delta_K_LH, K_Jet, K_LH, D_Jet, D_LH, PAX
):
    """
    This function calculates the cost of the airport infrastructure based on the capital expenditure (Capex) and
    operational expenditure (Opex) for Jet A1 and LH2 aircraft stands, as well as terminal operations.

    Args:
        Param (dict): Parameter Dictionary
        delta_K_Jet (np.array): Capacity Change Vector for Jet A1
        delta_K_LH (np.array): Capacity Change Vector for LH2
        K_Jet (np.array): Stand Capacity Vector for Jet A1
        K_LH (np.array): Stand Capacity Vector for LH2
        D_Jet (np.array): Stand Demand Vector for Jet A1
        D_LH (np.array): Stand Demand Vector for LH2
        PAX (np.array): Number of Passengers per Year

    Returns:
        Total_cost (np.array): Total Cost for installation and operation of the airport stand infrastructure in USD

    To call the function use following syntax:
        Opex_Jet(Param, K_Jet, D_Jet)
    """
    # Installation Cost
    CI_Jet = Capex_Jet(Param, delta_K_Jet)
    CI_LH = Capex_LH(Param, delta_K_LH, D_LH)

    # Operating Cost
    CO_Jet = Opex_Jet(Param, K_Jet, D_Jet)
    CO_LH = Opex_LH(Param, K_LH, D_LH)
    CO_Terminal = Opex_Terminal(Param, PAX)

    # Total cost as the sum of capital expenditure and operational expenditure
    Total_cost = CI_Jet + CI_LH + CO_Jet + CO_LH + CO_Terminal

    return Total_cost


def Revenue_Jet(Param, K_Jet, D_Jet):
    """
    This function calculates the revenue from Jet A1 aircraft stand fees based on stand capacity K and stand demand D.

    Args:
        Param (dict): Parameter Dictionary
        K_Jet (np.array): Stand Capacity Vector
        D_Jet (np.array): Stand Demand Vector

    Returns:
        R_Jet (np.array): Revenue from Jet A1 aircraft stands in USD

    To call the function use following syntax:
        Revenue_Jet(Param, K_Jet, D_Jet)
    """
    # Parameters
    p_dock = Param["p_Dock"]
    re_Dock_Jet = Param["re_Dock_Jet"]  # Revenue from docking for Jet A1
    re_Open_Jet = Param["re_Open_Jet"]  # Revenue from open Jet A1
    rf_Jet = Param["rf_Jet"]  # Fixed revenue for Jet A1
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust K_Jet shape
    if K_Jet.ndim == 2:
        K_Jet = K_Jet
    elif K_Jet.ndim == 3:
        K_Jet = np.sum(K_Jet, axis=2)
    else:
        raise ValueError("K_Jet has neiter shape 2D or 3D")

    # If condition to adjust D_Jet shape
    if D_Jet.ndim == 2:
        D_Jet = D_Jet
    elif D_Jet.ndim == 3:
        D_Jet = np.sum(D_Jet, axis=2)
    else:
        raise ValueError("D_Jet has neiter shape 2D or 3D")

    # Calculating the difference matrix
    diff = K_Jet - D_Jet

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculate the operational expenditure based on the demand and capacity
    # K_Jet > D_Jet
    R_Jet_greater = (
        greater * D_Jet * (p_dock * re_Dock_Jet) + (1 - p_dock) * re_Open_Jet + rf_Jet
    )
    # K_Jet <= D_Jet
    R_Jet_less_equal = (
        less_equal * K_Jet * (p_dock * re_Dock_Jet)
        + (1 - p_dock) * re_Open_Jet
        + rf_Jet
    )
    # Combine the two conditions to get the final operational expenditure
    R_Jet = np.round(R_Jet_greater + R_Jet_less_equal, 2)

    return R_Jet


def Revenue_LH(Param, K_LH, D_LH):
    """
    This function calculates the revenue from LH2 aircraft stand fees based on stand capacity K and stand demand D.

    Args:
        Param (dict): Parameter Dictionary
        K_LH (np.array): Stand Capacity Vector
        D_LH (np.array): Stand Demand Vector

    Returns:
        R_LH (np.array): Revenue from LH2 aircraft stands in USD

    To call the function use following syntax:
        Revenue_LH(Param, K_LH, D_LH)
    """
    # Parameters
    p_dock = Param["p_Dock"]  # Probability of docking
    re_Dock_LH = Param["re_Dock_LH"]  # Revenue from docking for LH2
    re_Open_LH = Param["re_Open_LH"]  # Revenue from open LH2
    rf_LH = Param["rf_LH"]  # Fixed revenue for LH2
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust K_LH shape
    if K_LH.ndim == 2:
        K_LH = K_LH
    elif K_LH.ndim == 3:
        K_LH = np.sum(K_LH, axis=2)
    else:
        raise ValueError("K_LH has neiter shape 2D or 3D")

    # If condition to adjust D_LH shape
    if D_LH.ndim == 2:
        D_LH = D_LH
    elif D_LH.ndim == 3:
        D_LH = np.sum(D_LH, axis=2)
    else:
        raise ValueError("K_LH has neiter shape 2D or 3D")

    # Calculating the difference matrix
    diff = K_LH - D_LH

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculate the operational expenditure based on the demand and capacity
    # K_LH > D_LH
    R_LH_greater = (
        greater * D_LH * (p_dock * re_Dock_LH) + (1 - p_dock) * re_Open_LH + rf_LH
    )
    # K_LH <= D_LH
    R_LH_less_equal = (
        less_equal * K_LH * (p_dock * re_Dock_LH) + (1 - p_dock) * re_Open_LH + rf_LH
    )
    # Combine the two conditions to get the final operational expenditure
    R_LH = np.round(R_LH_greater + R_LH_less_equal, 2)

    return R_LH


def Reveneue_Pax(Param, PAX):
    """
    This function calculates the operating revenues from the terminal based on Passenger numbers.

    Args:
        Param (dict): Parameter Dictionary
        PAX (np.array): Number of Passengers per Year

    Returns:
        R_PAX (np.array): Operating revenue from passengers in USD

    To call the function use following syntax:
        Reveneue_Pax(Param, PAX)
    """
    # Parameters
    re_Pax = Param["re_Pax"]  # Revenue per passenger within the terminal in USD

    # Calculate the total revenue from passengers
    R_Pax = PAX * re_Pax

    return R_Pax


def Revenue_Rent(Param, K_Jet, K_LH):
    """
    This function calculates the revenues from renting out spaces within the terminal.

    Args:
        Param (dict): Parameter Dictionary
        K (np.array): Total Capacity

    Returns:
        R_Rent (np.array): Revenue from renting in USD

    To call the function use following syntax:
        Reveneue_Rent(Param, K)
    """
    # Parameters
    re_Rent = Param[
        "re_Rent"
    ]  # Revenue from renting spaces in USD per unit of capacity K

    if K_Jet.ndim == 2:
        K_Jet = K_Jet
    elif K_Jet.ndim == 3:
        K_Jet = np.sum(K_Jet, axis=2)
    else:
        raise ValueError("K_Jet has neither 2D or 3D shape")

    if K_LH.ndim == 2:
        K_LH = K_LH
    elif K_LH.ndim == 3:
        K_LH = np.sum(K_LH, axis=2)
    else:
        raise ValueError("K_LH has neither 2D or 3D shape")

    # Calculate the total revenue from renting
    R_Rent = (K_Jet + K_LH) * re_Rent

    return R_Rent


def Total_Revenue_calculation(Param, K_Jet, K_LH, D_Jet, D_LH, PAX):
    """
    This function calculates the total revenues from the airport infrastructure based on
    the Jet A1 and LH2 aircraft stand operation, as well as terminal and rental revenues.

    Args:
        Param (dict): Parameter Dictionary
        K (np.array): Total Capacity
        K_Jet (np.array): Stand Capacity Vector for Jet A1
        K_LH (np.array): Stand Capacity Vector for LH2
        D_Jet (np.array): Stand Demand Vector for Jet A1
        D_LH (np.array): Stand Demand Vector for LH2
        PAX (np.array): Number of Passengers per Year

    Returns:
        Total_revenue (np.array): Total revenue from the airport infrastructure in USD

    To call the function use following syntax:
        Total_Revenue_calculation(Param, K, K_Jet, K_LH, D_Jet, D_LH, PAX)
    """
    # Calculate the revenue for Jet A1
    R_Jet = Revenue_Jet(Param, K_Jet, D_Jet)

    # Calculate the revenue for LH2
    R_LH = Revenue_LH(Param, K_LH, D_LH)

    # Calculate the revenue from passengers
    R_Pax = Reveneue_Pax(Param, PAX)

    # Calculate the revenue from renting
    R_Rent = Revenue_Rent(Param, K_Jet, K_LH)

    # Total revenue is the sum of all revenues
    Total_revenue = R_Jet + R_LH + R_Pax + R_Rent

    return Total_revenue


def NPV_calculation(Param, delta_K_Jet, delta_K_LH, ATM, S_value_matrix, PAX):
    """
    This function calculates the Net Present Value (NPV) of the airport infrastructure project based on the
    capital expenditure (Capex), operational expenditure (Opex), and revenues from Jet A1 and LH2 aircraft stands, as well as terminal and rental revenues.

    Args:
        Param (dict): Parameter Dictionary
        delta_K (np.array): Capacity Change Vector
        ATM (np.array): Total Air Traffic Movements
        S_value_matrix (np.array): S-Curve Matrix for Technology Adoption
        PAX (np.array): Number of Passengers per Year

    Returns:
        NPV (np.array): Net Present Value of the airport infrastructure project in USD

    To call the function use following syntax:
        NPV_calculation(Param, delta_K, ATM, S_value_matrix, PAX)
    """
    # Parameters
    delta_K_Jet_Matrix = delta_K_Jet  # Jet A1 capacity change
    delta_K_LH_Matrix = delta_K_LH  # LH2 capacity change
    discount_rate = Param["discount_rate"]  # Discount rate for NPV calculation
    Initial_Investment = Param["Initial_Investment"]  # Initial investment

    if ATM.ndim == 2:
        D_Jet = ATM * (1 - S_value_matrix)  # Jet A1 aircraft demand
        D_LH = ATM * S_value_matrix  # LH2 aircraft demand
    elif ATM.ndim == 3:
        D_Jet = np.round((ATM * (1 - S_value_matrix)[:, :, np.newaxis]), 0)
        D_LH = np.round(ATM * S_value_matrix[:, :, np.newaxis], 0)
    K_Jet = Capacity_3D(Param, delta_K_Jet_Matrix)  # Jet A1 aircraft stand capacity
    K_LH = Capacity_3D(Param, delta_K_LH_Matrix, True)  # LH2 aircraft stand capacity

    Cost = Total_Cost_calculation(
        Param, delta_K_Jet, delta_K_LH, K_Jet, K_LH, D_Jet, D_LH, PAX
    )
    Revenue = Total_Revenue_calculation(Param, K_Jet, K_LH, D_Jet, D_LH, PAX)
    Profit = Revenue - Cost
    # Discount per year:
    Discount = 1 / ((1 + discount_rate) ** np.arange(Profit.shape[1]))
    # Apply discount per year to each row (scenario) — element-wise
    Discounted_Profit = Profit * Discount
    # Sum discounted profits over time
    NPV = np.sum(Discounted_Profit, axis=1) - Initial_Investment

    return NPV


def ENPV_calculation(Param, delta_K_Jet, delta_K_LH, ATM, S_values, Pax):
    """
    This function calculates the Expected Net Present Value (NPV) of the airport infrastructure project based on the Net Present Value (NPV) calculation for each scenario.

    Args:
        Param (dict): Parameter Dictionary
        delta_K (np.array): Capacity Change Vector
        ATM (np.array): Total Air Traffic Movements
        S_value_matrix (np.array): S-Curve Matrix for Technology Adoption
        PAX (np.array): Number of Passengers per Year

    Returns:
        ENPV (float): Expected Net Present Value of the airport infrastructure project in USD

    To call the function use following syntax:
        ENPV_calculation(Param, delta_K, ATM, S_value_matrix, PAX)
    """
    # Parameters
    No_Forecasts = Param["No_Forecasts"]

    if delta_K_Jet.ndim == 1:
        delta_K_Jet = np.tile(delta_K_Jet, (No_Forecasts, 1))
    else:
        delta_K_Jet = delta_K_Jet

    if delta_K_LH.ndim == 1:
        delta_K_LH = np.tile(delta_K_LH, (No_Forecasts, 1))
    else:
        delta_K_LH = delta_K_LH

    # Calculate NPV for each scenario
    NPV = NPV_calculation(Param, delta_K_Jet, delta_K_LH, ATM, S_values, Pax)
    Sum_NPV = np.sum(NPV)  # Sum of NPV values

    ENPV = Sum_NPV / len(NPV)  # Expected NPV is the average of NPV values

    return ENPV


def GA_dual(Param, ATM, S_values, PAX):
    """
    This function evolves delta_K_Jet_Mix and delta_K_LH_Mix using a genetic algorithm to maximize ENPV.

    Args:
        Param (dict): Parameter dictionary
        ATM (np.array): Air Traffic Movements, shape (No_Forecasts, Fth, Mix) or (No_Forecasts, Fth)
        S_values (np.array): S-curve technology adoption matrix
        PAX (np.array): Passengers per year

    Returns:
        delta_K_Jet_Mix (np.array): Jet capacity change, shape (No_Forecasts, Fth, Mix)
        delta_K_LH_Mix (np.array): LH2 capacity change, shape (No_Forecasts, Fth, Mix)

    To call this function use:
        GA_dual(Param, ATM, S_values, PAX)
    """
    # Initialize core parameters
    value_vector = Param["allowed_values"]
    Fth = Param["Fth"] + 1
    mix_dim = len(Param["Mix"])
    n_vars = 2 * Fth * mix_dim  # Variables for Jet and LH2

    # Convert 2D ATM to 3D by applying mix
    if ATM.ndim == 2:
        mix = np.array(Param["Mix"])
        ATM = ATM[:, :, np.newaxis] * mix[np.newaxis, np.newaxis, :]

    # Register fitness and individual structures
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initialize a random individual from allowed values
    def init_individual():
        return np.array([random.choice(value_vector) for _ in range(n_vars)])

    # Register GA components
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, init_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Mutates individual values with fixed probability
    def mutate_individual(individual):
        for i in range(len(individual)):
            if random.random() < 0.2:
                individual[i] = random.choice(value_vector)

    toolbox.register("mutate", mutate_individual)

    # Evaluate individual's fitness using ENPV
    def evaluate(individual):
        gene_array = np.array(individual)

        # Reshape flat genes into (Fth, Mix) matrices
        delta_K_Jet_Mix = gene_array[: Fth * mix_dim].reshape(Fth, mix_dim)
        delta_K_LH_Mix = gene_array[Fth * mix_dim :].reshape(Fth, mix_dim)

        # Sum over mix axis → shape (Fth,)
        delta_K_Jet = np.sum(delta_K_Jet_Mix, axis=1)
        delta_K_LH = np.sum(delta_K_LH_Mix, axis=1)
        # delta_K = delta_K_Jet + delta_K_LH

        # Broadcast to (No_Forecasts, Fth, Mix) for ENPV calculation
        delta_K_Jet_full = np.broadcast_to(delta_K_Jet_Mix, ATM.shape)
        delta_K_LH_full = np.broadcast_to(delta_K_LH_Mix, ATM.shape)

        enpv = ENPV_calculation(
            Param, delta_K_Jet_full, delta_K_LH_full, ATM, S_values, PAX
        )
        return (enpv,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=Param["population"])
    cxpb, mutpb, ngen = 0.5, 0.2, 10  # Crossover, mutation, generations

    # Run GA for ngen generations
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring  # Replace population

    # Extract best solution
    best_ind = tools.selBest(population, 1)[0]
    gene_array = np.array(best_ind)

    # Reshape best solution into (Fth, Mix)
    delta_K_Jet_Mix = gene_array[: Fth * mix_dim].reshape(Fth, mix_dim)
    delta_K_LH_Mix = gene_array[Fth * mix_dim :].reshape(Fth, mix_dim)

    # Expand to (No_Forecasts, Fth, Mix)
    No_Forecasts = Param["No_Forecasts"]
    delta_K_Jet_Mix_full = np.broadcast_to(
        delta_K_Jet_Mix, (No_Forecasts, Fth, mix_dim)
    )
    delta_K_LH_Mix_full = np.broadcast_to(delta_K_LH_Mix, (No_Forecasts, Fth, mix_dim))

    return delta_K_Jet_Mix_full, delta_K_LH_Mix_full


def Decision_Rule(Param, K0, D, theta, condition):
    """
    This function creates a new delta capacity vector while considering a decision rule.

    Args:
        K0 (int): Initial Capacity
        D (ndarray): Demand Matrix (2D or 3D)
        theta (ndarray or int): Capacity Change Vector
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        delta_K_Flex (ndarray): Delta capacity vector(s) considering the decision rule
                                Shape matches input D: (scenarios, time) or (scenarios, time, mixes)
    """
    if D.ndim == 2:
        # 2D Case: (scenarios, time)
        K_Flex = np.full(D.shape, K0, dtype=D.dtype)

        for t in range(1, D.shape[1]):
            diff = K_Flex[:, t - 1] - D[:, t]
            over_capacity = np.greater(diff, condition).astype(int)
            under_capacity = np.less_equal(diff, condition).astype(int)
            K_Flex[:, t] = over_capacity * K_Flex[:, t - 1] + under_capacity * (
                K_Flex[:, t - 1] + theta
            )

        delta_K = np.diff(K_Flex - K0, axis=1)
        delta_K_Flex = np.insert(delta_K, 0, 0, axis=1)

    elif D.ndim == 3:
        # 3D Case: (scenarios, time, mixes)
        K0_mix = Param["Mix"] * K0
        K_Flex = np.full(D.shape, K0_mix, dtype=D.dtype)

        for mix in range(D.shape[2]):
            for t in range(1, D.shape[1]):
                diff = K_Flex[:, t - 1, mix] - D[:, t, mix]
                over_capacity = np.greater(diff, condition).astype(int)
                under_capacity = np.less_equal(diff, condition).astype(int)
                K_Flex[:, t, mix] = over_capacity * K_Flex[
                    :, t - 1, mix
                ] + under_capacity * (K_Flex[:, t - 1, mix] + theta)

        delta_K = np.diff(K_Flex - K0, axis=1)
        delta_K_Flex = np.insert(delta_K, 0, 0, axis=1)

    else:
        raise ValueError("Input demand matrix D must be either 2D or 3D.")

    return delta_K_Flex


def Optimization_parameters(Param, n):
    """
    This function creates a list of tuples consisiting of each pair of theta and
    condition, it reduces the list to a random sample of size n

    Args:
        Param (dict): Parameter Dictionary
        n (int): Sample Size

    Returns:
        optimization_params (list of tuples): List of Theta and Condition Tuple Pairs

    To call this function use the following syntax:
        Optimization(Param, n)
    """
    # Theta
    lower_theta = Param["lower_theta"]
    upper_theta = Param["upper_theta"]
    stepsize_theta = Param["stepsize_theta"]

    # Condition
    lower_cond = Param["lower_condition"]
    upper_cond = Param["upper_condition"]
    stepsize_cond = Param["stepsize_condition"]

    # Creation of a List of Tuples
    condition = np.arange(lower_cond, upper_cond, stepsize_cond)
    theta = np.arange(lower_theta, upper_theta, stepsize_theta)
    optimization_params = list(itertools.product(theta, condition))

    indices = np.random.choice(
        len(optimization_params), size=(min(n, len(optimization_params))), replace=False
    )
    optimization_params_sample = [optimization_params[i] for i in indices]

    return optimization_params_sample


def CDF_Plot(Vector1, Vector2, label1="Vector1", label2="Vector2"):
    """
    This function is Plotting the Cumulative Density Function of the NPVs
    Args:
        Vector1 (ndarray): Input Vector 1
        Vector2 (ndarray): Input Vector 2
        label1 (str): First CDF Curve
        label2 (str): Second CDF Curve
        label3 (str): First ENPV Value
        label4 (str): Second ENPV Value

    Returns:
        Plot of all input Vectors in a CDF Graphic
        + Visualisation of the 10th, 90th Percentile of the Input Vectors

    To call this Function use following syntax:
        CDF_Plot(Vector1, Vector2, label1, label2, label3, label4)
    """
    percentile_10a = np.percentile(Vector1, 10)
    percentile_90a = np.percentile(Vector1, 90)
    percentile_10b = np.percentile(Vector2, 10)
    percentile_90b = np.percentile(Vector2, 90)

    # Creating a subplot
    fig, ax = plt.subplots()

    # Step plot code with specific values
    ax.plot(
        np.sort(Vector1),
        np.arange(1, len(Vector1) + 1) / float(len(Vector1)),
        linestyle="-",
        label=label1 + " CDF Curve",
        linewidth=2,
        color="green",
        alpha=0.7,
    )

    ax.plot(
        np.sort(Vector2),
        np.arange(1, len(Vector2) + 1) / float(len(Vector2)),
        linestyle="-",
        label=label2 + " CDF Curve",
        linewidth=2,
        color="blue",
        alpha=0.7,
    )

    mean1 = np.mean(Vector1)
    Vector3 = np.full_like(Vector1, mean1)
    ax.plot(
        np.sort(Vector3),
        np.arange(1, len(Vector3) + 1) / float(len(Vector3)),
        linestyle="--",
        label=label1 + " ENPV",
        linewidth=2,
        color="green",
        alpha=0.7,
    )
    mean2 = np.mean(Vector2)
    Vector4 = np.full_like(Vector2, mean2)
    ax.plot(
        np.sort(Vector4),
        np.arange(1, len(Vector4) + 1) / float(len(Vector4)),
        linestyle="-.",
        label=label2 + " ENPV",
        linewidth=2,
        color="blue",
        alpha=0.7,
    )
    ax.axhline(
        0.9,
        color="orange",
        linestyle="--",
        label="90th Percentile",
    )

    ax.axhline(
        0.1,
        color="red",
        linestyle="-.",
        label="10th Percentile",
    )

    # Add crosshair at the specified point
    ax.plot(percentile_90a, 0.9, marker="X", color="black", markersize=6)
    ax.plot(percentile_10a, 0.1, marker="X", color="black", markersize=6)
    ax.plot(percentile_90b, 0.9, marker="X", color="black", markersize=6)
    ax.plot(percentile_10b, 0.1, marker="X", color="black", markersize=6)

    ax.grid(True)
    ax.set_title("Cumulative Distribution Function (CDF)")
    ax.set_xlabel("NPVs")
    ax.set_ylabel("Cumulative Probability [%]")
    ax.legend()
    plt.show()
    percentiles = [percentile_10a, percentile_90a, percentile_10b, percentile_90b]
    return percentiles


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


# def Decision_Rule(K0, D, theta, condition):
#     """
#     This function creates new delta capacity vector while considering a decision rule

#     Args:
#         K0 (int): Initial Capacity
#         D (ndarray): Demand Matrix
#         theta (ndarray): Capacity Change Vector
#         condition (int): Condition for Capacity Increase (difference of K and D)

#     Returns:
#         delta_K_Flex (ndarray): delta capacity vector considering a decision rule

#     To call this function use the following syntax:
#         Decision_Rule(K0, D, theta, condition)
#     """
#     # Creation of an array with the same shape as D initialized with initial capacity K0
#     K_Flex = np.full(D.shape, K0, dtype=D.dtype)

#     # For loop to iterate over all values of a Scenario
#     for t in range(1, D.shape[1]):  # Start from t=1
#         # Calculate the Difference Matrix
#         diff = K_Flex[:, t - 1] - D[:, t]

#         # Create an Index Matrix for the condition of over- and undercapacity
#         over_capacity = np.greater_equal(diff, condition).astype(int)
#         under_capacity = np.less(diff, condition).astype(int)

#         # Update K_Flex for the next iteration
#         K_Flex[:, t] = over_capacity * K_Flex[:, t - 1] + under_capacity * (
#             K_Flex[:, t - 1] + theta
#         )

#         # Calculation of the delta capacity vector delta_K
#         delta_K = np.diff((K_Flex) - K0)
#         delta_K_Flex = np.insert(delta_K, 0, 0, axis=1)

#     return delta_K_Flex


# def NPV_Flexible(delta_K, Param, D, condition):
#     """
#     This function calculates the Net Present Value for the flexible case by calling the
#     Capacity and NPV calculation functions

#     Args:
#         delta_K (ndarray): Delta Capacity Vector
#         Param (dict): Parameter Dictionary
#         condition (int): Condition for Capacity Increase (difference of K and D)

#     Returns:
#         NPV (ndarray): Net Present Value for the Flexible Case

#     To call this function use the following syntax:
#         NPV_Flexible(delta_K, Param)
#     """
#     # Parameter
#     K0 = Param["K0"]  # Initial Capacity

#     # Calling the Capacity function for the Capacity matrix
#     K_Flex = Capacity(delta_K, Param)

#     # Calling the NPV calculation function for the NPV vector
#     NPV = NPV_calculation(K_Flex, D, delta_K, Param, condition)

#     return NPV


# def ENPV_Flexible(theta, condition, Param, D):
#     """
#     This function calculates the Expected Net Present Value by calling the Decision Rule
#     and NPV Flexible functions and using the theta and condition values

#     Args:
#         theta (ndarray): Capacity increase value
#         condition (int): Condition for Capacity increase (difference of K and D)
#         Param (dict): Parameter Dictionary
#         D (ndarray): Demand Matrix

#     Returns:
#         ENPV (ndarray): Expected Net Present Value in the Flexible case

#     To call this function use the following syntax:
#         ENPV_Flexible(theta, condition, Param, D)
#     """
#     # Parameter
#     K0 = Param["K0"]  # Initial Capacity

#     # Calling the Decision Rule function for the delta capacity matrix
#     delta_K = Decision_Rule(K0, D, theta, condition)

#     # Calling the NPV Flexible function for the NPV vector
#     NPV = NPV_Flexible(delta_K, Param, D, condition)

#     # Calculating the mean of all NPVs to get the ENPV
#     ENPV = np.mean(NPV)

#     return ENPV


# def Optimization(Param, n):
#     """
#     This function creates a list of tuples consisiting of each pair of theta and
#     condition, it reduces the list to a random sample of size n

#     Args:
#         Param (dict): Parameter Dictionary
#         n (int): Sample Size

#     Returns:
#         optimization_params (list of tuples): List of Theta and Condition Tuple Pairs

#     To call this function use the following syntax:
#         Optimization(Param, n)
#     """
#     # Theta
#     lower_theta = Param["lower_theta"]
#     upper_theta = Param["upper_theta"]
#     stepsize_theta = Param["stepsize_theta"]

#     # Condition
#     lower_cond = Param["lower_cond"]
#     upper_cond = Param["upper_cond"]
#     stepsize_cond = Param["stepsize_cond"]

#     # Creation of a List of Tuples
#     condition = np.arange(lower_cond, upper_cond, stepsize_cond)
#     theta = np.arange(lower_theta, upper_theta, stepsize_theta)
#     optimization_params = list(itertools.product(theta, condition))

#     indices = np.random.choice(len(optimization_params), size=n, replace=False)
#     optimization_params_sample = [optimization_params[i] for i in indices]

#     return optimization_params_sample


# def Evaluation(Param, D, n=1000):
#     """
#     This function first calls the Optimization function to generate a list of tuples
#     consisiting of each pair (defined sample size n) of theta and conditon, it then
#     continues to evaluates all the tuples by iterating over each pair to find the
#     maximum ENPV value

#     Args:
#         Param (dict): Parameter Dictionary
#         D (ndarray): Demand Matrix
#         n (int): Sample Size

#     Returns:
#         max_enpv (int): Maximum value of the ENPV,
#         max_theta (int): optimal value of theta,
#         max_cond (int): optimal value of the condition

#     To call this function use the following syntax:
#         Evaluation(Param, D, n)
#     """
#     # Calling the Optimization function to get the list of tuples
#     optimization_params = Optimization(Param, n)

#     # Initialize the maximum values
#     max_enpv = float("-inf")
#     max_theta = None
#     max_cond = None

#     for theta, condition in optimization_params:
#         ENPV = ENPV_Flexible(theta, condition, Param, D)
#         if ENPV > max_enpv:
#             max_enpv = ENPV
#             max_theta = theta
#             max_cond = condition

#     return max_enpv, max_theta, max_cond
