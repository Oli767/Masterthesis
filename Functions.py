# Import of Packages for Functions
from matplotlib.legend import Legend
import numpy as np
import random
import matplotlib.pyplot as plt
from deap import base, creator, tools
import itertools

# Import of Packages for Data Handling
import os

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

    # Creating arrays for indices
    scenarios = np.arange(0, Forecasts)

    # Random values for spread of the scenario
    random_values = np.random.normal(0, 1, size=(len(scenarios), Fth))

    # Calculating the demand matrix with the GBM equation
    D = Dt0 * np.exp((mu * dt + sigma * np.sqrt(dt) * random_values).cumsum(axis=1))

    return np.round(D, 0)


# def Scenario_plot(
#     Param,
#     Scenarios,
#     NoStep=True,
#     Title="Demand Scenarios",
#     label="Passenger Numbers",
# ):
#     """
#     This function plots any data vector or matrix against the forecast time
#     horizon vector Fth, allowing visualization of a selected number (n) of plots.

#     Parameters:
#         Param (dict): Parameter Dictionary
#         Scenarios (ndarray): Scenario (Plotting) Data
#         NoStep (bool): If True, uses a Line pPlot; Otherwise, a Step Plot
#         Title (str): Title of the plot
#         label (str): Y-Axis Description of the Plot

#     Returns:
#         None: Displays the Plot

#     To call the function use following syntax:
#         Scenario_plot(Param, Scenarios, NoStep, Title, label)
#     """
#     # Parameters
#     Fth = Param["Fth"] + 1
#     n = Param["No_Forecasts_plot"]
#     colors = ["blue", "green", "red"]  # Define colors for the last dimension

#     # Setting the random seed for reproducibility
#     np.random.seed(Param["seed"])

#     if isinstance(Scenarios, tuple):
#         raise TypeError(
#             "Scenarios is a Tuple! Please provide a NumPy array. At the function call Shock_generation the value of display should be set to false!"
#         )

#     # Checking if the input is 1D, if so reshaping to 2D
#     if Scenarios.ndim == 1:
#         Scenarios = Scenarios.reshape(1, -1)

#     indices = np.random.choice(Scenarios.shape[0], size=n)
#     Small_Scenario = Scenarios[indices]
#     plotvector = np.arange(Fth)

#     # Checking if the input is 3D
#     if Small_Scenario.ndim == 3 and Small_Scenario.shape[2] == 3:
#         for i in range(3):
#             for scenario in Small_Scenario[:, :, i]:
#                 if NoStep:
#                     plt.plot(plotvector, scenario, color=colors[i], alpha=0.5)
#                 else:
#                     plt.step(
#                         plotvector, scenario, where="post", color=colors[i], alpha=0.5
#                     )
#     else:
#         for scenario in Small_Scenario:
#             if NoStep:
#                 plt.plot(plotvector, scenario, label="Scenario")
#             else:
#                 plt.step(plotvector, scenario, where="post", label="Scenario")

#     plt.grid(True)
#     plt.xlabel("Years")
#     plt.ylabel(label)
#     plt.title(Title)
#     plt.show()


def Scenario_plot(
    Param,
    Scenarios,
    NoStep=True,
    Title="Demand Scenarios",
    label="Passenger Numbers",
    save_plot=False,
    run_name="Unnamed_Run",
    output_base_folder="Plots",
):
    """
    This function plots any data vector or matrix against the forecast time
    horizon vector Fth, allowing visualization of a selected number (n) of plots.
    Additionally, it saves the plot to a specified folder named after the run.
    The function automatically avoids overwriting saved plots by incrementing filenames.

    Parameters:
        Param (dict): Parameter Dictionary
        Scenarios (ndarray): Scenario (Plotting) Data
        NoStep (bool): If True, uses a Line pPlot; Otherwise, a Step Plot
        Title (str): Title of the plot
        label (str): Y-Axis Description of the Plot
        save_plot (bool): If True, saves the plot to a file
        output_base_folder (str): Base folder for saving plots, default is "Plots"

    Returns:
        None: Displays the Plot, if save_plot is True, saves the plot to a file.

    To call the function use following syntax:
        Scenario_plot(Param, Scenarios, NoStep, Title, label, save_plot, output_base_folder)
    """

    # Parameters
    Fth = Param["Fth"] + 1
    n = Param["No_Forecasts_plot"]
    colors = ["blue", "green", "red"]

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    # Validating the scenario input shape
    if isinstance(Scenarios, tuple):
        raise TypeError(
            "Scenarios is a Tuple! Please provide a NumPy array. At the function call Shock_generation the value of display should be set to false!"
        )

    # Checking if the input is 1D, if so reshaping to 2D
    if Scenarios.ndim == 1:
        Scenarios = Scenarios.reshape(1, -1)

    indices = np.random.choice(Scenarios.shape[0], size=n)
    Small_Scenario = Scenarios[indices]
    plotvector = np.arange(Fth)

    # Create figure
    plt.figure()

    # Checking if the input is 3D
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
                plt.plot(plotvector, scenario)
            else:
                plt.step(plotvector, scenario, where="post")

    plt.grid(True)
    plt.xlabel("Years")
    plt.ylabel(label)
    plt.title(Title)

    # Saving the plot
    if save_plot:
        # Folder: Plots/run_name/
        folder_path = os.path.join(output_base_folder, run_name)
        os.makedirs(folder_path, exist_ok=True)

        # Createing a safe filename
        filename_safe_title = Title.replace(" ", "_").replace("/", "_")
        base_filename = f"{filename_safe_title}.png"
        plot_path = os.path.join(folder_path, base_filename)

        # Checking for duplicates and increment
        counter = 1
        while os.path.exists(plot_path):
            base_filename = f"{filename_safe_title}_{counter}.png"
            plot_path = os.path.join(folder_path, base_filename)
            counter += 1

        # Saving the figure
        plt.savefig(plot_path)
        print(f"Plot saved to: {os.path.abspath(plot_path)}")

    plt.show()


def Shock_generation(Param, Forecast_input, num_shocks=None, display=False):
    """
    This function simulates a demand shock and recovery process over a forecast period for multiple vectors
    by calling the Shock_vector function multiple times

    Parameters:
        Param (dict): Parameter Dictionary
        Forecast (np.array): Time Series Forecast Data
        num_shocks (int): Number of shocks (optional)
        display (bool): Display Number of Shocks -> Creates a Tuple -> Not for Suitable for Plotting Function

    Returns:
        Forecast_input (np.array): Updated Forecast with Shocks and Recovery Adjustments

    To call the function use following syntax:
        Shock_generation(Param, Forecast, num_shocks, display)
    """

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    if (
        len(Forecast_input.shape) == 1
    ):  # If Forecast is a single vector, apply shock normally
        return Shock_vector(Param, Forecast_input)

    num_vectors = Forecast_input.shape[0]

    # Determining number of vectors with shocks (default: exponential distribution)
    if num_shocks is None:
        num_shocks = min(
            num_vectors, max(1, int(np.random.exponential(scale=Param["num_shocks"])))
        )

    # Randomly selecting `num_shocks` vectors to apply the shock
    chosen_vectors = np.random.choice(num_vectors, size=num_shocks, replace=False)

    for vector_index in chosen_vectors:
        Forecast_input[vector_index] = Shock_vector(Param, Forecast_input[vector_index])

    if display == True:
        return Forecast_input, num_shocks
    else:
        return Forecast_input


def Shock_vector(Param, Forecast_vector):
    """
    This function applies a demand shock and recovery process to a single given forecast vector.

    Parameters:
        Param (dict): Parameter Dictionary
        Forecast_vector (np.array): Given Vector to Apply a Shock

    Returns:
        Forecast_vector (np.array): Updated forecast vector with shock and recovery adjustments.

    To call the function use following syntax:
        Shock_vector(Param, Forecast_vector)
    """
    # Parameters
    shock_time_scale = Param["shock_scale"]
    recovery_time_scale = Param["recovery_scale"]
    recovery_time_sigma = Param["recovery_sigma"]
    shock_drop_scale = Param["shock_drop_scale"]
    recovery_steepness = Param["recovery_steepness"]

    # Using independent RNG for randomness per vector
    rng = np.random.default_rng()

    # Calculating shock duration
    duration_shock = min(
        max(int(rng.exponential(scale=shock_time_scale) + 1), 2), int(Param["Fth"] / 2)
    )  # Randomizing shock duration
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
    # Randomizing shock recovery
    duration_combined = duration_shock + duration_recovery
    duration = min(Param["Fth"], duration_combined)

    # Determining start index for the shock event
    max_start_index = len(Forecast_vector) - duration
    start_index = rng.integers(0, max_start_index)

    # Determining the shock and recovery parameters
    D0 = Forecast_vector[start_index]  # Initial demand value before shock
    target = Forecast_vector[start_index + duration]  # Target demand after recovery

    # Calculating the shock
    shock_drop = Param["Dt0"] * (shock_drop_scale / 100)
    delta_demand = max(
        -rng.exponential(scale=shock_drop), -D0
    )  # Randomizing shock intensity
    raw_splits = np.sort(rng.uniform(0, 1, duration_shock - 1))
    raw_splits = np.insert(raw_splits, 0, 0)
    raw_splits = np.append(raw_splits, 1)
    shock_vector = D0 + delta_demand * raw_splits

    # Calculating the shock recovery
    k = abs(rng.normal(loc=recovery_steepness, scale=0.1))  # Random recovery rate
    t = np.arange(1, duration_recovery)
    recovery_vector = target - (target - D0) * np.exp(-k * t)

    # Combining the shock and the shock recovery
    combined_vector = np.round(np.concatenate((shock_vector, recovery_vector)), 2)
    Forecast_vector[start_index : start_index + duration] = combined_vector

    return Forecast_vector


def DHL_Calculation(Param, Scenario):
    """
    This function calculates the Demand Hour Load (DHL) based on the given Scenario, the Demand Hour Factors and its corresponding limits.

    Parameters:
        Param (dict): Parameter Dictionary
        Scenario (ndarray): Passenger Demand in the DHL Scenario Matrix

    Returns:
        DHL (ndarray): Demand Hour Load Passenger Scenario Matrix

    To call the function use following syntax:
        DHL(Param, Scenario)
    """
    # Parameters
    DHL_L = Param["DHL_Limits"] * 1000000
    DHL_F = Param["DHL_Factors"] / 100

    # Checking if the scenario values are within the limits
    Limit = (Scenario < DHL_L[0]).sum()
    if Limit > 0:
        raise ValueError(
            f"Scenario Pax values below the minimum limit of DHL_Factor: {DHL_L[0]} Pax exist."
        )

    # Creating the index matrix with 1s where the condition is met, else 0
    index_1 = ((Scenario >= DHL_L[0]) & (Scenario < DHL_L[1])).astype(int) * DHL_F[0]
    index_2 = ((Scenario >= DHL_L[1]) & (Scenario < DHL_L[2])).astype(int) * DHL_F[1]
    index_3 = ((Scenario >= DHL_L[2]) & (Scenario < DHL_L[3])).astype(int) * DHL_F[2]
    index_4 = (Scenario >= DHL_L[3]).astype(int) * DHL_F[3]

    # Summing the index matrices to get the final index matrix
    index_matrix = sum([index_1, index_2, index_3, index_4])

    # Calculating the DHL by multiplying the Scenarios with the index matrix
    DHL = Scenario * index_matrix

    return DHL


def ATM_yearly(Param, Scenario):
    """
    This function calculates the ATM Demand per year based on the Demand Hour Load (DHL) given the Demand Hour Factors and its corresponding limits.
    Lowerlimit set to 0 ATMs, accoridng to the literature it should start at approximately 2 ATMs in the DHL which corresponds to 0.05 Million Passengers.
    To adjust it, change first index lower boundary from limit_0 to limit_00 or adjust the first DHL_Limit from 0 to 0.5

    Parameters:
        Param (dict): Parameter Dictionary
        Scenario (ndarray): ATM Demand in the DHL Scenario Matrix

    Returns:
        DHL_yearly, DHL_yearly_mix (ndarray): Yearly ATMs

    To call the function use following syntax:
        ATM_yearly(Param, Scenario)
    """
    # Parameters
    Mix = Param["Mix"]
    Pax_capacity = Param["Pax_capacity"]
    Average_Pax_capacity = np.round(np.sum(Mix * Pax_capacity), 0)
    DHL_L = Param["DHL_Limits"] * 1000000
    DHL_F = Param["DHL_Factors"] / 100

    # Checking dimensions of Scenario
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

    # Creating the index matrix with 1s where the condition is met, else 0
    index_1 = ((Scenario >= limit_0) & (Scenario < limit_1)).astype(int) / DHL_F[0]
    index_2 = ((Scenario >= limit_01) & (Scenario < limit_10)).astype(int) / DHL_F[1]
    index_3 = ((Scenario >= limit_010) & (Scenario < limit_20)).astype(int) / DHL_F[2]
    index_4 = (Scenario >= limit_020).astype(int) / DHL_F[3]

    # Summing the index matrices to get the final index matrix
    index_matrix = sum([index_1, index_2, index_3, index_4])

    # Calculating the DHL by multiplying the Scenarios with the index matrix
    DHL_yearly = Scenario * index_matrix

    # Applying the Mix to the DHL matrix
    DHL_yearly_mix = DHL_yearly[:, :, np.newaxis] * Mix

    return np.round(DHL_yearly, 0), np.round(DHL_yearly_mix, 0)


def K_yearly(Param, Capacity):
    """
    This function calculates the yearly capacity based on the given parameters and the DHL capacity matrix.

     Parameters:
         Param (dict): Parameter Dictionary
         Capacity (ndarray): DHL Capacity Matrix

     Returns:
         K_yearly, K_yearly_mix (ndarray): Yearly capacity matrices
    """
    # Parameters
    Mix = Param["Mix"]
    DHL_F = Param["DHL_Factors"] / 100
    limit = Param["K_Yearly_Limits"]

    # Checking dimensions of Capacity
    if Capacity.ndim == 2:
        Capacity = Capacity
    elif Capacity.ndim == 3:
        Capacity = np.sum(Capacity, axis=2)
    else:
        raise ValueError("Capacity must be a 2D or 3D array.")

    # Creating the index matrix with the DHL Factors where the condition is met, else 0
    index_1 = ((Capacity >= limit[0]) & (Capacity < limit[1])).astype(int) * DHL_F[0]
    index_2 = ((Capacity >= limit[1]) & (Capacity < limit[2])).astype(int) * DHL_F[1]
    index_3 = ((Capacity >= limit[2]) & (Capacity < limit[3])).astype(int) * DHL_F[2]
    index_4 = (Capacity >= limit[3]).astype(int) * DHL_F[3]

    # Summing the index matrices to get the final index matrix
    index_matrix = sum([index_1, index_2, index_3, index_4])

    # Calculating the yearly capacity K by dividing the DHL capacity with the index matrix
    K_yearly = Capacity / index_matrix

    # Applying the Mix to the yearly capacity matrix
    K_yearly_mix = K_yearly[:, :, np.newaxis] * Mix

    return np.round(K_yearly, 0), np.round(K_yearly_mix, 0)


def Model(t, D0, mu, sigma):
    """
    In this function the model for the approximation of the demand, the load factor and more.
    The model is defined and returned. The current model represents an exponential function.

    Parameters:
        t (int): Time Step in Years
        D0 (float: Initial Demand at t=0
        mu (float): Growth Rate of Demand per Year
        sigma (float): Volatility of Demand Development per Year

    Returns:
        Model: Approximation Model for the demand, load factor and more

    To call the function use following syntax:
        Model(t, D0, mu, sigma)
    """
    return D0 * np.exp(mu * t + sigma * np.sqrt(t))


def exponential_fit(x, y):
    """
    This function fits the defined approxiamtion function to the given data and returns the fitted parameters.

    Parameters:
        x (array-like): Independent Variable Data (time).
        y (array-like): Dependent Variable Data (demand, load factor or any other dependent value).

    Returns:
        params (ndarray): Approximation vector in shape (Fth,) if the input is a vector, else the shape is (n_rows, Fth)

    To call the function use following syntax:
        exponential_fit(x, y)
    """

    # Calling the defined model function to fit the data
    def model_call(t, a, mu, sigma):
        return Model(t, a, mu, sigma)

    # Fitting the model to the data using curve_fit, starting with an initial guess
    initial_guess = [y[0], 0.1, 0.1]
    params, _ = opt.curve_fit(Model, x, y, p0=initial_guess)

    return params


def interpolate_exponential(data):
    """
    This function fits exponential parameters given in a list for a vector or matrix of inputdata and returns the fitted parameters in a vector or matrix format.
    The parameters contain the following values:
    - a: Initial Demand at t=0
    - mu: Growth Rate of Demand per Year
    - sigma: Volatility of Demand Development per Year

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
    # For 1D data, fitting a single exponential function
    if data.ndim == 1:
        x = np.linspace(0, len(data) - 1, len(data))
        return exponential_fit(x, data)

    # For 2D data, fitting an exponential function to each row
    elif data.ndim == 2:
        _, n_cols = data.shape
        x = np.linspace(0, n_cols - 1, n_cols)

        def fit_row(y):
            return exponential_fit(x, y)

        param_matrix = np.apply_along_axis(fit_row, axis=1, arr=data)
        return param_matrix

    else:
        raise ValueError("Input data must be 1D or 2D array.")


def trend_approximation(Param, CurveParameters, Adjust_mu=False):
    """
    This function computes trend approximation for a single vector or matrix of curve parameters by calling the previously defined model function.

    Parameters:
        Param (dict): Parameter Dictionary
        CurveParameters (ndarray):
            - Inital Value a
            - Growth Rate mu
            - Volatility sigma
        Adjust_mu (Bool): Condition to subtract Adjust_mu_factor from mu

    Returns:
        trend (ndarray): Approximation Trend for the Demand, Load Factor or any other Dependent Value According to the Model Function

    To call the function use following syntax:
        trend_approximation(Param, CurveParameters, Adjust_mu=True)
    """
    # Parameters
    time = Param["time"]
    Adjust_mu_factor = Param["adjusted_mu"]

    # Adjusting the shape to (1, Fth)
    time_expanded = time[np.newaxis, :]

    # Ensuring the input is at least of 2D shape
    CurveParameters = np.atleast_2d(CurveParameters)

    # Extracting parameters from CurveParameters
    initial_value = CurveParameters[:, [0]]  # (n, 1)
    mu = CurveParameters[:, [1]]  # (n, 1)
    sigma = CurveParameters[:, [2]]  # (n, 1)

    # Adjusting mu if condition applies
    if Adjust_mu:
        mu = mu - Adjust_mu_factor

    # Computing the trend using the model function
    trend = Model(time_expanded, initial_value, mu, sigma)

    # Readjusting the shape of the output if the input was a 1D vector
    return trend[0] if CurveParameters.shape[0] == 1 else trend


def Load_Factor_matrix(Param, Scenario):
    """
    This function calcuates the Load Factor according to the forecasts and a calculated trendline.
    The function uses the exponential fit model function to calculate the trendline.

    Parameters:
        Param (dict): Paremter dictionary
        Scenario (ndarray): Passenger Demand Scenario Matrix

    Returns:
        Load_Factor (ndarray): Load Factor Matrix

    To call the function use following syntax:
        Load_Factor_matrix(Param, Scenario)
    """
    # Parameters
    LF = Param["LF"]

    # Interpolating the Demand-Scenario
    params_list = interpolate_exponential(Scenario)

    # Approximating the Demand-Scenario-Trendline
    trend = trend_approximation(Param, params_list)

    # Smoothing the Demand-Scenario-Trend
    smoothing_factor_demand = Param["smoothing_factor_demand"]
    smoothed_scenario = trend + (smoothing_factor_demand * (Scenario - trend))

    # Interpolating the smoothed Demand-Scenario
    params_scenario = interpolate_exponential(smoothed_scenario)
    trend_smoothed_shifted = trend_approximation(Param, params_scenario, True) / LF

    # Calculating the Load Factor
    Load_Factor = smoothed_scenario / trend_smoothed_shifted

    # Adjust Loadfactor for initial setting
    delta = LF - Load_Factor[:, 0]
    Load_Factor = Load_Factor + delta.reshape(-1, 1)
    Load_Factor = np.clip(Load_Factor, 0.01, 1)

    return Load_Factor


def ATM(Param, Demand, Load_Factor):
    """
    This function calculates the Air Traffic Movements (ATM) based on the demand, load factor, aircraft mix and passenger capacity.

    Parameters:
        Param (dict): Parameter Dictionary
        Demand (ndarray): Demand Scenario Matrix
        Load_Factor (ndarray): Load Factor Matrix

    Returns:
        ATM (ndarray): Total Air Traffic Movements

    To call the function use following syntax:
        ATM(Param, Demand, Load_Factor)
    """
    # Parameters
    Pax_capacity = Param["Pax_capacity"]
    Mix = Param["Mix"]

    # Calculating the Transported Passengers per Aircraft Category
    Pax_s = Load_Factor * Pax_capacity[0]
    Pax_m = Load_Factor * Pax_capacity[1]
    Pax_l = Load_Factor * Pax_capacity[2]

    # Calculating the Air Traffic Movements (ATM)
    ATM = Demand / (Mix[0] * Pax_s + Mix[1] * Pax_m + Mix[2] * Pax_l)
    ATM = np.round(ATM, 0)

    # Applying the Mix to the ATM matrix
    ATM_mix = ATM[:, :, np.newaxis] * Mix

    return ATM, ATM_mix


def ATM_plot(ATM_Fleet, Param):
    """
    This function is plotting the Air Traffic Movement (ATM) mix matrix against the forecast time
    horizon vector (Fth), it only shows a selected number (No_Forecasts_plot) of plots

    Parameters:
        ATM_Fleet: ATM Szenario Matrix
        Param (dict): Parameter Dictionary

    Returns:
        None: Plot of "No_Forecasts_plot" Air Traffic Mix Forecasts Against the Forecast Time Horizon

    To call the function use following syntax:
        ATM_plot(ATM_Fleet, Param)

    """
    # Parameters
    time = Param["time"]

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    # Making sure the minimum number of plots to be shown does not lay below the available number of plots
    No_Forecasts_plot = min(Param["No_Forecasts_plot"], Param["No_Forecasts"])
    selected_indices = np.random.choice(
        Param["No_Forecasts"], No_Forecasts_plot, replace=False
    )

    plt.figure(figsize=(12, 6))

    # Looping through selected simulation runs
    for i in selected_indices:
        Shorthaul_Fleet = ATM_Fleet[i, :, 0]  # Short-haul for this run
        Mediumhaul_Fleet = ATM_Fleet[i, :, 1]  # Medium-haul for this run
        Longhaul_Fleet = ATM_Fleet[i, :, 2]  # Long-haul for this run

        # Plotting each run as a thin line to see individual trends
        plt.plot(time, Shorthaul_Fleet, color="blue", alpha=0.3)
        plt.plot(time, Mediumhaul_Fleet, color="green", alpha=0.3)
        plt.plot(time, Longhaul_Fleet, color="red", alpha=0.3)

    # Adding labels
    plt.xlabel("Years")
    plt.ylabel("Number of Aircraft Movements")
    plt.title("Total Air Traffic Movements per Aircarft Category")

    # Adding a single bold line to represent one of the runs for visibility in legend
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
        matrix (ndarray): Step Matrix

    To call the function use following syntax:
        exponential_matrix(Param)
    """
    # Parameters
    Fth = Param["Fth"] + 1
    No_Forecasts = Param["No_Forecasts"]
    scale_up = Param["scale_up"]
    p_down = Param["p_down"]
    scale_down = Param["scale_down"]
    offset_scale = Param["Fth"] * Param["S_curve_offset"] / 100

    # Setting the random seed for reproducibility
    np.random.seed(Param["seed"])

    # Generating positive and negative steps
    pos_steps = np.random.exponential(scale=scale_up, size=(No_Forecasts, Fth - 1))
    neg_steps = np.random.exponential(scale=scale_down, size=(No_Forecasts, Fth - 1))

    # Creating a mask for negative steps
    is_negative = np.random.rand(No_Forecasts, Fth - 1) < p_down
    steps = np.where(is_negative, -neg_steps, pos_steps)

    # Generating a random offsets per forecast
    t_offset = np.random.exponential(scale=offset_scale, size=(No_Forecasts, 1))

    # Initializing a matrix with zeros
    matrix = np.zeros((No_Forecasts, Fth))

    # Computing the cumulative sum with the offset
    matrix[:, 1:] = np.cumsum(steps, axis=1) - t_offset

    return np.clip(matrix, 0, Fth)


def S_curve(Param):
    """
    This function calculates S-Curve values according to the set parameters and the
    matrix from the exponential_matrix function

    Parameters:
        Param (dict): Parameter Dictionary

    Returns:
        S_values_matrix (ndarray): S-Curve Values Matrix

    To call the function use following syntax:
        S_curve(Param)
    """
    # Parameters
    t_max = Param["Fth"] + 1
    t0 = Param["t0_factor"] * t_max
    L = Param["L"]
    k = Param["k"]

    # Generating an exponential matrix
    # The matrix contains the time steps for each forecast
    matrix = exponential_matrix(Param)

    # Base sigmoid without scaling
    sigmoid = 1 / (1 + np.exp(-k * (matrix - t0)))

    # Normalize so that S(0) = 0 and S(∞) = L
    f0 = 1 / (1 + np.exp(k * t0))  # value at t=0
    S_values_matrix = L * (sigmoid - f0) / (1 - f0)

    return S_values_matrix


# def S_curve_plot(Param, S_Values):
#     """
#     This function plots the set S-Curve with a set of randomly choosen S-Curve values from the
#     exponential_matrix and S_curve functions.

#     Parameters:
#         Param (dict): Parameter Dictionary
#         S_Values (ndarray): S-Curve values Matrix

#     Returns:
#         None: Plot of the S-Curve and a Random set of the S-Curve Values

#     To call the function use following syntax:
#         S_curve_plot(Param, S_Values)

#     """
#     # Parameters
#     Fth = Param["Fth"] + 1
#     time = Param["time"]
#     L = Param["L"]
#     k = Param["k"]
#     t0 = Param["t0_factor"]
#     t0_val = t0 * Fth

#     # Normalized sigmoid curve
#     sigmoid = 1 / (1 + np.exp(-k * (time - t0_val)))
#     f0 = 1 / (1 + np.exp(k * t0_val))  # sigmoid value at time = 0
#     S = L * (sigmoid - f0) / (1 - f0)

#     # Setting the random seed for reproducibility
#     np.random.seed(Param["seed"])

#     plt.figure(figsize=(10, 6))
#     plt.plot(time, S, label="S-Curve", color="blue")

#     # Randomly selecting indices to plot
#     No_Forecasts_plot = min(Param["No_Forecasts_plot"], Param["No_Forecasts"])
#     selected_indices = np.random.choice(
#         S_Values.shape[0], No_Forecasts_plot, replace=False
#     )

#     # Ploting the selected forecasts
#     plt.plot(time, S_Values[selected_indices].T, alpha=0.7)
#     plt.xlabel("Time [yrs]")
#     plt.ylabel("Percentage [%]")
#     plt.legend()
#     plt.grid()
#     plt.title("Technology Adoption with Standard S-Curve")
#     plt.show()


def S_curve_plot(
    Param,
    S_Values,
    save_plot=False,
    run_name="Undefined_Run",
    output_base_folder="Plots",
):
    """
    This function calculates S-Curve values according to the set parameters and the
    matrix from the exponential_matrix function.
    Additionally, it saves the plots the S-Curve if the save_plot parameter is set to True.

    Parameters:
        Param (dict): Parameter Dictionary
        S_Values (ndarray): S-Curve values Matrix
        run_name (str): Name of the current run (used to create the saving folder)
        save_plot (bool): Whether to save the plot
        output_base_folder (str): Base directory for saving plots

    Returns:
        S_values_matrix (ndarray): S-Curve Values Matrix

    To call the function use following syntax:
        S_curve(Param, S_Values, save_plot=False, run_name="Undefined_Run", output_base_folder="Plots")
    """

    # Parameters
    Fth = Param["Fth"] + 1
    time = Param["time"]
    L = Param["L"]
    k = Param["k"]
    t0 = Param["t0_factor"]
    t0_val = t0 * Fth

    # Normalized sigmoid curve
    sigmoid = 1 / (1 + np.exp(-k * (time - t0_val)))
    f0 = 1 / (1 + np.exp(k * t0_val))  # sigmoid value at time = 0
    S = L * (sigmoid - f0) / (1 - f0)

    # Setting seed for reproducibility
    np.random.seed(Param["seed"])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time, S, label="S-Curve", color="blue")

    # Selecting random forecast paths
    No_Forecasts_plot = min(Param["No_Forecasts_plot"], Param["No_Forecasts"])
    selected_indices = np.random.choice(
        S_Values.shape[0], No_Forecasts_plot, replace=False
    )
    plt.plot(time, S_Values[selected_indices].T, alpha=0.7)

    plt.xlabel("Time [yrs]")
    plt.ylabel("Percentage [%]")
    plt.legend()
    plt.grid()
    plt.title("Technology Adoption with Standard S-Curve")

    # Saving the plot
    if save_plot:
        folder_path = os.path.join(output_base_folder, run_name)
        os.makedirs(folder_path, exist_ok=True)

        # Preparing a safe file name and avoiding overwrites
        filename_safe_title = "S_Curve_Plot"
        base_filename = f"{filename_safe_title}.png"
        plot_path = os.path.join(folder_path, base_filename)

        counter = 1
        while os.path.exists(plot_path):
            base_filename = f"{filename_safe_title}_{counter}.png"
            plot_path = os.path.join(folder_path, base_filename)
            counter += 1

        plt.savefig(plot_path)
        print(f"S-Curve plot saved to: {os.path.abspath(plot_path)}")

    plt.show()


def LH2_technology_adoption(Param, S_values, ATM_matrix):
    """
    This function calculates the LH2 technology adoption based on the S-curve adaption scenarios and the ATM mix matrix.

    Parameters:
        Param (dict): Parameter dictionary
        S_values (ndarray): S-curve Values for Technology Adoption
        ATM_matrix (ndarray): ATM Scenario Matrix

    Returns:
        LH2_adoption, LH2_mix_adoption (ndarray): LH2 Technology Adoption Values

    To call the function use following syntax:
        LH2_technology_adoption(Param, S_values, ATM_matrix)
    """
    # Parameters
    Mix = Param["Mix"]  # Aircraft Mix

    # Checking the shape of the ATM matrix
    if ATM_matrix.ndim == 2:
        ATM_matrix = ATM_matrix
    elif ATM_matrix.ndim == 3:
        ATM_matrix = np.sum(ATM_matrix, axis=2)
    else:
        raise ValueError("ATM_matrix must be either 2D or 3D.")

    # Calcualting the LH2 technology adoption
    LH2_adoption = np.round(ATM_matrix * S_values, 0)

    # Calculating the LH2 mix adoption
    LH2_mix_adoption = np.round(LH2_adoption[:, :, np.newaxis] * Mix, 0)

    # Alternative calculation of the LH2 mix adoption
    # LH2_mix_adoption2 = np.round(ATM_matrix[1] * S_values[:, :, np.newaxis], 0)

    return LH2_adoption, LH2_mix_adoption


def Capacity_2D(Param, delta_K, adjustK0=False):
    """
    This function calculates the capacity based on the initial capacity (K0) and the cumulative sum of delta_K.
    It returns the capacity as a matrix. The capacity is calculated for each forecast time step.

    Parameters:
        Param (dict): Parameter Dictionary
        delta_K (np.array): Capacity Change Vector
        adjustK0 (bool): If True, adjusts K0 to K0_LH

    Returns:
        K (np.array): Capacity Vectors in 2D shape

    To call the function use following syntax:
        Capacity_2D(Param, delta_K)
    """
    # Parameters
    K0 = Param["K0"]  # Initial Capacity of Jet A1 aircraft stands
    K0_LH = Param["K0_LH"]  # Initial Capacity of LH2 aircraft stands
    No_Forecasts = Param["No_Forecasts"]
    TTF = Param["TTF"][1]

    # Adjusting K0 for LH2 initial Capacity
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
    It returns the capacity as a matrix depending on the matrix parameter. The capacity is calculated for each forecast time step.

    Parameters:
        Param (dict): Parameter Dictionary
        delta_K (np.array): Capacity Change Vector
        adjustK0 (bool): If True, adjusts K0 to K0_LH

    Returns:
        K (np.array): Capacity Vector or Matrix Depending on the Matrix Parameter.

    To call the function use following syntax:
        Capacity_3D(Param, delta_K)
    """
    # Parameters
    K0 = Param["K0"] * Param["Mix"]  # Initial Capacity of Jet A1 aircraft stands
    K0_LH = Param["K0_LH"] * Param["Mix"]  # Initial Capacity of LH2 aircraft stands
    Mix = Param["Mix"]  # Aircraft Mix
    TTF = Param["TTF"]  # Turnaround time factor

    # Adjusting K0 for LH2 initial Capacity
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

    Parameters:
        Param (dict): Parameter Dictionary
        delta_K_Jet (np.array): Capacity Change Vector

    Returns:
        CI_Jet (np.array): Installation Cost for Jet A1 Aircraft Stands

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


def Capex_LH(Param, delta_K_LH, D_LH_yearly):
    """
    This function calculates the capital expenditure (Capex) for LH2 aircraft stands based on the change in capacity (delta_K).

    Parameters:
        Param (dict): Parameter Dictionary
        delta_K_LH (np.array): Capacity Change Vector
        D_LH_yearly (np.array): LH2 Aircraft Movement Demand per year

    Returns:
        CI_LH (np.array): Installation Cost for LH2 aircraft stands in USD

    To call the function use following syntax:
        Capex_LH(Param, delta_K_LH, D_LH_yearly)
    """
    # Parameters
    TS_D_LH = Param["TS_D_LH"]  # LH2 Yearly Demand Threshold for Pipeline construction
    D_LH_Fuel_ATM = Param["D_LH_Fuel_ATM"]  # LH2 DHL Demand per Aircraft Movement
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

    # If condition to adjust d_LH shape
    if D_LH_yearly.ndim == 2:
        D_LH_yearly = D_LH_yearly
    elif D_LH_yearly.ndim == 3:
        D_LH_yearly = np.sum(D_LH_yearly, axis=2)
    else:
        raise ValueError("D_LH_yearly has neiter shape 2D or 3D")

    D_LH_Fuel = D_LH_yearly * D_LH_Fuel_ATM  # Demand for LH2 Fuel based on yearly ATMs

    # Calculating the difference matrix
    diff = TS_D_LH - D_LH_Fuel

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
    # Combining the two conditions to get the final Installation Cost for LH2 stands
    CI_LH = np.round(CI_LH_greater + CI_LH_less_equal, 2)

    return CI_LH


def Opex_Jet(Param, k_Jet, d_Jet, K_Jet_yearly, D_Jet_yearly):
    """
    This function calculates the operating expenditure (Opex) for Jet A1 aircraft stands based on the stand capacity K and stand demand D.

    Parameters:
        Param (dict): Parameter Dictionary
        k_Jet (np.array): Stand Capacity Vector DHL
        d_Jet (np.array): Stand Demand Vector DHL
        K_Jet_yearly (np.array): Yearly Capacity Vector yearly
        D_Jet_yearly (np.array): Yearly Demand Vector yearly

    Returns:
        CO_Jet (np.array): Operating Cost for Jet A1 aircraft stands in USD

    To call the function use following syntax:
        Opex_Jet(Param, k_Jet, d_Jet, K_Jet_yearly, D_Jet_yearly)
    """
    # Parameters
    p_Dock = Param["p_Dock"]
    CE_Dock_Jet = Param["CE_Dock_Jet"]  # Cost of energy for Jet A1
    CE_Open_Jet = Param["CE_Open_Jet"]  # Cost of energy for open Jet A1
    CM_Over_Jet = Param["CM_Over_Jet"]  # Cost of maintenance for over capacity
    CM_Under_Jet = Param["CM_Under_Jet"]  # Cost of maintenance for under capacity
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust k_Jet shape
    if k_Jet.ndim == 2:
        k_Jet = k_Jet
    elif k_Jet.ndim == 3:
        k_Jet = np.sum(k_Jet, axis=2)
    else:
        raise ValueError("k_Jet has neiter shape 2D or 3D")

    # If condition to adjust d_Jet shape
    if d_Jet.ndim == 2:
        d_Jet = d_Jet
    elif d_Jet.ndim == 3:
        d_Jet = np.sum(d_Jet, axis=2)
    else:
        raise ValueError("d_Jet has neiter shape 2D or 3D")
    # If condition to adjust K_Jet shape
    if K_Jet_yearly.ndim == 2:
        K_Jet_yearly = K_Jet_yearly
    elif K_Jet_yearly.ndim == 3:
        K_Jet_yearly = np.sum(K_Jet_yearly, axis=2)
    else:
        raise ValueError("K_Jet_yearly has neiter shape 2D or 3D")

    # If condition to adjust D_Jet shape
    if D_Jet_yearly.ndim == 1:
        D_Jet_yearly = np.tile(
            D_Jet_yearly[:, :, np.newaxis], (1, 1, len(Param["Mix"]))
        )
    elif D_Jet_yearly.ndim == 2:
        D_Jet_yearly = D_Jet_yearly
    elif D_Jet_yearly.ndim == 3:
        D_Jet_yearly = np.sum(D_Jet_yearly, axis=2)
    else:
        raise ValueError("D_Jet_yearly has neiter shape 1D, 2D or 3D")

    # Calculating the difference matrix capacity minus demand in the DHL
    diff = k_Jet - d_Jet

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculating the operational expenditure based on the demand and capacity
    # k_Jet > d_Jet
    CO_Jet_greater = greater * (
        D_Jet_yearly * (p_Dock * CE_Dock_Jet + (1 - p_Dock) * CE_Open_Jet)
        + (K_Jet_yearly - D_Jet_yearly) * CM_Over_Jet
    )
    # k_Jet <= d_Jet
    CO_Jet_less_equal = less_equal * (
        K_Jet_yearly * (p_Dock * CE_Dock_Jet + (1 - p_Dock) * CE_Open_Jet)
        + (D_Jet_yearly - K_Jet_yearly) * CM_Under_Jet
    )

    # Combine the two conditions to get the final operational expenditure
    CO_Jet = np.round(CO_Jet_greater + CO_Jet_less_equal, 2)

    return CO_Jet


def Opex_LH(Param, k_LH, d_LH, K_LH_yearly, D_LH_yearly):
    """
    This function calculates the operating expenditure (Opex) for LH2 aircraft stands based on the stand capacity K and stand demand D.

    Parameters:
        Param (dict): Parameter Dictionary
        k_LH (np.array): Stand Capacity Vector in the DHL
        d_LH (np.array): Stand Demand Vector in the DHL
        K_LH_yearly (np.array): Yearly Capacity Vector
        D_LH_yearly (np.array): Yearly Demand Vector

    Returns:
        CO_LH (np.array): Operating Cost for LH2 aircraft stands in USD

    To call the function use following syntax:
        Opex_LH(Param, k_LH, d_LH, K_LH_yearly, D_LH_yearly)
    """
    # Parameters
    p_Dock = Param["p_Dock"]  # Share of aircraft beeing handeled at dock stands
    CE_Dock_LH = Param["CE_Dock_LH"]  # Cost of energy for LH2
    CE_Open_LH = Param["CE_Open_LH"]  # Cost of energy for open LH2
    CM_Over_LH = Param["CM_Over_LH"]  # Cost of maintenance for over capacity
    CM_Under_LH = Param["CM_Under_LH"]  # Cost of maintenance for under capacity
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust k_Jet shape
    if k_LH.ndim == 2:
        k_LH = k_LH
    elif k_LH.ndim == 3:
        k_LH = np.sum(k_LH, axis=2)
    else:
        raise ValueError("k_Jet has neiter shape 2D or 3D")

    # If condition to adjust d_Jet shape
    if d_LH.ndim == 2:
        d_LH = d_LH
    elif d_LH.ndim == 3:
        d_LH = np.sum(d_LH, axis=2)
    else:
        raise ValueError("d_Jet has neiter shape 2D or 3D")

    # If condition to adjust K_LH shape
    if K_LH_yearly.ndim == 2:
        K_LH_yearly = K_LH_yearly
    elif K_LH_yearly.ndim == 3:
        K_LH_yearly = np.sum(K_LH_yearly, axis=2)
    else:
        raise ValueError("K_LH_yearly has neiter shape 2D or 3D")

    # If condition to adjust D_LH shape
    if D_LH_yearly.ndim == 1:
        D_LH_yearly = np.tile(D_LH_yearly[:, :, np.newaxis], (1, 1, len(Param["Mix"])))
    elif D_LH_yearly.ndim == 2:
        D_LH_yearly = D_LH_yearly
    elif D_LH_yearly.ndim == 3:
        D_LH_yearly = np.sum(D_LH_yearly, axis=2)
    else:
        raise ValueError("D_LH_yearly has neiter shape 1D, 2D or 3D")

    # Calculating the difference matrix capacity minus demand
    diff = k_LH - d_LH

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculating the operational expenditure based on the demand and capacity
    # k_LH > d_LH
    CO_LH_greater = greater * (
        D_LH_yearly * (p_Dock * CE_Dock_LH + (1 - p_Dock) * CE_Open_LH)
        + (K_LH_yearly - D_LH_yearly) * CM_Over_LH
    )
    # k_LH <= d_LH
    CO_LH_less_equal = less_equal * (
        K_LH_yearly * (p_Dock * CE_Dock_LH + (1 - p_Dock) * CE_Open_LH)
        + (D_LH_yearly - K_LH_yearly) * CM_Under_LH
    )
    # Combine the two conditions to get the final operational expenditure
    CO_LH = np.round(CO_LH_greater + CO_LH_less_equal, 2)

    return CO_LH


def Opex_Terminal(Param, PAX_yearly):
    """
    This function calculates the operating expenditure (Opex) for a terminal based on Passenger numbers.

    Parameters:
        Param (dict): Parameter Dictionary
        PAX_yearly (np.array): Number of Passengers per Year

    Returns:
        CO_Terminal (np.array): Operating Cost for the Terminal

    To call the function use following syntax:
        Opex_Terminal(Param, PAX_yearly)
    """
    # Parameters
    CE_Terminal = Param[
        "CE_Terminal"
    ]  # Cost of terminal operations in USD per passenger

    CO_Terminal = np.round(PAX_yearly * CE_Terminal, 2)

    return CO_Terminal


def Total_Cost_calculation(
    Param,
    delta_K_Jet,
    delta_K_LH,
    k_Jet,
    k_LH,
    d_Jet,
    d_LH,
    PAX_yearly,
    K_Jet_yearly,
    K_LH_yearly,
    D_Jet_yearly,
    D_LH_yearly,
):
    """
    This function calculates the cost of the airport infrastructure project based on the capital expenditure (Capex) and
    operational expenditure (Opex) for Jet A1 and LH2 aircraft stands, as well as terminal operations.

    Parameters:
        Param (dict): Parameter Dictionary
        delta_K_Jet (np.array): Capacity Change Vector for Jet A1
        delta_K_LH (np.array): Capacity Change Vector for LH2
        k_Jet (np.array): Stand Capacity Vector for Jet A1 in the DHL
        k_LH (np.array): Stand Capacity Vector for LH2 in the DHL
        d_Jet (np.array): Stand Demand Vector for Jet A1 in the DHL
        d_LH (np.array): Stand Demand Vector for LH2 in the DHL
        PAX_yearly (np.array): Number of Passengers per Year
        K_Jet_yearly (np.array): Yearly Jet A1 Stand Capacity
        K_LH_yearly (np.array): Yearly LH2 Stand Capacity
        D_Jet_yearly (np.array): Yearly Jet A1 Stand Demand
        D_LH_yearly (np.array): Yearly LH2 Stand Demand

    Returns:
        Total_cost (np.array): Total Cost

    To call the function use following syntax:
        Total_Cost_calculation(Param, delta_K_Jet, delta_K_LH, k_Jet, k_LH, d_Jet,
        d_LH, PAX_yearly, K_Jet_yearly, K_LH_yearly, D_Jet_yearly, D_LH_yearly)
    """
    # Installation Cost
    CI_Jet = Capex_Jet(Param, delta_K_Jet)
    CI_LH = Capex_LH(Param, delta_K_LH, D_LH_yearly)

    # Operating Cost
    CO_Jet = Opex_Jet(Param, k_Jet, d_Jet, K_Jet_yearly, D_Jet_yearly)
    CO_LH = Opex_LH(Param, k_LH, d_LH, K_LH_yearly, D_LH_yearly)
    CO_Terminal = Opex_Terminal(Param, PAX_yearly)

    # Total cost as the sum of capital expenditure and operational expenditure
    Total_cost = CI_Jet + CI_LH + CO_Jet + CO_LH + CO_Terminal

    return Total_cost


def Revenue_Jet(Param, k_Jet, d_Jet, K_Jet_yearly, D_Jet_yearly):
    """
    This function calculates the revenue from Jet A1 aircraft stand fees based on stand capacity K and stand demand D.

    Parameters:
        Param (dict): Parameter Dictionary
        k_Jet (np.array): Stand Capacity Vector in the DHL
        d_Jet (np.array): Stand Demand Vector in the DHL
        K_Jet_yearly (np.array): Yearly Jet A1 Stand Capacity
        D_Jet_yearly (np.array): Yearly Jet A1 Stand Demand

    Returns:
        R_Jet (np.array): Revenue from Jet A1 aircraft stands

    To call the function use following syntax:
        Revenue_Jet(Param, k_Jet, d_Jet, K_Jet_yearly, D_Jet_yearly)
    """
    # Parameters
    p_dock = Param["p_Dock"]  # Share of aircrafts beeing handeled at dock stands
    re_Dock_Jet = Param["re_Dock_Jet"]  # Revenue from Jet A1 aircraft at dock stands
    re_Open_Jet = Param["re_Open_Jet"]  # Revenue from Jet A1 aircraft at open stands
    rf_Jet = Param["rf_Jet"]  # Fixed revenue for Jet A1
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust K_Jet shape
    if k_Jet.ndim == 2:
        k_Jet = k_Jet
    elif k_Jet.ndim == 3:
        k_Jet = np.sum(k_Jet, axis=2)
    else:
        raise ValueError("k_Jet has neiter shape 2D or 3D")

    # If condition to adjust D_Jet shape
    if d_Jet.ndim == 2:
        d_Jet = d_Jet
    elif d_Jet.ndim == 3:
        d_Jet = np.sum(d_Jet, axis=2)
    else:
        raise ValueError("d_Jet has neiter shape 2D or 3D")

    # If condition to adjust K_Jet shape
    if K_Jet_yearly.ndim == 2:
        K_Jet_yearly = K_Jet_yearly
    elif K_Jet_yearly.ndim == 3:
        K_Jet_yearly = np.sum(K_Jet_yearly, axis=2)
    else:
        raise ValueError("k_Jet_yearly has neiter shape 2D or 3D")

    # If condition to adjust D_Jet shape
    if D_Jet_yearly.ndim == 2:
        D_Jet_yearly = D_Jet_yearly
    elif D_Jet_yearly.ndim == 3:
        D_Jet_yearly = np.sum(D_Jet_yearly, axis=2)
    else:
        raise ValueError("d_Jet_yearly has neiter shape 2D or 3D")

    # Calculating the difference matrix
    diff = k_Jet - d_Jet

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculating the operational expenditure based on the demand and capacity
    # k_Jet > d_Jet
    R_Jet_greater = (
        greater * D_Jet_yearly * (p_dock * re_Dock_Jet)
        + (1 - p_dock) * re_Open_Jet
        + rf_Jet
    )
    # k_Jet <= d_Jet
    R_Jet_less_equal = (
        less_equal * K_Jet_yearly * (p_dock * re_Dock_Jet)
        + (1 - p_dock) * re_Open_Jet
        + rf_Jet
    )
    # Combining the two conditions to get the final operational expenditure
    R_Jet = np.round(R_Jet_greater + R_Jet_less_equal, 2)

    return R_Jet


def Revenue_LH(Param, k_LH, d_LH, K_LH_yearly, D_LH_yearly):
    """
    This function calculates the revenue from LH2 aircraft stand fees based on stand capacity K and stand demand D.

    Parameters:
        Param (dict): Parameter Dictionary
        k_LH (np.array): Stand Capacity Vector
        d_LH (np.array): Stand Demand Vector
        K_LH_yearly (np.array): Yearly LH2 Stand Capacity
        D_LH_yearly (np.array): Yearly LH2 Stand Demand

    Returns:
        R_LH (np.array): Revenue from LH2 aircraft stands

    To call the function use following syntax:
        Revenue_LH(Param, k_LH, d_LH, K_LH_yearly, D_LH_yearly)
    """
    # Parameters
    p_dock = Param["p_Dock"]  # Share of aircraft beeing handeled at dock stands
    re_Dock_LH = Param["re_Dock_LH"]  # Revenue from LH2 aircraft at dock stands
    re_Open_LH = Param["re_Open_LH"]  # Revenue from LH2 aircraft at open stands
    rf_LH = Param["rf_LH"]  # Revenue for LH2 refueling
    condition = Param["condition"]  # Condition for the difference matrix

    # If condition to adjust k_Jet shape
    if k_LH.ndim == 2:
        k_LH = k_LH
    elif k_LH.ndim == 3:
        k_LH = np.sum(k_LH, axis=2)
    else:
        raise ValueError("k_Jet has neiter shape 2D or 3D")

    # If condition to adjust d_Jet shape
    if d_LH.ndim == 2:
        d_LH = d_LH
    elif d_LH.ndim == 3:
        d_LH = np.sum(d_LH, axis=2)
    else:
        raise ValueError("d_Jet has neiter shape 2D or 3D")

    # If condition to adjust K_LH shape
    if K_LH_yearly.ndim == 2:
        K_LH_yearly = K_LH_yearly
    elif K_LH_yearly.ndim == 3:
        K_LH_yearly = np.sum(K_LH_yearly, axis=2)
    else:
        raise ValueError("K_LH_yearly has neiter shape 2D or 3D")

    # If condition to adjust D_LH shape
    if D_LH_yearly.ndim == 2:
        D_LH_yearly = D_LH_yearly
    elif D_LH_yearly.ndim == 3:
        D_LH_yearly = np.sum(D_LH_yearly, axis=2)
    else:
        raise ValueError("K_LH_yearly has neiter shape 2D or 3D")

    # Calculating the difference matrix
    diff = k_LH - d_LH

    # Creating indent matrices for the given conditions
    greater = np.greater(diff, condition).astype(int)
    less_equal = np.less_equal(diff, condition).astype(int)

    # Calculating the operational expenditure based on the demand and capacity
    # k_LH > d_LH
    R_LH_greater = (
        greater * D_LH_yearly * (p_dock * re_Dock_LH)
        + (1 - p_dock) * re_Open_LH
        + rf_LH
    )
    # k_LH <= d_LH
    R_LH_less_equal = (
        less_equal * K_LH_yearly * (p_dock * re_Dock_LH)
        + (1 - p_dock) * re_Open_LH
        + rf_LH
    )
    # Combining the two conditions to get the final operational expenditure
    R_LH = np.round(R_LH_greater + R_LH_less_equal, 2)

    return R_LH


def Revenue_Pax(Param, PAX_yearly):
    """
    This function calculates the operating revenues from the terminal based on Passenger numbers.

    Parameters:
        Param (dict): Parameter Dictionary
        PAX_Yearly (np.array): Number of Passengers per Year

    Returns:
        R_PAX (np.array): Operating revenue from passengers

    To call the function use following syntax:
        Reveneue_Pax(Param, PAX_yearly)
    """
    # Parameters
    re_Pax = Param["re_Pax"]  # Revenue per passenger within the terminal
    return PAX_yearly * re_Pax


def Revenue_Rent(Param, k_Jet, k_LH):
    """
    This function calculates the revenues from renting out spaces within the terminal.

    Parameters:
        Param (dict): Parameter Dictionary
        k_Jet (np.array): Total Capacity in the DHL for Jet A1 aircraft stands
        k_LH (np.array): Total Capacity in the DHL for LH2 aircraft stands

    Returns:
        R_Rent (np.array): Revenue from renting out spaces in USD

    To call the function use following syntax:
        Reveneue_Rent(Param, K)
    """
    # Parameters
    re_Rent = Param["re_Rent"]  # Revenue from renting spaces per unit of capacity K

    # If condition to adjust k_Jet shape
    if k_Jet.ndim == 2:
        k_Jet = k_Jet
    elif k_Jet.ndim == 3:
        k_Jet = np.sum(k_Jet, axis=2)
    else:
        raise ValueError("k_Jet has neither 2D or 3D shape")

    # If condition to adjust k_LH shape
    if k_LH.ndim == 2:
        k_LH = k_LH
    elif k_LH.ndim == 3:
        k_LH = np.sum(k_LH, axis=2)
    else:
        raise ValueError("k_LH has neither 2D or 3D shape")

    # Calculating the total revenue from renting
    R_Rent = (k_Jet + k_LH) * re_Rent

    return R_Rent


def Total_Revenue_calculation(
    Param,
    k_Jet,
    k_LH,
    d_Jet,
    d_LH,
    PAX_yearly,
    K_Jet_yearly,
    K_LH_yearly,
    D_Jet_yearly,
    D_LH_yearly,
):
    """
    This function calculates the total revenues from the airport infrastructure project based on
    the Jet A1 and LH2 aircraft stand operation, as well as terminal and rental revenues.

    Parameters:
        Param (dict): Parameter Dictionary
        K (np.array): Total Stand Capacity
        k_Jet (np.array): Stand Capacity Vector for Jet A1 Aircraft Stands in the DHL
        k_LH (np.array): Stand Capacity Vector for LH2 Aircraft Stands in the DHL
        d_Jet (np.array): Stand Demand Vector for Jet A1 Aircraft Stands in the DHL
        d_LH (np.array): Stand Demand Vector for LH2 Aircraft Stands in the DHL
        PAX (np.array): Number of Passengers per Year
        K_Jet_yearly (np.array): Yearly Jet A1 Aircraft Stand  Capacity
        K_LH_yearly (np.array): Yearly LH2 Aircraft Stand Capacity
        D_Jet_yearly (np.array): Yearly Jet A1 Aircraft Stand Demand
        D_LH_yearly (np.array): Yearly LH2 Aircraft Stand Demand

    Returns:
        Total_revenue (np.array): Total revenue from the airport infrastructure

    To call the function use following syntax:
        Total_Revenue_calculation(Param, k_Jet, k_LH, d_Jet, d_LH, PAX_yearly,
        K_Jet_yearly, K_LH_yearly, D_Jet_yearly, D_LH_yearly)
    """
    # Calculating the revenue for Jet A1
    R_Jet = Revenue_Jet(Param, k_Jet, d_Jet, K_Jet_yearly, D_Jet_yearly)

    # Calculating the revenue for LH2
    R_LH = Revenue_LH(Param, k_LH, d_LH, K_LH_yearly, D_LH_yearly)

    # Calculating the revenue from passengers
    R_Pax = Revenue_Pax(Param, PAX_yearly)

    # Calculating the revenue from renting
    R_Rent = Revenue_Rent(Param, k_Jet, k_LH)

    # Calculating the total revenue as the sum of all revenues
    Total_revenue = R_Jet + R_LH + R_Pax + R_Rent

    return Total_revenue


def NPV_calculation(Param, delta_K_Jet, delta_K_LH, d_ATM, S_values, PAX_yearly):
    """
    This function calculates the Net Present Value (NPV) of the airport infrastructure project based on the
    capital expenditure (Capex), operational expenditure (Opex), and revenues from Jet A1 and LH2 aircraft stands, as well as terminal and rental revenues.

    Parameters:
        Param (dict): Parameter Dictionary
        delta_K (np.array): Aircraft Stand Capacity Change Vector
        d_ATM (np.array): Total Air Traffic Movements
        S_value_matrix (np.array): S-Curve Matrix for Technology Adoption
        PAX_yearly (np.array): Number of Passengers per Year

    Returns:
        NPV (np.array): Net Present Value of the Airport Infrastructure Project

    To call the function use following syntax:
        NPV_calculation(Param, delta_K_Jet, delta_K_LH, d_ATM, S_values, PAX_yearly)
    """
    # Parameters
    discount_rate = Param["discount_rate"]  # Discount rate for NPV calculation
    Initial_Investment = Param["Initial_Investment"]  # Initial investment

    # If condition to adjust d_ATM shape
    if d_ATM.ndim == 2:
        d_Jet = d_ATM * (1 - S_values)  # Jet A1 aircraft demand
        d_LH = d_ATM * S_values  # LH2 aircraft demand
    elif d_ATM.ndim == 3:
        d_Jet = np.round((d_ATM * (1 - S_values)[:, :, np.newaxis]), 0)
        d_LH = np.round(d_ATM * S_values[:, :, np.newaxis], 0)
    k_Jet = Capacity_3D(Param, delta_K_Jet)  # Jet A1 aircraft stand capacity
    k_LH = Capacity_3D(Param, delta_K_LH, True)  # LH2 aircraft stand capacity

    # Calculting the yearly capacit and yearly demand
    K_Jet_yearly, K_Jet_mix_yearly = K_yearly(Param, k_Jet)
    K_LH_yearly, K_LH_mix_yearly = K_yearly(Param, k_LH)
    D_Jet_yearly, D_Jet_mix_yearly = ATM_yearly(Param, d_Jet)
    D_LH_yearly, D_LH_mix_yearly = ATM_yearly(Param, d_LH)

    Cost = Total_Cost_calculation(
        Param,
        delta_K_Jet,
        delta_K_LH,
        k_Jet,
        k_LH,
        d_Jet,
        d_LH,
        PAX_yearly,
        K_Jet_yearly,
        K_LH_yearly,
        D_Jet_yearly,
        D_LH_yearly,
    )
    Revenue = Total_Revenue_calculation(
        Param,
        k_Jet,
        k_LH,
        d_Jet,
        d_LH,
        PAX_yearly,
        K_Jet_yearly,
        K_LH_yearly,
        D_Jet_yearly,
        D_LH_yearly,
    )
    Profit = Revenue - Cost
    # Discount per year:
    Discount = 1 / ((1 + discount_rate) ** np.arange(Profit.shape[1]))
    # Applying the discount per year to each row (scenario) — element-wise
    Discounted_Profit = Profit * Discount
    # Summing the discounted profits over time
    NPV = np.sum(Discounted_Profit, axis=1) - Initial_Investment

    return NPV


def ENPV_calculation(Param, delta_K_Jet, delta_K_LH, d_ATM, S_values, PAX_yearly):
    """
    This function calculates the Expected Net Present Value (NPV) of the airport infrastructure project based on the Net Present Value (NPV) calculation for each scenario.

    Parameters:
        Param (dict): Parameter Dictionary
        delta_K_Jet (np.array): Capacity Change Vector for Jet A1 Aircraft Stand Capacity
        delta_K_LH (np.array): Capacity Change Vector for LH2 Aircraft Stand Capacity
        d_ATM (np.array): ATMs in the DHL
        S_values (np.array): S-Curve Matrix for Technology Adoption
        PAX_yearly (np.array): Number of Passengers per Year

    Returns:
        ENPV (float): Expected Net Present Value of the Airport Infrastructure Project in USD

    To call the function use following syntax:
        ENPV_calculation(Param, delta_K_Jet, delta_K_LH, d_ATM, S_values, PAX_yearly)
    """
    # Parameters
    No_Forecasts = Param["No_Forecasts"]

    # If condition to adjust delta_K_Jet shape
    if delta_K_Jet.ndim == 1:
        delta_K_Jet = np.tile(delta_K_Jet, (No_Forecasts, 1))
    else:
        delta_K_Jet = delta_K_Jet

    # If condition to adjust delta_K_LH shape
    if delta_K_LH.ndim == 1:
        delta_K_LH = np.tile(delta_K_LH, (No_Forecasts, 1))
    else:
        delta_K_LH = delta_K_LH

    # Calculating the NPV for each scenario
    NPV = NPV_calculation(Param, delta_K_Jet, delta_K_LH, d_ATM, S_values, PAX_yearly)
    Sum_NPV = np.sum(NPV)  # Summing the NPV values

    ENPV = Sum_NPV / len(NPV)  # Expected NPV is the expected value of the NPV values

    return ENPV


def GA_dual(Param, d_ATM, S_values, PAX_yearly):
    """
    This function evaluates the delta_K_Jet_Mix and delta_K_LH_Mix using a genetic algorithm to maximize the ENPV.

    Parameters:
        Param (dict): Parameter dictionary
        d_ATM (np.array): DHL ATMs
        S_values (np.array): S-curve Technology Adoption mMatrix
        PAX (np.array): Passengers per Year

    Returns:
        delta_K_Jet_Mix, delta_K_LH_Mix (np.array): Jet A1 and LH2 Aircraft Stand Capacity Change Mix Matrix

    To call this function use:
        GA_dual(Param, d_ATM, S_values, PAX)
    """
    # Parameters
    value_vector = Param["allowed_values"]
    Fth = Param["Fth"] + 1
    mix_dim = len(Param["Mix"])
    n_vars = 2 * Fth * mix_dim  # Variables for Jet and LH2

    # Convert 2D d_ATM to 3D by applying the Mix
    if d_ATM.ndim == 2:
        mix = np.array(Param["Mix"])
        d_ATM = d_ATM[:, :, np.newaxis] * mix[np.newaxis, np.newaxis, :]

    # Registering the fitness and individual structures
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Initializing a random individual from the allowed values
    def init_individual():
        return np.array([random.choice(value_vector) for _ in range(n_vars)])

    # Registering  GA components
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, init_individual
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Mutating individual values with fixed probability
    def mutate_individual(individual):
        for i in range(len(individual)):
            if random.random() < 0.2:
                individual[i] = random.choice(value_vector)

    toolbox.register("mutate", mutate_individual)

    # Evaluating individual's fitness using the ENPV function
    def evaluate(individual):
        gene_array = np.array(individual)

        # Reshapeing the flat genes into (Fth, Mix) matrices
        delta_K_Jet_Mix = gene_array[: Fth * mix_dim].reshape(Fth, mix_dim)
        delta_K_LH_Mix = gene_array[Fth * mix_dim :].reshape(Fth, mix_dim)

        # Summing over mix axis → shape (Fth,)
        delta_K_Jet = np.sum(delta_K_Jet_Mix, axis=1)
        delta_K_LH = np.sum(delta_K_LH_Mix, axis=1)
        # delta_K = delta_K_Jet + delta_K_LH

        # Broadcasting to (No_Forecasts, Fth, Mix) for the ENPV calculation
        delta_K_Jet_full = np.broadcast_to(delta_K_Jet_Mix, d_ATM.shape)
        delta_K_LH_full = np.broadcast_to(delta_K_LH_Mix, d_ATM.shape)

        enpv = ENPV_calculation(
            Param, delta_K_Jet_full, delta_K_LH_full, d_ATM, S_values, PAX_yearly
        )
        return (enpv,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initializing the population
    population = toolbox.population(n=Param["population"])
    cxpb, mutpb, ngen = 0.5, 0.2, 10  # Crossover, mutation, generations

    # Running the GA for ngen generations
    for gen in range(ngen):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Applying the crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Applying mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluating the invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        population[:] = offspring  # Replacing the population

    # Extracting the best solution
    best_ind = tools.selBest(population, 1)[0]
    gene_array = np.array(best_ind)

    # Reshaping the best solution into (Fth, Mix)
    delta_K_Jet_Mix = gene_array[: Fth * mix_dim].reshape(Fth, mix_dim)
    delta_K_LH_Mix = gene_array[Fth * mix_dim :].reshape(Fth, mix_dim)

    # Expanding shape to (No_Forecasts, Fth, Mix)
    No_Forecasts = Param["No_Forecasts"]
    delta_K_Jet_Mix_full = np.broadcast_to(
        delta_K_Jet_Mix, (No_Forecasts, Fth, mix_dim)
    )
    delta_K_LH_Mix_full = np.broadcast_to(delta_K_LH_Mix, (No_Forecasts, Fth, mix_dim))

    return delta_K_Jet_Mix_full, delta_K_LH_Mix_full


def Decision_Rule(Param, K0, d, theta, condition):
    """
    This function creates a new delta capacity vector while considering a decision rule.
    The parameters of the decision rule are given and optimized in different functions.

    Parameters:
        Param (dict): Parameter Dictionary
        K0 (int): Initial Capacity
        D (ndarray): Demand Matrix (2D or 3D)
        theta (ndarray or int): Capacity Increase Value
        condition (int): Condition for Capacity Increase (difference of K and D)

    Returns:
        delta_K_Flex (ndarray): Delta Capacity Matrix Considering the Decision Rule

    To call the function use following syntax:
        Decision_Rule(Param, K0, d, theta, condition)
    """
    # If condition to adjust delta_K_LH shape
    if d.ndim == 2:
        # 2D Case: (scenarios, time)
        K_Flex = np.full(d.shape, K0, dtype=d.dtype)

        # Looping over Fth
        for t in range(1, d.shape[1]):
            # Creating a difference matrix with over and under capacity conditions
            diff = K_Flex[:, t - 1] - d[:, t]
            over_capacity = np.greater_equal(diff, condition).astype(int)
            under_capacity = np.less(diff, condition).astype(int)
            # Combining the two condition matrices
            K_Flex[:, t] = over_capacity * K_Flex[:, t - 1] + under_capacity * (
                K_Flex[:, t - 1] + theta
            )

        delta_K = np.diff(K_Flex - K0, axis=1)
        delta_K_Flex = np.insert(delta_K, 0, 0, axis=1)

    # If condition to adjust delta_K_LH shape
    elif d.ndim == 3:
        # 3D Case: (scenarios, time, mixes)
        K0_mix = Param["Mix"] * K0
        K_Flex = np.full(d.shape, K0_mix, dtype=d.dtype)

        for mix in range(d.shape[2]):
            for t in range(1, d.shape[1]):
                diff = K_Flex[:, t - 1, mix] - d[:, t, mix]
                over_capacity = np.greater_equal(diff, condition).astype(int)
                under_capacity = np.less(diff, condition).astype(int)
                K_Flex[:, t, mix] = over_capacity * K_Flex[
                    :, t - 1, mix
                ] + under_capacity * (K_Flex[:, t - 1, mix] + theta)

        delta_K = np.diff(K_Flex - K0, axis=1)
        delta_K_Flex = np.insert(delta_K, 0, 0, axis=1)

    else:
        raise ValueError("Input demand matrix D must be either 2D or 3D.")

    return delta_K_Flex


def Parameter_combinations(Param, n=1000, Apply_Mix=False):
    """
    This function generates all possible combinations of parameters for the optimization process from the given ranges.

    Parameters:
        Param (dict): Parameter Dictionary
        n (int): Number of Random Samples to Return
        Apply_Mix (bool): If True, Applies Mix to the Parameter Combinations

    Returns:
        sampled_combinations (list): List of Sampled Parameter Combinations

    To call the function use following syntax:
        Parameter_combinations(Param, n=1000, Apply_Mix=False)
    """
    # Theta -> Capacity Increase Value
    lower_theta = Param["lower_theta"]
    upper_theta = Param["upper_theta"]
    stepsize_theta = Param["stepsize_theta"]

    # Condition -> Capacity Increase Condition (K - D)
    lower_cond = Param["lower_condition"]
    upper_cond = Param["upper_condition"]
    stepsize_cond = Param["stepsize_condition"]

    #  Defining integer ranges for each variable (inclusive)
    theta_Jet = np.arange(lower_theta, upper_theta + stepsize_theta, stepsize_theta)
    condition_Jet = np.arange(lower_cond, upper_cond + stepsize_cond, stepsize_cond)
    # Copying the ranges for the LH2 case
    theta_LH = theta_Jet
    condition_LH = condition_Jet

    if Apply_Mix == False:
        # Generating all possible combinations
        all_combinations = list(
            itertools.product(
                theta_Jet,
                condition_Jet,
                theta_LH,
                condition_LH,
            )
        )

    elif Apply_Mix == True:
        theta_Jet_short = theta_Jet
        theta_Jet_medium = theta_Jet
        theta_Jet_long = theta_Jet
        theta_LH_short = theta_LH
        theta_LH_medium = theta_LH
        theta_LH_long = theta_LH

        all_combinations = list(
            itertools.product(
                theta_Jet_short,
                theta_Jet_medium,
                theta_Jet_long,
                condition_Jet,
                theta_LH_short,
                theta_LH_medium,
                theta_LH_long,
                condition_LH,
            )
        )

    sample_size = min(n, len(all_combinations))
    sampled_combinations = random.sample(all_combinations, sample_size)
    return sampled_combinations


def Parameter_Evaluation(
    Param, d_ATM_Jet, d_ATM_LH, d_ATM_mix, S_values, PAX_yearly, n=1000, Apply_Mix=False
):
    """
    This function evaluates the expected net present value (ENPV) for different parameter
    combinations for the decision rule function and returns the maximum ENPV along with the best parameter combination.

    Parameters:
        Param (dict): Parameter Dictionary
        d_ATM_Jet (np.array): Jet A1 ATMs in the DHL
        d_ATM_LH (np.array): LH2 ATMs in the DHL
        d_ATM_mix (np.array): Total ATMs with Mix in the DHL
        S_values (np.array): S-Curve Matrix for Technology Adoption
        PAX_yearly (np.array): Number of Passengers per Year
        n (int): Number of Samples to Return
        Apply_Mix (bool): If True, Applies Mix to the Parameter Combinations

    Returns:
        max_enpv (float): Maximum ENPV Value
        best_params (tuple): Best Parameter Combination that Yields the Maximum ENPV

    To call the function use following syntax:
        Parameter_Evaluation(Param, d_ATM_Jet, d_ATM_LH, d_ATM_mix, S_values, PAX_yearly, n, Apply_Mix)
    """
    # Parameters
    max_enpv = -np.inf
    best_params = None
    Mix = Param["Mix"]
    K0 = Param["K0"]
    K0_LH = Param["K0_LH"]
    Optimization_parameters = Parameter_combinations(Param, n, Apply_Mix)

    # If condition to check d_ATM_Jet shape
    if Apply_Mix == False:
        for sample in Optimization_parameters:
            # Applying the decision rule to calculate delta_K_Jet and delta_K_LH
            delta_K_Jet = Decision_Rule(Param, K0, d_ATM_Jet, sample[0], sample[1])
            delta_K_LH = Decision_Rule(Param, K0_LH, d_ATM_LH, sample[2], sample[3])
            # Calculating the ENPV for the current parameter combination
            ENPV = ENPV_calculation(
                Param,
                delta_K_Jet,
                delta_K_LH,
                d_ATM_mix,
                S_values,
                PAX_yearly,
            )
            # Updating the maximum ENPV and best parameters if the current ENPV is greater
            if ENPV > max_enpv:
                max_enpv = ENPV
                best_params = sample

    # If condition to check d_ATM_Jet shape
    if Apply_Mix == True:
        for sample in Optimization_parameters:
            delta_K_Jet_short = Decision_Rule(
                Param, K0 * Mix[0], d_ATM_Jet[:, :, 0], sample[0], sample[3]
            )
            delta_K_Jet_medium = Decision_Rule(
                Param, K0 * Mix[1], d_ATM_Jet[:, :, 1], sample[1], sample[3]
            )
            delta_K_Jet_long = Decision_Rule(
                Param, K0 * Mix[2], d_ATM_Jet[:, :, 2], sample[2], sample[3]
            )
            delta_K_Jet = np.stack(
                [delta_K_Jet_short, delta_K_Jet_medium, delta_K_Jet_long], axis=2
            )
            delta_K_LH_short = Decision_Rule(
                Param, K0_LH * Mix[0], d_ATM_LH[:, :, 0], sample[4], sample[7]
            )
            delta_K_LH_medium = Decision_Rule(
                Param, K0_LH * Mix[1], d_ATM_LH[:, :, 1], sample[5], sample[7]
            )
            delta_K_LH_long = Decision_Rule(
                Param, K0_LH * Mix[2], d_ATM_LH[:, :, 2], sample[6], sample[7]
            )
            delta_K_LH = np.stack(
                [delta_K_LH_short, delta_K_LH_medium, delta_K_LH_long], axis=2
            )
            ENPV = ENPV_calculation(
                Param,
                delta_K_Jet,
                delta_K_LH,
                d_ATM_mix,
                S_values,
                PAX_yearly,
            )
            if ENPV > max_enpv:
                max_enpv = ENPV
                best_params = sample

    return max_enpv, best_params


# def CDF_Plot(Vector1, Vector2, label1="Vector1", label2="Vector2"):
#     """
#     This function is plotting the Cumulative Density Function of the NPVs

#     Parameters:
#         Vector1 (ndarray): Input Vector 1
#         Vector2 (ndarray): Input Vector 2
#         label1 (str): Label for the First CDF Curve
#         label2 (str): Label for the Second CDF Curve

#     Returns:
#         None: Plot of all Input Vectors in a CDF Graphic
#         + Visualisation of the 10th, 90th Percentile of the Input Vectors

#     To call this Function use following syntax:
#         CDF_Plot(Vector1, Vector2, label1, label2, label3, label4)
#     """
#     percentile_10a = np.percentile(Vector1, 10)
#     percentile_90a = np.percentile(Vector1, 90)
#     percentile_10b = np.percentile(Vector2, 10)
#     percentile_90b = np.percentile(Vector2, 90)

#     # Creating a subplot
#     fig, ax = plt.subplots()

#     # Step plot code with specific values
#     ax.plot(
#         np.sort(Vector1),
#         np.arange(1, len(Vector1) + 1) / float(len(Vector1)),
#         linestyle="-",
#         label=label1 + " CDF Curve",
#         linewidth=2,
#         color="green",
#         alpha=0.7,
#     )

#     ax.plot(
#         np.sort(Vector2),
#         np.arange(1, len(Vector2) + 1) / float(len(Vector2)),
#         linestyle="-",
#         label=label2 + " CDF Curve",
#         linewidth=2,
#         color="blue",
#         alpha=0.7,
#     )

#     mean1 = np.mean(Vector1)
#     Vector3 = np.full_like(Vector1, mean1)
#     ax.plot(
#         np.sort(Vector3),
#         np.arange(1, len(Vector3) + 1) / float(len(Vector3)),
#         linestyle="--",
#         label=label1 + " ENPV",
#         linewidth=2,
#         color="green",
#         alpha=0.7,
#     )
#     mean2 = np.mean(Vector2)
#     Vector4 = np.full_like(Vector2, mean2)
#     ax.plot(
#         np.sort(Vector4),
#         np.arange(1, len(Vector4) + 1) / float(len(Vector4)),
#         linestyle="-.",
#         label=label2 + " ENPV",
#         linewidth=2,
#         color="blue",
#         alpha=0.7,
#     )
#     ax.axhline(
#         0.9,
#         color="orange",
#         linestyle="--",
#         label="90th Percentile",
#     )

#     ax.axhline(
#         0.1,
#         color="red",
#         linestyle="-.",
#         label="10th Percentile",
#     )

#     # Adding crosshair at the specified points
#     ax.plot(percentile_90a, 0.9, marker="X", color="black", markersize=6)
#     ax.plot(percentile_10a, 0.1, marker="X", color="black", markersize=6)
#     ax.plot(percentile_90b, 0.9, marker="X", color="black", markersize=6)
#     ax.plot(percentile_10b, 0.1, marker="X", color="black", markersize=6)


#     ax.grid(True)
#     ax.set_title("Cumulative Distribution Function (CDF)")
#     ax.set_xlabel("NPVs")
#     ax.set_ylabel("Cumulative Probability [%]")
#     ax.legend()
#     plt.show()
#     print(f"10th Percentile {label1}: {percentile_10a}")
#     print(f"90th Percentile {label1}: {percentile_90a}")
#     print(f"10th Percentile {label2}: {percentile_10b}")
#     print(f"90th Percentile {label2}: {percentile_90b}")
#     print(f"{label1}: {mean1}")
#     print(f"{label2}: {mean2}")
#     return


def CDF_Plot(
    Vector1,
    Vector2,
    label1="Vector1",
    label2="Vector2",
    save_plot=False,
    run_name="Unnamed_Run",
    output_base_folder="Plots",
):
    """
    This function is plotting the Cumulative Density Function of the NPVs

    Parameters:
        Vector1 (ndarray): Input Vector 1
        Vector2 (ndarray): Input Vector 2
        label1 (str): Label for the First CDF Curve
        label2 (str): Label for the Second CDF Curve
        save_plot (bool): If True, saves the plot to a file
        run_name (str): Name of the run for saving the plot
        output_base_folder (str): Base folder to save the plot

    Returns:
        None: Plot of all Input Vectors in a CDF Graphic
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

    # Adding crosshair at the specified points
    ax.plot(percentile_90a, 0.9, marker="X", color="black", markersize=6)
    ax.plot(percentile_10a, 0.1, marker="X", color="black", markersize=6)
    ax.plot(percentile_90b, 0.9, marker="X", color="black", markersize=6)
    ax.plot(percentile_10b, 0.1, marker="X", color="black", markersize=6)

    ax.grid(True)
    ax.set_title("Cumulative Distribution Function (CDF)")
    ax.set_xlabel("NPVs")
    ax.set_ylabel("Cumulative Probability [%]")
    ax.legend()

    # SAVE the plot if requested
    if save_plot and run_name is not None:
        import os

        folder_path = os.path.join(output_base_folder, run_name)
        os.makedirs(folder_path, exist_ok=True)

        filename_safe_title = "CDF_Plot"
        base_filename = f"{filename_safe_title}.png"
        plot_path = os.path.join(folder_path, base_filename)

        counter = 1
        while os.path.exists(plot_path):
            base_filename = f"{filename_safe_title}_{counter}.png"
            plot_path = os.path.join(folder_path, base_filename)
            counter += 1

        plt.savefig(plot_path)
        print(f"CDF plot saved to: {os.path.abspath(plot_path)}")

    plt.show()
    print(f"10th Percentile {label1}: {percentile_10a}")
    print(f"90th Percentile {label1}: {percentile_90a}")
    print(f"10th Percentile {label2}: {percentile_10b}")
    print(f"90th Percentile {label2}: {percentile_90b}")
    print(f"{label1}: {mean1}")
    print(f"{label2}: {mean2}")
    return
