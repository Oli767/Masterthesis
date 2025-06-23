print("Hello")


# Old Functions
# def generate_exponential_vector(Param):
#     """
#     Generate a vector following an exponential distribution with occasional negative steps.

#     Parameters:
#         Param (dict): Dictionary containing:
#             - "Fth" (int): Length of the output vector
#             - "dt" (float): Average step size for positive movement
#             - "p_down" (float): Probability of a step being negative
#             - "scale_down" (float): Scale for negative steps (should be much smaller than "dt")

#     Returns:
#         np.ndarray: The generated vector

#     To call the function use following syntax:
#         generate_exponential_vector(Param)
#     """
#     Fth = Param["Fth"]
#     max_value = Fth
#     Average_step = Param["dt"]
#     # p_down = Param["p_down"]
#     p_down = Param.get("p_down", 0.1)  # Probability of a downward step
#     scale_down = Param.get("scale_down", 0.5)  # Scale for downward steps (small)

#     # Generate positive steps (exponential)
#     pos_steps = np.random.exponential(scale=Average_step, size=Fth - 1)

#     # Generate negative steps (exponential, but much smaller)
#     neg_steps = np.random.exponential(scale=scale_down, size=Fth - 1)

#     # Create a mask for when to apply negative steps
#     is_negative = np.random.rand(Fth - 1) < p_down  # Bernoulli distribution

#     # Apply negative steps where the mask is True
#     steps = np.where(is_negative, -neg_steps, pos_steps)

#     # Cumulative sum to create the vector
#     vector = np.cumsum(np.insert(steps, 0, 0))

#     # Ensure values stay within bounds [0, max_value]
#     vector = np.clip(vector, 0, max_value)

#     return vector


# def plot_s_curve_with_random_points(Param, L=100, k=0.3):
#     """
#     Plots an S-Curve (logistic function) and returns all hit curve values.

#     Parameters:
#         L (float): Carrying capacity (max value).
#         k (float): Growth rate.
#         t0 (float): Midpoint (where growth is fastest).
#         Fth (int): Length of the generated vector.

#     Returns:
#         np.ndarray: All values of the curve that have been hit.

#     To call the function use following syntax:
#         plot_s_curve_with_random_points(Param, L=100, k=0.3)
#     """
#     t_max = Param["Fth"]
#     t0 = 0.5 * t_max  # Ensure curve aligns with Fth

#     # Time range (continuous)
#     time = Param["time"]

#     # Get the vector for development
#     vector = generate_exponential_vector(Param)
#     S_values = L / (1 + np.exp(-k * (vector - t0)))  # Compute corresponding S values

#     # Plot function
#     S = L / (1 + np.exp(-k * (time - t0)))
#     # Plot S-Curve
#     plt.plot(time, S, label="S-Curve", color="blue")
#     plt.scatter(vector, S_values, color="red", label="Hit Points", zorder=3)

#     # Labels and grid
#     plt.xlabel("Time [yrs]")
#     plt.ylabel("Percentage [%]")
#     plt.legend()
#     plt.grid()
#     plt.show()
#     print(vector)

#     return S_values

# # Example usage
# hit_values = plot_s_curve_with_random_points(Param)
# print("Hit values:", hit_values)
