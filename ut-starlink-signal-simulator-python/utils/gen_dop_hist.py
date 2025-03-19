from scipy.constants import speed_of_light

def gen_dop_hist(s):
    NMAX = 32000000

    c = speed_of_light / 1e3  # Convert to km/s

    lamda = c / s["fc"]

    n_samples = s["fs"] * s["tdur"]