from scipy.signal import savgol_filter

def smoothen_data(data):
    return savgol_filter(data, 51, 3)