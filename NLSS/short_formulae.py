import numpy as np

def find_flight_path_angle(eccentricity : float , true_anomaly : float) -> float:
    """Calculating flight path angle or angle from perpendicular tangent

    Args:
        eccentricity (float)
        true_anomaly (float)

    Returns:
        gamma (float): flight path angle in radians
    """
    tangam = (eccentricity * np.sin(true_anomaly))/(1 + eccentricity * np.cos(true_anomaly))
    gamma = np.arctan(tangam)
    return gamma


def utc_to_j0(year : int, month : int, day : int) -> float:
    """Converts a UTC datetime to J0 (midnight) of the corresponding
    Julian date.

    Args:
        utc (dt.datetime): UTC date

    Returns:
        float: J0 Julian date
    """

    j0 = 367 * year - np.floor((7 * (year + np.floor((month + 9) / 12))) / 4) \
        + np.floor((275 * month) / 9) + day + 1721013.5    # TODO: Equation (5.48)
    return j0


def utc_to_jd(year : int, month : int, day : int, 
              hour: int =0, minute: int =0, second: int =0) -> float:
    """
    Convert UTC date and time to Julian Date.   
    """

    j0 = utc_to_j0(year, month, day)
    UT = (hour + minute / 60 + second / 3600)
    jd = j0 + UT/24

    return jd

def seconds_to_julian_date(seconds_since_epoch: float, 
                           epoch_jd: float) -> float:
    """
    Convert seconds since an epoch to Julian date.
    
    Args:
        seconds_since_epoch: Time in seconds
        epoch_jd: Julian date of epoch (days since J2000)
    
    Returns:
        Julian date (days since J2000)
    """
    days_elapsed = seconds_since_epoch / 86400.0  # seconds to days
    return epoch_jd + days_elapsed