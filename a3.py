"""
ENGG1001 Assignment 3
Semester 1, 2021
"""

__author__ = "William Sawyer 46963608"
__email__ = "w.sawyer@uqconnect.edu.au"
__date__ = "Friday 21 May, 2021"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

F_PHC = 0.8         # Peak-hour capacity factor.
TRAF_GROWTH = 0.1   # Projected growth rate of traffic.
START_YR = 2021     # First year of projection.
END_YR = 2030       # Last year of projection.
INIT_TRAF = 6500    # Initial volume of traffic entering road network.


class Link:
    """ A segment of the road network.

    Attributes
    ----------
    number : int
        The index of the segment within the imported data. Used for
        indexing matrices.
    start : int
        The index of the location which is at the beginning of the
        segment, indicated by 'From' in the imported data.
    end : int
        The index of the location which is at the end of the segment,
        indicated by 'To' in the imported data.
    """
    def __init__(self, network_df, locations, segment):
        """
        Parameters
        ----------
        network_df : pd.DataFrame
            Imported data about the road segments.
        locations : np.ndarray
            Locations in the road network.
        segment : str
            Name of segment used to create the Link.
        """

        self.number = list(network_df.index).index(segment)
        self.start = list(locations).index(network_df.loc[segment, 'From'])
        self.end = list(locations).index(network_df.loc[segment, 'To'])


def network_import():
    """ Import data from CSV file to a DataFrame. Alter index of the
    DataFrame. Add columns for Capacity and Conductance of each road segment.
    Export altered DataFrame to a CSV file. Return DataFrame and array of
    locations.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        network_df - Imported data about the road segments.
        locations - Locations in the road network.
    """

    network_df = pd.read_csv('road_network_data.csv').set_index('Segment')
    network_df['Capacity (veh/hr)'] = (network_df['Lanes (-)'] / 2) / F_PHC * (
                                    1000 + 12 * network_df['Speed (km/hr)'])
    network_df['Conductivity (-)'] = network_df['Speed (km/hr)'] * (network_df[
                                    'Lanes (-)'] / 2) / 80
    locations = network_df['From'].append(network_df['To']).unique()
    network_df.to_csv(r'network_df.csv')
    return network_df, locations


def create_adj(network_df, locations):
    """ Return an adjacency matrix to represent connections between locations
    in the road network.

    Parameters
    ----------
    network_df : pd.DataFrame
        Imported data about the road segments.
    locations : np.ndarray
        Locations in the road network.

    Returns
    -------
    adj_arr : np.ndarray
        Adjacency matrix representing the road network.
    """

    adj_arr = np.zeros((len(locations), len(locations)))
    for segment in network_df.index:
        link = Link(network_df, locations, segment)
        adj_arr[link.start, link.end] = 1
        adj_arr[link.end, link.start] = 1
    return adj_arr


def create_inc(network_df, locations):
    """ Return an incidence matrix to represent information about the
    segments of the road network in a way which can be utilised mathematically.

    Parameters
    ----------
    network_df : pd.DataFrame
        Imported data about the road segments.
    locations : np.ndarray
        Locations in the road network.

    Returns
    -------
    inc_arr : np.ndarray
        Incidence matrix representing the road network.
    """

    inc_arr = np.zeros((len(network_df), len(locations)))
    for segment in network_df.index:
        link = Link(network_df, locations, segment)
        inc_arr[link.number, link.start] = -1
        inc_arr[link.number, link.end] = 1
    return inc_arr


def create_cond(network_df):
    """ Return a conductance matrix to represent the conductivity of road
    segments within the road network in a way which can be utilised
    mathematically.

    Parameters
    ----------
    network_df : pd.DataFrame
        Imported data about the road segments.

    Returns
    -------
    cond_arr : np.ndarray
        Conductance matrix representing the road network.
    """

    cond_arr = np.diag(network_df['Conductivity (-)'])
    return cond_arr


def calc_flows(entering_traffic, network_df, cond_arr, inc_arr):
    """ Calculate the traffic potentials at each location and the traffic
    volumes and volume capacity ratios of each road segment.

    Parameters
    ----------
    entering_traffic : int
        Hourly traffic volume entering the road network.
    network_df : pd.DataFrame
        Imported data about the road segments.
    cond_arr : np.ndarray
        Conductance matrix representing the road network.
    inc_arr : np.ndarray
        Incidence matrix representing the road network.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        potentials - Traffic potentials at each location within the ro
        network.
        flows - Hourly traffic volumes flowing through each segment of the road
        network.
        vcrs - Volume-to-capacity ratios of each segment of the road network.
    """

    ext_flows = np.zeros(len(inc_arr.T))
    ext_flows[0], ext_flows[-1] = entering_traffic, -entering_traffic
    potentials = np.zeros_like(ext_flows)
    potentials[0] = 0
    potentials[1:] = np.linalg.solve((inc_arr.T @ cond_arr @ inc_arr)[1:, 1:],
                                     ext_flows[1:])
    flows = -1 * cond_arr @ inc_arr @ potentials
    vcrs = np.abs(flows / network_df['Capacity (veh/hr)'])
    return potentials, flows, vcrs


def traffic_proj(first_yr, last_yr, first_traf, growth):
    """ Project future hourly volumes of traffic entering the road network.
    Plot volume of entering traffic against year. Return years and their
    projected traffic volumes

    Parameters
    ----------
    first_yr : int
        Year in which traffic projection begins.
    last_yr : int
        Year in which traffic projection ends.
    first_traf : int
        Initial volume of traffic using the road network.
    growth : float
        Rate at which the volume of traffic grows.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        years - List of years between given first and last years.
        traffic - Projected hourly traffic volumes entering the road network
        each year.
    """

    years = np.arange(first_yr, last_yr + 1)
    traffic = first_traf * (1 + growth) ** (years - first_yr)

    plt.plot(years, traffic, color='black', marker='o')
    plt.title('Projected volume of traffic using the road network')
    plt.xlabel('Year')
    plt.ylabel('Projected traffic volumes (PCEVs per hour)')
    plt.savefig('projected_traffic_volumes.png')
    plt.show()
    return years, traffic


def calc_yearsflow(years, traffic, network_df, cond_arr, inc_arr):
    """ Calculate the traffic potentials at each node and the traffic volumes
    and volume capacity ratios of each segment over a number of years.

    Parameters
    ----------
    years : np.ndarray
        List of years between given first and last years.
    traffic : np.ndarray
        Projected hourly traffic volumes entering the road network each year.
    network_df : pd.DataFrame
        Imported data about the road segments.
    cond_arr : np.ndarray
        Conductance matrix representing the road network.
    inc_arr : np.ndarray
        Incidence matrix representing the road network.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        potentials - Traffic potentials at each location within the ro
        network.
        flows - Hourly traffic volumes flowing through each segment of the road
        network.
        vcrs - Volume-to-capacity ratios of each segment of the road network.
    """

    potentials = np.zeros((len(years), len(inc_arr.T)))
    flows = np.zeros((len(years), len(network_df)))
    vcrs = np.zeros((len(years), len(network_df)))
    for i in range(len(years)):
        potentials[i, :], flows[i, :], vcrs[i, :] = calc_flows(traffic[i],
                                                network_df, cond_arr, inc_arr)
    return potentials, flows, vcrs


def main():
    """ Call other functions to create data to represent the road network.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]
        adj_arr - Adjacency matrix representing the road network.
        inc_arr - Incidence matrix representing the road network.
        traffic - Hourly volume of traffic projected to enter the road network
        each year.
        cond_arr - Conductance matrix representing the road network.
        potentials - Traffic potentials at each location of the road network.
        flows - Hourly traffic volumes flowing through each segment of the
        road network.
        vcrs - Volume-to-capacity ratios in each segment of the road network.
    """

    network_df, locations = network_import()
    adj_arr = create_adj(network_df, locations)
    inc_arr = create_inc(network_df, locations)
    cond_arr = create_cond(network_df)
    years, traffic = traffic_proj(START_YR, END_YR, INIT_TRAF, TRAF_GROWTH)
    potentials, flows, vcrs = calc_yearsflow(years, traffic, network_df,
                                             cond_arr, inc_arr)

    return adj_arr, inc_arr, traffic, cond_arr, potentials, flows, vcrs


if __name__ == "__main__":
    adj_arr, inc_arr, traffic, cond_arr, potentials, flows, vcrs = main()

# Year first segment to exceed capacity: 2023

# Minimum additional lanes: 4

# Next problem segment and year of failure: K 2028
