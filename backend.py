import numpy as np
from dataclasses import dataclass
import pandas as pd
from climate_classification import (
    KoppenClassification,
    TrewarthaClassification,
    DLClassification,
)
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import h5py
from TorchModel import DLModel
from numba import njit
from functools import lru_cache
from matplotlib.colors import to_rgb


LATEST_YEAR = 2024
TEMP_RANGE = [
    [-25, 40],  # °C
    [-8, 96],   # °F
]  
TEMP_TICKVALS = [
    [-20, -10, 0, 10, 20, 30, 40],  # °C
    [0, 16, 32, 48, 64, 80, 96],   # °F
]  
PREC_RANGE = [
    [0, 390],  # mm
    [0, 13],   # inch
]  
PREC_TICKVALS = [
    [30, 90, 150, 210, 270, 330, 390],  # mm
    [1, 3, 5, 7, 9, 11, 13],            # inch
]  
VARIABLE_TYPE_INDICES = {
    "Coldest Month Mean Temperature": 0,
    "Hottest Month Mean Temperature": 1,
    "Coldest Month Mean Daily Minimum": 2,
    "Hottest Month Mean Daily Maximum": 3,
    "Annual Mean Temperature": 4,
    "Wettest Month Precipitation": 5,
    "Driest Month Precipitation": 6,
    "Annual Total Precipitation": 7,
    "Thermal Index": 8,
    "Aridity Index": 9,
}


@njit
def celcius_to_fahrenheit(temp: np.ndarray) -> np.ndarray:
    return temp * np.float32(9 / 5) + np.float32(32)


@njit
def mm_to_inch(prec: np.ndarray) -> np.ndarray:
    return prec / np.float32(25.4)


def lighten(color, amount):
    c = np.array(to_rgb(color))
    return tuple(1 - amount * (1 - c))


@njit
def calc_trend_theil_sen(x: np.ndarray, y: np.ndarray) -> np.float32:
    """use Theil-Sen estimator to calculate the trend"""
    n = x.shape[0]
    slopes = np.zeros(n * (n - 1) // 2, dtype=np.float32)
    k = 0

    # calculate all possible slopes
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[j] - x[i]) > 1e-10:
                slopes[k] = (y[j] - y[i]) / (x[j] - x[i])
                k += 1

    # return the median slope
    return np.float32(np.median(slopes[:k]))


@njit
def process_variables_trend(
    data: np.ndarray, x: np.ndarray, map_type_idx: int, convert_unit: bool = False
) -> np.ndarray:
    """batch process climate data and calculate the trend"""
    n_points = data.shape[0]
    values = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        y = data[i, :, map_type_idx]
        if convert_unit:
            if map_type_idx in [0, 1, 2, 3, 4]:
                y = celcius_to_fahrenheit(y)
            elif map_type_idx in [5, 6, 7]:
                y = mm_to_inch(y)
        values[i] = calc_trend_theil_sen(x, y)

    return values


@njit
def calc_variables(data: np.ndarray) -> np.ndarray:
    """calculate all climate indicators at once"""
    n_samples = len(data)
    # pre-allocate the result array
    res = np.zeros((n_samples, 8), dtype=np.float32)

    for i in range(n_samples):
        temp = data[i, 0, :]  # temperature data
        min_temp = data[i, 1, :]  # minimum temperature data
        max_temp = data[i, 2, :]  # maximum temperature data
        precip = data[i, 3, :]  # precipitation data

        res[i, 0] = np.min(temp)  # 'Coldest Month Mean Temperature'
        res[i, 1] = np.max(temp)  # 'Hottest Month Mean Temperature'
        res[i, 2] = np.min(min_temp)  # 'Coldest Month Mean Daily Minimum'
        res[i, 3] = np.max(max_temp)  # 'Hottest Month Mean Daily Maximum'
        res[i, 4] = np.mean(temp)  # 'Mean Annual Temperature'
        res[i, 5] = np.max(precip)  # 'Wettest Month Precipitation'
        res[i, 6] = np.min(precip)  # 'Driest Month Precipitation'
        res[i, 7] = np.sum(precip)  # 'Total Annual Precipitation'

    return res


# only used for change rate chart
def calc_change_rate(
    variable_file: h5py.File,
    indices: np.ndarray,
    elev: np.ndarray,
    variable_type: str,
    yr: int,
    yrs: int,
    unit: bool = False,
) -> pd.DataFrame:
    x = np.array([i for i in range(yr, yr + yrs)], dtype=np.float32)
    data = variable_file.get("res")[:-1, yr: yr + yrs, :]

    values = process_variables_trend(
        data, x, VARIABLE_TYPE_INDICES[variable_type], unit
    )
    lats, lons = zip(*indices)

    return pd.DataFrame({"lat": lats, "lon": lons, "value": values, "elev": elev})


@dataclass
class ClimateData:
    __slots__ = ["tmp", "pre", "pet", "tmn", "tmx", "elev"]

    tmp: np.ndarray
    pre: np.ndarray
    pet: np.ndarray
    tmn: np.ndarray
    tmx: np.ndarray
    elev: float

    def get_dl_data(self) -> np.ndarray:
        return np.vstack([self.tmn, self.pre, self.tmx])
    
    def get_classic_data(self) -> np.ndarray:
        return np.vstack([self.tmp, self.pre])
    
    def get_variable_data(self) -> np.ndarray:
        return np.vstack([self.tmp, self.tmn, self.tmx, self.pre])
    
    def get_all_data(self) -> np.ndarray:
        return np.vstack([self.tmp, self.tmn, self.tmx, self.pre, self.pet])
    
    def get_elev(self) -> float:
        return self.elev


@dataclass
class ClimateDataset:
    __slots__ = ["data", "probabilities", "thermal_index", "aridity_index", "variables"]

    data: dict[tuple[float, float], ClimateData]
    probabilities: np.ndarray     
    thermal_index: np.ndarray     
    aridity_index: np.ndarray     
    variables: np.ndarray

    def __getitem__(self, idx: tuple[float, float]) -> ClimateData:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def prepare_dl(self, dl_network: DLModel, indices: np.ndarray) -> None:
        batch_size = 4096
        n_samples = len(self)

        self.aridity_index = np.zeros((n_samples,), dtype=np.float32)
        self.thermal_index = np.zeros((n_samples,), dtype=np.float32)
        self.probabilities = np.zeros((n_samples, 26), dtype=np.float32)
        for i in range(0, n_samples, batch_size):
            batch = np.array(
                [
                    self.data[(indices[j, 0], indices[j, 1])].get_dl_data()
                    for j in range(i, i + batch_size)
                    if j < n_samples
                ]
            )
            (
                self.thermal_index[i: i + batch_size],
                self.aridity_index[i: i + batch_size],
                self.probabilities[i: i + batch_size, :],
            ) = dl_network(batch)

    def prepare_variables(self) -> None:
        self.variables = calc_variables(
            np.array([v.get_variable_data() for v in self.data.values()])
        )

    def get_koppen(self, cd_threshold: float, kh_mode: str) -> list[str]:
        return [
            KoppenClassification.classify(v.get_classic_data(), cd_threshold, kh_mode)
            for v in self.data.values()
        ]

    def get_trewartha(self) -> list[str]:
        return [
            TrewarthaClassification.classify(v.get_classic_data()) for v in self.data.values()
        ]

    def get_dl(self) -> list[str]:
        return DLClassification.classify(self.probabilities)

    def get_thermal_index(self) -> np.ndarray:
        return self.thermal_index

    def get_aridity_index(self) -> np.ndarray:
        return self.aridity_index

    def get_coldest_month_tmp(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 0]
        return celcius_to_fahrenheit(res) if unit else res

    def get_hottest_month_tmp(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 1]
        return celcius_to_fahrenheit(res) if unit else res

    def get_coldest_month_daily_min(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 2]
        return celcius_to_fahrenheit(res) if unit else res
    
    def get_hottest_month_daily_max(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 3]
        return celcius_to_fahrenheit(res) if unit else res
    
    def get_mean_temp(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 4]
        return celcius_to_fahrenheit(res) if unit else res

    def get_wettest_month_pre(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 5]
        return mm_to_inch(res) if unit else res

    def get_driest_month_pre(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 6]
        return mm_to_inch(res) if unit else res

    def get_total_pr(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 7]
        return mm_to_inch(res) if unit else res

    def prepare_map_data(
        self,
        map_type: str,
        koppen_cd_mode: str = "",
        koppen_kh_mode: str = "",
        unit: bool = False,
        class_name: str = "Af",
    ) -> pd.DataFrame:
        values = []

        if map_type == "Köppen-Geiger Classification":
            cd_threshold = -3 if koppen_cd_mode == "-3&deg;C" else 0
            kh_criterion = (
                "mean_temp"
                if koppen_kh_mode == "annual mean temp 18&deg;C"
                else "coldest_month"
            )
            values = self.get_koppen(cd_threshold, kh_criterion)
        elif map_type == "Trewartha Classification":
            values = self.get_trewartha()
        elif map_type == "DeepEcoClimate":
            values = self.get_dl()
        # elif map_type == "Dm probability":
        #     # Find index of 'Dm' in DLClassification.class_map
        #     from climate_classification import DLClassification
        #     dm_idx = DLClassification.class_map.index('Dm')
        #     # self.probabilities shape: (n, 26)
        #     probs = self.probabilities[:, dm_idx]
        #     # Only keep points with prob > 0.01
        #     lats, lons = zip(*self.data.keys())
        #     elevs = [v.elev for v in self.data.values()]
        #     df = pd.DataFrame({
        #         'lat': lats,
        #         'lon': lons,
        #         'value': probs,
        #         'elev': elevs,
        #     })
        #     df = df[df['value'] > 0.01].reset_index(drop=True)
        #     return df
        elif map_type == "DeepEcoClimate Class Probability":
            class_idx = DLClassification.class_map.index(class_name)
            probs = self.probabilities[:, class_idx]
            lats, lons = zip(*self.data.keys())
            elevs = [v.elev for v in self.data.values()]
            df = pd.DataFrame({
                'lat': lats,
                'lon': lons,
                'value': probs,
                'elev': elevs,
            })
            df = df[df['value'] >= 0.02].reset_index(drop=True)
            return df
        elif map_type == "Annual Mean Temperature":
            values = self.get_mean_temp(unit)
        elif map_type == "Annual Total Precipitation":
            values = self.get_total_pr(unit)
        elif map_type == "Aridity Index":
            values = self.get_aridity_index()
        elif map_type == "Aridity Index (Discretized)":
            values = discretize_aridity_index(self.get_aridity_index())
        elif map_type == "Thermal Index":
            values = self.get_thermal_index()
        elif map_type == "Thermal Index (Discretized)":
            values = discretize_thermal_index(self.get_thermal_index())
        elif map_type == "Coldest Month Mean Temperature":
            values = self.get_coldest_month_tmp(unit)
        elif map_type == "Hottest Month Mean Temperature":
            values = self.get_hottest_month_tmp(unit)
        elif map_type == "Coldest Month Mean Daily Minimum":
            values = self.get_coldest_month_daily_min(unit)
        elif map_type == "Hottest Month Mean Daily Maximum":
            values = self.get_hottest_month_daily_max(unit)
        elif map_type == "Mean Annual Precipitation":
            values = self.get_total_pr(unit)
        elif map_type == "Driest Month Precipitation":
            values = self.get_driest_month_pre(unit)
        elif map_type == "Wettest Month Precipitation":
            values = self.get_wettest_month_pre(unit)

        lats, lons = zip(*self.data.keys())

        return pd.DataFrame(
            {
                "lat": lats,
                "lon": lons,
                "value": values,
                "elev": [v.elev for v in self.data.values()],
            }
        )


def get_average(
    yr: int, yrs: int, datafile: h5py.File, indices: np.ndarray, elev: np.ndarray
) -> ClimateDataset:
    """
    process climate data

    Args:
        yr: start year
        yrs: years span
        tmp: temperature data shape: (300, 720, years, 12)
        pre: precipitation data shape: (300, 720, years, 12)
        pet: potential evapotranspiration data shape: (300, 720, years, 12)
        tmn: minimum temperature data shape: (300, 720, years, 12)
        tmx: maximum temperature data shape: (300, 720, years, 12)
        elev: elevation data shape: (300, 720)

    Returns:
        ClimateDataset object, containing processed climate data
    """
    res = {}

    # pre-calculate the monthly average of all data
    tmp_mean = np.mean(datafile["tmp"][:, yr: yr + yrs, :], axis=1)
    pre_mean = np.mean(datafile["pre"][:, yr: yr + yrs, :], axis=1)
    pet_mean = np.mean(datafile["pet"][:, yr: yr + yrs, :], axis=1)
    tmn_mean = np.mean(datafile["tmn"][:, yr: yr + yrs, :], axis=1)
    tmx_mean = np.mean(datafile["tmx"][:, yr: yr + yrs, :], axis=1)

    for i in range(tmp_mean.shape[0]):

        res[(indices[i, 0], indices[i, 1])] = ClimateData(
            tmp=tmp_mean[i], pre=pre_mean[i], pet=pet_mean[i], tmn=tmn_mean[i], tmx=tmx_mean[i], elev=elev[i]
        )

    return ClimateDataset(
        data=res,
        probabilities=np.array([]),
        thermal_index=np.array([]),
        aridity_index=np.array([]),
        variables=np.array([]),
    )


class LocationService:
    def __init__(self, user_agent: str = "climate_viz"):
        self.geolocator = Nominatim(user_agent=user_agent)
        self.max_retries = 3
        self.retry_delay = 1

    @lru_cache(maxsize=1000)
    def get_location_info(self, location: tuple[float, float], local_lang: bool = False) -> str:
        """
        get the location information, with cache
        
        Args:
            location: location coordinates (lat, lon)
            local_lang: whether to use local language
            
        Returns:
            string containing location information
        """
        for attempt in range(self.max_retries):
            try:
                location_info = self.geolocator.reverse(
                    location, 
                    language="en" if not local_lang else False, 
                    zoom=10,
                )
                
                if location_info and location_info.raw.get("address"):
                    return location_info.address
                    
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except (GeocoderTimedOut, GeocoderUnavailable):
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                
        return "Unknown Location"

    @lru_cache(maxsize=1000)
    def search_location(self, query: str) -> tuple[float, float] | None:
        """
        search location and get the coordinates, with cache
        
        Args:
            query: location name or address
            
        Returns:
            coordinates (lat, lon) or None
        """
        for attempt in range(self.max_retries):
            try:
                location = self.geolocator.geocode(query)
                if location:
                    # round to the nearest 0.25 degree or 0.75 degree
                    lat = location.latitude
                    lon = location.longitude
                    # round to 0.5 degree first
                    lat_rounded = round(lat * 2) / 2
                    lon_rounded = round(lon * 2) / 2

                    lat_rounded = lat_rounded + 0.25 if lat >= lat_rounded else lat_rounded - 0.25
                    lon_rounded = lon_rounded + 0.25 if lon >= lon_rounded else lon_rounded - 0.25

                    return (lat_rounded, lon_rounded)
                    
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    
            except (GeocoderTimedOut, GeocoderUnavailable):
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                
        return None


def create_climate_chart(
    climate_data: ClimateData,
    title: str,
    subtitle: str,
    july_first: bool,
    unit: bool,
    auto_scale: bool,
) -> go.Figure:
    """
    create climate chart

    Args:
        climate_data: ClimateData object
        title: title
        subtitle: subtitle
        july_first: whether to start from July
        unit: whether to use °F/inch
        auto_scale: whether to auto scale
    Returns:
        plotly.graph_objects.Figure object
    """
    # prepare data
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    temp = climate_data.tmp  # monthly mean temperature
    prec = climate_data.pre  # monthly precipitation
    evap = climate_data.pet  # monthly evaporation
    tmax = climate_data.tmx  # daily maximum temperature
    tmin = climate_data.tmn  # daily minimum temperature

    # if start from July, adjust the data order
    if july_first:
        months = months[6:] + months[:6]
        temp = np.concatenate((temp[6:], temp[:6]))
        prec = np.concatenate((prec[6:], prec[:6]))
        evap = np.concatenate((evap[6:], evap[:6]))
        tmax = np.concatenate((tmax[6:], tmax[:6]))
        tmin = np.concatenate((tmin[6:], tmin[:6]))

    if unit:
        temp = celcius_to_fahrenheit(temp)
        tmax = celcius_to_fahrenheit(tmax)
        tmin = celcius_to_fahrenheit(tmin)
        prec = mm_to_inch(prec)
        evap = mm_to_inch(evap)

    fig = go.Figure()

    # add precipitation bar chart
    fig.add_trace(
        go.Bar(
            x=months,
            y=prec,
            name="Precipitation",
            marker_color="rgba(0, 135, 189, 0.5)",
            yaxis="y2",
            showlegend=False,
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # add evaporation bar chart
    fig.add_trace(
        go.Bar(
            x=months,
            y=evap,
            name="Evaporation",
            marker_color="rgba(255, 211, 0, 0.5)",
            yaxis="y2",
            showlegend=False,
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # add temperature line chart
    fig.add_trace(
        go.Scatter(
            x=months,
            y=temp,
            name="Mean Temperature",
            line=dict(color="rgba(196, 2, 52, 0.8)", width=2),
            mode="lines+markers",
            showlegend=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=tmax - temp,
                arrayminus=temp - tmin,
            ),
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # add temperature range
    fig.add_trace(
        go.Scatter(
            x=months,
            y=tmax,
            mode="markers",
            name="Daily Maximum",
            marker=dict(color="rgba(196, 2, 52, 0.8)", size=8),
            showlegend=False,
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # add temperature range
    fig.add_trace(
        go.Scatter(
            x=months,
            y=tmin,
            mode="markers",
            name="Daily Minimum",
            marker=dict(color="rgba(196, 2, 52, 0.8)", size=8),
            showlegend=False,
            hovertemplate="(%{x}, %{y:.1f})",
        )
    )

    # update layout
    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(text=subtitle, font=dict(size=13)),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(t=60, l=60, r=60, b=30),  # add top margin to leave space for the title
        height=400,  # set the chart height, ensure the chart size is fixed
        yaxis=dict(
            title="Temperature (°C)" if not unit else "Temperature (°F)",
            range=TEMP_RANGE[unit],
            autorange=auto_scale,
            showgrid=not auto_scale,
            gridcolor="lightgray" if not auto_scale else None,
            zeroline=(not unit),
            zerolinecolor="black",
            linecolor="rgb(196, 2, 52)",
            tickcolor="rgb(196, 2, 52)",
            tickfont=dict(color="rgb(196, 2, 52)"),
            tickvals=TEMP_TICKVALS[unit] if not auto_scale else None,
        ),
        yaxis2=dict(
            title=(
                "Precipitation/Evaporation (mm)"
                if not unit
                else "Precipitation/Evaporation (inch)"
            ),
            range=PREC_RANGE[unit],
            autorange=auto_scale,
            showgrid=not auto_scale,
            gridcolor="lightgray" if not auto_scale else None,
            overlaying="y",
            side="right",
            zeroline=True,
            zerolinecolor="black",
            linecolor="rgb(0, 135, 189)",
            tickvals=PREC_TICKVALS[unit] if not auto_scale else None,
            tickcolor="rgb(0, 135, 189)",
            tickfont=dict(color="rgb(0, 135, 189)"),
        ),
        xaxis=dict(gridcolor="lightgray"),
        shapes=(
            [
                dict(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=32 if unit else 0,
                    y1=32 if unit else 0,
                    xref="paper",
                    yref="y",
                    line=dict(color="black", width=0.5),
                )
            ]
            if unit
            else None
        ),  # add this line only when unit is Fahrenheit
        barmode="overlay",
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


def moving_average(a, n=30):
    ret = np.cumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def create_variable_chart(
    y: np.ndarray,
    location: tuple[float, float] | None,
    title: str | None,
    subtitle: str | None,
    map_type: str,
    unit: bool,
    mov_avg: bool = False,
) -> go.Figure:
    fig = go.Figure()


    if map_type in [
        "Annual Mean Temperature",
        "Coldest Month Mean Temperature",
        "Hottest Month Mean Temperature",
        "Coldest Month Mean Daily Minimum",
        "Hottest Month Mean Daily Maximum",
    ]:
        if unit:
            y = celcius_to_fahrenheit(y)
            if not location and title is not None:
                title += " (°F)"
        else:
            if not location and title is not None:
                title += " (°C)"
    elif map_type in [
        "Annual Total Precipitation",
        "Driest Month Precipitation",
        "Wettest Month Precipitation",
    ]:
        if unit:
            y = mm_to_inch(y)
            if not location and title is not None:
                title += " (inch)"
        else:
            if not location and title is not None:
                title += " (mm)"

    if mov_avg:
        y = moving_average(y)
        x = [i for i in range(1931, LATEST_YEAR + 1)]
    else:
        x = [i for i in range(1901, LATEST_YEAR + 1)]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines" if mov_avg else "lines+markers",
            showlegend=False,
            hovertemplate="(%{x}, %{y:.4f})<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text=title if title is not None else "",
            subtitle=dict(text=subtitle, font=dict(size=13)) if subtitle is not None else None,
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=14 if location else 15),
        ),
        margin=dict(t=60, l=60, r=60, b=30),  # add top margin to leave space for the title
        height=400,
        xaxis=dict(
            range=[1931, 2030] if mov_avg else [1901, 2030],
            tickvals=(
                np.arange(1931, 2030, 10) if mov_avg else np.arange(1901, 2030, 10)
            ),
            gridcolor="lightgray",
        ),
        yaxis=dict(
            title=map_type if location else "",
            autorange=True,
            gridcolor="lightgray",
        ),
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def create_probability_chart(
    probabilities: np.ndarray,
    class_map: list[str],
    color_map: dict[str, str],
    title: str,
    subtitle: str,
) -> go.Figure:
    """
    create climate type probability distribution chart

    Args:
        probabilities: probability array, shape is (n_classes,)
        class_map: climate type name list
        color_map: climate type color mapping dictionary
        title: title
        subtitle: sub-title
    Returns:
        plotly.graph_objects.Figure object
    """
    # get the indices of the top 3 types
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    
    # prepare data
    classes = [class_map[i] for i in top_3_indices]
    probs = probabilities[top_3_indices]
    colors = [color_map[cls] for cls in classes]

    # calculate remaining probability
    remaining_prob = 1.0 - np.sum(probs)
    
    # add "Others (Uncertainty)" if remaining probability >= 0.05
    if remaining_prob >= 0.05:
        classes.append("Others (Uncertainty)")
        probs = np.append(probs, remaining_prob)
        # use a distinct color that's different from climate type colors
        colors.append("#808080")  # gray, different from climate type colors

    fig = go.Figure()

    # add probability bar chart
    fig.add_trace(
        go.Bar(
            x=classes,
            y=probs,
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],  # show percentage
            textposition="auto",
            textfont=dict(size=13),
            hoverinfo="skip",  # disable hover tooltip
        )
    )

    # update layout
    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(text=subtitle, font=dict(size=13)),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(t=60, l=60, r=60, b=30),
        height=400,
        yaxis=dict(
            title="Probability",
            range=[0, 1],
            tickformat=".0%",
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            tickfont=dict(size=13),
        ),
        xaxis=dict(
            tickfont=dict(size=13),
        ),
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


def discretize_thermal_index(arr):
    arr = np.asarray(arr)
    bins = np.array([-np.inf, -1, 0, 1, np.inf])
    labels = ['cold', 'cool temperate', 'warm temperate', 'hot']
    inds = np.digitize(arr, bins) - 1
    return np.array([labels[i] for i in inds])


def discretize_aridity_index(arr):
    arr = np.asarray(arr)
    bins = np.array([-np.inf, -1, 0, 1, np.inf])
    labels = ['humid', 'sub-humid', 'semi-arid', 'arid']
    inds = np.digitize(arr, bins) - 1
    return np.array([labels[i] for i in inds])
