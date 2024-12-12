import numpy as np
from dataclasses import dataclass
import pandas as pd
from climate_classification import *
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import h5py
from DLModel import Network
from numba import njit

LATEST_YEAR = 2023
TEMP_RANGE = [
    [-25, 40],  # °C
    [-8, 96],
]  # °F
TEMP_TICKVALS = [
    [-20, -10, 0, 10, 20, 30, 40],  # °C
    [0, 16, 32, 48, 64, 80, 96],
]  # °F
PREC_RANGE = [
    [0, 390],  # mm
    [0, 13],
]  # inch
PREC_TICKVALS = [
    [30, 90, 150, 210, 270, 330, 390],  # mm
    [1, 3, 5, 7, 9, 11, 13],
]  # inch
MAP_TYPE_INDICES = {
    "Lowest Monthly Temperature": 0,
    "Highest Monthly Temperature": 1,
    "Highest Monthly Precipitation": 2,
    "Lowest Monthly Precipitation": 3,
    "Annual Mean Temperature": 4,
    "Annual Total Precipitation": 5,
    "Aridity Index": 6,
    "Cryohumidity": 7,
    "Continentality": 8,
    "Seasonality": 9,
}


@njit
def celcius_to_fahrenheit(temp: np.ndarray) -> np.ndarray:
    return temp * np.float32(9 / 5) + np.float32(32)


@njit
def mm_to_inch(prec: np.ndarray) -> np.ndarray:
    return prec / np.float32(25.4)


@njit
def calc_trend_theil_sen(x: np.ndarray, y: np.ndarray) -> np.float32:
    """使用Theil-Sen估计器计算趋势"""
    n = x.shape[0]
    slopes = np.zeros(n * (n - 1) // 2, dtype=np.float32)
    k = 0

    # 计算所有可能的斜率
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[j] - x[i]) > 1e-10:
                slopes[k] = (y[j] - y[i]) / (x[j] - x[i])
                k += 1

    # 返回中位数斜率
    return np.float32(np.median(slopes[:k]))


@njit
def process_variables_trend(
    data: np.ndarray, x: np.ndarray, map_type_idx: int, convert_unit: bool = False
) -> np.ndarray:
    """批量处理气候数据并计算趋势"""
    n_points = data.shape[0]
    values = np.zeros(n_points, dtype=np.float32)

    for i in range(n_points):
        y = data[i, :, map_type_idx]
        if convert_unit:
            if map_type_idx in [0, 1, 4]:
                y = celcius_to_fahrenheit(y)
            elif map_type_idx in [2, 3, 5]:
                y = mm_to_inch(y)
        values[i] = calc_trend_theil_sen(x, y)

    return values


@njit
def calc_variables(data: np.ndarray) -> np.ndarray:
    """一次性计算所有气候指标"""
    n_samples = len(data)
    # 预分配结果数组
    res = np.zeros((n_samples, 7), dtype=np.float32)

    for i in range(n_samples):
        temp = data[i, 0, :]  # 温度数据
        precip = data[i, 1, :]  # 降水数据
        evap = data[i, 2, :]  # 蒸发数据

        res[i, 0] = np.min(temp)  # 'Coldest Month Temperature'
        res[i, 1] = np.max(temp)  # 'Hottest Month Temperature'
        res[i, 2] = np.max(precip)  # 'Wettest Month Precipitation'
        res[i, 3] = np.min(precip)  # 'Driest Month Precipitation'
        res[i, 4] = np.mean(temp)  # 'Mean Annual Temperature'
        res[i, 5] = np.sum(precip)  # 'Total Annual Precipitation'
        res[i, 6] = res[i, 5] / np.sum(evap)  # 'Aridity Index'

    return res


# 只用于change rate绘图
def calc_change_rate(
    variable_file: h5py.File,
    indices: np.ndarray,
    elev: np.ndarray,
    map_type: str,
    yr: int,
    yrs: int,
    unit: bool = False,
) -> pd.DataFrame:
    x = np.array([i for i in range(yr, yr + yrs)], dtype=np.float32)
    data = variable_file.get("variables")[:-1, yr : yr + yrs, :]

    values = process_variables_trend(data, x, MAP_TYPE_INDICES[map_type], unit)
    lats, lons = zip(*indices)

    return pd.DataFrame({"lat": lats, "lon": lons, "value": values, "elev": elev})


@dataclass
class ClimateData:
    __slots__ = ["ori", "tmn", "tmx", "elev"]

    ori: np.ndarray
    tmn: np.ndarray
    tmx: np.ndarray
    elev: float


@dataclass
class ClimateDataset:
    __slots__ = ["data", "pca_features", "veg_indices", "variables"]

    data: dict[tuple[float, float], ClimateData]
    pca_features: np.ndarray
    veg_indices: np.ndarray
    variables: np.ndarray

    def __getitem__(self, idx: tuple[float, float]) -> ClimateData:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def prepare_dl(self, dl_network: Network, indices: np.ndarray) -> None:
        batch_size = 4096
        n_samples = len(self)
        self.pca_features = np.zeros(
            (n_samples, dl_network.pca_components), dtype=np.float32
        )
        self.veg_indices = np.zeros(n_samples, dtype=np.int32)

        for i in range(0, n_samples, batch_size):
            batch = np.array(
                [
                    self.data[(indices[j, 0], indices[j, 1])].ori
                    for j in range(i, i + batch_size)
                    if j < n_samples
                ]
            )
            (
                self.pca_features[i : i + batch_size],
                self.veg_indices[i : i + batch_size],
            ) = dl_network(batch)

    def prepare_variables(self) -> None:
        self.variables = calc_variables(np.array([v.ori for v in self.data.values()]))

    def get_koppen(self, cd_threshold: float, kh_mode: str) -> list[str]:
        return [
            KoppenClassification.classify(v.ori, cd_threshold, kh_mode)
            for v in self.data.values()
        ]

    def get_trewartha(self) -> list[str]:
        return [TrewarthaClassification.classify(v.ori) for v in self.data.values()]

    def get_veg(self, veg_names: list[str]) -> list[str]:
        return [veg_names[i] for i in self.veg_indices]

    def get_dl(self, dl_classifier: DLClassification) -> list[str]:
        return dl_classifier.classify(self.pca_features)

    def get_cryohumidity(self) -> np.ndarray:
        return self.pca_features[:, 0]

    def get_continentality(self) -> np.ndarray:
        return self.pca_features[:, 1]

    def get_seasonality(self) -> np.ndarray:
        return self.pca_features[:, 2]

    def get_coldest_month_tmp(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 0]
        return celcius_to_fahrenheit(res) if unit else res

    def get_hottest_month_tmp(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 1]
        return celcius_to_fahrenheit(res) if unit else res

    def get_wettest_month_pre(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 2]
        return mm_to_inch(res) if unit else res

    def get_driest_month_pre(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 3]
        return mm_to_inch(res) if unit else res

    def get_mean_temp(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 4]
        return celcius_to_fahrenheit(res) if unit else res

    def get_total_pr(self, unit: bool = False) -> np.ndarray:
        res = self.variables[:, 5]
        return mm_to_inch(res) if unit else res

    def get_aridity_index(self) -> np.ndarray:
        return self.variables[:, 6]

    def prepare_map_data(
        self,
        map_type: str,
        koppen_cd_mode: str = "",
        koppen_kh_mode: str = "",
        dl_classifier: DLClassification | None = None,
        veg_names: list[str] | None = None,
        unit: bool = False,
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
        elif (
            map_type == "Data-driven Ecological - Basic"
            or map_type == "Data-driven Ecological - Advanced"
        ):
            values = self.get_dl(dl_classifier)
        elif map_type == "Predicted Land Cover":
            values = self.get_veg(veg_names)
        elif map_type == "Annual Mean Temperature":
            values = self.get_mean_temp(unit)
        elif map_type == "Annual Total Precipitation":
            values = self.get_total_pr(unit)
        elif map_type == "Aridity Index":
            values = self.get_aridity_index()
        elif map_type == "Cryohumidity":
            values = self.get_cryohumidity()
        elif map_type == "Continentality":
            values = self.get_continentality()
        elif map_type == "Seasonality":
            values = self.get_seasonality()
        elif map_type == "Lowest Monthly Temperature":
            values = self.get_coldest_month_tmp(unit)
        elif map_type == "Highest Monthly Temperature":
            values = self.get_hottest_month_tmp(unit)
        elif map_type == "Highest Monthly Precipitation":
            values = self.get_wettest_month_pre(unit)
        elif map_type == "Lowest Monthly Precipitation":
            values = self.get_driest_month_pre(unit)

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
    处理气候数据并计算Köppen气候分类

    参数:
        yr: 起始年份
        yrs: 年份跨度
        tmp: 温度数据 shape: (300, 720, years, 12)
        pre: 降水数据 shape: (300, 720, years, 12)
        pet: 潜在蒸发量数据 shape: (300, 720, years, 12)
        tmn: 最低温度数据 shape: (300, 720, years, 12)
        tmx: 最高温度数据 shape: (300, 720, years, 12)
        elev: 海拔数据 shape: (300, 720)

    返回:
        ClimateDataset对象，包含处理后的气候数据
    """
    res = {}

    # 预先计算所有数据的多年月平均值
    tmp_mean = np.mean(datafile["tmp"][:, yr : yr + yrs, :], axis=1)
    pre_mean = np.mean(datafile["pre"][:, yr : yr + yrs, :], axis=1)
    pet_mean = np.mean(datafile["pet"][:, yr : yr + yrs, :], axis=1)
    tmn_mean = np.mean(datafile["tmn"][:, yr : yr + yrs, :], axis=1)
    tmx_mean = np.mean(datafile["tmx"][:, yr : yr + yrs, :], axis=1)

    for i in range(tmp_mean.shape[0]):

        pic = np.vstack([tmp_mean[i], pre_mean[i], pet_mean[i]])

        res[(indices[i, 0], indices[i, 1])] = ClimateData(
            ori=pic, tmn=tmn_mean[i], tmx=tmx_mean[i], elev=elev[i]
        )

    return ClimateDataset(
        data=res,
        pca_features=np.array([]),
        veg_indices=np.array([]),
        variables=np.array([]),
    )


def get_location_info(location: tuple[float, float], local_lang: bool = False) -> str:
    """
    获取地理位置信息

    参数:
        location: 地理位置坐标 (lat, lon)
        local_lang: 是否使用当地语言

    返回:
        包含地理信息的字符串
    """
    geolocator = Nominatim(user_agent="climate_viz")
    tries = 3  # 重试次数

    while tries > 0:
        try:
            location_info = geolocator.reverse(
                location, language="en" if not local_lang else False
            )
            if location_info and location_info.raw.get("address"):
                return location_info.address
            else:
                return "Unknown (Coastal Location or Network Error)"
        except GeocoderTimedOut:
            tries -= 1
            time.sleep(1)  # 等待1秒后重试
        except Exception:
            return "Unknown (Coastal Location or Network Error)"

    return "Unknown (Coastal Location or Network Error)"


def search_location(query: str) -> tuple[float, float] | None:
    """搜索地点获取经纬度"""
    try:
        geolocator = Nominatim(user_agent="climate_viz")
        location = geolocator.geocode(query)
        if location:
            return (round(location.latitude * 2) / 2, round(location.longitude * 2) / 2)
        return None
    except (GeocoderTimedOut, Exception):
        return None


def create_climate_chart(
    climate_data: ClimateData,
    location: tuple[float, float],
    subtitle: str,
    local_lang: bool,
    july_first: bool,
    unit: bool,
    auto_scale: bool,
) -> go.Figure:
    """
    创建气候图

    参数:
        climate_data: ClimateData 对象
        location: 坐标 (lat, lon)
        local_lang: 是否使用当地语言
        july_first: 是否从七月开始
        unit: 是否使用°F/inch
        auto_scale: 是否自动缩放

    返回:
        plotly.graph_objects.Figure 对象
    """
    # 准备数据
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
    temp = climate_data.ori[0, :]  # 月均温
    prec = climate_data.ori[1, :]  # 月降水量
    evap = climate_data.ori[2, :]  # 月蒸发量
    tmax = climate_data.tmx  # 日均高温
    tmin = climate_data.tmn  # 日均低温

    # 如果从七月开始，调整数据顺序
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

    # 添加降水量柱状图
    fig.add_trace(
        go.Bar(
            x=months,
            y=prec,
            name="Precipitation",
            marker_color="rgba(0, 135, 189, 0.5)",
            yaxis="y2",
            showlegend=False,
        )
    )

    # 添加蒸发量柱状图
    fig.add_trace(
        go.Bar(
            x=months,
            y=evap,
            name="Evaporation",
            marker_color="rgba(255, 211, 0, 0.5)",
            yaxis="y2",
            showlegend=False,
        )
    )

    # 添加温度折线图
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
        )
    )

    # 添加温度范围
    fig.add_trace(
        go.Scatter(
            x=months,
            y=tmax,
            mode="markers",
            name="Daily Maximum",
            marker=dict(color="rgba(196, 2, 52, 0.8)", size=8),
            showlegend=False,
        )
    )

    # 添加温度范围
    fig.add_trace(
        go.Scatter(
            x=months,
            y=tmin,
            mode="markers",
            name="Daily Minimum",
            marker=dict(color="rgba(196, 2, 52, 0.8)", size=8),
            showlegend=False,
        )
    )

    # 更新布局
    fig.update_layout(
        title=dict(
            text=get_location_info(location, local_lang),
            subtitle=dict(text=subtitle, font=dict(size=13)),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(t=60, l=60, r=60, b=30),  # 增加顶部边距给标题留空间
        height=400,  # 设置图表高度，确保图表本身大小不变
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
        ),  # 只在华氏度时添加这条线
        barmode="overlay",
        plot_bgcolor="white",
        showlegend=False,
    )

    return fig


def moving_average(a, n=30):
    ret = np.cumsum(a, dtype=np.float32)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def create_variable_chart(
    y: np.ndarray,
    location: tuple[float, float] | None,
    subtitle: str,
    map_type: str,
    unit: bool,
    local_lang: bool,
    mov_avg: bool = False,
) -> go.Figure:
    fig = go.Figure()
    title = (
        get_location_info(location, local_lang)
        if location
        else ("Global Average " + map_type)
    )

    if map_type in [
        "Annual Mean Temperature",
        "Lowest Monthly Temperature",
        "Highest Monthly Temperature",
    ]:
        if unit:
            y = celcius_to_fahrenheit(y)
            if not location:
                title += " (°F)"
        else:
            if not location:
                title += " (°C)"
    elif map_type in [
        "Annual Total Precipitation",
        "Lowest Monthly Precipitation",
        "Highest Monthly Precipitation",
    ]:
        if unit:
            y = mm_to_inch(y)
            if not location:
                title += " (inch)"
        else:
            if not location:
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
        )
    )
    fig.update_layout(
        title=dict(
            text=title,
            subtitle=dict(text=subtitle, font=dict(size=13)),
            x=0.5,
            xanchor="center",
            y=0.95,
            yanchor="top",
            font=dict(size=14 if location else 15),
        ),
        margin=dict(t=60, l=60, r=60, b=30),  # 增加顶部边距给标题留空间
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
