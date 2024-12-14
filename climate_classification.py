"""
This file contains the definition of the climate classification methods

Author: Tracy Van
Date: 2024-12-08
"""

import numpy as np
from DLModel import som
from dataclasses import dataclass

EPS = 1e-4

# Basic DECC
SIMPLE_MAP = ["Ax", "Dx", "Fx", "Am", "Dm", "Fm", "As", "Ds", "Fs", "Af", "Df", "Ff"]

SIMPLE_ORDER = ["Af", "Am", "As", "Ax", "Df", "Dm", "Ds", "Dx", "Ff", "Fm", "Fs", "Fx"]

SIMPLE_COLOR_MAP = {
    # A组 - 暖热气候
    "Af": "#960000",  # 温暖湿润 (深红)
    "Am": "#CC9900",  # 温暖半干旱 (金色，参考Koppen BSh)
    "As": "#FF6666",  # 干湿季 (浅红，参考Trewartha Aw)
    "Ax": "#FF9933",  # 炎热干旱 (橙色，参考Koppen BWh)
    # D组 - 适中气候
    "Df": "#33CC33",  # 温和 (深绿，参考Trewartha Cf)
    "Dm": "#006600",  # （寒凉）海洋性 (深绿，参考Koppen Cfc)
    "Ds": "#6666FF",  # （大陆性）高地 (浅紫，参考Trewartha Eo)
    "Dx": "#FFCC00",  # 大陆性干旱半干旱 (土黄，参考Koppen BWk)
    # F组 - 寒冷气候
    "Ff": "#0033CC",  # 大陆性季风 (深蓝，参考Koppen Dwc)
    "Fm": "#003366",  # 大陆性湿润 (深蓝，参考Koppen Dfd)
    "Fs": "#99FFFF",  # 苔原 (浅青，参考Trewartha Ft)
    "Fx": "#FFFFFF",  # 冰原 (白色，参考Trewartha Fi)
}

# Advanced DECC
DETAILED_MAP = [
    "Fsk",
    "Fsh",
    "DW",
    "Es",
    "As",
    "DSk",
    "DSh",
    "Cs",
    "AW",
    "Emh",
    "Cm",
    "EW",
    "Ds",
    "Fm",
    "Cf",
    "Emk",
    "CW",
    "Gk",
    "Gh",
    "DH",
    "Efh",
    "Am",
    "EH",
    "Ff",
    "Efk",
    "Af",
]

DETAILED_COLOR_MAP = {
    # A组 - 热带气候
    "Af": "#960000",  # 热带雨林 (深红，保持一致)
    "Am": "#CC3300",  # 热带温和 (红褐)
    "As": "#FF6666",  # 热带干湿季 (浅红，保持一致)
    "AW": "#FF9933",  # 热带沙漠 (橙色，参考Koppen BWh)
    # C组 - 亚热带气候
    "Cf": "#33CC33",  # 亚热带湿润 (深绿，参考Trewartha Cf)
    "Cm": "#66CC66",  # 亚热带温和 (中绿)
    "Cs": "#CC9900",  # 暖热半干旱 (金色，参考Koppen BSh)
    "CW": "#FFCC00",  # 亚热带沙漠 (土黄，参考Koppen BWk)
    # D组 - 大陆性气候
    "Ds": "#0066CC",  # 大陆性季风 (深蓝，参考Koppen Dwb)
    "DSh": "#CC6600",  # 暖温带大陆性半干旱 (深橙，参考Koppen BSk)
    "DSk": "#996600",  # 冷温带大陆性半干旱 (褐色)
    "DW": "#FFCC66",  # 大陆性沙漠 (浅橙)
    "DH": "#6666FF",  # 大陆性高地 (浅紫，参考Trewartha Eo)
    # E组 - 温带气候
    "Efh": "#009933",  # 暖温带海洋性 (深绿，参考Koppen Cfb)
    "Efk": "#006600",  # 冷温带海洋性 (暗绿，参考Koppen Cfc)
    "Emh": "#00CC99",  # 暖温带半海洋性 (青绿)
    "Emk": "#009999",  # 冷温带半海洋性 (深青)
    "Es": "#99FF99",  # 温带干湿季 (浅绿，参考Trewartha Cs)
    "EW": "#FFE5CC",  # 温带沙漠 (极浅橙)
    "EH": "#9999FF",  # 温和高地 (中紫)
    # F组 - 亚寒带气候
    "Ff": "#003366",  # 亚寒带海洋性 (深蓝，参考Koppen Dfc)
    "Fm": "#0033CC",  # 亚寒带半海洋性 (深蓝，参考Koppen Dwc)
    "Fsh": "#3366CC",  # 一般亚寒带大陆性 (中蓝，调亮)
    "Fsk": "#66CCFF",  # 极端亚寒带大陆性 (浅蓝，接近极地色系)
    # G组 - 极地气候
    "Gh": "#99FFFF",  # 极地苔原 (浅青，参考Trewartha Ft)
    "Gk": "#FFFFFF",  # 极地冰原 (白色，参考Trewartha Fi)
}

DETAILED_ORDER = [
    "Af",
    "Am",
    "As",
    "AW",
    "Cf",
    "Cm",
    "Cs",
    "CW",
    "Ds",
    "DSh",
    "DSk",
    "DW",
    "DH",
    "Efh",
    "Efk",
    "Emh",
    "Emk",
    "Es",
    "EW",
    "EH",
    "Ff",
    "Fm",
    "Fsh",
    "Fsk",
    "Gh",
    "Gk",
]

# Land Cover
VEG_MAP = [
    "evergreen needleleaf forest",
    "deciduous needleleaf forest",
    "evergreen broadleaf forest",
    "deciduous broadleaf forest",
    "mixed forest",
    "closed shrubland",
    "open shrubland",
    "woody savanna",
    "savanna",
    "grassland",
    "permanent wetland",
    "cropland mosaics",
    "snow and ice",
    "barren",
]

VEG_COLOR_MAP = {
    "evergreen needleleaf forest": "#1A9850",  # 深绿色
    "deciduous needleleaf forest": "#66BD63",  # 中绿色
    "evergreen broadleaf forest": "#006837",  # 暗绿色
    "deciduous broadleaf forest": "#A6D96A",  # 浅绿色
    "mixed forest": "#D9EF8B",  # 黄绿色
    "closed shrubland": "#BF812D",  # 深棕色
    "open shrubland": "#DFC27D",  # 浅棕色
    "woody savanna": "#F6E8C3",  # 米黄色
    "savanna": "#FED98E",  # 浅黄色
    "grassland": "#FFFFBF",  # 淡黄色
    "permanent wetland": "#80CDC1",  # 青绿色
    "cropland mosaics": "#C7EAE5",  # 浅青色
    "snow and ice": "#FFFFFF",  # 白色
    "barren": "#F4A582",  # 浅褐色
}


class KoppenClassification:
    order = [
        "Af",
        "Am",
        "Aw",
        "BWh",
        "BWk",
        "BSh",
        "BSk",
        "Cfa",
        "Cfb",
        "Cfc",
        "Csa",
        "Csb",
        "Csc",
        "Cwa",
        "Cwb",
        "Cwc",
        "Dfa",
        "Dfb",
        "Dfc",
        "Dfd",
        "Dsa",
        "Dsb",
        "Dsc",
        "Dsd",
        "Dwa",
        "Dwb",
        "Dwc",
        "Dwd",
        "ET",
        "EF",
    ]

    color_map = {
        # A组 - 热带气候
        "Af": "#960000",  # 热带雨林
        "Am": "#FF0000",  # 热带季风
        "Aw": "#FF6666",  # 热带草原
        # B组 - 干燥气候
        "BWh": "#FF9933",  # 热带沙漠
        "BWk": "#FFCC00",  # 温带沙漠
        "BSh": "#CC9900",  # 热带草原
        "BSk": "#CC6600",  # 温带草原
        # C组 - 温带气候
        "Cfa": "#33CC33",  # 温带湿润
        "Cfb": "#009900",  # 海洋性
        "Cfc": "#006600",  # 亚寒带海洋性
        "Csa": "#66FF66",  # 地中海夏干
        "Csb": "#00FF00",  # 温和地中海
        "Csc": "#004D00",  # 亚寒带地中海
        "Cwa": "#99FF99",  # 亚热带季风
        "Cwb": "#CCFFCC",  # 高地热带
        "Cwc": "#669966",  # 亚寒带季风
        # D组 - 大陆性气候
        "Dfa": "#00CCFF",  # 温带大陆性
        "Dfb": "#0099CC",  # 亚寒带大陆性
        "Dfc": "#006699",  # 亚寒带
        "Dfd": "#003366",  # 极寒带大陆性
        "Dsa": "#99CCFF",  # 地中海型大陆性
        "Dsb": "#6699CC",  # 温和地中海型大陆性
        "Dsc": "#336699",  # 亚寒带地中海型大陆性
        "Dsd": "#1A334D",  # 极寒带地中海型大陆性
        "Dwa": "#CCE5FF",  # 季风型大陆性
        "Dwb": "#3366CC",  # 温和季风型大陆性
        "Dwc": "#0033CC",  # 亚寒带季风型大陆性
        "Dwd": "#000066",  # 极寒带季风型大陆性
        # E组 - 极地气候
        "ET": "#99FFFF",  # 苔原
        "EF": "#FFFFFF",  # 冰原
    }

    @staticmethod
    def classify(climate_data: np.ndarray, cd_threshold: float, kh_mode: str) -> str:
        """
        cd_threshold: threshold for C and D classes, usually -3 or 0
        kh_mode: classification mode for B classes ('coldest_month' coldest month < cd_threshold is k, otherwise h, or 'mean_temp' mean temperature < 18 is k, otherwise h)
        """
        # Sort by temperature
        sorted_indices = np.argsort(climate_data[0, :])
        t_monthly = climate_data[:, sorted_indices]

        # Calculate basic climate indicators
        total_pr = np.sum(t_monthly[1, :])
        pr_percent = np.sum(t_monthly[1, 5:12]) / (
            total_pr + EPS
        )  # Summer half-year precipitation ratio
        mean_temp = np.mean(t_monthly[0, :])

        # Calculate aridity threshold
        thresh = 20 * mean_temp
        if pr_percent >= 0.7:
            thresh += 280
        elif pr_percent >= 0.3:
            thresh += 140

        cls = ""

        # Determine climate type
        if total_pr < 0.5 * thresh:  # Desert climate BW
            cls = "BW"
            if kh_mode == "coldest_month":
                if t_monthly[0, 0] < cd_threshold:
                    cls += "k"
                else:
                    cls += "h"
            elif kh_mode == "mean_temp":
                if mean_temp < 18:
                    cls += "k"
                else:
                    cls += "h"

        elif total_pr < thresh:  # Steppe climate BS
            cls = "BS"
            if kh_mode == "coldest_month":
                if t_monthly[0, 0] < cd_threshold:
                    cls += "k"
                else:
                    cls += "h"
            elif kh_mode == "mean_temp":
                if mean_temp < 18:
                    cls += "k"
                else:
                    cls += "h"

        else:
            if t_monthly[0, 0] >= 18:  # Tropical climate A
                cls = "A"
                if np.min(t_monthly[1, :]) >= 60:
                    cls += "f"
                else:
                    thresh_2 = 100 - total_pr / 25
                    if np.min(t_monthly[1, :]) < thresh_2:
                        cls += "w"
                    else:
                        cls += "m"

            elif t_monthly[0, -1] < 10:  # Polar climate E
                cls = "E"
                if t_monthly[0, -1] < 0:
                    cls += "F"
                else:
                    cls += "T"

            elif t_monthly[0, 0] > cd_threshold:  # Temperate climate C
                cls = "C"
                # Determine precipitation characteristics
                if np.max(t_monthly[1, 9:12]) > 10 * np.min(t_monthly[1, 0:3]):
                    cls += "w"
                elif (
                    np.max(t_monthly[1, 0:3]) > 3 * np.min(t_monthly[1, 9:12])
                    and np.min(t_monthly[1, 9:12]) < 30
                ):
                    cls += "s"
                else:
                    cls += "f"

                # Determine temperature characteristics
                if np.sum(t_monthly[0, :] > 10) < 4:
                    cls += "c"
                elif t_monthly[0, -1] > 22:
                    cls += "a"
                else:
                    cls += "b"

            else:  # Continental climate D
                cls = "D"
                # Determine precipitation characteristics
                if np.max(t_monthly[1, 9:12]) > 10 * np.min(t_monthly[1, 0:3]):
                    cls += "w"
                elif (
                    np.max(t_monthly[1, 0:3]) > 3 * np.min(t_monthly[1, 9:12])
                    and np.min(t_monthly[1, 9:12]) < 30
                ):
                    cls += "s"
                else:
                    cls += "f"

                # Determine temperature characteristics
                if np.sum(t_monthly[0, :] > 10) < 4:
                    if t_monthly[0, 0] <= -38:
                        cls += "d"
                    else:
                        cls += "c"
                elif t_monthly[0, -1] > 22:
                    cls += "a"
                else:
                    cls += "b"

        return cls


class TrewarthaClassification:
    order = ["Ar", "Aw", "BW", "BS", "Cf", "Cs", "Do", "Dc", "Eo", "Ec", "Ft", "Fi"]

    color_map = {
        # A组 - 热带气候
        "Ar": "#960000",  # 热带雨林 (深红)
        "Aw": "#FF6666",  # 热带草原 (浅红)
        # B组 - 干燥气候
        "BW": "#FF9933",  # 沙漠 (橙色)
        "BS": "#CC9900",  # 草原 (金色)
        # C组 - 副热带气候
        "Cf": "#33CC33",  # 湿润副热带 (深绿)
        "Cs": "#99FF99",  # 地中海副热带 (浅绿)
        # D组 - 温带气候
        "Do": "#00CCFF",  # 海洋性温带 (浅蓝)
        "Dc": "#0066CC",  # 大陆性温带 (深蓝)
        # E组 - 亚寒带气候
        "Eo": "#6666FF",  # 海洋性亚寒带 (浅紫)
        "Ec": "#333399",  # 大陆性亚寒带 (深紫)
        # F组 - 极地气候
        "Ft": "#99FFFF",  # 苔原 (浅青)
        "Fi": "#FFFFFF",  # 冰原 (白色)
    }

    @staticmethod
    def classify(climate_data: np.ndarray) -> str:
        # Sort by temperature
        sorted_indices = np.argsort(climate_data[0, :])
        t_monthly = climate_data[:, sorted_indices]

        # Calculate basic climate indicators
        total_pr = np.sum(t_monthly[1, :])
        pr_percent = np.sum(t_monthly[1, 0:6]) / (
            total_pr + EPS
        )  # Winter half-year precipitation ratio
        mean_temp = np.mean(t_monthly[0, :])

        # Calculate aridity threshold (Trewartha's formula)
        # de Castro M, Gallardo C, Jylha K, Tuomenvirta H (2007) The use of a climate-type classification for assessing climate change effects in Europe form an ensemble of nine regional climate models. Clim Change 81: 329−341
        thresh = 23 * mean_temp - 6.4 * pr_percent + 410

        cls = ""

        # Determine climate type
        if total_pr < 0.5 * thresh:  # Desert climate BW
            cls = "BW"

        elif total_pr < thresh:  # Steppe climate BS
            cls = "BS"

        else:
            if t_monthly[0, 0] >= 18:  # Tropical climate A
                cls = "A"
                # Determine precipitation characteristics
                if (
                    np.sum(t_monthly[1, :] >= 60) >= 10
                ):  # At least 10 months with precipitation >= 60mm
                    cls += "r"
                else:
                    cls += "w"

            elif t_monthly[0, -1] < 10:  # Polar climate F
                cls = "F"
                if t_monthly[0, -1] < 0:  # Warmest month temperature < 0℃
                    cls += "i"
                else:  # Warmest month temperature 0-10℃
                    cls += "t"

            elif np.sum(t_monthly[0, :] > 10) >= 8:  # Subtropical climate C
                cls = "C"
                # Determine precipitation characteristics
                if (
                    np.max(t_monthly[1, 0:3]) > 3 * np.min(t_monthly[1, 9:12])
                    and np.min(t_monthly[1, 9:12]) < 30
                    and total_pr < 890
                ):
                    cls += "s"
                else:
                    cls += "f"

            elif np.sum(t_monthly[0, :] > 10) >= 4:  # Temperate climate D
                cls = "D"
                if t_monthly[0, 0] < 0:  # Coldest month < 0℃
                    cls += "c"
                else:  # Coldest month >= 0℃
                    cls += "o"

            else:  # Subarctic climate E
                cls = "E"
                if t_monthly[0, 0] < -10:  # Coldest month < -10℃
                    cls += "c"
                else:  # Coldest month >= -10℃
                    cls += "o"

        return cls


@dataclass
class DLClassification:
    __slots__ = ["order", "class_map", "color_map", "centroid"]

    order: list[str]
    class_map: list[str]
    color_map: dict[str, str]

    centroid: np.ndarray

    # bulk operation
    # pca_features: shape (N, 15)
    def classify(self, pca_features: np.ndarray) -> list[str]:
        cls_indices = som(pca_features, self.centroid)
        return [self.class_map[i] for i in cls_indices]


# usage
if __name__ == "__main__":
    from DLModel import Network
    import h5py

    example_data = np.array(
        [
            [
                [
                    6.6,
                    9.3,
                    13.4,
                    18.5,
                    22.1,
                    25.1,
                    27.8,
                    27.9,
                    23.0,
                    17.9,
                    13.3,
                    8.1,
                ],  # temperature in Celsius
                [
                    15.8,
                    16.7,
                    35.5,
                    78.7,
                    126.3,
                    156.9,
                    168.8,
                    137.4,
                    125.2,
                    84.6,
                    38.3,
                    17.4,
                ],  # precipitation in mm
                [
                    28.4,
                    37.7,
                    59.2,
                    79.6,
                    96.4,
                    98.6,
                    114.4,
                    116.7,
                    74.2,
                    50.5,
                    34.6,
                    26.8,
                ],  # PET in mm
            ]
        ]
    )
    # shape should be (batch_size, 3, 12)
    print("Input Shape:", example_data.shape)

    # data may also be loaded from the downloaded data file
    # with h5py.File('climate_data_land.h5', 'r') as f:
    #     indices = f.get('indices')[:]       # latitude and longitude of all the locations
    #     # example is given as a single location and single year, you can also do bulk processing
    #     lat, lon = 30.6, 105.8
    #     year = 2010
    #     # round to the nearest 0.5 degree
    #     lat = round(lat * 2) / 2
    #     lon = round(lon * 2) / 2
    #     idx = np.where((indices[:, 0] == lat) & (indices[:, 1] == lon))[0]
    #     tmp = f.get('tmp')[idx, year - 1901, :]
    #     pre = f.get('pre')[idx, year - 1901, :]
    #     pet = f.get('pet')[idx, year - 1901, :]
    #     example_data = np.array([tmp, pre, pet])
    #     example_data = example_data.transpose(1, 0, 2)

    with h5py.File("weights.h5", "r") as f:
        network = Network(f)
        pca_features, veg_indices = network(example_data)
        advanced_dl_classifier = DLClassification(
            order=DETAILED_ORDER,
            class_map=DETAILED_MAP,
            color_map=DETAILED_COLOR_MAP,
            centroid=f.get("centroid_detailed")[:],
        )
        basic_dl_classifier = DLClassification(
            order=SIMPLE_ORDER,
            class_map=SIMPLE_MAP,
            color_map=SIMPLE_COLOR_MAP,
            centroid=f.get("centroid_322")[:],
        )
        print("Output Features Shape:", pca_features.shape)
        print("Output Land Cover Indices Shape:", veg_indices.shape)

        print("Cryohumidity:", pca_features[0, 0])
        print("Continentality:", pca_features[0, 1])
        print("Seasonality:", pca_features[0, 2])

        print("Advanced DECC Type:", advanced_dl_classifier.classify(pca_features)[0])
        print("Basic DECC Type:", basic_dl_classifier.classify(pca_features)[0])
        print("Land Cover Type:", VEG_MAP[veg_indices[0]])

        # For the classic climate classification, the input shape should be (>=2, 12) for a single location
        # first row is temperature, second row is precipitation, other rows are omitted
        # bulk operation is not supported
        example_data = example_data.squeeze()
        print(
            "Köppen-Geiger Type:",
            KoppenClassification.classify(
                example_data, cd_threshold=0, kh_mode="mean_temp"
            ),
        )
        print("Trewartha Type:", TrewarthaClassification.classify(example_data))
