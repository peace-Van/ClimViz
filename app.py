import streamlit as st
import numpy as np
import plotly.express as px
from backend import (
    ClimateDataset,
    get_average,
    create_climate_chart,
    calc_change_rate,
    search_location,
    create_variable_chart,
    create_probability_chart,
    VARIABLE_TYPE_INDICES,
    LATEST_YEAR,
)
import h5py
from climate_classification import *
from DLModel import Network
import gc


MAP_TYPES = {
    "Classification": [
        "Data-driven Ecological - Advanced",
        "Data-driven Ecological - Basic",
        "Köppen-Geiger Classification",
        "Trewartha Classification",
        "Predicted Land Cover",
    ],
    "Variable": [
        "Cryohumidity",
        "Continentality",
        "Seasonality",
        "Annual Mean Temperature",
        "Annual Total Precipitation",
        "Aridity Index",
        "Lowest Monthly Temperature",
        "Highest Monthly Temperature",
        "Highest Monthly Precipitation",
        "Lowest Monthly Precipitation",
    ],
}

RANGES = {
    "Annual Mean Temperature": [
        [
            (-10, 30),  # value in degree Celsius
            (14, 86),  # value in degree Fahrenheit
        ],
        [
            (-0.1, 0.1),  # change rate in degree Celsius
            (-0.2, 0.2),  # change rate in degree Fahrenheit
        ],
    ],
    "Annual Total Precipitation": [
        [
            (0, 1600),  # value in mm
            (0, 63),  # value in inches
        ],
        [
            (-10, 10),  # change rate in mm
            (-0.4, 0.4),  # change rate in inches
        ],
    ],
    "Aridity Index": [
        [
            (0, 2),  # value
            (0, 2),
        ],
        [
            (-0.02, 0.02),  # change rate
            (-0.02, 0.02),
        ],
    ],
    "Cryohumidity": [
        [
            (-8, 8),  # value
            (-8, 8),
        ],
        [
            (-0.1, 0.1),  # change rate
            (-0.1, 0.1),
        ],
    ],
    "Continentality": [
        [
            (-8, 8),  # value
            (-8, 8),
        ],
        [
            (-0.1, 0.1),  # change rate
            (-0.1, 0.1),
        ],
    ],
    "Seasonality": [
        [
            (-6, 6),  # value
            (-6, 6),
        ],
        [
            (-0.1, 0.1),  # change rate
            (-0.1, 0.1),
        ],
    ],
    "Lowest Monthly Temperature": [
        [
            (-30, 30),  # value in degree Celsius
            (-22, 86),  # value in degree Fahrenheit
        ],
        [
            (-0.2, 0.2),  # change rate in degree Celsius
            (-0.4, 0.4),  # change rate in degree Fahrenheit
        ],
    ],
    "Highest Monthly Temperature": [
        [
            (0, 40),  # value in degree Celsius
            (32, 104),  # value in degree Fahrenheit
        ],
        [
            (-0.2, 0.2),  # change rate in degree Celsius
            (-0.4, 0.4),  # change rate in degree Fahrenheit
        ],
    ],
    "Highest Monthly Precipitation": [
        [
            (0, 400),  # value in mm
            (0, 16),  # value in inches
        ],
        [
            (-5, 5),  # change rate in mm
            (-0.2, 0.2),  # change rate in inches
        ],
    ],
    "Lowest Monthly Precipitation": [
        [
            (0, 100),  # value in mm
            (0, 4),  # value in inches
        ],
        [
            (-5, 5),  # change rate in mm
            (-0.2, 0.2),  # change rate in inches
        ],
    ],
}

COLOR_SCHEMES = {
    "Annual Mean Temperature": "Portland",
    "Annual Total Precipitation": "Earth",
    "Aridity Index": "RdYlBu",
    "Cryohumidity": "Spectral",
    "Continentality": "Cividis",
    "Seasonality": "tempo",
    "Lowest Monthly Temperature": "delta",
    "Highest Monthly Temperature": "Temps",
    "Highest Monthly Precipitation": "deep",
    "Lowest Monthly Precipitation": "speed",
}


@st.cache_resource
def load_resources() -> tuple[h5py.File, h5py.File, np.ndarray, np.ndarray, h5py.File]:
    data_file = h5py.File("dataset/climate_data_land.h5", "r", swmr=True)
    weight_file = h5py.File("weights.h5", "r", swmr=True)
    indices = data_file.get("indices")[:]
    elev = data_file.get("elev")[:]
    variable_file = h5py.File("dataset/climate_variables.h5", "r", swmr=True)
    return data_file, weight_file, indices, elev, variable_file


@st.cache_resource
def load_default_data(
    _data_file: h5py.File, _indices: np.ndarray, _elev: np.ndarray, _network: Network
) -> ClimateDataset:
    return calc_climate_normals(90, 30, _data_file, _indices, _elev, _network)


def calc_climate_normals(
    start_year: int,
    years: int,
    data_file: h5py.File,
    indices: np.ndarray,
    elev: np.ndarray,
    network: Network,
) -> ClimateDataset:
    res = get_average(start_year, years, data_file, indices, elev)
    res.prepare_dl(network, indices)
    res.prepare_variables()
    return res


@st.cache_resource
def get_simple_classifier(_weight_file: h5py.File) -> DLClassification:
    centroid = _weight_file.get("centroid_322")[:]
    return DLClassification(
        order=SIMPLE_ORDER,
        class_map=SIMPLE_MAP,
        color_map=SIMPLE_COLOR_MAP,
        centroid=centroid,
    )


@st.cache_resource
def get_detailed_classifier(_weight_file: h5py.File) -> DLClassification:
    centroid = _weight_file.get("centroid_detailed")[:]
    return DLClassification(
        order=DETAILED_ORDER,
        class_map=DETAILED_MAP,
        color_map=DETAILED_COLOR_MAP,
        centroid=centroid,
    )


@st.cache_resource
def get_network(_weight_file: h5py.File) -> Network:
    return Network(_weight_file)


def sync_slider():
    """同步滑动条和数字输入框的值"""
    if "year_slider" in st.session_state:
        st.session_state["start_year"] = st.session_state["year_slider"][0]
        st.session_state["end_year"] = st.session_state["year_slider"][1]
        st.session_state["year_range"] = st.session_state["year_slider"]
        st.session_state["settings_changed"] = True
        st.session_state["year_range_changed"] = True


def sync_number_input():
    """同步数字输入框到滑动条的值"""
    if "start_year" in st.session_state and "end_year" in st.session_state:
        st.session_state["year_slider"] = (
            st.session_state["start_year"],
            st.session_state["end_year"],
        )
        st.session_state["year_range"] = st.session_state["year_slider"]
        st.session_state["settings_changed"] = True
        st.session_state["year_range_changed"] = True


def sync_settings_changed():
    st.session_state["settings_changed"] = True


def clear_text_location():
    st.session_state["text_location"] = ""


def sync_update_map():
    clear_text_location()
    st.session_state["settings_changed"] = False


def find_point_index(fig, location) -> tuple[str, int] | tuple[None, None]:
    for trace in fig.data:
        for i in range(len(trace.lat)):
            if trace.lat[i] == location[0] and trace.lon[i] == location[1]:
                return (trace.name, i)
    return (None, None)


def update_points_dict(
    fig, points_location: list[tuple[float, float]]
) -> dict[tuple[str, int], tuple[float, float]]:
    res = {}
    for point in points_location:
        trace_name, point_index = find_point_index(fig, point)
        if trace_name is not None:
            res[(trace_name, point_index)] = point
    return res


def update_selected_points(fig, points_key: list[tuple[str, int]]):
    if not points_key:
        fig.update_traces(selectedpoints=None)
        return

    point_dict = {}
    for trace_name, point_index in points_key:
        if trace_name not in point_dict:
            point_dict[trace_name] = []
        point_dict[trace_name].append(point_index)

    for trace in fig.data:
        if trace.name in point_dict:
            fig.update_traces(
                selectedpoints=point_dict[trace.name], selector=dict(name=trace.name)
            )
        else:
            fig.update_traces(selectedpoints=[], selector=dict(name=trace.name))


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="ClimViz - World Climate Explorer")
    # st.logo("icon.png")
    # 注入自定义 CSS 来调整顶部间距
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
            }
        </style>
        <script>
            window.onerror = function(msg, url, lineNo, columnNo, error) {
                // 发送错误信息到 Streamlit
                window.parent.postMessage({
                    type: 'streamlit:error',
                    message: {
                        message: msg,
                        stack: error ? error.stack : '',
                        url: url,
                        line: lineNo,
                        column: columnNo        
                    }
                }, '*');
                return false;
            };
        </script>
    """,
        unsafe_allow_html=True,
    )
    # 设置页面标题
    st.title("ClimViz - World Climate Explorer")

    if "fig" not in st.session_state:
        st.session_state["fig"] = None

    if "year_range" not in st.session_state:
        st.session_state["year_range"] = (1991, 2020)

    if "selected_points" not in st.session_state:
        st.session_state["selected_points"] = {}

    if "text_location" not in st.session_state:
        st.session_state["text_location"] = ""

    if "climate_data" not in st.session_state:
        st.session_state["climate_data"] = None

    if "settings_changed" not in st.session_state:
        st.session_state["settings_changed"] = False

    if "year_range_changed" not in st.session_state:
        st.session_state["year_range_changed"] = False

    if "change_rate" not in st.session_state:
        st.session_state["change_rate"] = False

    data_file, weight_file, indices, elev, variable_file = load_resources()
    network = get_network(weight_file)
    default_data = load_default_data(data_file, indices, elev, network)
    simple_classifier = get_simple_classifier(weight_file)
    detailed_classifier = get_detailed_classifier(weight_file)

    gc.collect()

    with st.sidebar:
        # 侧边栏控件
        st.subheader("Select Display Content")
        st.radio(
            "zzz",
            ["Classification", "Variable"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key="cat_val",
            on_change=sync_settings_changed,
        )
        st.selectbox(
            "zzz",
            MAP_TYPES[st.session_state["cat_val"]],
            index=0,
            label_visibility="collapsed",
            key="map_type",
            on_change=sync_settings_changed,
        )

        if st.session_state["cat_val"] == "Variable":
            st.toggle(
                "Annual Change Rate",
                value=False,
                key="change_rate",
                on_change=sync_settings_changed,
            )

        st.markdown("---")  # 添加分隔线

        # 条件渲染 Köppen 设置
        if st.session_state["map_type"] == "Köppen-Geiger Classification":
            st.subheader("Köppen Classification Settings")  # 添加小标题
            st.radio(
                "Temperature threshold between **C** and **D**",
                ["0&deg;C", "-3&deg;C"],
                key="koppen_cd_mode",
                index=0,
                horizontal=True,
                on_change=sync_settings_changed,
            )
            st.radio(
                "Criterion between **Bh** and **Bk**",
                [
                    "annual mean temp 18&deg;C",
                    "coldest month %s" % st.session_state["koppen_cd_mode"],
                ],
                key="koppen_kh_mode",
                index=0,
                horizontal=True,
                on_change=sync_settings_changed,
            )
            st.markdown("---")  # 添加分隔线

        if (
            st.session_state["cat_val"] == "Variable"
            and st.session_state["change_rate"]
        ):
            st.subheader("During the period")
        else:
            st.subheader("Based on the climate normals")

        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "From",
                min_value=1901,
                max_value=LATEST_YEAR,
                value=st.session_state["year_range"][0],
                key="start_year",
                on_change=sync_number_input,
                help="Press Enter to apply",
            )

        with col2:
            st.number_input(
                "To",
                min_value=1901,
                max_value=LATEST_YEAR,
                value=st.session_state["year_range"][1],
                key="end_year",
                on_change=sync_number_input,
                help="Press Enter to apply",
            )

        st.slider(
            "zzz",
            min_value=1901,
            max_value=LATEST_YEAR,
            key="year_slider",
            value=st.session_state["year_range"],
            on_change=sync_slider,
            label_visibility="collapsed",
        )

        st.session_state["year_range"] = st.session_state["year_slider"]

        st.markdown("---")  # 添加分隔线

        st.write("Search for a location (e.g. London, UK)")
        st.caption("Press Enter to apply")
        new_location = st.text_input(
            label="zzz", value=None, key="text_location", label_visibility="collapsed"
        )
        if new_location and st.session_state["fig"] is not None:
            nearest_point = search_location(new_location)
            if nearest_point:
                trace_name, point_index = find_point_index(
                    st.session_state["fig"], nearest_point
                )
                if trace_name is not None and point_index is not None:
                    if (trace_name, point_index) not in st.session_state[
                        "selected_points"
                    ]:
                        if len(st.session_state["selected_points"]) < 3:
                            st.session_state["selected_points"][
                                (trace_name, point_index)
                            ] = nearest_point
                            update_selected_points(
                                st.session_state["fig"],
                                list(st.session_state["selected_points"].keys()),
                            )
                        else:
                            st.toast("At most 3 locations could be displayed")
                    # else:
                    #     st.toast("Location already selected")
                else:
                    st.toast("Coastal location. Try a nearby inland location.")
            else:
                st.toast("Location not found or network error")

        st.markdown("---")  # 添加分隔线

        if st.session_state["settings_changed"]:
            st.info("Click Update Map to apply new settings")

        st.toggle("Plot global trend", value=True, key="show_global_trend", help="This applies only when no points are selected on the map")
        st.toggle("Plot class probability", value=False, key="show_probability", help="This applies only when DECC or Predicted Land Cover is selected and there are points selected on the map")
        if st.session_state["map_type"] in [
            "Annual Mean Temperature",
            "Annual Total Precipitation",
            "Highest Monthly Temperature",
            "Lowest Monthly Temperature",
            "Highest Monthly Precipitation",
            "Lowest Monthly Precipitation",
        ]:
            st.toggle(
                "&deg;F/inch",
                value=False,
                key="unit",
                on_change=sync_settings_changed,
            )
        else:
            st.toggle("&deg;F/inch", value=False, key="unit")

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.button(
                "Update Map",
                key="update_btn",
                on_click=sync_update_map,
                help="Update the map with the current settings",
            )
            if st.session_state["year_range"][0] > st.session_state["year_range"][1]:
                st.toast("Start year must be less than end year")
                submitted = False
        with col2:
            if (
                st.button(
                    "Clear Locations",
                    key="clear_btn",
                    type="primary",
                    on_click=clear_text_location,
                    help="Clear all marked locations",
                )
                and st.session_state["fig"] is not None
            ):
                st.session_state["selected_points"] = {}
                update_selected_points(
                    st.session_state["fig"],
                    list(st.session_state["selected_points"].keys()),
                )

        # 在表单之后添加额外信息
        st.markdown("---")
        st.subheader("Click a scatter point on the map to select a location")
        st.subheader("Double click on one legend label to isolate the climate type")
        with st.expander("FAQ", icon=":material/help:"):
            st.markdown(
                "[What is Data-driven Ecological Classification - Advanced?](https://peace-van.github.io/climate/2023/11/23/sec7.html)"
            )
            st.markdown(
                "[What is Data-driven Ecological Classification - Basic?](https://peace-van.github.io/climate/2023/11/14/sec4.html)"
            )
            st.markdown(
                "What is Köppen-Geiger Classification? [My explanation](https://peace-van.github.io/climate/2023/11/05/koppen.html), [Wikipedia](https://en.wikipedia.org/wiki/K%C3%B6ppen_climate_classification)"
            )
            st.markdown(
                "[What is Trewartha Classification?](https://en.wikipedia.org/wiki/Trewartha_climate_classification)"
            )
            st.markdown(
                "[What are cryohumidity, continentality, and seasonality?](https://peace-van.github.io/climate/2023/11/14/sec4.html)"
            )
            st.markdown(
                "[What is aridity index?](https://en.wikipedia.org/wiki/Aridity_index) (We use Precipitation / PET)"
            )
            st.markdown(
                "How is annual change rate calculated? [Theil-Sen estimator](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator)"
            )
        st.markdown(
            """
            Data source: [CRU TS v4.08](https://crudata.uea.ac.uk/cru/data/hrg/), [GMTED2000](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-multi-resolution-terrain-elevation)
            """
        )

    if submitted or st.session_state["fig"] is None:
        # 清理旧的图表
        if "fig" in st.session_state:
            st.session_state["fig"] = None

        # 处理年份范围
        start_year_ = st.session_state["year_range"][0] - 1901  # 转换为数组索引
        years = (
            st.session_state["year_range"][1] - st.session_state["year_range"][0] + 1
        )

        if start_year_ == 90 and years == 30:
            st.session_state["climate_data"] = default_data
        elif st.session_state["year_range_changed"]:
            with st.spinner("Calculating... (may take one minute)"):
                st.session_state["climate_data"] = calc_climate_normals(
                    start_year_, years, data_file, indices, elev, network
                )
                st.session_state["year_range_changed"] = False
        gc.collect()

        with st.spinner("Preparing map data..."):
            if st.session_state["cat_val"] == "Classification":
                match st.session_state["map_type"]:
                    case "Köppen-Geiger Classification":
                        df = st.session_state["climate_data"].prepare_map_data(
                            st.session_state["map_type"],
                            st.session_state["koppen_cd_mode"],
                            st.session_state["koppen_kh_mode"],
                        )
                    case "Data-driven Ecological - Basic":
                        df = st.session_state["climate_data"].prepare_map_data(
                            st.session_state["map_type"],
                            dl_classifier=simple_classifier,
                        )
                    case "Data-driven Ecological - Advanced":
                        df = st.session_state["climate_data"].prepare_map_data(
                            st.session_state["map_type"],
                            dl_classifier=detailed_classifier,
                        )
                    case "Predicted Land Cover":
                        df = st.session_state["climate_data"].prepare_map_data(
                            st.session_state["map_type"], veg_names=VEG_MAP
                        )
                    case "Trewartha Classification":
                        df = st.session_state["climate_data"].prepare_map_data(
                            st.session_state["map_type"]
                        )
            else:
                if st.session_state["change_rate"]:
                    df = calc_change_rate(
                        variable_file,
                        indices,
                        elev,
                        st.session_state["map_type"],
                        start_year_,
                        years,
                        unit=st.session_state["unit"],
                    )
                else:
                    df = st.session_state["climate_data"].prepare_map_data(
                        st.session_state["map_type"], unit=st.session_state["unit"]
                    )

        gc.collect()

        if st.session_state["cat_val"] == "Classification":
            match st.session_state["map_type"]:
                case "Köppen-Geiger Classification":
                    cm = KoppenClassification.color_map
                    classes = KoppenClassification.order
                case "Trewartha Classification":
                    cm = TrewarthaClassification.color_map
                    classes = TrewarthaClassification.order
                case "Data-driven Ecological - Basic":
                    cm = simple_classifier.color_map
                    classes = simple_classifier.order
                case "Data-driven Ecological - Advanced":
                    cm = detailed_classifier.color_map
                    classes = detailed_classifier.order
                case "Predicted Land Cover":
                    cm = VEG_COLOR_MAP
                    classes = VEG_MAP

            fig = px.scatter_geo(
                df,
                lat="lat",
                lon="lon",
                color="value",
                color_discrete_map=cm,
                hover_data={"elev": True, "value": True},
                opacity=0.8,
                category_orders={"value": classes},
            )

            # 自定义悬停提示
            fig.update_traces(
                hovertemplate=(
                    # "point index: %{pointIndex}<br>" +
                    "lat: %{lat:.1f}<br>"
                    + "lon: %{lon:.1f}<br>"
                    + "elev: %{customdata[0]:.0f}m<br>"
                    + "type: %{customdata[1]}<br>"
                    + "<extra></extra>"  # 这行用来去除第二个框
                )
            )

            fig.update_layout(
                legend=dict(
                    title="",
                    itemsizing="constant",
                    font=dict(size=15),
                    yanchor="bottom",
                    y=0.01,
                )
            )
        else:
            fig = px.scatter_geo(
                df,
                lat="lat",
                lon="lon",
                color="value",
                color_continuous_scale=COLOR_SCHEMES[st.session_state["map_type"]],
                range_color=RANGES[st.session_state["map_type"]][
                    st.session_state["change_rate"]
                ][st.session_state["unit"]],
                hover_data={"elev": True, "value": True},
                opacity=0.8,
            )

            fig.update_traces(
                hovertemplate=(
                    # "point index: %{pointIndex}<br>" +
                    "lat: %{lat:.1f}<br>"
                    + "lon: %{lon:.1f}<br>"
                    + "elev: %{customdata[0]:.0f}m<br>"
                    + "value: %{customdata[1]:.2f}<br>"
                    + "<extra></extra>"
                )
            )
            # 更新图例样式
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="",
                    tickfont=dict(size=15),
                ),
                legend=dict(
                    yanchor="bottom",
                    y=0.01,
                ),
            )

        fig.update_geos(
            projection_type="equirectangular",  # 等距矩形投影
            showcoastlines=True,  # 显示海岸线
            coastlinecolor="Black",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue",
            showcountries=True,  # 显示国界线
            countrycolor="darkgray",  # 国界线颜色
            showframe=False,
            resolution=50,
            lonaxis_range=[-180, 180],  # 经度范围
            lataxis_range=[-60, 90],  # 纬度范围
        )

        fig.update_traces(
            marker=dict(size=5),
            unselected=dict(marker=dict(opacity=0.2)),
        )

        st.session_state["selected_points"] = update_points_dict(
            fig, list(st.session_state["selected_points"].values())
        )
        update_selected_points(fig, list(st.session_state["selected_points"].keys()))

        fig.update_layout(
            height=800,
            autosize=True,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            uirevision=True,
        )
        st.session_state["fig"] = fig
        gc.collect()

    if st.session_state["fig"] is not None:
        clicked_point = st.plotly_chart(
            st.session_state["fig"],
            use_container_width=True,
            on_select="rerun",  # 当有点击事件时重新运行 app
            selection_mode="points",
            key="map",  # 必须提供 key 参数
        )

        # 处理点击事件
        if clicked_point and clicked_point.selection.points:  # 检查是否有选择事件
            if len(clicked_point.selection.points) > 3:
                st.toast("At most 3 locations can be displayed")

            ps = clicked_point.selection.points[:3]
            # 保存选中的点的数据到 session_state
            st.session_state["selected_points"].update(
                {
                    (
                        point["legendgroup"] if "legendgroup" in point else "",
                        point["point_index"],
                    ): (point["lat"], point["lon"])
                    for point in ps
                }
            )
            update_selected_points(
                st.session_state["fig"],
                list(st.session_state["selected_points"].keys()),
            )
            st.rerun()

        cols = st.columns(3)

        # 如果有选中的点，显示气候图
        if st.session_state["selected_points"]:
            if len(st.session_state["selected_points"]) > 3:
                st.toast("At most 3 locations can be displayed")

            points_to_remove = []  # 记录需要移除的点

            for i, (point_key, point_location) in enumerate(
                st.session_state["selected_points"].items()
            ):
                customdata = []
                for trace in st.session_state["fig"].data:
                    if trace.name == point_key[0]:
                        customdata = trace.customdata[point_key[1]]
                        break

                subtitle = f"lat: {point_location[0]:.1f}, lon: {point_location[1]:.1f}, elev: {customdata[0]:.0f}m"
                if not st.session_state["change_rate"]:
                    if isinstance(customdata[1], str):
                        subtitle += f", type: {customdata[1]}"
                    else:
                        subtitle += f", value: {customdata[1]:.2f}"

                with cols[i]:
                    with st.empty():
                        with st.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.checkbox(
                                    "Auto scale axes",
                                    value=st.session_state["change_rate"],
                                    key=f"auto_scale_{i}",
                                    disabled=st.session_state["change_rate"],
                                )
                            with col2:
                                st.checkbox(
                                    "Local language", value=False, key=f"local_lang_{i}"
                                )
                            with col3:
                                st.checkbox(
                                    "July first",
                                    value=False,
                                    key=f"july_first_{i}",
                                    disabled=st.session_state["change_rate"],
                                )

                            if not st.session_state["change_rate"]:
                                if st.session_state["show_probability"] and (st.session_state["map_type"].startswith("Data-driven Ecological") or st.session_state["map_type"] == "Predicted Land Cover"):
                                    idx = np.where(
                                        (indices[:, 0] == point_location[0])
                                        & (indices[:, 1] == point_location[1])
                                    )[0]
                                    if st.session_state["map_type"].startswith("Data-driven Ecological"):
                                        dl_classifier = simple_classifier if st.session_state["map_type"].endswith("Basic") else detailed_classifier
                                        pca_features = st.session_state["climate_data"].pca_features[idx]
                                        probs = dl_classifier.probability(pca_features)[0]  # 获取单个位置的概率分布
                                        fig = create_probability_chart(
                                            probabilities=probs,
                                            class_map=dl_classifier.class_map,
                                            color_map=dl_classifier.color_map,
                                            location=point_location,
                                            subtitle=subtitle,
                                            local_lang=st.session_state[f"local_lang_{i}"],
                                        )
                                    else:
                                        probs = st.session_state["climate_data"].veg_probabilities[idx, :].squeeze()
                                        fig = create_probability_chart(
                                            probabilities=probs,
                                            class_map=VEG_MAP,
                                            color_map=VEG_COLOR_MAP,
                                            location=point_location,
                                            subtitle=subtitle,
                                            local_lang=st.session_state[f"local_lang_{i}"],
                                        )
                                else:
                                    fig = create_climate_chart(
                                        st.session_state["climate_data"][point_location],
                                        point_location,
                                        subtitle,
                                        st.session_state[f"local_lang_{i}"],
                                        st.session_state[f"july_first_{i}"],
                                        st.session_state["unit"],
                                        st.session_state[f"auto_scale_{i}"],
                                    )
                            else:
                                x = [i for i in range(1901, LATEST_YEAR + 1)]
                                idx = np.where(
                                    (indices[:, 0] == point_location[0])
                                    & (indices[:, 1] == point_location[1])
                                )[0]
                                y = variable_file.get("variables")[
                                    idx,
                                    :,
                                    VARIABLE_TYPE_INDICES[st.session_state["map_type"]],
                                ].squeeze()
                                st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key=f"mov_avg_{i}",
                                )
                                fig = create_variable_chart(
                                    y,
                                    point_location,
                                    subtitle,
                                    st.session_state["map_type"],
                                    st.session_state["unit"],
                                    st.session_state[f"local_lang_{i}"],
                                    mov_avg=st.session_state[f"mov_avg_{i}"],
                                )

                            st.plotly_chart(fig, use_container_width=True)

                            gc.collect()

                            with col4:
                                if st.button(
                                    "Clear",
                                    key=f"clear_{i}",
                                    type="primary",
                                    on_click=clear_text_location,
                                ):
                                    points_to_remove.append(point_key)

                if i == 2:
                    break

            if points_to_remove:
                for point_key in points_to_remove:
                    st.session_state["selected_points"].pop(point_key)

                # 更新地图的选中状态
                update_selected_points(
                    st.session_state["fig"],
                    list(st.session_state["selected_points"].keys()),
                )

                st.rerun()

        elif st.session_state["show_global_trend"]:
            x = [i for i in range(1901, LATEST_YEAR + 1)]
            global_avg = variable_file.get("variables")[-1, :, :]
            match st.session_state["map_type"]:
                case "Köppen-Geiger Classification" | "Trewartha Classification":
                    with cols[0]:
                        with st.empty():
                            with st.container():
                                # 绘制全球平均'Annual Mean Temperature'的折线图
                                y = global_avg[
                                    :, VARIABLE_TYPE_INDICES["Annual Mean Temperature"]
                                ]
                                st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_0",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "",
                                    "Annual Mean Temperature",
                                    st.session_state["unit"],
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_0"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    with cols[1]:
                        with st.empty():
                            with st.container():
                                # 绘制全球平均'Annual Total Precipitation'的折线图
                                y = global_avg[
                                    :,
                                    VARIABLE_TYPE_INDICES["Annual Total Precipitation"],
                                ]
                                st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_1",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "",
                                    "Annual Total Precipitation",
                                    st.session_state["unit"],
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_1"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    with cols[2]:
                        with st.empty():
                            with st.container():
                                # 绘制全球平均'Aridity Index'的折线图
                                y = global_avg[
                                    :, VARIABLE_TYPE_INDICES["Aridity Index"]
                                ]
                                st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_2",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "",
                                    "Aridity Index",
                                    False,
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_2"],
                                )
                                st.plotly_chart(fig, use_container_width=True)

                case (
                    "Data-driven Ecological - Basic"
                    | "Data-driven Ecological - Advanced"
                    | "Predicted Land Cover"
                ):
                    with cols[0]:
                        with st.empty():
                            with st.container():
                                # 绘制全球平均'Cryohumidity'的折线图
                                y = global_avg[:, VARIABLE_TYPE_INDICES["Cryohumidity"]]
                                st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_0",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "",
                                    "Cryohumidity",
                                    False,
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_0"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    with cols[1]:
                        with st.empty():
                            with st.container():
                                # 绘制全球平均'Continentality'的折线图
                                y = global_avg[
                                    :, VARIABLE_TYPE_INDICES["Continentality"]
                                ]
                                mov_avg = st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_1",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "",
                                    "Continentality",
                                    False,
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_1"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    with cols[2]:
                        with st.empty():
                            with st.container():
                                # 绘制全球平均'Seasonality'的折线图
                                y = global_avg[:, VARIABLE_TYPE_INDICES["Seasonality"]]
                                mov_avg = st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_2",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "",
                                    "Seasonality",
                                    False,
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_2"],
                                )
                                st.plotly_chart(fig, use_container_width=True)

                case _:
                    with cols[1]:
                        with st.empty():
                            with st.container():
                                # 绘制全球平均'选中变量'的折线图
                                y = global_avg[
                                    :,
                                    VARIABLE_TYPE_INDICES[st.session_state["map_type"]],
                                ]
                                st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_1",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "",
                                    st.session_state["map_type"],
                                    st.session_state["unit"],
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_1"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
