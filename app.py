import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import torch
from backend import (
    ClimateData,
    ClimateDataset,
    get_average,
    create_climate_chart,
    calc_change_rate,
    LocationService,
    create_variable_chart,
    create_probability_chart,
    VARIABLE_TYPE_INDICES,
    LATEST_YEAR,
    lighten, 
)
import h5py
from climate_classification import (
    KoppenClassification,
    TrewarthaClassification,
    DLClassification,
)
from TorchModel import DLModel
import gc
torch.classes.__path__ = []


MAP_TYPES = {
    "Classification": [
        "DeepEcoClimate",
        "Köppen-Geiger Classification",
        "Trewartha Classification",
    ],
    "Variable": [
        "DeepEcoClimate Class Probability",
        "Thermal Index",
        "Thermal Index (Discretized)", 
        "Aridity Index",
        "Aridity Index (Discretized)", 
        "Elevation",
        "Annual Mean Temperature",
        "Annual Total Precipitation",
        "Coldest Month Mean Temperature",
        "Hottest Month Mean Temperature",
        "Coldest Month Mean Daily Minimum",
        "Hottest Month Mean Daily Maximum",
        "Wettest Month Precipitation",
        "Driest Month Precipitation",
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
            (-2, 2),  # value
            (-2, 2),
        ],
        [
            (-0.04, 0.04),  # change rate
            (-0.04, 0.04),
        ],
    ],
    "Thermal Index": [
        [
            (-2, 2),  # value
            (-2, 2),
        ],
        [
            (-0.02, 0.02),  # change rate
            (-0.02, 0.02),
        ],
    ],
    "Coldest Month Mean Temperature": [
        [
            (-30, 30),  # value in degree Celsius
            (-22, 86),  # value in degree Fahrenheit
        ],
        [
            (-0.2, 0.2),  # change rate in degree Celsius
            (-0.4, 0.4),  # change rate in degree Fahrenheit
        ],
    ],
    "Hottest Month Mean Temperature": [
        [
            (-30, 30),  # value in degree Celsius
            (-22, 86),  # value in degree Fahrenheit
        ],
        [
            (-0.2, 0.2),  # change rate in degree Celsius
            (-0.4, 0.4),  # change rate in degree Fahrenheit
        ],
    ],
    "Wettest Month Precipitation": [
        [
            (0, 400),  # value in mm
            (0, 16),  # value in inches
        ],
        [
            (-5, 5),  # change rate in mm
            (-0.2, 0.2),  # change rate in inches
        ],
    ],
    "Driest Month Precipitation": [
        [
            (0, 100),  # value in mm
            (0, 4),  # value in inches
        ],
        [
            (-2, 2),  # change rate in mm
            (-0.1, 0.1),  # change rate in inches
        ],
    ],
    "Coldest Month Mean Daily Minimum": [
        [
            (-40, 20),  # value in degree Celsius
            (-40, 68),  # value in degree Fahrenheit
        ],
        [
            (-0.2, 0.2),  # change rate in degree Celsius
            (-0.4, 0.4),  # change rate in degree Fahrenheit
        ],
    ],
    "Hottest Month Mean Daily Maximum": [
        [
            (0, 40),  # value in degree Celsius
            (32, 104),  # value in degree Fahrenheit
        ],
        [
            (-0.2, 0.2),  # change rate in degree Celsius
            (-0.4, 0.4),  # change rate in degree Fahrenheit
        ],
    ],
}

COLOR_SCHEMES = {
    "Annual Mean Temperature": "Portland",
    "Annual Total Precipitation": "Earth",
    "Aridity Index": "Geyser",
    "Aridity Index (Discretized)": {
        "humid": "rgb(0, 128, 128)",
        "sub-humid": "rgb(112, 164, 148)",
        "semi-arid": "rgb(246, 237, 189)",
        "arid": "rgb(202, 86, 44)",
    },
    "Thermal Index": "balance",
    "Thermal Index (Discretized)": {
        "arctic/subarctic": "rgb(41, 58, 143)",
        "cool temperate": "rgb(69, 144, 185)",
        "warm temperate": "rgb(230, 210, 204)",
        "tropical/subtropical": "rgb(172, 43, 36)",
    },
    "Coldest Month Mean Temperature": "delta",
    "Hottest Month Mean Temperature": "Temps",
    "Wettest Month Precipitation": "deep",
    "Driest Month Precipitation": "speed",
    "Coldest Month Mean Daily Minimum": "delta",
    "Hottest Month Mean Daily Maximum": "Temps",
    "Elevation": "Cividis",
}


@st.cache_resource
def load_resources() -> tuple[h5py.File, np.ndarray, np.ndarray, h5py.File, LocationService]:
    data_file = h5py.File("dataset/climate_data_land.h5", "r", swmr=True)
    indices = data_file.get("indices")[:]
    elev = data_file.get("elev")[:].squeeze()
    variable_file = h5py.File("dataset/climate_variables.h5", "r", swmr=True)
    locationService = LocationService()
    return data_file, indices, elev, variable_file, locationService


@st.cache_resource
def load_default_data(
    _data_file: h5py.File, _indices: np.ndarray, _elev: np.ndarray, _network: DLModel
) -> ClimateDataset:
    return calc_climate_normals(94, 30, _data_file, _indices, _elev, _network)


def calc_climate_normals(
    start_year: int,
    years: int,
    data_file: h5py.File,
    indices: np.ndarray,
    elev: np.ndarray,
    network: DLModel,
) -> ClimateDataset:
    res = get_average(start_year, years, data_file, indices, elev)
    res.prepare_dl(network, indices)
    res.prepare_variables()
    return res


@st.cache_resource
def get_network(_weight_file: str) -> DLModel:
    model = DLModel('cpu', 'inference')
    model = torch.compile(model)
    model.load_state_dict(torch.load(_weight_file, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_download_data(climate_data: ClimateData) -> tuple[float, str]:
    elev = climate_data.get_elev()
    data = climate_data.get_all_data()
    data = pd.DataFrame(data, columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], index=["tmp", "tmn", "tmx", "pre", "pet"])
    return elev, data.to_csv().encode('utf-8')


def sync_slider():
    """synchronize the value of the slider and the number input box"""
    if "year_slider" in st.session_state:
        st.session_state["start_year"] = st.session_state["year_slider"][0]
        st.session_state["end_year"] = st.session_state["year_slider"][1]
        st.session_state["year_range"] = st.session_state["year_slider"]
        st.session_state["settings_changed"] = True
        st.session_state["year_range_changed"] = True


def sync_number_input():
    """synchronize the value of the number input box to the slider"""
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
    # CSS injection to adjust the top spacing
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
                // send error information to Streamlit
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
    # set the page title
    st.title("ClimViz - World Climate Explorer")

    if "fig" not in st.session_state:
        st.session_state["fig"] = None

    if "year_range" not in st.session_state:
        st.session_state["year_range"] = (1995, 2024)

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

    data_file, indices, elev, variable_file, locationService = load_resources()
    network = get_network("model.pth")
    default_data = load_default_data(data_file, indices, elev, network)

    gc.collect()

    with st.sidebar:
        # sidebar widgets
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

        if st.session_state["map_type"] == "DeepEcoClimate Class Probability":
            st.selectbox(
                "Select climate type", 
                DLClassification.order,  
                index=0, 
                key="selected_class", 
                on_change=sync_settings_changed,
            )

        elif (
            st.session_state["cat_val"] == "Variable"
            and "Discretized" not in st.session_state["map_type"]
            and st.session_state["map_type"] != "Elevation"
        ):
            st.toggle(
                "Annual Change Rate",
                value=False,
                key="change_rate",
                on_change=sync_settings_changed,
            )

        # Elevation max range control
        if (
            st.session_state["cat_val"] == "Variable"
            and st.session_state["map_type"] == "Elevation"
        ):
            st.slider(
                "Max elevation (m)",
                min_value=1000,
                max_value=5500,
                step=500,
                value=5000,
                key="elev_max",
                on_change=sync_settings_changed,
            )

        st.markdown("---")  # add a separator

        # conditional rendering Köppen settings
        if st.session_state["map_type"] == "Köppen-Geiger Classification":
            st.subheader("Köppen Classification Settings")  # add a subheader
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
            st.markdown("---")  # add a separator

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
                disabled=st.session_state["map_type"] == "Elevation",
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
                disabled=st.session_state["map_type"] == "Elevation",
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
            disabled=st.session_state["map_type"] == "Elevation",
        )

        st.session_state["year_range"] = st.session_state["year_slider"]

        st.markdown("---")  # add a separator

        st.write("Search for a location (e.g. London, UK)")
        st.caption("Press Enter to apply")
        new_location = st.text_input(
            label="zzz", value=None, key="text_location", label_visibility="collapsed"
        )
        if new_location and st.session_state["fig"] is not None:
            nearest_point = locationService.search_location(new_location)
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
                else:
                    st.toast("Coastal location. Try a nearby inland location.")
            else:
                st.toast("Location not found or network error")

        st.markdown("---")  # add a separator

        if st.session_state["settings_changed"]:
            st.info("Click Update Map to apply new settings")

        st.toggle(
            "Plot global trend", 
            value=False, 
            key="show_global_trend", 
            help="This applies only when no points are selected on the map", 
            # disabled=len(st.session_state["selected_points"]) > 0, 
            )
        st.toggle(
            "Plot class probability for locations", 
            value=False, 
            key="show_probability", 
            help="This applies only when DeepEcoClimate is selected and there are points selected on the map", 
            # disabled=len(st.session_state["selected_points"]) == 0 or "DeepEcoClimate" not in st.session_state["map_type"], 
            )
        if st.session_state["map_type"] in [
            "Annual Mean Temperature",
            "Annual Total Precipitation",
            "Coldest Month Mean Temperature",
            "Hottest Month Mean Temperature",
            "Coldest Month Mean Daily Minimum",
            "Hottest Month Mean Daily Maximum",
            "Wettest Month Precipitation",
            "Driest Month Precipitation",
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

        # add extra information after the form
        st.markdown("---")
        st.subheader("Click a scatter point on the map to show a location")
        st.subheader("Double click on one legend label to isolate the climate type")
        # with st.expander("FAQ", icon=":material/help:"):
        #     st.markdown(
        #         "[What is Data-driven Ecological Classification - Advanced?](https://peace-van.github.io/climate/2023/11/23/sec7.html)"
        #     )
        #     st.markdown(
        #         "[What is Data-driven Ecological Classification - Basic?](https://peace-van.github.io/climate/2023/11/14/sec4.html)"
        #     )
        #     st.markdown(
        #         "What is Köppen-Geiger Classification? [My explanation](https://peace-van.github.io/climate/2023/11/05/koppen.html), [Wikipedia](https://en.wikipedia.org/wiki/K%C3%B6ppen_climate_classification)"
        #     )
        #     st.markdown(
        #         "[What is Trewartha Classification?](https://en.wikipedia.org/wiki/Trewartha_climate_classification)"
        #     )
        #     st.markdown(
        #         "[What are cryohumidity, continentality, and seasonality?](https://peace-van.github.io/climate/2023/11/14/sec4.html)"
        #     )
        #     st.markdown(
        #         "[What is aridity index?](https://en.wikipedia.org/wiki/Aridity_index) (We use Precipitation / PET)"
        #     )
        #     st.markdown(
        #         "How is annual change rate calculated? [Theil-Sen estimator](https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator)"
        #     )
        st.markdown(
            """
            Data source: [CRU TS v4.09](https://crudata.uea.ac.uk/cru/data/hrg/)
            """
        )

        st.page_link("https://climcalc.streamlit.app", icon=":material/query_stats:", label="ClimCalc", help="Get the climate type for your location or data")
        st.page_link("https://www.youtube.com/watch?v=bF0Mck-yqhw", icon=":material/play_circle:", label="YouTube", help="Watch the introduction video")

    if submitted or st.session_state["fig"] is None:
        # clean the old chart
        if "fig" in st.session_state:
            st.session_state["fig"] = None

        # handle the year range
        start_year_ = st.session_state["year_range"][0] - 1901  # convert to array index
        years = (
            st.session_state["year_range"][1] - st.session_state["year_range"][0] + 1
        )

        if start_year_ == 94 and years == 30:
            st.session_state["climate_data"] = default_data
        elif st.session_state["year_range_changed"]:
            with st.spinner("Calculating..."):
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
                    case "DeepEcoClimate":
                        df = st.session_state["climate_data"].prepare_map_data(
                            st.session_state["map_type"],
                        )
                    case "Trewartha Classification":
                        df = st.session_state["climate_data"].prepare_map_data(
                            st.session_state["map_type"]
                        )
            else:
                if st.session_state["change_rate"] and st.session_state["map_type"] != "Elevation":
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
                case "DeepEcoClimate":
                    cm = DLClassification.color_map
                    classes = DLClassification.order

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

            # custom hover tooltip
            fig.update_traces(
                hovertemplate=(
                    # "point index: %{pointIndex}<br>" +
                    "lat: %{lat:.2f}<br>"
                    + "lon: %{lon:.2f}<br>"
                    + "elev: %{customdata[0]:.0f}m<br>"
                    + "type: %{customdata[1]}<br>"
                    + "<extra></extra>"  # this line is used to remove the second box
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
            if st.session_state["map_type"] == "Thermal Index (Discretized)":
                fig = px.scatter_geo(
                    df,
                    lat="lat",
                    lon="lon",
                    color="value",
                    color_discrete_map=COLOR_SCHEMES["Thermal Index (Discretized)"],
                    hover_data={"elev": True, "value": True},
                    opacity=0.8,
                    category_orders={"value": ["arctic/subarctic", "cool temperate", "warm temperate", "tropical/subtropical"]},
                )
                fig.update_traces(
                    hovertemplate=(
                        "lat: %{lat:.2f}<br>"
                        + "lon: %{lon:.2f}<br>"
                        + "elev: %{customdata[0]:.0f}m<br>"
                        + "zone: %{customdata[1]}<br>"
                        + "<extra></extra>"
                    )
                )
                fig.update_layout(
                    legend=dict(
                        title="",
                        itemsizing="constant",
                        font=dict(size=15),
                        yanchor="bottom",
                        y=0.01,
                    ),
                    coloraxis_showscale=False,
                )
            elif st.session_state["map_type"] == "Aridity Index (Discretized)":
                fig = px.scatter_geo(
                    df,
                    lat="lat",
                    lon="lon",
                    color="value",
                    color_discrete_map=COLOR_SCHEMES["Aridity Index (Discretized)"],
                    hover_data={"elev": True, "value": True},
                    opacity=0.8,
                    category_orders={"value": ["humid", "sub-humid", "semi-arid", "arid"]},
                )
                fig.update_traces(
                    hovertemplate=(
                        "lat: %{lat:.2f}<br>"
                        + "lon: %{lon:.2f}<br>"
                        + "elev: %{customdata[0]:.0f}m<br>"
                        + "zone: %{customdata[1]}<br>"
                        + "<extra></extra>"
                    )
                )
                fig.update_layout(
                    legend=dict(
                        title="",
                        itemsizing="constant",
                        font=dict(size=15),
                        yanchor="bottom",
                        y=0.01,
                    ),
                    coloraxis_showscale=False,
                )
            elif st.session_state["map_type"] == "DeepEcoClimate Class Probability":
                class_name = st.session_state["selected_class"]
                df = st.session_state["climate_data"].prepare_map_data("DeepEcoClimate Class Probability", class_name=class_name)
                main_color = DLClassification.color_map[class_name]
                light_color = '#%02x%02x%02x' % tuple(int(x*255) for x in lighten(main_color, 0.02))
                fig = px.scatter_geo(
                    df,
                    lat="lat",
                    lon="lon",
                    color="value",
                    color_continuous_scale=[(0, light_color), (1, main_color)],
                    range_color=(0, 1),
                    hover_data={"elev": True, "value": True},
                    opacity=0.8,
                )
                fig.update_traces(
                    hovertemplate=(
                        "lat: %{lat:.2f}<br>"
                        + "lon: %{lon:.2f}<br>"
                        + "elev: %{customdata[0]:.0f}m<br>"
                        + f"{class_name} probability: %{{customdata[1]:.2f}}<br>"
                        + "<extra></extra>"
                    )
                )
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title=f"{class_name} probability",
                        tickfont=dict(size=15),
                    ),
                    legend=dict(
                        yanchor="bottom",
                        y=0.01,
                    ),
                )
            else:
                if st.session_state["map_type"] == "Elevation":
                    # Apply user-defined max range for elevation
                    elev_max = st.session_state.get("elev_max", 5000)
                    fig = px.scatter_geo(
                        df,
                        lat="lat",
                        lon="lon",
                        color="value",
                        color_continuous_scale=COLOR_SCHEMES[st.session_state["map_type"]],
                        range_color=(0, elev_max),
                        hover_data={"elev": True, "value": True},
                        opacity=0.8,
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
                        "lat: %{lat:.2f}<br>"
                        + "lon: %{lon:.2f}<br>"
                        + "elev: %{customdata[0]:.0f}m<br>"
                        + "value: %{customdata[1]:.2f}<br>"
                        + "<extra></extra>"
                    )
                )
                # update the legend style
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
            projection_type="equirectangular",  # equirectangular projection
            showcoastlines=True,  # show the coastline
            coastlinecolor="Black",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue",
            showcountries=True,  # show the country boundary
            countrycolor="darkgray",  # country boundary color
            showframe=False,
            resolution=50,
            lonaxis_range=[-180, 180],  # longitude range
            lataxis_range=[-60, 90],  # latitude range
        )

        fig.update_traces(
            marker=dict(size=4),
            unselected=dict(marker=dict(opacity=0.25)),
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
            on_select="rerun",  # when there is a click event, rerun the app
            selection_mode="points",
            key="map",  # must provide the key parameter
        )

        # handle the click event
        if clicked_point and clicked_point.selection.points:  # check if there is a selection event
            if len(clicked_point.selection.points) > 3:
                st.toast("At most 3 locations can be displayed")

            ps = clicked_point.selection.points[:3]
            # save the selected points data to session_state
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

        # if there are selected points, show the climate chart
        if st.session_state["selected_points"]:
            if len(st.session_state["selected_points"]) > 3:
                st.toast("At most 3 locations can be displayed")

            cols = st.columns(3)

            points_to_remove = []  # record the points to remove

            for i, (point_key, point_location) in enumerate(
                st.session_state["selected_points"].items()
            ):
                customdata = []
                for trace in st.session_state["fig"].data:
                    if trace.name == point_key[0]:
                        customdata = trace.customdata[point_key[1]]
                        break

                subtitle = f"lat: {point_location[0]:.2f}, lon: {point_location[1]:.2f}, elev: {float(customdata[0]):.0f}m"
                if not st.session_state["change_rate"]:
                    if isinstance(customdata[1], str):
                        subtitle += f", type: {customdata[1]}"
                    else:
                        subtitle += f", value: {customdata[1]:.2f}"

                with cols[i]:
                    with st.empty():
                        with st.container():
                            download_data = None
                            elev_ = None
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.checkbox(
                                    "Auto scale axes",
                                    value=st.session_state["change_rate"],
                                    key=f"auto_scale_{i}",
                                    disabled=st.session_state["change_rate"] or st.session_state["show_probability"],
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
                                    disabled=st.session_state["change_rate"] or st.session_state["show_probability"],
                                )

                            if not st.session_state["change_rate"]:
                                title = locationService.get_location_info(point_location, st.session_state[f"local_lang_{i}"])

                                if st.session_state["show_probability"] and "DeepEcoClimate" in st.session_state["map_type"]:
                                    # print(st.session_state["climate_data"].data[point_location].get_dl_data())
                                    idx = np.where(
                                        (indices[:, 0] == point_location[0])
                                        & (indices[:, 1] == point_location[1])
                                    )[0]
                                    # if st.session_state["map_type"] == "DeepEcoClimate":
                                    thermal_index = st.session_state["climate_data"].thermal_index[idx][0]
                                    aridity_index = st.session_state["climate_data"].aridity_index[idx][0]

                                    # print(thermal_index, aridity_index)
                                    probs = st.session_state["climate_data"].probabilities[idx, :].squeeze()
                                    fig = create_probability_chart(
                                        probabilities=probs,
                                        class_map=DLClassification.class_map,
                                        color_map=DLClassification.color_map,
                                        title=title,
                                        subtitle=f"Thermal Index: {thermal_index:.2f}, Aridity Index: {aridity_index:.2f}",
                                    )
                                else:
                                    fig = create_climate_chart(
                                        st.session_state["climate_data"][point_location],
                                        title=title,
                                        subtitle=subtitle,
                                        july_first=st.session_state[f"july_first_{i}"],
                                        unit=st.session_state["unit"],
                                        auto_scale=st.session_state[f"auto_scale_{i}"],
                                    )
                                    elev_, download_data = get_download_data(st.session_state["climate_data"][point_location])
                            else:
                                x = [i for i in range(1901, LATEST_YEAR + 1)]
                                idx = np.where(
                                    (indices[:, 0] == point_location[0])
                                    & (indices[:, 1] == point_location[1])
                                )[0]
                                y = variable_file.get("res")[
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
                                    title,
                                    subtitle,
                                    st.session_state["map_type"],
                                    st.session_state["unit"],
                                    mov_avg=st.session_state[f"mov_avg_{i}"],
                                )

                            st.plotly_chart(fig, use_container_width=True)

                            if download_data:
                                col1_, col2_, col3_ = st.columns(3)
                                with col2_:
                                    st.download_button(
                                        label="Download data (.csv)",
                                        data=download_data,
                                        file_name=f"{title} ({elev_:.0f}m).csv",
                                        mime="text/csv",
                                        icon=":material/download:",
                                        help="All downloaded data use &deg;C/mm unit. For full dataset download, please see [here](https://data.mendeley.com/datasets/dnk6839b86/2).",
                                        use_container_width=True,
                                    )

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

                # update the selected state of the map
                update_selected_points(
                    st.session_state["fig"],
                    list(st.session_state["selected_points"].keys()),
                )

                st.rerun()

        elif st.session_state["show_global_trend"]:
            x = [i for i in range(1901, LATEST_YEAR + 1)]
            global_avg = variable_file.get("res")[-1, :, :]
            match st.session_state["map_type"]:
                case "Köppen-Geiger Classification" | "Trewartha Classification":
                    cols = st.columns(3)
                    with cols[0]:
                        with st.empty():
                            with st.container():
                                # draw the line chart of the global average 'Annual Mean Temperature'
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
                                    "Global Average Annual Mean Temperature",
                                    "",
                                    "Annual Mean Temperature",
                                    st.session_state["unit"],
                                    mov_avg=st.session_state["mov_avg_global_0"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    with cols[2]:
                        with st.empty():
                            with st.container():
                                # draw the line chart of the global average 'Annual Total Precipitation'
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
                                    "Global Average Annual Total Precipitation",
                                    "",
                                    "Annual Total Precipitation",
                                    st.session_state["unit"],
                                    mov_avg=st.session_state["mov_avg_global_1"],
                                )
                                st.plotly_chart(fig, use_container_width=True)

                case "DeepEcoClimate" | "DeepEcoClimate Class Probability":
                    cols = st.columns(3)
                    with cols[0]:
                        with st.empty():
                            with st.container():
                                # draw the line chart of the global average 'Thermal Index'
                                y = global_avg[:, VARIABLE_TYPE_INDICES["Thermal Index"]]
                                st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_0",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "Global Average Thermal Index",
                                    "",
                                    "Thermal Index",
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_0"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    with cols[2]:
                        with st.empty():
                            with st.container():
                                # draw the line chart of the global average 'Aridity Index'
                                y = global_avg[
                                    :, VARIABLE_TYPE_INDICES["Aridity Index"]
                                ]
                                mov_avg = st.toggle(
                                    "30-year moving average",
                                    value=True,
                                    key="mov_avg_global_1",
                                )
                                fig = create_variable_chart(
                                    y,
                                    None,
                                    "Global Average Aridity Index",
                                    "",
                                    "Aridity Index",
                                    False,
                                    mov_avg=st.session_state["mov_avg_global_1"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                case _:
                    cols = st.columns(3)
                    with cols[1]:
                        with st.empty():
                            with st.container():
                                # draw the line chart of the global average 'selected variable'
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
                                    "Global Average " + st.session_state["map_type"],
                                    "",
                                    st.session_state["map_type"],
                                    st.session_state["unit"],
                                    mov_avg=st.session_state["mov_avg_global_1"],
                                )
                                st.plotly_chart(fig, use_container_width=True)
