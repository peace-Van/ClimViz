# ClimViz - World Climate Explorer

![DeepEcoClimateLogo](logo.png)

![DeepEcoClimateMap](world.png)

[ClimViz](https://climviz.streamlit.app/) is an interactive web application for exploring global climate data, visualizing climate classification world maps, and analyzing climate variables and trends. It leverages deep learning, advanced climate indices, and interactive charts to provide a comprehensive climate data exploration experience.

## Features

- **Interactive World Map**: Visualize climate classifications (DeepEcoClimate, Köppen-Geiger, Trewartha) and climate variables (temperature, precipitation, aridity, etc.) on a global scale.
- **Customizable Time Range**: Select any period from 1901 to 2024 for analysis.
- **Change Rate Analysis**: View annual change rates for climate variables (Theil-Sen estimator).
- **Location Search**: Find and analyze climate data for any location worldwide.
- **Detailed Climate Charts**: Generate and download climate charts and data for selected locations.
- **Global Trends**: Visualize global average trends for key climate variables.
- **Probability Visualization**: For DeepEcoClimate, view class probability distributions for selected points.

## Data Sources

- Climate normals [CRU TS v4.09](https://crudata.uea.ac.uk/cru/data/hrg/)
- Land cover [MCD12C1](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd12c1-061)
- Elevation [GMTED2010](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-global-multi-resolution-terrain-elevation)
- For the compiled dataset, see [here](https://data.mendeley.com/datasets/dnk6839b86/2)

## Deploy Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/peace-Van/ClimViz.git
   cd ClimViz
   ```

2. **(Recommended) Create and activate a virtual environment:**
   - Using `venv` (standard Python):
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Or using `conda`:
     ```bash
     conda create -n climviz python=3.10
     conda activate climviz
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## DeepEcoClimate deep learning model usage

```python
    from TorchModel import DLModel
    from climate_classification import DLClassification
    model = DLModel('cpu', 'inference')
    model = torch.compile(model)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    # data shape: (batch, 3, 12)
    # 3 rows are mean daily minimum temperature (°C), precipitation (mm), mean daily maximum temperature (°C)
    # 12 columns are 12 months
    thermal, aridity, class_probabilities = model(data)
    DEC_types = DLClassification.classify(class_probabilities)
```

## Usage

- Use the sidebar to select the type of map (classification or variable), time range, and other settings.
- Click on the map to select up to 3 locations for detailed analysis (display climate charts and class probabilities for DeepEcoClimate).
- Use the search box to find locations by name.
- Download climate data for selected locations as CSV files.
- Toggle between units (°C/mm and °F/inch).
- View global trends of climate variables.

## Project Structure

```
.
├── app.py                # Main Streamlit app
├── backend.py            # Data processing and chart generation
├── TorchModel.py         # Deep learning model definition
├── structured_kmeans.py  # Structured KMeans clustering
├── climate_classification.py # Climate classification logic for Köppen-Geiger, Trewartha and DeepEcoClimate
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
└── model.pth             # Trained DeepEcoClimate model weights
└── world.png             # Header image: DeepEcoClimate world map for 1995-2024 climate normals
└── LICENSE               # GNU GPLv3 License
```

## Citation

If you use DeepEcoClimate in your research, please cite this repo and/or [ClimCalc](https://github.com/peace-Van/ClimCalc/tree/main) repo.

## License

This project is licensed under the [GNU GPLv3 License](LICENSE).

## Related Project

[ClimCalc](https://climcalc.streamlit.app/) - an interactive web app to get the climate type for your location or data

## Acknowledgements

- Data provided by CRU TS, MCD12C1 and GMTED2010
- Deep learning and clustering powered by MATLAB and PyTorch
- Web-based visualization powered by Streamlit and Plotly

