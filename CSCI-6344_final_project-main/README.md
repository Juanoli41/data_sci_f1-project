# F1 Telemetry Analysis & Driver Classification

## Project Overview
This project analyzes Formula 1 telemetry data using the `FastF1` library. The goal is to compare "Elite" vs "Midfield" drivers using various metrics and visualizations from the 2023 Formula 1 season, controlling for car performance to isolate driver skill differences.

## Features

### 🏎️ Dynamic Driver Classification
The project implements an intelligent team-based classification system that accounts for mid-season performance changes:
- **Elite Teams**: Red Bull, Ferrari, Mercedes
- **Dynamic Teams**: 
  - **McLaren**: Classified as Midfield for early 2023 races (before major upgrades at Austria/Silverstone), then Elite
  - **Aston Martin**: Classified as Elite for early 2023 races (strong podium contention), then Midfield
- **Midfield Teams**: Williams, Haas, Alpine, Alfa Romeo/Sauber, AlphaTauri/RB

### 📊 Multi-Race Analysis
The analysis covers four representative races from 2023, each representing different track characteristics:
- **Bahrain**: Balanced circuit with emphasis on traction
- **Monaco**: Street circuit with high downforce requirements
- **Silverstone**: High-speed cornering emphasis
- **Monza**: Low downforce, top-speed focused

### 📈 Comprehensive Visualizations
1. **Qualifying Gap Distribution**: Box plot comparing Elite vs Midfield lap time deltas
2. **Speed Trace Comparison**: Detailed speed profiles throughout a lap
3. **Top Speed vs. Delta Analysis**: Scatter plot examining speed trap data
4. **Throttle Application Distribution**: Box plot showing throttle consistency
5. **Speed Distribution Density**: Histogram comparing speed profiles by tier

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd CSCI-6344_final_project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Packages
- `fastf1`: Formula 1 telemetry and timing data
- `pandas`: Data manipulation and analysis
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `numpy`: Numerical computing
- `pyarrow`: Efficient data serialization (for Parquet format)

## Usage

### Running the Analysis
1. Open `f1_analysis.ipynb` in Jupyter Notebook or VS Code with Jupyter extension
2. Run all cells sequentially (Ctrl+Shift+P → "Run All Cells")

### Data Collection
The notebook automatically:
- Downloads telemetry data from FastF1 API
- Caches data locally in `fastf1_cache/` directory to avoid repeated downloads
- Processes and validates telemetry for each driver
- Exports processed data to CSV/Parquet files

### Output Files
- `processed_laps.csv`: Aggregated lap metrics (gap to pole, speed trap, tier classification)
- `processed_telemetry.parquet` or `processed_telemetry.csv`: High-frequency telemetry data (speed, throttle, distance)

## Methodology

### Data Validation
The code implements strict data quality checks:
1. **Valid Lap Time**: Filters out laps with missing or zero lap times
2. **Valid Telemetry**: Ensures telemetry data exists for each lap
3. **Clean Missing Values**: Drops laps with >10% missing data points

### Dynamic Team Tier Logic
The classification function `get_driver_tier(team_name, year, gp)` dynamically assigns tiers based on:
- Team name
- Season year
- Specific Grand Prix (to account for mid-season upgrades)

This approach controls for the dominant variable in F1 performance (the car) while allowing analysis of driver-specific skills.

## Project Structure
```
CSCI-6344_final_project/
├── f1_analysis.ipynb          # Main analysis notebook
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── processed_laps.csv          # Generated: Lap metrics
├── fastf1_cache/              # Generated: FastF1 API cache
└── processed_telemetry.*      # Generated: Telemetry data
```

## Key Findings
The analysis enables comparison of:
- **Qualifying Performance**: Average gap to pole position by tier
- **Speed Profiles**: Cornering and straight-line speed differences
- **Throttle Application**: Consistency in throttle control
- **Braking Performance**: Braking points and profiles (via speed traces)

## Notes

### Why Team-Based Classification?
F1 performance is dominated by car performance. Comparing a Williams driver directly to a Red Bull driver on raw lap time measures the car more than the driver. By grouping into "Elite" vs "Midfield" machinery, we can identify driver skill patterns that transcend equipment differences.

### Cache Management
- FastF1 downloads can be large (hundreds of MB)
- The `fastf1_cache/` directory stores `.ff1pkl` files to avoid re-downloading
- Cache is excluded from git via `.gitignore`
- To refresh data, delete the cache directory and re-run the notebook

### Performance Considerations
- Initial data download may take 10-15 minutes depending on internet speed
- Subsequent runs use cached data and complete in seconds
- Telemetry concatenation for multiple races may take 30-60 seconds

## Contributing
This project was developed as part of CSCI-6344. For questions or suggestions, please open an issue or contact the project maintainer.

## License
This project is for educational purposes as part of a university course.

## Acknowledgments
- [FastF1 Library](https://github.com/theOehrly/Fast-F1) for providing access to F1 telemetry data
- Formula 1 for making historical data available
