# LNA Survey Visualizer

An interactive visualization tool for Low Noise Amplifier (LNA) survey data, built with Dash and Plotly.

> **Note:** This project was developed as a side exploration during a course, experimenting with interactive data visualization for RF/microwave component analysis.

## 📚 Original Survey Citation

This visualization is based on the LNA survey compiled by **ETH Zürich Integrated Devices, Electronics, And Systems (IDEAS)**:

> **[LNA Survey - ETH Zürich IDEAS](https://ideas.ethz.ch/Surveys/lna-survey.html)**

If you use this data in your research, please cite the original survey from ETH Zürich IDEAS.

## 🌐 Live Demo

**[View the Interactive Visualization →](https://aissus.github.io/LNA_survey_Viz/)**

## Features

- **Linked Scatter Plots**: Explore NF, Gain, Power, IIP3, and FOM vs Frequency
- **Gain vs NF Cartesian Plot**: Direct comparison of key LNA metrics
- **Interactive Filters**:
  - Technology/Process selection
  - Frequency range (log scale with octave ticks)
  - Year range
  - Metric-specific limits (NF, Gain, Power)
- **Point Selection**:
  - Click to select individual points
  - Box/lasso selection for multiple points
  - Selection synced across all charts
- **Detailed Info Panel**: View paper details with Google Scholar link
- **Manual Data Points**: Add your own design points for comparison
- **Minor Grid Toggle**: Optional minor gridlines for precise reading
- **SVG/PNG Export**: Download charts via the modebar

## Installation

```bash
# Clone and navigate to the directory
cd examples/Survey_Viz

# Install dependencies
pip install -r requirements.txt
# or with uv
uv pip install -r requirements.txt
```

## Usage

### Local Development

```bash
python app_dash.py
```

Then open http://127.0.0.1:8050/ in your browser.

### Production Deployment

The app can be deployed to any platform supporting Python WSGI applications (Heroku, Railway, Render, etc.) or as a static site via GitHub Pages using the included workflow.

## Project Structure

```
Survey_Viz/
├── app_dash.py           # Main Dash application
├── data/
│   └── lna_data.csv      # Processed LNA survey data
├── LNA_Survey_v3.0.xlsx  # Original source data
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── .github/
    └── workflows/
        └── deploy.yml    # GitHub Pages deployment
```

## Data Source

The LNA survey data is compiled from published research papers across various conferences and journals, covering different technology nodes and frequency ranges. 

**Original data source:** [ETH Zürich IDEAS LNA Survey](https://ideas.ethz.ch/Surveys/lna-survey.html)

## License

This project is provided for educational and research purposes.

## Acknowledgments

- **Original survey data:** [ETH Zürich Integrated Devices, Electronics, And Systems (IDEAS)](https://ideas.ethz.ch/)
- Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/)
