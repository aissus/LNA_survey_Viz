"""
LNA Survey Visualizer - Dash Version
=====================================
A clean, interactive visualization tool for LNA survey data with:
- Linked scatter plots across multiple metrics
- Double-click to select and view point details
- Interactive filters (Technology, Frequency Range, Year, Metric Limits)
- Paper links to Google Scholar
- Export functionality

Run: python app_dash.py
Then navigate to http://127.0.0.1:8050/
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import pandas as pd
import os
# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_data():
    """Load and preprocess the LNA survey data."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "data", "lna_data.csv")
        df = pd.read_csv(file_path)
        
        numeric_cols = ['Year', 'Node', 'f@NFmin', 'NFmin', 'fc', 'NF', 'Gain', 'P1dB', 'IIP3', '3dB_BW', 'Power']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Group Processes
        if 'Process' in df.columns:
            def group_process(p):
                p = str(p).lower()
                if 'cmos' in p and 'soi' in p:
                    return 'CMOS SOI'
                if 'cmos' in p:
                    return 'CMOS Bulk'
                if 'sige' in p:
                    return 'SiGe'
                if 'finfet' in p:
                    return 'FinFET'
                if 'gaas' in p or 'inp' in p or 'gan' in p:
                    return 'III-V'
                return 'Other'
            df['Process_Group'] = df['Process'].apply(group_process)
        
        # Create Search Link
        if 'Paper_Title' in df.columns:
            df['Search_Link'] = df['Paper_Title'].apply(
                lambda x: f"https://scholar.google.com/scholar?q={x}" if pd.notnull(x) else None
            )
        
        # Add Type column
        df['Type'] = 'Survey'
        df['unique_id'] = df.index.astype(str)
        
        return df
    except FileNotFoundError:
        # Return sample data for demo
        return pd.DataFrame({
            'fc': [1, 5, 10, 28, 60],
            'NF': [1.5, 2.0, 2.5, 3.0, 4.0],
            'Gain': [20, 18, 15, 12, 10],
            'Power': [10, 15, 20, 25, 30],
            'IIP3': [-10, -8, -5, -3, 0],
            'Year': [2020, 2021, 2022, 2023, 2024],
            'Process_Group': ['CMOS Bulk', 'CMOS SOI', 'SiGe', 'FinFET', 'III-V'],
            'Paper_Title': ['Paper A', 'Paper B', 'Paper C', 'Paper D', 'Paper E'],
            'Type': ['Survey'] * 5,
            'unique_id': [f'sample_{i}' for i in range(5)],
            'Search_Link': [f'https://scholar.google.com/scholar?q=Paper+{chr(65+i)}' for i in range(5)]
        })

# Load data
df = load_data()
freq_col = 'fc' if 'fc' in df.columns else 'f@NFmin'

# Calculate FOM
if all(col in df.columns for col in ['Gain', freq_col, 'Power', 'NF']):
    df['F_lin'] = 10 ** (df['NF'] / 10)
    df['Gain_lin'] = 10 ** (df['Gain'] / 10)
    df['FOM'] = (df['Gain_lin'] * df[freq_col]) / (df['Power'] * (df['F_lin'] - 1))

# =============================================================================
# COLOR SCHEME
# =============================================================================
COLOR_MAP = {
    'CMOS Bulk': '#636EFA',
    'CMOS SOI': '#00CC96',
    'SiGe': '#EF553B',
    'FinFET': '#AB63FA',
    'III-V': '#FFA15A',
    'Other': '#19D3F3',
    'User': '#FFD700'
}

SYMBOL_MAP = {
    'CMOS Bulk': 'circle',
    'CMOS SOI': 'square',
    'SiGe': 'diamond',
    'FinFET': 'triangle-up',
    'III-V': 'cross',
    'Other': 'x'
}

freq_min = float(df[freq_col].min()) if not df.empty else 0.0
freq_max = float(df[freq_col].max()) if not df.empty else 0.0
year_min = int(df['Year'].min()) if 'Year' in df.columns else 1990
year_max = int(df['Year'].max()) if 'Year' in df.columns else 2025
nf_min = float(df['NF'].min()) if 'NF' in df.columns else 0.0
nf_max = float(df['NF'].max()) if 'NF' in df.columns else 0.0
gain_min = float(df['Gain'].min()) if 'Gain' in df.columns else 0.0
gain_max = float(df['Gain'].max()) if 'Gain' in df.columns else 0.0
power_min = float(df['Power'].min()) if 'Power' in df.columns else 0.0
power_max = float(df['Power'].max()) if 'Power' in df.columns else 0.0
tech_options = sorted(df['Process_Group'].dropna().unique()) if 'Process_Group' in df.columns else []

# =============================================================================
# DASH APP INITIALIZATION
# =============================================================================
app = dash.Dash(__name__)
app.title = "LNA Survey Visualizer"

# =============================================================================
# APP LAYOUT
# =============================================================================
app.layout = html.Div([
    # Stores for shared state
    dcc.Store(id='selected-point-store', data=None),
    dcc.Store(id='selected-ids-store', data=[]),
    dcc.Store(id='axis-range-store', data=None),
    dcc.Store(id='pinned-point-store', data=None),
    dcc.Store(id='hidden-traces-store', data=[]),
    dcc.Store(id='manual-points-store', data=[]),

    # Header
    html.Div([
        html.H1("LNA Survey Visualizer", style={'marginBottom': '8px'}),
        html.P("Explore RF/microwave LNAs, filter by specs, and inspect linked papers",
               style={'color': '#555', 'margin': 0})
    ], style={'textAlign': 'center', 'marginBottom': '25px'}),

    # Main content wrapper
    html.Div([
        # Sidebar filters
        html.Div([
            html.H3("Filters", style={'marginTop': 0}),

            html.Label("Technology / Process:", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='tech-filter',
                options=[{'label': tech, 'value': tech} for tech in tech_options],
                value=tech_options,
                inline=False,
                labelStyle={'display': 'block', 'marginBottom': '6px'}
            ),

            html.Label("Frequency Range (GHz):", style={'fontWeight': 'bold', 'marginTop': '16px'}),
            html.Div(id='freq-range-display', style={'fontSize': '13px', 'color': '#555', 'marginBottom': '6px'}),
            dcc.RangeSlider(
                id='freq-range',
                min=freq_min,
                max=freq_max,
                value=[freq_min, freq_max],
                marks={freq_min: f'{freq_min:.0f}', freq_max: f'{freq_max:.0f}'},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),

            html.Label("Year Range:", style={'fontWeight': 'bold', 'marginTop': '24px'}),
            dcc.RangeSlider(
                id='year-range',
                min=year_min,
                max=year_max,
                value=[year_min, year_max],
                marks={year_min: str(year_min), year_max: str(year_max)},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),

            html.Label("NF Range (dB):", style={'fontWeight': 'bold', 'marginTop': '24px'}),
            dcc.RangeSlider(
                id='nf-range',
                min=nf_min,
                max=nf_max,
                value=[nf_min, nf_max],
                marks={nf_min: f'{nf_min:.1f}', nf_max: f'{nf_max:.1f}'},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),

            html.Label("Gain Range (dB):", style={'fontWeight': 'bold', 'marginTop': '24px'}),
            dcc.RangeSlider(
                id='gain-range',
                min=gain_min,
                max=gain_max,
                value=[gain_min, gain_max],
                marks={gain_min: f'{gain_min:.1f}', gain_max: f'{gain_max:.1f}'},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),

            html.Label("Power Range (mW):", style={'fontWeight': 'bold', 'marginTop': '24px'}),
            dcc.RangeSlider(
                id='power-range',
                min=power_min,
                max=power_max,
                value=[power_min, power_max],
                marks={power_min: f'{power_min:.1f}', power_max: f'{power_max:.1f}'},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),

            html.Hr(style={'marginTop': '24px', 'marginBottom': '16px'}),
            html.H4("➕ Add Manual Point", style={'marginTop': 0, 'marginBottom': '12px'}),
            html.Label("Label:", style={'fontSize': '12px'}),
            dcc.Input(id='manual-label', type='text', placeholder='My Design', style={'width': '100%', 'marginBottom': '8px', 'padding': '4px'}),
            html.Div([
                html.Div([
                    html.Label("Freq (GHz):", style={'fontSize': '11px'}),
                    dcc.Input(id='manual-freq', type='number', placeholder='28', style={'width': '100%', 'padding': '4px'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                html.Div([
                    html.Label("NF (dB):", style={'fontSize': '11px'}),
                    dcc.Input(id='manual-nf', type='number', placeholder='2.5', style={'width': '100%', 'padding': '4px'})
                ], style={'width': '48%', 'display': 'inline-block'}),
            ], style={'marginBottom': '8px'}),
            html.Div([
                html.Div([
                    html.Label("Gain (dB):", style={'fontSize': '11px'}),
                    dcc.Input(id='manual-gain', type='number', placeholder='15', style={'width': '100%', 'padding': '4px'})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                html.Div([
                    html.Label("Power (mW):", style={'fontSize': '11px'}),
                    dcc.Input(id='manual-power', type='number', placeholder='20', style={'width': '100%', 'padding': '4px'})
                ], style={'width': '48%', 'display': 'inline-block'}),
            ], style={'marginBottom': '8px'}),
            html.Div([
                html.Label("IIP3 (dBm):", style={'fontSize': '11px'}),
                dcc.Input(id='manual-iip3', type='number', placeholder='-5', style={'width': '60%', 'padding': '4px'})
            ], style={'marginBottom': '12px'}),
            html.Button('Add Point', id='add-manual-point-btn', n_clicks=0, style={
                'padding': '8px 16px',
                'border': 'none',
                'borderRadius': '4px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'cursor': 'pointer',
                'fontSize': '12px',
                'marginRight': '8px'
            }),
            html.Button('Clear All', id='clear-manual-points-btn', n_clicks=0, style={
                'padding': '8px 16px',
                'border': 'none',
                'borderRadius': '4px',
                'backgroundColor': '#f44336',
                'color': 'white',
                'cursor': 'pointer',
                'fontSize': '12px'
            }),
            html.Div(id='manual-points-list', style={'marginTop': '12px', 'fontSize': '11px', 'color': '#666'}),

        ], style={
            'width': '22%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'backgroundColor': '#f9f9f9',
            'borderRight': '1px solid #ddd'
        }),

        # Main content area
        html.Div([
            html.Div([
                html.Div(id='selected-point-details', style={
                    'backgroundColor': '#e3f2fd',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'marginBottom': '10px',
                    'border': '1px solid #90caf9'
                }),
                html.Button('✕ Clear Selection', id='clear-selection-btn', n_clicks=0, style={
                    'marginBottom': '20px',
                    'padding': '6px 14px',
                    'border': 'none',
                    'borderRadius': '4px',
                    'backgroundColor': '#e0e0e0',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                    'marginRight': '10px'
                }),
                html.Button('⬚ Clear Area Selection', id='clear-area-selection-btn', n_clicks=0, style={
                    'marginBottom': '20px',
                    'padding': '6px 14px',
                    'border': 'none',
                    'borderRadius': '4px',
                    'backgroundColor': '#e0e0e0',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                    'marginRight': '10px'
                }),
                dcc.Checklist(
                    id='minor-grid-toggle',
                    options=[{'label': ' Minor Grid', 'value': 'on'}],
                    value=[],
                    inline=True,
                    style={'display': 'inline-block', 'marginBottom': '20px', 'fontSize': '12px'}
                ),
            ]),

            html.Div(id='data-stats', style={'marginBottom': '15px', 'color': '#666'}),

            html.Div([
                html.Div([
                    dcc.Graph(id='nf-chart', style={'height': '400px'},
                              config={'toImageButtonOptions': {'format': 'svg', 'filename': 'nf_vs_freq', 'scale': 2},
                                      'displaylogo': False})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='gain-chart', style={'height': '400px'},
                              config={'toImageButtonOptions': {'format': 'svg', 'filename': 'gain_vs_freq', 'scale': 2},
                                      'displaylogo': False})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),
            ]),

            html.Div([
                html.Div([
                    dcc.Graph(id='power-chart', style={'height': '400px'},
                              config={'toImageButtonOptions': {'format': 'svg', 'filename': 'power_vs_freq', 'scale': 2},
                                      'displaylogo': False})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='iip3-chart', style={'height': '400px'},
                              config={'toImageButtonOptions': {'format': 'svg', 'filename': 'iip3_vs_freq', 'scale': 2},
                                      'displaylogo': False})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),
            ]),

            html.Div([
                html.Div([
                    dcc.Graph(id='fom-chart', style={'height': '400px'},
                              config={'toImageButtonOptions': {'format': 'svg', 'filename': 'fom_vs_freq', 'scale': 2},
                                      'displaylogo': False})
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='gain-nf-chart', style={'height': '400px'},
                              config={'toImageButtonOptions': {'format': 'svg', 'filename': 'gain_vs_nf', 'scale': 2},
                                      'displaylogo': False})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),
            ]),

            html.H3("📋 Data Table", style={'marginTop': '30px'}),
            html.Div(id='data-table')

        ], style={
            'width': '76%',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'padding': '20px',
            'paddingLeft': '2%'
        })
    ], style={'display': 'flex', 'width': '100%'}),

], style={'fontFamily': 'Arial, sans-serif', 'padding': '15px'})

# =============================================================================
# CALLBACKS
# =============================================================================

def get_filtered_data(tech_values, freq_range, year_range, nf_range, gain_range, power_range):
    """Filter data based on current filter states."""
    filtered = df.copy()
    
    if tech_values:
        filtered = filtered[filtered['Process_Group'].isin(tech_values)]
    
    def _within(series, bounds):
        low, high = bounds
        mask = series.between(low, high, inclusive='both')
        return mask | series.isna()
    
    filtered = filtered[_within(filtered[freq_col], freq_range)]
    filtered = filtered[_within(filtered['Year'], year_range)]
    if 'NF' in filtered:
        filtered = filtered[_within(filtered['NF'], nf_range)]
    if 'Gain' in filtered:
        filtered = filtered[_within(filtered['Gain'], gain_range)]
    if 'Power' in filtered:
        filtered = filtered[_within(filtered['Power'], power_range)]
    
    return filtered

def create_scatter_plot(filtered_data, y_col, title, y_title, selected_id=None, selected_ids=None, x_range=None, hidden_traces=None, manual_points=None, show_minor_grid=False, clear_selections=False):
    """Create a scatter plot with selection highlighting and custom tooltips."""
    fig = go.Figure()
    selected_set = {str(s) for s in (selected_ids or []) if s is not None}
    selected_point = str(selected_id) if selected_id is not None else None
    has_selection = bool(selected_set)
    hidden_set = set(hidden_traces or [])
    manual_pts = manual_points or []

    for tech in sorted(filtered_data['Process_Group'].unique()):
        tech_data = filtered_data[filtered_data['Process_Group'] == tech]
        if tech_data.empty:
            continue

        uid_list = tech_data['unique_id'].astype(str).tolist()
        # Dynamic marker styling based on stored selection
        marker_sizes = []
        marker_opacities = []
        marker_line_widths = []
        marker_line_colors = []
        for uid in uid_list:
            is_focused = selected_point and uid == selected_point
            is_selected = uid in selected_set
            marker_sizes.append(16 if is_focused else 11)
            # Thick black stroke for focused point, thin gray for others
            marker_line_widths.append(3.5 if is_focused else 0.8)
            marker_line_colors.append('#000' if is_focused else '#444')
            if has_selection:
                # High contrast: selected = full opacity, unselected = very faint
                marker_opacities.append(1.0 if is_selected else 0.08)
            else:
                marker_opacities.append(0.85)

        hover_text = []
        for idx, row in tech_data.iterrows():
            title_text = str(row['Paper_Title'])[:50]
            author = str(row.get('Last_Name', '')) if pd.notna(row.get('Last_Name')) else ''
            pub = str(row.get('Publication', '')) if pd.notna(row.get('Publication')) else ''
            year = int(row['Year']) if pd.notna(row['Year']) else ''
            citation = f"({author}, {pub} {year})" if author or pub else f"({year})"
            hover_text.append(
                f"<b>{title_text}{'...' if len(str(row['Paper_Title'])) > 50 else ''}</b><br>"
                f"{citation}<br>"
                f"Freq: {row[freq_col]:.2f} GHz | NF: {row['NF']:.2f} dB<br>"
                f"Gain: {row['Gain']:.1f} dB | Power: {row['Power']:.1f} mW<br>"
                f"IIP3: {row['IIP3']:.1f} dBm | Process: {row['Process_Group']}"
            )

        base_color = COLOR_MAP.get(tech, '#999')
        
        fig.add_trace(go.Scatter(
            x=tech_data[freq_col],
            y=tech_data[y_col],
            mode='markers',
            name=tech,
            marker=dict(
                size=marker_sizes,
                color=base_color,
                symbol=SYMBOL_MAP.get(tech, 'circle'),
                line=dict(width=marker_line_widths, color=marker_line_colors),
                opacity=marker_opacities
            ),
            customdata=uid_list,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            meta={'tech': tech},
            visible=True if tech not in hidden_set else 'legendonly'
        ))

    # Add manual data points (user-added)
    if manual_pts:
        y_key = y_col.lower()  # 'NF' -> 'nf', 'Gain' -> 'gain', etc.
        manual_x = [p['freq'] for p in manual_pts if y_key in p]
        manual_y = [p[y_key] for p in manual_pts if y_key in p]
        manual_labels = [p['label'] for p in manual_pts if y_key in p]
        if manual_x:
            hover_text = [f"<b>{lbl}</b><br>Freq: {fx:.2f} GHz | {y_title}: {fy:.2f}" 
                          for lbl, fx, fy in zip(manual_labels, manual_x, manual_y)]
            fig.add_trace(go.Scatter(
                x=manual_x,
                y=manual_y,
                mode='markers',
                name='User Points',
                marker=dict(
                    size=14,
                    color='#FFD700',
                    symbol='star',
                    line=dict(width=2, color='#000')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))

    # Octave-based tick values for frequency axis (10, 20, 50, 100, 200, 500, etc.)
    freq_ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    # Calculate minor grid config with auto-scaling nticks
    if show_minor_grid:
        # For log scale x-axis, use 5 minor ticks per decade
        x_minor_config = dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.15)', nticks=5)
        # For linear y-axis, calculate based on data range
        y_data = filtered_data[y_col].dropna()
        if not y_data.empty:
            y_span = y_data.max() - y_data.min()
            # Aim for ~4-5 minor ticks between major ticks
            y_minor_config = dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.15)', nticks=5)
        else:
            y_minor_config = dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.15)', nticks=5)
    else:
        x_minor_config = {}
        y_minor_config = {}
    
    xaxis_config = dict(
        title='Frequency (GHz)', 
        type='log',
        tickvals=freq_ticks,
        ticktext=[str(v) for v in freq_ticks],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.3)',
        minor=x_minor_config,
        dtick=None  # Let tickvals control
    )
    if x_range:
        xaxis_config['range'] = x_range
    
    fig.update_layout(
        title=title,
        xaxis=xaxis_config,
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            minor=y_minor_config
        ),
        hovermode='closest',
        height=400,
        showlegend=True,
        template='plotly_white',
        dragmode='select',
        uirevision='cleared' if clear_selections else 'freq-scatter-state',
        newselection_mode='gradual',
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.9)',
            font_size=12,
            font_color='#111',
            bordercolor='#888'
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#ccc',
            borderwidth=1
        )
    )
    
    # Enable SVG/PNG download via modebar
    fig.update_layout(
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )

    return fig

def create_cartesian_plot(filtered_data, x_col, y_col, x_title, y_title, title, selected_id=None, selected_ids=None, hidden_traces=None, manual_points=None, show_minor_grid=False, clear_selections=False):
    """Create a Cartesian scatter plot (not vs Frequency)."""
    fig = go.Figure()
    selected_set = {str(s) for s in (selected_ids or []) if s is not None}
    selected_point = str(selected_id) if selected_id is not None else None
    has_selection = bool(selected_set)
    hidden_set = set(hidden_traces or [])
    manual_pts = manual_points or []
    
    for tech in sorted(filtered_data['Process_Group'].unique()):
        tech_data = filtered_data[filtered_data['Process_Group'] == tech]
        if tech_data.empty:
            continue
        uid_list = tech_data['unique_id'].astype(str).tolist()
        marker_sizes = []
        marker_opacities = []
        marker_line_widths = []
        marker_line_colors = []
        for uid in uid_list:
            is_focused = selected_point and uid == selected_point
            is_selected = uid in selected_set
            marker_sizes.append(16 if is_focused else 11)
            # Thick black stroke for focused point, thin gray for others
            marker_line_widths.append(3.5 if is_focused else 0.8)
            marker_line_colors.append('#000' if is_focused else '#444')
            if has_selection:
                # High contrast: selected = full opacity, unselected = very faint
                marker_opacities.append(1.0 if is_selected else 0.08)
            else:
                marker_opacities.append(0.85)
        hover_text = []
        for idx, row in tech_data.iterrows():
            title_text = str(row['Paper_Title'])[:50]
            author = str(row.get('Last_Name', '')) if pd.notna(row.get('Last_Name')) else ''
            pub = str(row.get('Publication', '')) if pd.notna(row.get('Publication')) else ''
            year = int(row['Year']) if pd.notna(row['Year']) else ''
            citation = f"({author}, {pub} {year})" if author or pub else f"({year})"
            hover_text.append(
                f"<b>{title_text}{'...' if len(str(row['Paper_Title'])) > 50 else ''}</b><br>"
                f"{citation}<br>"
                f"Freq: {row[freq_col]:.2f} GHz | NF: {row['NF']:.2f} dB<br>"
                f"Gain: {row['Gain']:.1f} dB | Power: {row['Power']:.1f} mW<br>"
                f"IIP3: {row['IIP3']:.1f} dBm | Process: {row['Process_Group']}"
            )
        base_color = COLOR_MAP.get(tech, '#999')
        
        fig.add_trace(go.Scatter(
            x=tech_data[x_col],
            y=tech_data[y_col],
            mode='markers',
            name=tech,
            marker=dict(
                size=marker_sizes,
                color=base_color,
                symbol=SYMBOL_MAP.get(tech, 'circle'),
                line=dict(width=marker_line_widths, color=marker_line_colors),
                opacity=marker_opacities
            ),
            customdata=uid_list,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            meta={'tech': tech},
            visible=True if tech not in hidden_set else 'legendonly'
        ))

    # Add manual data points (user-added) for Cartesian plot (Gain vs NF)
    if manual_pts:
        x_key = x_col.lower()  # 'NF' -> 'nf'
        y_key = y_col.lower()  # 'Gain' -> 'gain'
        manual_x = [p[x_key] for p in manual_pts if x_key in p and y_key in p]
        manual_y = [p[y_key] for p in manual_pts if x_key in p and y_key in p]
        manual_labels = [p['label'] for p in manual_pts if x_key in p and y_key in p]
        if manual_x:
            hover_text = [f"<b>{lbl}</b><br>{x_title}: {fx:.2f} | {y_title}: {fy:.2f}" 
                          for lbl, fx, fy in zip(manual_labels, manual_x, manual_y)]
            fig.add_trace(go.Scatter(
                x=manual_x,
                y=manual_y,
                mode='markers',
                name='User Points',
                marker=dict(
                    size=14,
                    color='#FFD700',
                    symbol='star',
                    line=dict(width=2, color='#000')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))
    
    # Calculate minor grid config with auto-scaling nticks for both linear axes
    if show_minor_grid:
        # For linear axes, use nticks=5 for balanced grid density
        x_minor_config = dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.15)', nticks=5)
        y_minor_config = dict(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.15)', nticks=5)
    else:
        x_minor_config = {}
        y_minor_config = {}
    
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_title,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            minor=x_minor_config
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.3)',
            minor=y_minor_config
        ),
        hovermode='closest',
        height=400,
        showlegend=True,
        template='plotly_white',
        dragmode='select',
        uirevision='cleared' if clear_selections else 'cartesian-scatter-state',
        newselection_mode='gradual',
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.9)',
            font_size=12,
            font_color='#111',
            bordercolor='#888'
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#ccc',
            borderwidth=1
        ),
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )
    
    return fig


def extract_selected_ids(selection_payload):
    """Return ordered list of unique ids from a Plotly selectedData payload."""
    if not selection_payload or 'points' not in selection_payload:
        return []
    seen = set()
    ids = []
    for point in selection_payload['points']:
        cid = point.get('customdata')
        if cid is None:
            continue
        cid = str(cid)
        if cid in seen:
            continue
        seen.add(cid)
        ids.append(cid)
    return ids


def parse_axis_range(relayout_payload):
    """Extract x-axis range info from relayoutData, honoring autorange reset."""
    if not relayout_payload:
        return None, False
    if relayout_payload.get('xaxis.autorange'):
        return None, True
    start = relayout_payload.get('xaxis.range[0]')
    end = relayout_payload.get('xaxis.range[1]')
    if start is not None and end is not None:
        return [float(start), float(end)], True
    return None, False

@callback(
    [Output('nf-chart', 'figure'),
     Output('gain-chart', 'figure'),
     Output('power-chart', 'figure'),
     Output('iip3-chart', 'figure'),
     Output('fom-chart', 'figure'),
     Output('gain-nf-chart', 'figure'),
     Output('data-stats', 'children'),
     Output('selected-point-details', 'children'),
     Output('selected-point-store', 'data'),
     Output('selected-ids-store', 'data'),
     Output('axis-range-store', 'data'),
     Output('hidden-traces-store', 'data')],
    [Input('tech-filter', 'value'),
     Input('freq-range', 'value'),
     Input('year-range', 'value'),
     Input('nf-range', 'value'),
     Input('gain-range', 'value'),
     Input('power-range', 'value'),
     Input('clear-selection-btn', 'n_clicks'),
     Input('clear-area-selection-btn', 'n_clicks'),
     Input('minor-grid-toggle', 'value'),
     Input('nf-chart', 'clickData'),
     Input('nf-chart', 'selectedData'),
     Input('nf-chart', 'relayoutData'),
     Input('nf-chart', 'restyleData'),
     Input('gain-chart', 'clickData'),
     Input('gain-chart', 'selectedData'),
     Input('gain-chart', 'relayoutData'),
     Input('gain-chart', 'restyleData'),
     Input('power-chart', 'clickData'),
     Input('power-chart', 'selectedData'),
     Input('power-chart', 'relayoutData'),
     Input('power-chart', 'restyleData'),
     Input('iip3-chart', 'clickData'),
     Input('iip3-chart', 'selectedData'),
     Input('iip3-chart', 'relayoutData'),
     Input('iip3-chart', 'restyleData'),
     Input('fom-chart', 'clickData'),
     Input('fom-chart', 'selectedData'),
     Input('fom-chart', 'relayoutData'),
     Input('fom-chart', 'restyleData'),
     Input('gain-nf-chart', 'clickData'),
     Input('gain-nf-chart', 'selectedData'),
     Input('gain-nf-chart', 'restyleData'),
     Input('manual-points-store', 'data')],
    [State('selected-point-store', 'data'),
     State('selected-ids-store', 'data'),
     State('axis-range-store', 'data'),
     State('hidden-traces-store', 'data')]
)
def update_charts(tech_values, freq_range, year_range, nf_range, gain_range, power_range,
                  clear_clicks, clear_area_clicks, minor_grid_value,
                  click_nf, sel_nf, relayout_nf, restyle_nf,
                  click_gain, sel_gain, relayout_gain, restyle_gain,
                  click_power, sel_power, relayout_power, restyle_power,
                  click_iip3, sel_iip3, relayout_iip3, restyle_iip3,
                  click_fom, sel_fom, relayout_fom, restyle_fom,
                  click_gainf, sel_gainf, restyle_gainf,
                  manual_points,
                  selected_store, selected_ids_store, axis_range_store, hidden_traces_store):
    """Update all charts and details based on filters and selection."""
    
    ctx = dash.callback_context
    triggered_prop = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    
    # Check if minor grid is enabled
    show_minor_grid = 'on' in (minor_grid_value or [])

    # Initialize from stored state
    selected_id = selected_store
    selected_ids = list(selected_ids_store or [])
    axis_range = axis_range_store
    hidden_traces = list(hidden_traces_store or [])

    # Flag to clear UI selections (box/lasso)
    clear_selections = False
    
    # Handle clear button: reset all selections (point + area)
    if triggered_prop == 'clear-selection-btn.n_clicks':
        selected_id = None
        selected_ids = []
        clear_selections = True
    
    # Handle clear area selection button: only clear area selection, keep focused point
    if triggered_prop == 'clear-area-selection-btn.n_clicks':
        selected_ids = []
        clear_selections = True

    click_map = {
        'nf-chart': click_nf,
        'gain-chart': click_gain,
        'power-chart': click_power,
        'iip3-chart': click_iip3,
        'fom-chart': click_fom,
        'gain-nf-chart': click_gainf
    }
    selection_map = {
        'nf-chart': sel_nf,
        'gain-chart': sel_gain,
        'power-chart': sel_power,
        'iip3-chart': sel_iip3,
        'fom-chart': sel_fom,
        'gain-nf-chart': sel_gainf
    }
    relayout_map = {
        'nf-chart': relayout_nf,
        'gain-chart': relayout_gain,
        'power-chart': relayout_power,
        'iip3-chart': relayout_iip3,
        'fom-chart': relayout_fom
    }
    restyle_map = {
        'nf-chart': restyle_nf,
        'gain-chart': restyle_gain,
        'power-chart': restyle_power,
        'iip3-chart': restyle_iip3,
        'fom-chart': restyle_fom,
        'gain-nf-chart': restyle_gainf
    }

    # Process the triggered interaction (skip for clear buttons)
    if triggered_prop and triggered_prop not in ['clear-selection-btn.n_clicks', 'clear-area-selection-btn.n_clicks']:
        source, prop = triggered_prop.split('.') if '.' in triggered_prop else (triggered_prop, '')
        
        if prop == 'clickData':
            # Single click on a point - set as focused point
            payload = click_map.get(source)
            if payload and 'points' in payload and payload['points']:
                cid = payload['points'][0].get('customdata')
                selected_id = str(cid) if cid is not None else None
                
        elif prop == 'selectedData':
            # Area selection - this becomes the GLOBAL selection for all charts
            payload = selection_map.get(source)
            new_ids = extract_selected_ids(payload)
            if new_ids:
                # New area selection overrides previous
                selected_ids = new_ids
                # Set first selected as the focused point
                selected_id = selected_ids[0] if selected_ids else None
            # If payload is empty/None, keep previous selection (don't clear on deselect)
                
        elif prop == 'relayoutData':
            payload = relayout_map.get(source)
            new_range, should_update = parse_axis_range(payload)
            if should_update:
                axis_range = new_range
                
        elif prop == 'restyleData':
            # Legend click: toggle visibility of the clicked trace
            payload = restyle_map.get(source)
            if payload and len(payload) >= 2:
                visibility_changes = payload[0].get('visible', [])
                trace_indices = payload[1]
                all_techs = tech_options
                for vis, idx in zip(visibility_changes, trace_indices):
                    if idx < len(all_techs):
                        tech_name = all_techs[idx]
                        if vis == 'legendonly' and tech_name not in hidden_traces:
                            hidden_traces.append(tech_name)
                        elif vis is True and tech_name in hidden_traces:
                            hidden_traces = [t for t in hidden_traces if t != tech_name]

    # Get filtered data
    filtered_data = get_filtered_data(tech_values, freq_range, year_range, nf_range, gain_range, power_range)
    available_ids = set(filtered_data['unique_id'].astype(str))
    # Keep only selected IDs that are still in filtered data
    selected_ids = [sid for sid in selected_ids if sid in available_ids]
    if selected_id is not None and str(selected_id) not in available_ids:
        selected_id = selected_ids[0] if selected_ids else None
    
    # Create charts with synced legend visibility and manual points
    nf_fig = create_scatter_plot(filtered_data, 'NF', 'Noise Figure vs Frequency', 'NF (dB)', selected_id, selected_ids, axis_range, hidden_traces, manual_points, show_minor_grid, clear_selections)
    gain_fig = create_scatter_plot(filtered_data, 'Gain', 'Gain vs Frequency', 'Gain (dB)', selected_id, selected_ids, axis_range, hidden_traces, manual_points, show_minor_grid, clear_selections)
    power_fig = create_scatter_plot(filtered_data, 'Power', 'Power vs Frequency', 'Power (mW)', selected_id, selected_ids, axis_range, hidden_traces, manual_points, show_minor_grid, clear_selections)
    iip3_fig = create_scatter_plot(filtered_data, 'IIP3', 'IIP3 vs Frequency', 'IIP3 (dBm)', selected_id, selected_ids, axis_range, hidden_traces, manual_points, show_minor_grid, clear_selections)
    
    # FOM plot
    if 'FOM' in filtered_data.columns:
        fom_fig = create_scatter_plot(filtered_data, 'FOM', 'Figure of Merit vs Frequency', 'FOM', selected_id, selected_ids, axis_range, hidden_traces, manual_points, show_minor_grid, clear_selections)
    else:
        fom_fig = go.Figure()
        fom_fig.add_annotation(text="FOM data not available", x=0.5, y=0.5, showarrow=False)
        fom_fig.update_layout(template='plotly_white', height=400)
    
    # Gain vs NF plot (Cartesian plot instead of vs Frequency)
    gain_nf_fig = create_cartesian_plot(filtered_data, 'NF', 'Gain', 'Noise Figure (dB)', 'Gain (dB)', 
                                        'Gain vs Noise Figure', selected_id, selected_ids, hidden_traces, manual_points, show_minor_grid, clear_selections)
    
    # Stats
    stats = html.P(f"📊 Showing {len(filtered_data)} points out of {len(df)} total")
    
    # Selected point details
    if selected_id:
        point_row = df[df['unique_id'].astype(str) == str(selected_id)]
        if not point_row.empty:
            row = point_row.iloc[0]
            # Build citation string
            author = str(row.get('Last_Name', '')) if pd.notna(row.get('Last_Name')) else ''
            pub = str(row.get('Publication', '')) if pd.notna(row.get('Publication')) else ''
            year = int(row['Year']) if pd.notna(row['Year']) else ''
            citation = f"({author}, {pub} {year})" if author or pub else f"({year})"
            
            details = html.Div([
                html.H4("📍 Selected Point Details", style={'marginTop': '0', 'marginBottom': '12px'}),
                html.Div([
                    html.Div(row['Paper_Title'], 
                            style={'fontWeight': 'bold', 'marginBottom': '6px', 'fontSize': '0.95em'}),
                    html.Div(citation, 
                            style={'marginBottom': '10px', 'color': '#666', 'fontStyle': 'italic', 'fontSize': '0.9em'}),
                    html.Div(f"Process: {row['Process_Group']}", style={'marginBottom': '8px', 'color': '#555'}),
                    html.Div(f"Freq: {row[freq_col]:.2f} GHz  |  NF: {row['NF']:.2f} dB  |  Gain: {row['Gain']:.1f} dB", 
                            style={'marginBottom': '6px', 'color': '#555', 'fontSize': '0.9em'}),
                    html.Div(f"Power: {row['Power']:.1f} mW  |  IIP3: {row['IIP3']:.1f} dBm", 
                            style={'marginBottom': '10px', 'color': '#555', 'fontSize': '0.9em'}),
                    html.A('🔍 Search on Google Scholar', href=row['Search_Link'], target='_blank',
                          style={'color': '#2196F3', 'textDecoration': 'none', 'fontWeight': 'bold', 'fontSize': '0.9em'})
                ])
            ])
        else:
            details = html.P("Point not found", style={'color': '#999'})
    else:
        details = html.P("👆 Click on a data point to view details and access the paper link", style={'color': '#999'})
    
    return (nf_fig, gain_fig, power_fig, iip3_fig, fom_fig, gain_nf_fig,
            stats, details, selected_id, selected_ids, axis_range, hidden_traces)

@callback(
    Output('data-table', 'children'),
    [Input('tech-filter', 'value'),
     Input('freq-range', 'value'),
     Input('year-range', 'value'),
     Input('nf-range', 'value'),
     Input('gain-range', 'value'),
     Input('power-range', 'value')]
)
def update_table(tech_values, freq_range, year_range, nf_range, gain_range, power_range):
    """Update data table."""
    filtered_data = get_filtered_data(tech_values, freq_range, year_range, nf_range, gain_range, power_range)
    
    # Create table
    table_rows = [html.Tr([
        html.Th("Paper Title"),
        html.Th("Year"),
        html.Th("Freq (GHz)"),
        html.Th("NF (dB)"),
        html.Th("Gain (dB)"),
        html.Th("Power (mW)"),
        html.Th("IIP3 (dBm)"),
        html.Th("Process"),
        html.Th("Link")
    ], style={'borderBottom': '2px solid #ddd'})]
    
    for _, row in filtered_data.iterrows():
        table_rows.append(html.Tr([
            html.Td(row['Paper_Title'], style={'whiteSpace': 'normal', 'wordWrap': 'break-word'}),
            html.Td(int(row['Year'])),
            html.Td(f"{row[freq_col]:.2f}"),
            html.Td(f"{row['NF']:.2f}"),
            html.Td(f"{row['Gain']:.1f}"),
            html.Td(f"{row['Power']:.1f}"),
            html.Td(f"{row['IIP3']:.1f}"),
            html.Td(row['Process_Group']),
            html.Td(html.A('Scholar', href=row['Search_Link'], target='_blank'), style={'whiteSpace': 'nowrap'})
        ], style={'borderBottom': '1px solid #eee', 'verticalAlign': 'top'}))
    
    return html.Table(table_rows, style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'fontSize': '12px',
        'tableLayout': 'auto'
    })


@callback(Output('freq-range-display', 'children'), Input('freq-range', 'value'))
def update_freq_range_display(freq_range):
    """Show the numeric frequency range so the upper bound is visible."""
    if not freq_range or len(freq_range) != 2:
        return ""
    return f"{freq_range[0]:.1f} – {freq_range[1]:.1f} GHz"


# Callback to manage manual data points
@callback(
    [Output('manual-points-store', 'data'),
     Output('manual-points-list', 'children'),
     Output('manual-label', 'value'),
     Output('manual-freq', 'value'),
     Output('manual-nf', 'value'),
     Output('manual-gain', 'value'),
     Output('manual-power', 'value'),
     Output('manual-iip3', 'value')],
    [Input('add-manual-point-btn', 'n_clicks'),
     Input('clear-manual-points-btn', 'n_clicks')],
    [State('manual-points-store', 'data'),
     State('manual-label', 'value'),
     State('manual-freq', 'value'),
     State('manual-nf', 'value'),
     State('manual-gain', 'value'),
     State('manual-power', 'value'),
     State('manual-iip3', 'value')]
)
def manage_manual_points(add_clicks, clear_clicks, current_points, label, freq, nf, gain, power, iip3):
    """Add or clear manual data points."""
    ctx = dash.callback_context
    triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ''
    
    points = list(current_points or [])
    
    if triggered == 'clear-manual-points-btn.n_clicks':
        points = []
    elif triggered == 'add-manual-point-btn.n_clicks':
        # Validate inputs
        if freq is not None and nf is not None and gain is not None:
            new_point = {
                'label': label or f'Point {len(points) + 1}',
                'freq': float(freq),
                'nf': float(nf),
                'gain': float(gain),
                'power': float(power) if power is not None else 0,
                'iip3': float(iip3) if iip3 is not None else 0
            }
            points.append(new_point)
    
    # Generate list display
    if points:
        list_items = [html.Div(f"• {p['label']}: {p['freq']}GHz, NF={p['nf']}dB, G={p['gain']}dB") for p in points]
        list_display = html.Div(list_items)
    else:
        list_display = html.Div("No manual points added", style={'fontStyle': 'italic'})
    
    # Clear inputs after adding
    if triggered == 'add-manual-point-btn.n_clicks':
        return points, list_display, '', None, None, None, None, None
    
    return points, list_display, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# =============================================================================
# RUN APP
# =============================================================================
if __name__ == '__main__':
    print("🚀 Starting LNA Survey Visualizer")
    print("📱 Open http://127.0.0.1:8050/ in your browser")
    app.run(debug=False, host='127.0.0.1', port=8050)
