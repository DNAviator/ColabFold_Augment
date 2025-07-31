import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from Bio import AlignIO
import pandas as pd
from collections import Counter
import webbrowser
from threading import Timer


def plot_sequence_variability(clustal_file_path, port=8050):
    """
    Parses a Clustal file and launches an interactive Dash web application
    to visualize sequence variability.

    The app features:
    1. A main stacked bar chart showing amino acid distribution at each position.
    2. A pie chart showing amino acid distribution for a selected residue.
    3. A searchable, sortable table of raw data for the selected residue.

    Args:
        clustal_file_path (str): The path to the input Clustal file.
        port (int): The port on which to run the web application.
    """

    # --- 1. Data Parsing and Processing ---
    try:
        alignment = AlignIO.read(clustal_file_path, "fasta")
    except FileNotFoundError:
        print(f"Error: The file '{clustal_file_path}' was not found.")
        return
    except ValueError:
        print(
            f"Error: Could not parse '{clustal_file_path}'. Please ensure it's a valid Clustal file."
        )
        return

    num_sequences = len(alignment)
    alignment_length = alignment.get_alignment_length()
    sequence_ids = [rec.id for rec in alignment]

    # --- 2. Amino Acid Property-Based Coloring ---
    aa_colors = {
        # Negatively Charged (Reds)
        "D": "#E60A0A",
        "E": "#D43F3F",
        # Positively Charged (Blues)
        "K": "#145AFF",
        "R": "#3C74E6",
        "H": "#658FF0",
        # Polar Neutral (Greens)
        "S": "#40E040",
        "T": "#66E666",
        "N": "#8CF08C",
        "Q": "#A6F5A6",
        # Non-polar, Aliphatic (Greys)
        "A": "#C0C0C0",
        "V": "#A9A9A9",
        "I": "#808080",
        "L": "#696969",
        "G": "#505050",
        # Non-polar, Aromatic (Purples)
        "F": "#B284BE",
        "Y": "#9966CC",
        "W": "#7F4D99",
        # Special Cases (Yellow/Orange/Pink)
        "M": "#FFA500",
        "C": "#FF00DD",
        "P": "#FFC0CB",
        # Gap
        "-": "#F0F0F0",
    }

    # --- 3. Calculate Frequencies for the Main Plot ---
    all_aas = sorted(list(set("".join([str(rec.seq) for rec in alignment]))))

    plot_data = []
    for i in range(alignment_length):
        column = alignment[:, i]
        counts = Counter(column)

        position_data = {"Residue": i + 1}

        # Calculate frequency for each amino acid at the position
        for aa in all_aas:
            position_data[aa] = counts.get(aa, 0) / num_sequences * 100

        plot_data.append(position_data)

    df = pd.DataFrame(plot_data)

    # --- 4. Initialize and Configure Dash App ---
    app = dash.Dash(__name__)

    # Main figure (Stacked Bar Plot)
    main_fig = go.Figure()
    for aa in all_aas:
        main_fig.add_trace(
            go.Bar(
                x=df["Residue"],
                y=df[aa],
                name=aa,
                marker_color=aa_colors.get(aa, "#333333"),
            )
        )

    main_fig.update_layout(
        title="Sequence Variability Overview (Click a Bar for Details)",
        xaxis_title="Residue Position",
        yaxis_title="Amino Acid Frequency (%)",
        barmode="stack",  # This is the key change to stack the bars
        template="plotly_white",
        clickmode="event+select",
        legend_title="Amino Acid",
    )
    main_fig.update_yaxes(range=[0, 100])  # Ensure y-axis is fixed 0-100

    # --- 5. Define App Layout ---
    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "20px"},
        children=[
            html.H1(
                f"Interactive Sequence Analysis: {clustal_file_path}",
                style={"textAlign": "center"},
            ),
            html.P(
                "Click on a bar in the plot below to see detailed information for that residue position.",
                style={"textAlign": "center"},
            ),
            dcc.Graph(id="main-plot", figure=main_fig),  # Renamed for clarity
            html.Hr(),
            html.Div(
                id="details-section",
                style={"display": "none"},
                children=[
                    html.Div(
                        className="row",
                        style={"display": "flex", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                style={
                                    "width": "48%",
                                    "minWidth": "300px",
                                    "padding": "10px",
                                },
                                children=[
                                    html.H3(
                                        "Residue Distribution",
                                        style={"textAlign": "center"},
                                    ),
                                    dcc.Graph(id="pie-chart"),
                                ],
                            ),
                            html.Div(
                                style={
                                    "width": "48%",
                                    "minWidth": "300px",
                                    "padding": "10px",
                                },
                                children=[
                                    html.H3(
                                        "Raw Data at Selected Residue",
                                        style={"textAlign": "center"},
                                    ),
                                    dash_table.DataTable(
                                        id="raw-data-table",
                                        columns=[
                                            {
                                                "name": "Sequence ID",
                                                "id": "sequence_id",
                                            },
                                            {"name": "Amino Acid", "id": "amino_acid"},
                                        ],
                                        page_size=15,
                                        filter_action="native",
                                        sort_action="native",
                                        style_table={
                                            "overflowX": "auto",
                                            "height": "400px",
                                            "overflowY": "auto",
                                        },
                                        style_cell={
                                            "textAlign": "left",
                                            "padding": "5px",
                                        },
                                        style_header={
                                            "backgroundColor": "lightgrey",
                                            "fontWeight": "bold",
                                        },
                                    ),
                                ],
                            ),
                        ],
                    )
                ],
            ),
        ],
    )

    # --- 6. Define Callbacks for Interactivity ---
    @app.callback(
        [
            Output("pie-chart", "figure"),
            Output("raw-data-table", "data"),
            Output("details-section", "style"),
        ],
        [Input("main-plot", "clickData")],  # Updated to match the new graph ID
    )
    def update_on_click(clickData):
        if clickData is None:
            # Before first click, hide the details section
            pie_fig = go.Figure().update_layout(
                title="Click a residue bar to see details"
            )
            return pie_fig, [], {"display": "none"}

        # Get residue index from click event (x is 1-based, index is 0-based)
        residue_index = clickData["points"][0]["x"] - 1

        # Get data for that column
        column = alignment[:, residue_index]
        counts = Counter(column)

        # --- Create Pie Chart ---
        labels = list(counts.keys())
        values = list(counts.values())
        colors = [aa_colors.get(aa, "#333333") for aa in labels]

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.3,
                    marker_colors=colors,
                    textinfo="percent+label",
                )
            ]
        )
        pie_fig.update_layout(
            title_text=f"Amino Acid Distribution at Residue {residue_index + 1}",
            showlegend=False,
        )

        # --- Prepare Raw Data Table ---
        table_data = [
            {"sequence_id": seq_id, "amino_acid": aa}
            for seq_id, aa in zip(sequence_ids, column)
        ]

        # Show the details section
        details_style = {"display": "block", "marginTop": "20px"}

        return pie_fig, table_data, details_style

    # --- 7. Run the App ---
    url = f"http://127.0.0.1:{port}"
    Timer(1, lambda: webbrowser.open_new(url)).start()

    print(f"Starting Dash server on {url}")
    print("Press Ctrl+C to stop the server.")
    app.run_server(debug=False, port=port)


if __name__ == "__main__":
    CLUSTAL_FILE_PATH = "C:\\Users\\cramerj\\Downloads\\BLAST_500_G6PC1.aln"
    plot_sequence_variability(CLUSTAL_FILE_PATH)
