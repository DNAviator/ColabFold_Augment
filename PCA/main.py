"""
Main script to simplify setup for multiple runs
"""

import timeit
from multi_gene_PCA import PCACalculator, RunParams, AlignmentParams
from PCA_visualization import PCAVisualizer, ClusteringParams, VisualizationParams
from movement_analyzer import MovementAnalyzer, AnalysisParams


def time_function(func, *args, **kwargs):
    """Utility function to time the execution of a function."""
    start_time = timeit.default_timer()
    print(f"Starting function {func.__name__} || time: {start_time:.2f}")
    result = func(*args, **kwargs)
    elapsed = timeit.default_timer() - start_time
    print(f"Function {func.__name__} completed in {elapsed:.2f} seconds.")
    return result


if __name__ == "__main__":
    PDB_DIRECTORIES_BASE = (
        "C:\\Users\\cramerj\\OneDrive - Vanderbilt\\Documents\\AF2_result"
    )
    # Run Params
    RUN_PARAMS = RunParams(
        pdb_dirs=[
            f"{PDB_DIRECTORIES_BASE}\\{_}"
            for _ in [
                # "Penguin",
                "Penguin_w_dropout",
                # "Bos_mutus_w_dropout",
                # "Camelus_ferus_w_dropout",
                "Chrysemys_picta_bellii_w_dropout",
                "Dryophytes_japonicus_w_dropout",
                "Jaculus_jaculus_w_dropout",
                "Nanorana_parkeri_w_dropout",
                # "All_Wood_Frog",
                "Wood_Frog_w_dropout_all",
                # "Human_w_dropout",
                "Mus_Musculus_w_dropout",
                # "African_Elephant_w_dropout",
                "Chinese_Turtle_w_dropout",
                "Rattus_Norvegicus_w_dropout",
                "Human_Chimeric_Mus_Musculus_136-147_287-291_344-357_w_dropout",
                # "Human_Chimeric_Mus_Musculus_136-147_287-291_w_dropout",
                "Human_Chimeric_Mus_Musculus_136-147_344-357_w_dropout",
                # "Human_Chimeric_Mus_Musculus_136-147_w_dropout",
                "Human_Chimeric_Mus_Musculus_287-291_344-357_w_dropout",
                # "Human_Chimeric_Mus_Musculus_287-291_w_dropout",
            ]
        ],
        msa_path="C:\\Users\\cramerj\\Code\\ColabFold_Augmentation\\ColabFold_Augment\\PCA\\Alignments\\20_align_w_chimeric_Jul_31.clustal_num",
        output_dir="C:\\Users\\cramerj\\Code\\ColabFold_Augmentation\\ColabFold_Augment\\PCA\\PCA_output",
        run_name="No_Proline_Cluster_Dropout_BB_Jul_31_01",
    )

    ALIGNMENT_PARAMS = AlignmentParams(
        atom_selection="backbone",
        num_iterations=3,
        residues_to_exclude=[
            1,
            5,
            18,
            23,
            24,
            25,
            26,
            28,
            29,
            30,
            33,
            35,
            36,
            37,
            38,
            42,
            47,
            48,
            49,
            51,
            52,
            53,
            54,
            58,
            75,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            95,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            134,
            138,
            139,
            145,
            146,
            147,
            148,
            149,
            168,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            192,
            194,
            198,
            201,
            202,
            203,
            204,
            205,
            208,
            209,
            225,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            243,
            244,
            245,
            248,
            249,
            250,
            251,
            252,
            255,
            256,
            257,
            258,
            259,
            260,
            271,
            273,
            274,
            275,
            276,
            277,
            278,
            279,
            281,
            282,
            283,
            284,
            285,
            287,
            290,
            291,
            295,
            306,
            314,
            315,
            316,
            317,
            318,
            319,
            320,
            321,
            322,
            345,
            351,
            352,
            353,
            354,
            355,
            361,
        ],
    )

    CLUSTERING_PARAMS = ClusteringParams(
        range_pcs_for_clustering=[1, 3],
        min_cluster_size=15,
    )

    VISUALIZATION_PARAMS = VisualizationParams(
        pcs_to_plot_2d=[[1, 2], [1, 3], [2, 3]],
        pcs_to_plot_3d=[1, 2, 3],
        num_frames=100,
        cluster_start=0,
        cluster_end=5,
    )

    ANALYSIS_PARAMS = AnalysisParams(
        pcs_for_raw_report=[1,],
        pcs_for_summary_report=[1,],
        movement_threshold=0.05,
    )

    # PCA Calculator

    # Initialize the MultiGenePCA class
    pca = PCACalculator(
        run_params=RUN_PARAMS,
        alignment_params=ALIGNMENT_PARAMS,
    )
    time_function(pca.run)

    # PCA Visualizer
    pca_visualizer = PCAVisualizer(
        run_params=RUN_PARAMS,
        clustering_params=CLUSTERING_PARAMS,
        viz_params=VISUALIZATION_PARAMS,
    )
    time_function(pca_visualizer.run_clustering)
    time_function(pca_visualizer.generate_plots)
    # time_function(pca_visualizer.generate_animation)
    # time_function(pca_visualizer.generate_putty_models)

    # Movement Analyzer
    movement_analyzer = MovementAnalyzer(RUN_PARAMS, ANALYSIS_PARAMS)
    time_function(movement_analyzer.analyze)
