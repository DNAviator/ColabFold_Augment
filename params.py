class params:
    def __init__(self) -> None:

        # Input options
        self.sequence = None
        self.jobname = None
        self.copies = 1

        # MSA options
        self.msa_method = None #["mmseqs2","single_sequence", "custom_fas", "custom_a3m", "custom_sto"]
        self.pair_mode = None #["unpaired", "paired"]

        # Filtering options
        self.cov = 75 # ["0", "25", "50", "75", "90", "99"] {type:"raw"}
        self.id = 90 # ["90", "100"] {type:"raw"}
        self.qid = 0 # ["0", "10", "15", "20", "30"] {type:"raw"}
        self.do_not_filter = False # {type:"boolean"}

        # Template options
        self.template_mode = "none"  #  ["none", "mmseqs2", "custom"] {type:"string"}
        self.pdb = ""  #  {type:"string"}
        self.chain = "A"  #  {type:"string"}
        self.rm_template_seq = False  #  {type:"boolean"}
        self.propagate_to_copies = True  #  {type:"boolean"}
        self.do_not_align = False  #  {type:"boolean"}

        # Pre-analysis
        self.pre_analysis = "none"  #  ["none", "coevolution"]
        self.pre_analysis_dpi = 100  #  ["100", "200", "300"] {type:"raw"}

        # Model options
        self.model_type = "auto"  #  ["monomer (ptm)", "pseudo_multimer (v3)", "multimer (v3)", "auto"]
        self.rank_by = "auto"  #  ["auto", "plddt", "ptm"]
        self.debug = False  #  {type:"boolean"}
        self.use_initial_guess = False

        # MSA options
        self.num_msa = (
            512  #  ["1","2","4","8","16","32", "64", "128", "256", "512"] {type:"raw"}
        )
        self.num_extra_msa = 1024  #  ["1","2","4","8","16","32", "64", "128", "256", "512", "1024","2048","4096"] {type:"raw"}
        self.use_cluster_profile = True  #  {type:"boolean"}

        # Extended metrics (calculate pairwise ipTM, actifpTM and chain pTM)
        self.calc_extended_ptm = True  #  {type:"boolean"}

        # Run AlphaFold options
        self.model = "all"  #  ["1", "2", "3", "4", "5", "all"]
        self.num_recycles = 6  #  ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
        self.recycle_early_stop_tolerance = 0.5  #  ["0.0", "0.5", "1.0"] {type:"raw"}
        self.select_best_across_recycles = False  #  {type:"boolean"}

        # Stochastic options
        self.use_mlm = True  #  {type:"boolean"}
        self.use_dropout = False  #  {type:"boolean"}
        self.seed = 0  #  {type:"raw"}
        self.num_seeds = 1  #  ["1", "2", "4", "8", "16", "32", "64", "128"] {type:"raw"}

        # Extras
        self.show_images = True  #  {type:"boolean"}

        # Display best result (optional) {run: "auto"}
        self.color = "pLDDT"  #  ["pLDDT","chain","rainbow"]
        self.show_sidechains = False  #  {type:"boolean"}
        self.show_mainchains = False  #  {type:"boolean"}
        self.color_HP = True

        # Post analysis (optional)
        self.post_analysis = "animate_all_results"  #  ["none", "display_top5_results", "animate_all_results", "coevolution"]
        self.post_analysis_dpi = 100  #  ["100", "200"] {type:"raw"}

    def items(self):
        return {key: value for key, value in self.__dict__.items()}
