TODO: unwrap this to the rest of the obsidian notebook

flowchart TB

    subgraph MATH [Math]

        subgraph DIFFUSION [Diffusion]

            DIFFUSION[<a href='https://arxiv.org/abs/1503.03585'>Nonequilibrium Thermodynamics</a>]

            NOISESDE[<a href='https://arxiv.org/pdf/2011.13456'>NOISESDE</a>]

        end

        STOCHASTICINTERPOLANTS[<a href='https://arxiv.org/pdf/2303.08797'>Stochastic Interpolants</a>]

        NORMALIZINGFLOWS[<a href='https://arxiv.org/pdf/2209.15571'>Normalizing Flows</a>]

        MANIFOLDS[<a href='https://arxiv.org/pdf/2202.02763'>Riemannian</a>]

        SE3[<a href='https://arxiv.org/pdf/2310.02391'>SE3 Flow</a>]

        GEOMETRYFLOW[<a href='https://arxiv.org/pdf/2302.03660'>Flow on general Geometries</a>]

        ROFORMER[<a href='https://arxiv.org/pdf/2104.09864'>RoFormer</a>]

        MINIBATCHOT[<a href='https://arxiv.org/pdf/2302.00482'>Minibatch OT</a>]

        FLOWMATCHING[<a href='https://arxiv.org/pdf/2210.02747'>Flowmatching</a>]

        RECTIFIED[<a href='https://arxiv.org/pdf/2209.03003'>Rectified Flows</a>]

        MINIBATCHCOUPLINGS[<a href='https://arxiv.org/pdf/2304.14772'>Minibatch Couplings</a>]

    end

    subgraph METHODS [Methods]

        CONDGENMODELS[<a href='https://arxiv.org/pdf/2212.00362'>Cond-Generative Models</a>]

        FOLDSEEK[<a href='https://www.biorxiv.org/content/10.1101/2023.03.09.531927v1.full.pdf'>FoldSeek</a>]

        STABLEDIFFUSION3[<a href='https://arxiv.org/pdf/2403.03206'>StableDiffusion 3</a>]

        EVOLUTION500[<a href='https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1.full.pdf'>500 Million Years</a>]

        CLASSIFIERFREE[<a href='https://arxiv.org/abs/2207.12598'>Classifier Free Guidance</a>]

        DDPM[<a href='https://arxiv.org/abs/2006.11239'>DDPM</a>]

        LORA[<a href='https://arxiv.org/abs/2106.09685'>LORA</a>]

        ATTENTIONSINKS[<a href='https://arxiv.org/pdf/2309.17453'>Attention Sinks</a>]

        SE3BACKBONES[<a href='https://arxiv.org/pdf/2309.17453'>SE3 Backbones</a>]

        SE3BACKBONES2[<a href='https://arxiv.org/pdf/2302.02277'>SE3 Backbones2</a>]

        SE3BACKBONES3[<a href='https://arxiv.org/pdf/2401.04082'>SE3 Backbones3</a>]

    end

    subgraph TRANSFORMERS [Transformers]

        SCALING[<a href='https://arxiv.org/pdf/2302.05442'>Scaling</a>]

        PROTTRANS[<a href='https://arxiv.org/pdf/2007.06225'>ProtTrans</a>]

        REGISTERS[<a href='https://arxiv.org/pdf/2309.16588'>DINOv2</a>]

    end

    subgraph BACKGROUND [Background]

        ALHPA3[<a href='https://papers.baulab.info/papers/also/Abramson-2024.pdf'>Alpha Fold 3</a>]

        PROTEINMPNN[<a href='https://www.science.org/doi/10.1126/science.add2187'>ProteinMPNN</a>]

        CATH[<a href='https://academic.oup.com/nar/article/45/D1/D289/2605733'>Cath</a>]

        COMINGOFAGE[<a href='https://www.bakerlab.org/wp-content/uploads/2016/09/HuangBoyken_DeNovoDesign_Nature2016.pdf'>Coming of Age</a>]

        ALPHA1[<a href='https://www.nature.com/articles/s41586-021-03819-2'>Alphafold 1</a>]

        ADVANCES[<a href='https://sci-hub.se/10.1038/s41580-019-0163-x'>Advances 2019</a>]

        OG[<a href='https://sci-hub.se/10.1016/0968-0004_89_90070-4'>OG</a>]

        RFDIFFUSION[<a href='https://www.bakerlab.org/wp-content/uploads/2023/07/s41586-023-06415-8_reference.pdf'>RFDiffusion</a>]

    end

  

    PROTEINA[<a href='https://arxiv.org/pdf/2503.00710'>Proteina</a>]

    PROTEINA --> BACKGROUND

    PROTEINA --> MATH

    PROTEINA --> METHODS

    PROTEINA --> TRANSFORMERS

  
  

    %% TODO: Break up into subgraphs

    style PROTEINA fill:#008000