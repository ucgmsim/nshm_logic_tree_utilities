## Overview

This package has utilities for working with the logic tree used in New Zealand's National Seismic Hazard Model 
(NSHM) 2022. With this package, you can create modified logic trees, use them to calculate new hazard curves, and
plot the results. It depends on several packages developed by GNS Science that are mentioned in requirements.txt.
To extract individual realizations from logic tree branches, modified versions of toshi-hazard-post 
and toshi-hazard-store are needed from forks on ucgmsim's GitHub account. 

## Background

The 2022 revision of Aotearoa New Zealand’s National Seismic Hazard Model (NSHM 2022) is a significant improvement over 
the previous 2010 NSHM. NSHM 2022 uses a logic tree to better quantify modelling (epistemic) uncertainty.  

A logic tree consists of many branches that lead to different predictions reflecting epistemic uncertainty. 
A mean hazard model is produced by combining the individual branches based on their weight (degree of belief). 
The NSHM 2022 logic tree consists of approximately one million branches, making it challenging to intuitively understand 
which branches contribute the most to the hazard and uncertainty. However, this information could inform future research 
directions, making it highly valuable. Therefore, we systematically removed parts of the NSHM 2022 logic tree and 
re-computed the hazard curves.

## Method

GNS Science implemented the NSHM 2022 logic tree using  pre-computed components that can be combined in different ways 
(Di Caprio et al. 2024). We downloaded approximately 500 GB of pre-computed components 
(private communication with Christopher J. DiCaprio), and utilized GNS Science’s open-source Python-based software to 
generate hazard curves from modified logic trees.

With this code, we systematically isolated parts of the logic tree and examined the resulting hazard and epistemic 
uncertainty. We generated hazard curves for a range of intensity measures in the locations of Wellington, 
Christchurch, and Auckland, which have high, moderate, and low seismic hazard, respectively. 
We initially used a range of Vs30 values but as this had little effect, we adopted a constant Vs30 of 400 m/s for 
all locations. The NSHM 2022 logic tree consists of separate component logic trees for the source or seismicity rate 
models (SRM) and the ground motion characterization models (GMCMs). A schematic representation of the component logic 
trees for crustal tectonic region type is shown in Figure 1.

We determined the epistemic uncertainty contributions of the component logic trees by reducing one to the single 
highest weighted branch and leaving the other in its entirety. With this approach, all epistemic uncertainty in 
the generated hazard curves originated from the component logic tree that was left in its entirety. To investigate 
individual models within the SRM or GMCM logic trees, we repeated this procedure with smaller component logic trees. 
To compare the epistemic uncertainty of several models, we followed Bradley (2009) and related the mean prediction to 
the dispersion in predictions for each intensity measure level, as demonstrated in Figure 2.

![Figure 2](https://github.com/username/repository/blob/main/images/example.png)






