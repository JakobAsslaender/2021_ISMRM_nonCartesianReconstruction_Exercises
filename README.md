# Exercises for the ISMRM Lecture on the _Reconstruction of Non-Cartesian Data_

`Session: Image Reconstruction`

Live Q&A 5/16/2021 13:45 UTC

Join the live Q&A session if you have questions, or reach out via [email](mailto:jakob.asslaender@nyumc.org)!

### Run the exercises in the cloud
A click on the button below will start a virtual machine on a _binder_ server (can take a couple of minutes) and opens the notebook with the exercises in your browser:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JakobAsslaender/2021_ISMRM_nonCartesianReconstruction_Exercises/HEAD?urlpath=pluto/open?path=/home/jovyan/src/nonCart_PlutoNotebook.jl)


### Run the exercises on your own computer

The computational resources on _binder_ are somewhat limited. The notebook should run a bit faster on your own computer. The installation is rather simple and explained in the following:

- Download [Julia](https://julialang.org/downloads/) (tested on v1.6.1)

- Run Julia 

- Install the exercise package (including all dependencies):

`import Pkg; Pkg.develop(url="https://github.com/JakobAsslaender/2021_ISMRM_nonCartesianReconstruction_Exercises")`

- Import the package:

`import ISMRM_nonCartesianReconstruction_Exercises`

- Start the Pluto notebook with the exercises:

`ISMRM_nonCartesianReconstruction_Exercises.run()`