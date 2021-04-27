module ISMRM_nonCartesianReconstruction_Exercises

import Pluto
import Pkg

function run()
    Pluto.run(notebook=joinpath(dirname(pathof(ISMRM_nonCartesianReconstruction_Exercises)), "nonCart_PlutoNotebook.jl"))
end

end # module
