### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 65780f00-ed6b-11ea-1ecf-8b35523a7ac0
begin
	using Plots
	plotlyjs()
	using PlutoUI
	using FFTW
	using MRIReco
	using LinearAlgebra
	using IterativeSolvers
end

# ╔═╡ 29dfe3d6-c353-4081-8192-b12f374bf702
filter!(LOAD_PATH) do path
	path != "@v#.#"
end;

# ╔═╡ ac8ff080-ed61-11ea-3650-d9df06123e1f
md"""

# **Excercise for the ISMRM Lecture on _Reconstruction of Non-Cartesian Data_**
`Session: Image Reconstruction`, May 16 2021

`Live session`: **Sunday, 16 May 2021, 13:45 - 14:30**

This notebook contains _built-in, live answer checks_! In some exercises you will see a coloured box, which runs a test case on your code, and provides feedback based on the result. Simply edit the code, run it, and the check runs again.

Feel free to ask questions!
"""

# ╔═╡ 5f95e01a-ee0a-11ea-030c-9dba276aba92
md"""
#### Intializing packages

_When running this notebook for the first time, this could take up to 15 minutes. Julia has a package manager and the following lines load the packages needed for this excercise. Additionally, Julia is compiling the packages. All consequtive code will be compiled just in time to find a compromise between interactivity and speed. Hang in there!_
"""

# ╔═╡ 540ccfcc-ee0a-11ea-15dc-4f8120063397
md"""
## **Exercise 0** - _Julia 101_

Here we just create a numerical phantom and calculate the Cartesian k-space. If you are new to Julia, this code can give a plimpse at its syntax.

Let's create a Shepp-Logan phantom with the Images package:

"""

# ╔═╡ 26709e4b-1675-4ee6-a436-bf9a85683950
nx = 128

# ╔═╡ d72dc575-6d2e-489c-b512-aeae6d017f7f
image_org = reverse(shepp_logan(nx),dims=1)

# ╔═╡ 9391828f-041c-49db-93d8-3483bb672f3d
md"
And let's plot with the Plots package:
"

# ╔═╡ 1adbae4f-c732-457b-aad8-bc0f3a7be13d
heatmap(image_org, c=:grays)

# ╔═╡ c6be8293-7f7c-48a7-b112-92f7ae4fb7c8
md"
FFT functionality is provided by the FFTW package:
"

# ╔═╡ 35022e16-d376-426c-a9f8-b71760bcaaf7
k_Cart = fftshift(fft(image_org))

# ╔═╡ 40a94bcd-a0cc-4b1a-b370-6405adc0f92b
md"Dot syntax: In comparison to Matlab, Julia distinguishes more between functions acting on a matrix as a whole or on each element. Here, abs(kspace) would not work, as the absolute value is not defined for matrices. E.g., abs.(kspace), on the other hand applies takes the absolute value of each element. "

# ╔═╡ c0afaa73-aa46-4f61-a5b4-c49d071f4fb6
abs_k_Cart = abs.(k_Cart)

# ╔═╡ 8a3dc74a-4b20-4c0f-96e8-1ab0451a7da3
try
	heatmap(log.(abs_k_Cart))
catch
end

# ╔═╡ 515ae3f8-fb28-49b9-a023-38d143b6674a
begin
	function calculate_error(reco, org, k_mask)
		k_reco = fftshift(fft(reco))
		k_org  = fftshift(fft(org))
		return norm((k_reco .- k_org)[k_mask]) / norm(k_org[k_mask])
	end
	function calculate_error_norm(reco, org, k_mask)
		reco = reco ./ norm(reco)
		org  = org  ./ norm(org)
		return calculate_error(reco, org, k_mask)
	end
	
	k_mask = similar(k_Cart, Bool)
	for i ∈ CartesianIndices(k_mask)
		if (i[1]-nx/2)^2 + (i[2]-nx/2)^2 > (nx - 4)^2/4
			k_mask[i] = false
		else
			k_mask[i] = true
		end
	end
end

# ╔═╡ ad6a33b0-eded-11ea-324c-cfabfd658b56
md"""
$(html"<br>")
## Exerise 1 - _filtered back-projection_
Let's create a trajectory with the MRIReco.jl package
"""

# ╔═╡ 28668257-321b-41f2-bfe2-d5f35441ec8d
nspokes = Int(ceil(nx * pi/2)) + 20

# ╔═╡ 04962733-71dd-4e6f-9e23-5b29ee821d68
nr = 2 * nx

# ╔═╡ 7769e12c-7fb5-44df-bbd0-d8ee714098e8
trj = trajectory("Radial", nspokes, nr)

# ╔═╡ 4ded3325-4935-4843-b9c2-b0e29469bf3e
plot(trj.nodes[1,:], trj.nodes[2,:])

# ╔═╡ c0f04553-6031-4984-8c80-1260022eb23f
md"With the same package we can create a non-uniform FFT (NFFT) operator:"

# ╔═╡ 44818050-cafa-4087-bed6-b7ef4336fc7e
md"As `F` is a linear operator, it essentially acts like a matrix and we can calculate the k-space siganl with something that looks like a matrix-vector multiplication:"

# ╔═╡ f6d6fe5e-3381-4338-817d-366194b3f7ba
function calculate_k_radial(F, image_org)
	k_radial = F * vec(image_org)
	return k_radial
end

# ╔═╡ 424e1625-e14b-4fcb-bd5b-7e426c97d330
md"Let's try to reconstruct this data. As discussed in the lecture, we need to apply density compensation. Modify below line to compute the correct density compensation for radial imaging!

Tip: For radial imaging, the density is only a function of `r`, so you only need to replace `ones(nr)`."

# ╔═╡ 16ad1b20-f3c8-4516-b5d9-6150875c326c
density_compensation = sqrt.(trj.nodes[1,:].^2 + trj.nodes[2,:].^2)
#density_compensation = repeat(ones(nr), nspokes)

# ╔═╡ 9aff92eb-8094-4a83-ad27-242277b9aee1
md"With the correct density compensation, we can reconstruct the image:"

# ╔═╡ 9d70e2e6-6e45-4662-9792-b36ab36c35b4
md"""
## Exerise 2 - _iterative reconstruction_
In case you didn't get the density compensation right: don't worry. As discussed in the lecture, we can avoid the density compensation by using iterative algorithms. 

Here we will use conjugate gradients (CG), which is usually the method of choice for solving linear problem iteratively. To recall, CG solves

$\hat{x} = \arg \min ||Ax - b||_2^2$

where A is a symmetric, positive-definite matrix. In order to bring our imaging problem in the right form we have to define 

$A = F' F$

and

$b = F' s,$

where $F$ is our NFFT-Operator, $F'$ its adjoint, and $s$ is the measured signal. 

Great, so let's do that:
"""

# ╔═╡ 698d0344-ec5d-4c27-a2b1-b52825e398d6
md"Now we have everything in place to run a reconstruction with conjugate gradients:"

# ╔═╡ f6447d81-6612-428d-b3d7-acd7f8756df5
md"""
## Exerise 3 - _the puzzle_

Great, we have mastered back-projection and interative reconstructions. When you look at the definition of `F`, you might be confused by the definition of `kernelSize` and `oversamplingFactor`. In fact they are defined by these two sliders:
"""

# ╔═╡ 98a1c04c-58a3-44f6-995f-5d4d81095576
md"kernelSize = " 

# ╔═╡ e6af6037-7eb6-40be-a1b9-846ab1f50b9c
@bind kernelSize Slider(1:9, show_value=true)

# ╔═╡ 8075e80d-b6df-435a-ac4d-44a919909323
md"oversamplingFactor = "

# ╔═╡ e4d80b4c-faa3-4514-876b-63bce1ece81f
@bind oversamplingFactor Slider(1:0.05:2, show_value=true)

# ╔═╡ beccb301-1c40-4c41-8960-462a82eca3ad
F = NFFTOp(size(image_org), trj; oversamplingFactor=oversamplingFactor, kernelSize=kernelSize)

# ╔═╡ 32bc54d1-4ea8-4663-ba9a-03d357eb8c02
#A = randn(256, 256)
A = F' * F

# ╔═╡ 62934cdb-526d-4f97-afc0-20bcd9f9714e
md"""
So we have initialized the NFFT operator with `oversamplingFactor=1` and `kernelSize=1`, yet we get good results. This is in sharp contrast to theory discussed in the lecture. 

$(@bind ic_answer Select(["undefined" => "", "green" => "The image looks great, the box is green, and theory has rarely practical value. Let's enjoy the nice weather!", "teacher" => "The teacher said this can't be right, so I guess it's not...", "correct" => "I contemplated about the theory and it's correct. By logical conclusion, there must be something wrong here!"]))

"""

# ╔═╡ aa209a90-0dfb-41ff-bf0e-a26a6c1b8c8b
function dft(image_org, trj)
	k_radial_dft = zeros(ComplexF64, size(trj,2))
	nx2 = size(image_org,1)/2 + 1
	Threads.@threads for i = 1:length(k_radial_dft)
		for j ∈ CartesianIndices(image_org)
	    	k_radial_dft[i] += image_org[j] *
				exp(-1im * 2π * (trj.nodes[1,i] * (j[1]-nx2) + 
  							     trj.nodes[2,i] * (j[2]-nx2)))
		end
	end
	return k_radial_dft
end

# ╔═╡ a0e0fba0-39e1-4873-8d7e-ee5cb793062c
@bind use_dft Select(["no" => "Use NFFT Operator to simulate the signal (inverse crime)", "yes" => "Use discrete the discrete Fourier transform to calculate the signal (slow, but correct)"])

# ╔═╡ 86af3791-f3e6-45bc-bc4e-072803491d1f
if use_dft == "yes"
	k_radial = dft(image_org, trj)
else
	Fforward = NFFTOp(size(image_org), trj; oversamplingFactor=1, kernelSize=1)
	k_radial = calculate_k_radial(Fforward, image_org)
end

# ╔═╡ b114e592-0d39-43cb-aee3-c0948b2d2f88
reco_bp = reshape(F' * (k_radial .* density_compensation), (nx, nx))

# ╔═╡ b62f8883-02cd-4ab0-9c27-61283e4839f6
try
	heatmap(abs.(reco_bp), c=:grays)
catch
end

# ╔═╡ 51b2f56b-3880-4ba6-828a-104334bad1ec
#b = randn(256)
b = F' * k_radial

# ╔═╡ 57a2c882-760c-4ff9-9bcb-be8f358d7b8a
reco_cg = reshape(cg(A, b, maxiter=50), (isqrt(size(A,1)),isqrt(size(A,1))))

# ╔═╡ ce933bda-ac45-46ec-b577-c4168b1ce569
heatmap(abs.(reco_cg), c=:grays, clim=(0,1))

# ╔═╡ 756d150a-b7bf-4bf5-b372-5b0efa80d987
md"## Function library

Just some helper functions used in the notebook."

# ╔═╡ 4bc94bec-da39-4f8a-82ee-9953ed73b6a4
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

# ╔═╡ aefd5a3d-9b7d-40a1-9cc8-1a5b555d251d
md"""
You can find out more about any function by clicking on the Live Docs in the bottom right of this Pluto window, and typing a function name in the top.

![image](https://user-images.githubusercontent.com/6933510/107848812-c934df80-6df6-11eb-8a32-663d802f5d11.png)


![image](https://user-images.githubusercontent.com/6933510/107848846-0f8a3e80-6df7-11eb-818a-7271ecb9e127.png)

We recommend that you leave the window open while you work on Julia code. It will continually look up documentation for anything you type!
""" |> hint

# ╔═╡ 7906f103-65db-45c6-9d95-ef1483048c14
md"""
The sampling density of radial trajectories increases as we approach the k-space center, more precisely it is proportional to 1/r, where r is the distance from the k-space center. 

When back-projecting, we have too much signal in the center of k-space, hence the blurred image. To compensate, we, thus, have to multiply the k-space data with 1/f. Try again! 

If you still have not solved it, below hint will give you the solution. 
""" |> hint

# ╔═╡ c604ce61-08ac-46af-b231-a99e37f52bd6
md"""
The correct answer is

```
density_compensation = repeat(abs.(-nr/2 : nr/2-1), nspokes)
```

Alterantively, this is also correct:

```
density_compensation = sqrt.(trj.nodes[1,:].^2 + trj.nodes[2,:].^2)
```

""" |> hint

# ╔═╡ 8ce6ad06-819c-4af5-bed7-56ecc08c97be
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

# ╔═╡ dfa40e89-03fc-4a7a-825e-92d67ee217b2
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

# ╔═╡ 086ec1ff-b62d-4566-9973-5b2cc3353409
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

# ╔═╡ 2f6fb3a6-bb5d-4c7a-9297-84dd4b16c7c3
yays = [md"Fantastic!", md"Splendid!", md"Great!", md"Yay ❤", md"Great! 🎉", md"Well done!", md"Keep it up!", md"Good job!", md"Awesome!", md"You got the right answer!", md"Let's move on to the next section."];

# ╔═╡ c22f688b-dc04-4a94-b541-fe06266c5446
correct(text=rand(yays)) = Markdown.MD(Markdown.Admonition("correct", "Got it!", [text]))

# ╔═╡ ab3d1b70-88e8-4118-8d3e-601a8a68f72d
not_defined(variable_name) = Markdown.MD(Markdown.Admonition("danger", "Oopsie!", [md"Make sure that you define a variable called **$(Markdown.Code(string(variable_name)))**"]))

# ╔═╡ 810b38a1-490f-4e22-8edb-0e50a8af2f28
if !@isdefined(abs_k_Cart)
	not_defined(:abs_k_Cart)
elseif ismissing(abs_k_Cart)
	still_missing()
elseif !(abs_k_Cart isa Matrix)
	keep_working(md"`abs_kspace` should be a `Matrix`.")
elseif eltype(abs_k_Cart) != Float64
	mytype = typeof(abs_k_Cart[1]);
	almost(md"""
		You generated a matrix, but the elements are of type *$(mytype)*. Instead, they 		should be of type `Float64`.
		""")
elseif size(abs_k_Cart) != size(abs_k_Cart)
	keep_working(md"`abs_kspace` does not have the correct size.")
elseif abs_k_Cart == abs.(k_Cart)
	correct(md"Well done! A Julia Pro in the making! Now we can also plot the absolute value the k-space data (logarithmically):")
else
	keep_working(md"Still something not quite right...")
end

# ╔═╡ a05cce2b-b1b7-4d8c-a823-137b90473d70
begin
	bp_error = round(calculate_error_norm(reco_bp, image_org, k_mask) * 100)/100
	if !@isdefined(reco_bp)
		not_defined(:reco_bp)
	elseif ismissing(reco_bp)
		still_missing()
	elseif all(density_compensation .== 1)
		still_missing(md"modify `density_compensation` to correct for the sampling density of the radial k-space trajectory.")
	elseif !(reco_bp isa Matrix)
		keep_working(md"`reco` should be a `Matrix`.")
	elseif eltype(reco_bp) != ComplexF64
		type_reco = typeof(reco_bp[1]);
		almost(md"""
			You generated a matrix, but the elements are of type *$(type_reco)*. Instead, they 		should be of type `ComplexF64`.
			""")
	elseif size(reco_bp) != size(image_org)
		keep_working(md"`reco` does not have the correct size. I should be the same as `image_org`")
	elseif bp_error > 0.25
		keep_working(md"You are getting an image, but it's not quite looking like `image_org`: the normalized RMSE $bp_error. It should be below 0.25")
	else
		correct(md"Well done! You master the art of filtered back-projection! The normalized RMSE is $bp_error, that's about as good as it gets with filtered backprojection")
	end
end

# ╔═╡ a491a997-0a95-491a-8ca2-1ff85f5f39d3
begin
	if !@isdefined(A)
		not_defined(:A)
	elseif !@isdefined(b)
		not_defined(:b)
	elseif !@isdefined(reco_cg) || ismissing(A) || ismissing(b) || A isa Matrix{Float64}
		still_missing(md"define `A` and `b` correctly!")
	elseif !(A isa LinearOperator{ComplexF64})
		keep_working(md"`A` should be a `LinearOperator{ComplexF64}`.")
	elseif !(b isa Vector{ComplexF64})
		keep_working(md"`b` should be a `Vector{ComplexF64}`.")
	elseif size(reco_cg) != size(image_org)
		keep_working(md"`reco` does not have the correct size. I should be the same as `image_org`")
	else
		error_cg = round(calculate_error(reco_cg, image_org, k_mask) * 10000) / 10000
		if error_cg > 0.002
			keep_working(md"You are getting an image, but it's not quite looking like `image_org`. The normalized RMSE is $error_cg, but it should be below 0.002.")
		else
			correct(md"Well done! You master the art of iterative reconstructions! The normalized RMSE is $error_cg. ")
		end
	end
end

# ╔═╡ ed4ef764-e377-4af3-b3a5-86aa442396e8
if !@isdefined(ic_answer)
	not_defined(ic_answer)
elseif ic_answer == "undefined"
	still_missing(md"Choose an answer!")
elseif ic_answer == "green"
	keep_working(md"Nice weather solves almost everything, but not an inverse crime!")
elseif ic_answer == "teacher"
	almost(md"Science is the belief in the ignorance of experts - Richard P. Feynman")
else
	correct(md"What you have discovered in called an *inverse crime*. With `kernelSize=1` we essentially perform nearest-neighbor interpolation. Thus, we just copy the Cartesian k-space data point onto the non-Cartesian grid without changing it, and we copy the same data point back to the Cartesian grid when reconstructing. The approximation errors cancel each other out and we get great results that have little to do with reality. 
		
To overcome this issue, we should always simulate the forward problem as accurately as possible and use approximations only in the reconstruction. Here, we can simulate the non-Cartesian k-space data *by foot* with the slow discrete Fourier transform:
		")

end

# ╔═╡ 8cb0aee8-5774-4490-9b9e-ada93416c089
todo(text) = HTML("""<div
	style="background: rgb(220, 200, 255); padding: 2em; border-radius: 1em;"
	><h1>TODO</h1>$(repr(MIME"text/html"(), text))</div>""")

# ╔═╡ 4dd0670b-d84e-4f7f-bfda-ec0628d9ed51
bigbreak = html"<br><br><br><br><br>";

# ╔═╡ 4bccda27-d96b-4234-b5e4-933422b4a157
bigbreak

# ╔═╡ 91f4778e-ee20-11ea-1b7e-2b0892bd3c0f
bigbreak

# ╔═╡ Cell order:
# ╟─ac8ff080-ed61-11ea-3650-d9df06123e1f
# ╟─5f95e01a-ee0a-11ea-030c-9dba276aba92
# ╠═65780f00-ed6b-11ea-1ecf-8b35523a7ac0
# ╟─29dfe3d6-c353-4081-8192-b12f374bf702
# ╟─540ccfcc-ee0a-11ea-15dc-4f8120063397
# ╠═26709e4b-1675-4ee6-a436-bf9a85683950
# ╠═d72dc575-6d2e-489c-b512-aeae6d017f7f
# ╟─9391828f-041c-49db-93d8-3483bb672f3d
# ╠═1adbae4f-c732-457b-aad8-bc0f3a7be13d
# ╟─c6be8293-7f7c-48a7-b112-92f7ae4fb7c8
# ╠═35022e16-d376-426c-a9f8-b71760bcaaf7
# ╟─40a94bcd-a0cc-4b1a-b370-6405adc0f92b
# ╠═c0afaa73-aa46-4f61-a5b4-c49d071f4fb6
# ╟─810b38a1-490f-4e22-8edb-0e50a8af2f28
# ╟─8a3dc74a-4b20-4c0f-96e8-1ab0451a7da3
# ╟─aefd5a3d-9b7d-40a1-9cc8-1a5b555d251d
# ╠═515ae3f8-fb28-49b9-a023-38d143b6674a
# ╠═ad6a33b0-eded-11ea-324c-cfabfd658b56
# ╠═28668257-321b-41f2-bfe2-d5f35441ec8d
# ╠═04962733-71dd-4e6f-9e23-5b29ee821d68
# ╠═7769e12c-7fb5-44df-bbd0-d8ee714098e8
# ╠═4ded3325-4935-4843-b9c2-b0e29469bf3e
# ╟─c0f04553-6031-4984-8c80-1260022eb23f
# ╠═beccb301-1c40-4c41-8960-462a82eca3ad
# ╟─44818050-cafa-4087-bed6-b7ef4336fc7e
# ╠═f6d6fe5e-3381-4338-817d-366194b3f7ba
# ╟─424e1625-e14b-4fcb-bd5b-7e426c97d330
# ╠═16ad1b20-f3c8-4516-b5d9-6150875c326c
# ╟─a05cce2b-b1b7-4d8c-a823-137b90473d70
# ╟─7906f103-65db-45c6-9d95-ef1483048c14
# ╟─c604ce61-08ac-46af-b231-a99e37f52bd6
# ╟─9aff92eb-8094-4a83-ad27-242277b9aee1
# ╠═b114e592-0d39-43cb-aee3-c0948b2d2f88
# ╟─b62f8883-02cd-4ab0-9c27-61283e4839f6
# ╟─9d70e2e6-6e45-4662-9792-b36ab36c35b4
# ╠═32bc54d1-4ea8-4663-ba9a-03d357eb8c02
# ╠═51b2f56b-3880-4ba6-828a-104334bad1ec
# ╟─698d0344-ec5d-4c27-a2b1-b52825e398d6
# ╠═57a2c882-760c-4ff9-9bcb-be8f358d7b8a
# ╟─a491a997-0a95-491a-8ca2-1ff85f5f39d3
# ╟─ce933bda-ac45-46ec-b577-c4168b1ce569
# ╟─f6447d81-6612-428d-b3d7-acd7f8756df5
# ╟─98a1c04c-58a3-44f6-995f-5d4d81095576
# ╟─e6af6037-7eb6-40be-a1b9-846ab1f50b9c
# ╟─8075e80d-b6df-435a-ac4d-44a919909323
# ╟─e4d80b4c-faa3-4514-876b-63bce1ece81f
# ╟─62934cdb-526d-4f97-afc0-20bcd9f9714e
# ╟─ed4ef764-e377-4af3-b3a5-86aa442396e8
# ╠═aa209a90-0dfb-41ff-bf0e-a26a6c1b8c8b
# ╟─a0e0fba0-39e1-4873-8d7e-ee5cb793062c
# ╠═86af3791-f3e6-45bc-bc4e-072803491d1f
# ╟─4bccda27-d96b-4234-b5e4-933422b4a157
# ╟─91f4778e-ee20-11ea-1b7e-2b0892bd3c0f
# ╟─756d150a-b7bf-4bf5-b372-5b0efa80d987
# ╟─4bc94bec-da39-4f8a-82ee-9953ed73b6a4
# ╟─8ce6ad06-819c-4af5-bed7-56ecc08c97be
# ╟─dfa40e89-03fc-4a7a-825e-92d67ee217b2
# ╟─086ec1ff-b62d-4566-9973-5b2cc3353409
# ╟─2f6fb3a6-bb5d-4c7a-9297-84dd4b16c7c3
# ╟─c22f688b-dc04-4a94-b541-fe06266c5446
# ╟─ab3d1b70-88e8-4118-8d3e-601a8a68f72d
# ╟─8cb0aee8-5774-4490-9b9e-ada93416c089
# ╟─4dd0670b-d84e-4f7f-bfda-ec0628d9ed51
