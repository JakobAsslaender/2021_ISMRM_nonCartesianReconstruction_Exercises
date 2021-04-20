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
	import Pkg
	Pkg.activate("..")
	using Plots
	plotlyjs()
	using PlutoUI
	using FFTW
	using MRIReco
	using LinearAlgebra
	using IterativeSolvers
end

# ╔═╡ ac8ff080-ed61-11ea-3650-d9df06123e1f
md"""

# **Exercise for the ISMRM Lecture on the _Reconstruction of Non-Cartesian Data_**
`Session: Image Reconstruction`
"""

# ╔═╡ a24b4518-e6c3-4845-8b32-b203af6a2d69
html"""
Live Q&A in your local time zone:
<p id="demo"></p>
<script>
document.getElementById("demo").innerHTML = new Date('5/16/2021 13:45 UTC').toString();
</script>

"""

# ╔═╡ aff9b50d-d013-4c8b-8510-9a59e0c9fc2e
md"""
This notebook contains _built-in, live answer checks_! In some exercises you will see a colored box, which runs a test case on your code, and provides feedback based on the result. Simply edit the code, run it (`Shift` + `Enter`), and the check runs again as the notebook _reactive_: whenever you change a cell, all affected cells are evaluated, which will become important in Exercises 3-4.

Join the live Q&A session if you have questions, or reach out via [email](mailto:jakob.asslaender@nyumc.org)!
"""

# ╔═╡ 5f95e01a-ee0a-11ea-030c-9dba276aba92
md"""
#### Load packages

Most functionality in Julia in provided by so-called _packages_. On binder, I already installed all required packages. If you are running this exercise on your own computer, you installed everything with the `Pkg.develop(url="https://github.com/JakobAsslaender/2021_ISMRM_nonCartesianReconstruction_Exercises")` command. The only remaining step is to make those packages available:
"""

# ╔═╡ 540ccfcc-ee0a-11ea-15dc-4f8120063397
md"""
## **Exercise 0** - _Julia 101_

Let's start by creating a numerical phantom and calculate the Cartesian k-space. If you are new to Julia, this code will give a glimpse at its syntax.

Let's choose a (quadratic) matrix size:

"""

# ╔═╡ 26709e4b-1675-4ee6-a436-bf9a85683950
nx = 128

# ╔═╡ 97a4d58f-9c32-4de9-9142-31e34fc4e33b
md"""
If you are running this exercise on binder, I would suggest to stick to $n_x \leq 128$ as computational resources on binder are rather limited. If you are runnings it on your own beafy computer, feel free to increase this value!

Let's create a Shepp-Logan phantom:
"""

# ╔═╡ d72dc575-6d2e-489c-b512-aeae6d017f7f
image_org = reverse(shepp_logan(nx),dims=1)

# ╔═╡ 9391828f-041c-49db-93d8-3483bb672f3d
md"
And let's plot it with the `Plots` package:
"

# ╔═╡ 67bedf95-8ff4-430f-8737-a0d21ff368c8
md"Color scaling:"

# ╔═╡ eeed0b36-37be-4af4-a8c2-68c595d6bb2c
@bind cmax_org Slider(0.5:0.1:1.0, default=1.0)

# ╔═╡ c6be8293-7f7c-48a7-b112-92f7ae4fb7c8
md"
FFT functionality is provided by the `FFTW` package and the syntax is very similar to Matlab:
"

# ╔═╡ 35022e16-d376-426c-a9f8-b71760bcaaf7
k_Cart = fftshift(fft(image_org))

# ╔═╡ 40a94bcd-a0cc-4b1a-b370-6405adc0f92b
md"**Dot syntax**: In comparison to Matlab, Julia distinguishes more explicitly between functions that act on a matrix as a whole or on each element. E.g. `exp(x)` calculates the matrix exponential of `x`, while `exp.(x)` calculates the exponential of each element of `x`.

Use the dot-syntax to calculate the absolute value `abs` of `k_Cart`:
. "

# ╔═╡ c0afaa73-aa46-4f61-a5b4-c49d071f4fb6
abs_k_Cart = missing

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
## Exercise 1 - _filtered back-projection_

For Exercises 1-4, we will use a radial trajectory. In radial sampling, we need to acquire $\pi/2$-times as many _spokes_ compared to Cartesian k-space lines in order to fulfill the Nyquist theorem at the edge of k-space. This is the price you pay for a higher sampling density in the center:
"""

# ╔═╡ 28668257-321b-41f2-bfe2-d5f35441ec8d
nspokes = ceil(Int, nx * π/2 + 20)

# ╔═╡ 137ed83c-171f-48df-be8e-dc538ac18929
md"
The added 20 improve the numerical accuracy. In practice, this should not be necessary.

In the readout direction, we use the usual 2-fold oversampling:
"

# ╔═╡ 04962733-71dd-4e6f-9e23-5b29ee821d68
nr = 2 * nx

# ╔═╡ 74d30291-506f-4e57-9241-c791f0fb8264
md"We can use the MRIReco.jl package to create a trajectory:"

# ╔═╡ 319872dd-d646-4029-9319-4ffb957179a0
trj = trajectory("Radial", nspokes, nr)

# ╔═╡ 4ded3325-4935-4843-b9c2-b0e29469bf3e
plot(trj.nodes[1,:], trj.nodes[2,:], legend=false, xlabel="kx (rad/voxel/2pi)", ylabel="ky (rad/voxel/2pi)")

# ╔═╡ c0f04553-6031-4984-8c80-1260022eb23f
md"With the same package we can also create a non-uniform FFT (NFFT) operator:"

# ╔═╡ 05dafc16-c68f-47e8-8bdf-cfea8736bc94
html"""
<p style="line-height:1.5">
<code>F</code> is a linear operator and the <em>MRIReco.jl</em> package uses Julia's 

<a href="https://docs.julialang.org/en/v1/manual/methods/" target="_blank">multiple dispatch</a>
to implement a <code>*</code> method for this operator. Despit it being a much more efficient implemenentation, <code>F</code> essentially acts like a matrix and we can calculate the k-space signal with something that looks like a matrix-vector multiplication:
"""

# ╔═╡ f6d6fe5e-3381-4338-817d-366194b3f7ba
function calculate_k_radial(F, image_org)
	k_radial = F * vec(image_org)
	return k_radial
end

# ╔═╡ 424e1625-e14b-4fcb-bd5b-7e426c97d330
md"Let's try to reconstruct this data with a back-projection. As discussed in the lecture, we need to apply density compensation, othewise the image is blurry (see below). Modify below code line to compute the correct density compensation for radial imaging!

Tip: For radial imaging, the density is only a function of `r`, so you only need to replace `ones(nr)`."

# ╔═╡ 16ad1b20-f3c8-4516-b5d9-6150875c326c
density_compensation = repeat(ones(nr), nspokes)

# ╔═╡ 9aff92eb-8094-4a83-ad27-242277b9aee1
md"With the correct density compensation, we can reconstruct the image:"

# ╔═╡ 28c1a017-8977-40a8-91d2-1cac83a0d3cf
md"Color scaling:"

# ╔═╡ bfa47923-d80b-43ea-a21e-f3dad07b4958
md"""
If you change the color scaling, it should become apparent that the image still has artifacts. Take a look at the k-space trajectory plotted above, and compare it to the Cartesian k-space. You should see that the radial k-space trajectory does not acquire the corners of k-space, which results in the here observed artifacts. For comparison, you can cut out the Cartesian k-space and observe the "original image" and it's Cartesian k-space at the very top.
"""

# ╔═╡ 05d84e43-d768-4680-82eb-d3a5419edd85
@bind b_gt Select(["org" => "show original image", "cut" => "cut out k-space"])

# ╔═╡ 2af0c856-430b-41ec-a804-77b3367d1fc5
md"Throughout this notebook, the RMSE tests are calculated on the k-space data and only within the sampled area of k-space."

# ╔═╡ 94ca0194-811f-472f-8815-1f52c6a1619c
if b_gt == "org"
	image_gt = image_org;
	abs_k_Cart_disp = abs_k_Cart;
else
	image_gt = abs.(ifft(fft(image_org) .* ifftshift(k_mask)));
	abs_k_Cart_disp = abs_k_Cart .* k_mask
end;

# ╔═╡ 1adbae4f-c732-457b-aad8-bc0f3a7be13d
heatmap(image_gt, c=:grays, clim=(0,cmax_org))

# ╔═╡ 8a3dc74a-4b20-4c0f-96e8-1ab0451a7da3
try
	heatmap(log.(abs_k_Cart_disp))
catch
end

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

Great, so let's do that! Replace the `missing` in the following two cells:
"""

# ╔═╡ 32bc54d1-4ea8-4663-ba9a-03d357eb8c02
A = randn(256, 256)

# ╔═╡ 51b2f56b-3880-4ba6-828a-104334bad1ec
b = randn(256)

# ╔═╡ 698d0344-ec5d-4c27-a2b1-b52825e398d6
md"Now we have everything in place to run a reconstruction with conjugate gradients:"

# ╔═╡ 57a2c882-760c-4ff9-9bcb-be8f358d7b8a
reco_cg = reshape(cg(A, b, maxiter=50), (isqrt(size(A,1)),isqrt(size(A,1))));

# ╔═╡ 93f84f84-fdf3-41fa-88f8-2a26b8d6982a
md"Color scaling:"

# ╔═╡ 33b23b5c-85e7-49ba-9e4a-27d67943ee73
@bind cmax_cg Slider(0:.1:1, default=1)

# ╔═╡ ce933bda-ac45-46ec-b577-c4168b1ce569
heatmap(abs.(reco_cg), c=:grays, clim=(0,cmax_cg))

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

# ╔═╡ 62934cdb-526d-4f97-afc0-20bcd9f9714e
md"""
So we have initialized the NFFT operator with `oversamplingFactor=1` and `kernelSize=1`, yet we get good results. This is in sharp contrast to theory discussed in the lecture. 

$(@bind ic_answer Select(["undefined" => "", "green" => "The image looks great, the box is green, and theory rarely has practical value. Let's enjoy the nice weather!", "teacher" => "The teacher said this can't be right, so I guess it's not...", "correct" => "I contemplated about the theory and it's correct. By logical conclusion, there must be something wrong here!"]))

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
@bind use_dft Select(["no" => "Use NFFT Operator to simulate the signal (inverse crime)", "yes" => "Use the discrete Fourier transform to calculate the signal (slow, but more accurate)"])

# ╔═╡ 86af3791-f3e6-45bc-bc4e-072803491d1f
if use_dft == "yes"
	k_radial = dft(image_org, trj)
else
	Fforward = NFFTOp(size(image_org), trj; oversamplingFactor=1, kernelSize=1)
	k_radial = calculate_k_radial(Fforward, image_org)
end

# ╔═╡ b114e592-0d39-43cb-aee3-c0948b2d2f88
reco_bp = reshape(F' * (k_radial .* density_compensation), (nx, nx));

# ╔═╡ 179e6a9d-3728-4bde-8bae-5b6ac4490a74
@bind cmax_bp Slider(0.5*maximum(abs.(reco_bp)):0.1*maximum(abs.(reco_bp)):maximum(abs.(reco_bp)), default=maximum(abs.(reco_bp)))

# ╔═╡ b62f8883-02cd-4ab0-9c27-61283e4839f6
heatmap(abs.(reco_bp), c=:grays, clim=(minimum(abs.(reco_bp)),cmax_bp))

# ╔═╡ 843dcc64-3015-4682-bae7-ddd86a8615fa
md"When scrolling up, you should see that the RMSE test for the CG reconstruction fails again and the image has artifacts (use the slider to change the color scaling). Change the kernel size and oversampling until the RMSE test is once more passed and the ghosting-artifacts dissapear."

# ╔═╡ ac4504d8-ed24-4972-beff-e518a820dc83
md"""
## Exerise 4 - _computation time and accuracy_

Now that we have a working simulation and reconstruction framework, it is time to play with it a bit! I suggest you modify the following parameters, one at a time:

- `nx`
- `nspokes`
- `kernelSize`
- `oversamplingFactor`

At the bottom right of each cell you should see time it took to execute respective cell. When increasing `nx`, you should be able to observe how the reconstruction time increases moderately, but the DFT time will increase substantially. This effect is exactly what computational complexity refers to. When increasing the other three parameters, you should see modest increases in the reconstruction time.

Feel free to reduce `nspokes` below Nyquist and to see modest undersampling artifacts. Create your own cell to perform Cartesian undersampling for comparison!

"""

# ╔═╡ 9ce17af6-6aff-4615-933e-5e67b56d1038
md"""
## Exerise 5 - _off-resonance correction_

This exercise briefly introduces off-resonance correction. Let's assume we have measured a frequency map, e.g. with a dual-echo gradient echo pulse sequence and it looks like this:
"""

# ╔═╡ c04477fc-ab75-4806-9045-50f71f45c707
md"It is physically not particularily plausible and we magically managed to measure the frequency outside of our object, but it will serve the purpose of this excercise. A smart way to representing a frequency map is to store it as the imaginary part of a complex-valued matrix:"

# ╔═╡ a8339197-b1b5-400f-89f7-0daa87e5bf2c
ω = im * quadraticFieldmap(nx,nx,1e3)[:,:,1]

# ╔═╡ 5a9f1bf5-2f04-4a13-859e-0e09957fe20c
heatmap(imag.(ω), c=:berlin, title="ω (rad/s)")

# ╔═╡ 0368a211-0e20-4841-9e93-ba722b7b42f1
t_acq = 15e-3

# ╔═╡ 84c928bd-4dd4-468c-a62e-7347bcede7fd
md"The advantage of this format is that we joinlty describe signal decay in the real part and off-resonance in the imaginary part. The signal evolution in each voxel is then simply given by

$s(t) \propto exp(- t \cdot (1/T_2^* + i \omega)).$

For now, we will stick to off-resonance and assume $1/T_2^* = 0$, but feel free to experiment with $T_2^*$ correction!

For this exercise we will use a spiral trajectory as one of its primary advantage is the possiblity for long readouts. Here, we assume a $(t_acq*1e3) ms duration:
"

# ╔═╡ c393b9a8-2f4c-40e5-adc8-4605655dff89
n_windings = 10

# ╔═╡ d2936afa-b271-464e-b7ef-37f33e7306dc
md"After choosing the length of each spiral, we can calculate the number of spirals required for Nyquist sampling:"

# ╔═╡ 605ee395-e123-4128-9b56-c73a128572ec
n_spirals = ceil(Int, nx / n_windings)

# ╔═╡ 5a58c91c-1383-4c17-9341-fda706eb0c1a
md"And we set the number of readout points along the spiral:"

# ╔═╡ 882de061-061c-4538-b1b4-5aba0a385569
n_ADC = 2^12

# ╔═╡ 1fbbf883-cf0d-461a-98f0-a934395e276a
md"Now we have everything in place to create and plot the trajectory:"

# ╔═╡ bfd13cd5-87ba-4a4a-b59b-2f06fcb2055c
trj_spiral = trajectory("Spiral", n_spirals, n_ADC; windings=n_windings, AQ=t_acq)

# ╔═╡ 123f7947-71ef-4274-b0fc-25ae6d6b1396
md"Here we plot only the first two of the $(n_spirals) spirals:"

# ╔═╡ cfebce07-d5ee-4df4-97c4-28bc1576ccfe
begin
	plot(trj_spiral.nodes[1,1:2^12], trj_spiral.nodes[2,1:2^12], label="spiral 1/$n_spirals")
	plot!(trj_spiral.nodes[1,2^12+1:2^13], trj_spiral.nodes[2,2^12+1:2^13], label="spiral 2/$n_spirals", xlabel="kx (rad/voxel/2pi)", ylabel="ky (rad/voxel/2pi)")
end

# ╔═╡ 51ef7e32-cf88-405e-8b13-2cae310b93ca
md"To avoid an inverse crime, we perform the forward simulation with a discrete Fourier transform. Compare this function to the one in Exercise 3, it's only one additional phase factor!"

# ╔═╡ 1ea1974a-cebb-4c05-a885-89058a34cf15
function dft_or(image_org, trj, ω)
	k_radial_dft = zeros(ComplexF64, size(trj,2))
	nx2 = size(image_org,1)/2 + 1
	Threads.@threads for i = 1:length(k_radial_dft)
		for j ∈ CartesianIndices(image_org)
	    	k_radial_dft[i] += image_org[j] * exp.(-ω[j] * trj.times[i]) *
				exp(-1im * 2π * (trj.nodes[1,i] * (j[1]-nx2) + 
  							     trj.nodes[2,i] * (j[2]-nx2)))
		end
	end
	return k_radial_dft
end

# ╔═╡ 7cf1b6fc-2489-4010-aab0-dc973937a704
k_spiral_or = dft_or(image_org, trj_spiral, ω)

# ╔═╡ 814283c0-99e9-4d9e-929f-6aa721c28aee
md"Replace the normal `NFFTOp`-operator with the `FieldmapNFFTOp`-operator. Once you copy and pasted the constructor call, _Live docs_ a the bottom right will tell you the required and (optional) arguments of the function."

# ╔═╡ a49fa464-56f8-4c4a-8e14-00977cb033a6
F_spiral_or = NFFTOp(size(image_org), trj_spiral)

# ╔═╡ 8b894d85-2dcd-4927-9e0d-91eb93c8f3dd
reco_or = reshape(cg(F_spiral_or' * F_spiral_or, F_spiral_or' * k_spiral_or, maxiter=50), (isqrt(size(F_spiral_or,2)),isqrt(size(F_spiral_or,2))));

# ╔═╡ 220675f2-9ef6-4187-bdd9-1dbebb8af4ec
heatmap(abs.(reco_or), c=:grays, clim=(0,1))

# ╔═╡ 756d150a-b7bf-4bf5-b372-5b0efa80d987
md"## Function library

Just some helper functions used in the notebook."

# ╔═╡ 4bc94bec-da39-4f8a-82ee-9953ed73b6a4
hint(text) = Markdown.MD(Markdown.Admonition("hint", "Hint", [text]))

# ╔═╡ aefd5a3d-9b7d-40a1-9cc8-1a5b555d251d
md"""
You can find documentation by clicking on the Live docs in the bottom right of this Pluto window, and typing a function name in the top.

![image](https://user-images.githubusercontent.com/6933510/107848812-c934df80-6df6-11eb-8a32-663d802f5d11.png)


![image](https://user-images.githubusercontent.com/6933510/107848846-0f8a3e80-6df7-11eb-818a-7271ecb9e127.png)

I recommend that you leave the window open while you work on Julia code. It will continually look up documentation for anything you type!
""" |> hint

# ╔═╡ 7906f103-65db-45c6-9d95-ef1483048c14
md"""
The sampling density of radial trajectories increases as we approach the k-space center, more precisely it is proportional to `1/r`, where `r` is the distance from the k-space center. 

When back-projecting, we have too much signal in the center of k-space, hence the blurred image. To compensate, we, thus, have to multiply the k-space data with `r`. Try again! 
""" |> hint

# ╔═╡ bfa4b5ac-94b6-4edc-8de8-03b6ac184754
answer(text) = Markdown.MD(Markdown.Admonition("hint", "Answer", [text]))

# ╔═╡ c604ce61-08ac-46af-b231-a99e37f52bd6
md"""
The correct answer is

```
density_compensation = repeat(abs.(-nr/2 : nr/2-1), nspokes)
```

or, equivalently:

```
density_compensation = sqrt.(trj.nodes[1,:].^2 + trj.nodes[2,:].^2)
```

""" |> answer

# ╔═╡ 8233b774-7363-463b-9c27-13c2fecb9e25
md"
The solution is:

`F_spiral_or = FieldmapNFFTOp(size(image_org), trj_spiral, ω)`
" |> answer


# ╔═╡ 8ce6ad06-819c-4af5-bed7-56ecc08c97be
almost(text) = Markdown.MD(Markdown.Admonition("warning", "Almost there!", [text]))

# ╔═╡ dfa40e89-03fc-4a7a-825e-92d67ee217b2
still_missing(text=md"Replace `missing` with your answer.") = Markdown.MD(Markdown.Admonition("warning", "Here we go!", [text]))

# ╔═╡ 086ec1ff-b62d-4566-9973-5b2cc3353409
keep_working(text=md"The answer is not quite right.") = Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))

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
		You generated a matrix, but the elements are of type *$(mytype)*. Instead, they should be of type `Float64`.
		""")
elseif size(abs_k_Cart) != size(abs_k_Cart)
	keep_working(md"`abs_kspace` has the wrong size.")
elseif abs_k_Cart == abs.(k_Cart)
	correct(md"Well done! A Julia Pro in the making! Here is a plot of the absolute value the k-space data (logarithmically):")
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
		correct(md"Well done! You master the art of filtered back-projection! The normalized RMSE is $bp_error, that is about as good as it gets with filtered backprojection")
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
		keep_working(md"`reco_cg` does not have the correct size. I should be the same as `image_org`")
	else
		error_cg = round(calculate_error(reco_cg, image_org, k_mask) * 10000) / 10000
		if error_cg > 0.002
			keep_working(md"You are getting an image, but it's not quite looking like `image_org`. The normalized RMSE is $error_cg, but it should be below 0.002.")
		else
			correct(md"Well done! You master the art of iterative reconstructions! The normalized RMSE is $error_cg. ")
		end
	end
end

# ╔═╡ 2b0fb5c3-4f96-4559-a28c-dac0de27a328
begin
	if F_spiral_or isa NFFTOp
		still_missing(md"Replace the `NFFTOp` operator with an `FieldmapNFFTOp` operator!")
	elseif !(F_spiral_or isa FieldmapNFFTOp)
		keep_working(md"`F_spiral_or` should be of type `FieldmapNFFTOp`.")
	else
		error_or = round(calculate_error(reco_or, image_org, k_mask) * 10000) / 10000
		if error_or > 0.04
			keep_working(md"Not quiet. The normalized RMSE is $error_or, but it should be below 0.04.")
		else
			correct(md"Well done! Off-resonance artifacts are history! The normalized RMSE is $error_or. Note that it is larger compared to the reconstruction without any off-resonance, which is $error_cg with the current settings (Excercises 2-3). This is a result of the additional approximations made in the `FieldmapNFFTOp`-operator for speed purposes, as discussed in the lecture. However, spatial off-resonance variations can result in a loss of spatial information that we cannot correct for!")
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
	correct(md"What you have discovered in called an *inverse crime*. With `kernelSize=1` we essentially perform nearest-neighbor interpolation. Thus, we just copy the Cartesian k-space data point onto the non-Cartesian grid without changing it, and we copy the same data point back to the Cartesian grid in the reconstruction. The approximation errors cancel each other out and we get great results that have little to do with reality. 
		
To avoid this problem, we should always simulate the forward problem as accurately as possible and use approximations only in the reconstruction. Here, we can simulate the non-Cartesian k-space data *by foot* with the slow discrete Fourier transform:
		")

end

# ╔═╡ 4dd0670b-d84e-4f7f-bfda-ec0628d9ed51
bigbreak = html"<br><br><br><br><br>";

# ╔═╡ 4bccda27-d96b-4234-b5e4-933422b4a157
bigbreak

# ╔═╡ 91f4778e-ee20-11ea-1b7e-2b0892bd3c0f
bigbreak

# ╔═╡ Cell order:
# ╟─ac8ff080-ed61-11ea-3650-d9df06123e1f
# ╟─a24b4518-e6c3-4845-8b32-b203af6a2d69
# ╟─aff9b50d-d013-4c8b-8510-9a59e0c9fc2e
# ╟─5f95e01a-ee0a-11ea-030c-9dba276aba92
# ╠═65780f00-ed6b-11ea-1ecf-8b35523a7ac0
# ╟─540ccfcc-ee0a-11ea-15dc-4f8120063397
# ╠═26709e4b-1675-4ee6-a436-bf9a85683950
# ╟─97a4d58f-9c32-4de9-9142-31e34fc4e33b
# ╠═d72dc575-6d2e-489c-b512-aeae6d017f7f
# ╟─9391828f-041c-49db-93d8-3483bb672f3d
# ╟─67bedf95-8ff4-430f-8737-a0d21ff368c8
# ╟─eeed0b36-37be-4af4-a8c2-68c595d6bb2c
# ╟─1adbae4f-c732-457b-aad8-bc0f3a7be13d
# ╟─c6be8293-7f7c-48a7-b112-92f7ae4fb7c8
# ╠═35022e16-d376-426c-a9f8-b71760bcaaf7
# ╟─40a94bcd-a0cc-4b1a-b370-6405adc0f92b
# ╠═c0afaa73-aa46-4f61-a5b4-c49d071f4fb6
# ╟─810b38a1-490f-4e22-8edb-0e50a8af2f28
# ╟─8a3dc74a-4b20-4c0f-96e8-1ab0451a7da3
# ╟─aefd5a3d-9b7d-40a1-9cc8-1a5b555d251d
# ╟─515ae3f8-fb28-49b9-a023-38d143b6674a
# ╟─ad6a33b0-eded-11ea-324c-cfabfd658b56
# ╠═28668257-321b-41f2-bfe2-d5f35441ec8d
# ╟─137ed83c-171f-48df-be8e-dc538ac18929
# ╠═04962733-71dd-4e6f-9e23-5b29ee821d68
# ╟─74d30291-506f-4e57-9241-c791f0fb8264
# ╠═319872dd-d646-4029-9319-4ffb957179a0
# ╟─4ded3325-4935-4843-b9c2-b0e29469bf3e
# ╟─c0f04553-6031-4984-8c80-1260022eb23f
# ╠═beccb301-1c40-4c41-8960-462a82eca3ad
# ╟─05dafc16-c68f-47e8-8bdf-cfea8736bc94
# ╠═f6d6fe5e-3381-4338-817d-366194b3f7ba
# ╟─424e1625-e14b-4fcb-bd5b-7e426c97d330
# ╠═16ad1b20-f3c8-4516-b5d9-6150875c326c
# ╟─a05cce2b-b1b7-4d8c-a823-137b90473d70
# ╟─7906f103-65db-45c6-9d95-ef1483048c14
# ╟─c604ce61-08ac-46af-b231-a99e37f52bd6
# ╟─9aff92eb-8094-4a83-ad27-242277b9aee1
# ╠═b114e592-0d39-43cb-aee3-c0948b2d2f88
# ╟─28c1a017-8977-40a8-91d2-1cac83a0d3cf
# ╟─179e6a9d-3728-4bde-8bae-5b6ac4490a74
# ╟─b62f8883-02cd-4ab0-9c27-61283e4839f6
# ╟─bfa47923-d80b-43ea-a21e-f3dad07b4958
# ╟─05d84e43-d768-4680-82eb-d3a5419edd85
# ╟─2af0c856-430b-41ec-a804-77b3367d1fc5
# ╟─94ca0194-811f-472f-8815-1f52c6a1619c
# ╟─9d70e2e6-6e45-4662-9792-b36ab36c35b4
# ╠═32bc54d1-4ea8-4663-ba9a-03d357eb8c02
# ╠═51b2f56b-3880-4ba6-828a-104334bad1ec
# ╟─698d0344-ec5d-4c27-a2b1-b52825e398d6
# ╠═57a2c882-760c-4ff9-9bcb-be8f358d7b8a
# ╟─a491a997-0a95-491a-8ca2-1ff85f5f39d3
# ╟─93f84f84-fdf3-41fa-88f8-2a26b8d6982a
# ╟─33b23b5c-85e7-49ba-9e4a-27d67943ee73
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
# ╟─843dcc64-3015-4682-bae7-ddd86a8615fa
# ╟─ac4504d8-ed24-4972-beff-e518a820dc83
# ╟─9ce17af6-6aff-4615-933e-5e67b56d1038
# ╟─5a9f1bf5-2f04-4a13-859e-0e09957fe20c
# ╟─c04477fc-ab75-4806-9045-50f71f45c707
# ╠═a8339197-b1b5-400f-89f7-0daa87e5bf2c
# ╟─84c928bd-4dd4-468c-a62e-7347bcede7fd
# ╠═0368a211-0e20-4841-9e93-ba722b7b42f1
# ╠═c393b9a8-2f4c-40e5-adc8-4605655dff89
# ╟─d2936afa-b271-464e-b7ef-37f33e7306dc
# ╠═605ee395-e123-4128-9b56-c73a128572ec
# ╟─5a58c91c-1383-4c17-9341-fda706eb0c1a
# ╠═882de061-061c-4538-b1b4-5aba0a385569
# ╟─1fbbf883-cf0d-461a-98f0-a934395e276a
# ╠═bfd13cd5-87ba-4a4a-b59b-2f06fcb2055c
# ╟─123f7947-71ef-4274-b0fc-25ae6d6b1396
# ╟─cfebce07-d5ee-4df4-97c4-28bc1576ccfe
# ╟─51ef7e32-cf88-405e-8b13-2cae310b93ca
# ╠═1ea1974a-cebb-4c05-a885-89058a34cf15
# ╠═7cf1b6fc-2489-4010-aab0-dc973937a704
# ╟─814283c0-99e9-4d9e-929f-6aa721c28aee
# ╠═a49fa464-56f8-4c4a-8e14-00977cb033a6
# ╟─2b0fb5c3-4f96-4559-a28c-dac0de27a328
# ╠═8b894d85-2dcd-4927-9e0d-91eb93c8f3dd
# ╟─220675f2-9ef6-4187-bdd9-1dbebb8af4ec
# ╟─8233b774-7363-463b-9c27-13c2fecb9e25
# ╟─4bccda27-d96b-4234-b5e4-933422b4a157
# ╟─91f4778e-ee20-11ea-1b7e-2b0892bd3c0f
# ╟─756d150a-b7bf-4bf5-b372-5b0efa80d987
# ╟─4bc94bec-da39-4f8a-82ee-9953ed73b6a4
# ╟─bfa4b5ac-94b6-4edc-8de8-03b6ac184754
# ╟─8ce6ad06-819c-4af5-bed7-56ecc08c97be
# ╟─dfa40e89-03fc-4a7a-825e-92d67ee217b2
# ╟─086ec1ff-b62d-4566-9973-5b2cc3353409
# ╟─c22f688b-dc04-4a94-b541-fe06266c5446
# ╟─ab3d1b70-88e8-4118-8d3e-601a8a68f72d
# ╟─4dd0670b-d84e-4f7f-bfda-ec0628d9ed51
