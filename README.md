# Wave.jl
This is an unregistered Julia package implementing WaveNet using Flux.

## Usage

```
julia> using Flux, CuArrays, JuliaDB, Plots

julia> include("Wave/Wave.jl")
julia> Main.Wave

julia> tbl = load("mydata.juliadb")

julia> net = Wave.Net(10, cond=5)
:: Wave.Net ::
Hidden layers:     10
Receptive field:   1024
Residual channels: 64
Skip channels:     256
Conditioning vars: 5
GPU enabled:       yes

julia> dat = Wave.preptbl(tbl, dt=:sin, 
                        lc=(:lc1, :lc2), 
                        gc=(:sin, :sin2, :sin4), 
                        samples=1024, 
                        batchsize=8, 
                        stride=128)

julia> dat_t = gpu.(dat[1:end-1])
julia> dat_v = gpu(dat[end])

julia> opt = ADAM(0.001)
julia> loss(x, y) = Flux.crossentropy(net(x), y)
julia> function accuracy()
       println("Mean Average Error:  ", 
           mean(abs.(Flux.onecold(net(dat_v[1]), -128:127) - 
               Flux.onecold(dat_v[2], -128:127))),
           " out of 128")
           println("Cross Entropy:       ", loss(dat_v[1], dat_v[2]).data)
       end

julia> Flux.@epochs 4 Flux.train!(loss, params(net), dat_t, opt, cb = Flux.throttle(accuracy, 60))
Mean Average Error:  0.875 out of 128
Cross Entropy:       0.8963004
   
julia> testpred, testlbl = Wave.testmodel(net, filter(x -> 1200 < x.t, tbl), 
    dt=:sin, 
    lc=(:lc1, :lc2), 
    gc=(:sin, :sin2, :sin4))

julia> plot(hcat(testpred, testlbl), label=["prediction", "truth"])

```
