 using Flux

"""
`gate(x::T, y::T) where T <: AbstractArray`

* __input:__ arrays of equal length of Float32
* __output:__ array of Float32

The additional `y` vector is used to apply local and global conditioning.
`z = tanh(x[1:f÷2] + y[1:f÷2]) * σ(x[f÷2+1:end] + y[f÷2+1:end])`
"""
function gate(x)
    chan = size(x, 3) ÷ 2
    tanh.(x[:, :, 1:chan, :]) .* σ.(x[:, :, chan+1:end, :])
end

function gate(x, y)
    chan = size(x, 3) ÷ 2
    tanh.(x[:, :, 1:chan, :] .+ y[:, :, 1:chan, :]) .* 
    σ.(x[:, :, chan+1:end, :] .+ y[:, :, chan+1:end, :])
end

"""
`WConv(dilate::Int, skpflt::Int, resflt::Int)`

* __input:__ 
    * `dilate` how wide the dilation should be for this conv layer
    * `resflt` how many filters to return for the residual layer
    * `skpflt` how many filters to return for the skip layer
* __output:__ a WConv struct

`WConv(x::AbstractArray)`

* __input:__ a (data, 1, channels, batch) Array like a 1D Conv
* __output:__ an Array processed by gated activation, a skip layer applied, and residual values stored in the `WConv` struct.

`WConv(x::Tuple)`

* __input:__ a Tuple of arrays
* __output:__ same as above but also with conditioning array applied in gated activation

"""
mutable struct WConv
    dilconv::Conv
    resconv::Conv
    skpconv::Conv
    skp::DenseArray
    condconv::Conv

    function WConv(dil::Int, res::Int, skp::Int, cond::Int)
        dc = gpu(Conv((2, 1), res => res*2, identity, dilation=(dil, 1), pad=(dil, 0)))
        rc = gpu(Conv((1, 1), res => res, identity))
        sc = gpu(Conv((1, 1), res => skp, identity))
        s = sc(gpu(zeros(Float32, 2, 1, res, 1)))
        cc = gpu(Conv((1, 1), cond => res*2, tanh))
        new(dc, rc, sc, s, cc)
    end

    function WConv(dc::Conv, rc::Conv, sc::Conv, s::DenseArray, cc::Conv)
        new(dc, rc, sc, s, cc)
    end
end

function (wc::WConv)(x)
    w,h,c,b = size(x)
    dat = gate(wc.dilconv(x))[1:w,:,:,:]
    wc.skp = wc.skpconv(dat)
    x .+ wc.resconv(dat)
end

function (wc::WConv)(input::Tuple)
    x, y = input
    w,h,c,b = size(x)
    dat = gate(wc.dilconv(x)[1:w,:,:,:], wc.condconv(y))
    wc.skp = wc.skpconv(dat)
    (x .+ wc.resconv(dat), y)
end

function Base.show(io::IO, wc::WConv)
    ln1 = ":: Wave.WConv ::\n"
    ln2 = "Dilated Conv:   " * string(wc.dilconv) * 
    " dil: " * string(wc.dilconv.dilation) *
    " pad: " * string(wc.dilconv.pad) * "\n"
    ln3 = "Residual Conv:  " * string(wc.resconv) * "\n"
    ln4 = "Skip Conv:      " * string(wc.skpconv) * "\n"
    ln5 = "Cond Conv: " * string(wc.condconv) * "\n"
    print(io, ln1 * ln2 * ln3 * ln4 * ln5)
end

Flux.@treelike WConv

import Base: +
+(x::WConv, y::WConv) = x.skp + y.skp
+(x::DenseArray, y::WConv) = x + y.skp

"""
`Net(nlayers::Int; res=64, skip=256, cond=0)`

* __input:__ 
    * `nlayers` how many dilated layers to include
    * `res` how many channels to pass into residual layers
    * `skip` how many channels to pass into skip layers
    * `depth` reset dilation to 2 at every multiple of depth
    * `cond` how many variables are included in the conditioning tensor
* __output:__ a NetNet struct

`Net(x)` 

* __input:__ array of shape (samples, 1, 1channel, batchsize)
* __output:__ probability distribution (softmax) of wave amplitudes

`Net(x::Tuple)` 

* __input:__ Tuple of data and conditioning Array of shape (samples, condvars, batchsize)
* __output:__ probability distribution (softmax) of wave amplitudes

"""
mutable struct Net
    input::Conv
    layers::Chain
    output::Chain
    
    function Net(nlayers; res=64, skip=256, depth=9, cond=0)
        i = gpu(Conv((2, 1), 1=>res, tanh, pad=(1, 0)))
        l = Chain([WConv(2^(i%depth), res, skip, cond) for i in 1:nlayers]...)
        o = Chain(
            # x -> leakyrelu.(x),
            gpu(Conv((1, 1), skip=>256, leakyrelu)),
            gpu(Conv((1, 1), 256=>256, leakyrelu))
        )
        new(i, l, o)
    end

    function Net(i::Conv, l::Chain, o::Chain)
        new(i, l, o)
    end
end

Flux.@treelike Net

function (w::Net)(x)
    w.layers(w.input(x)[1:end-1,:,:,:])
    softmax(w.output(reduce(+, w.layers))[end, 1, :, :])
end

function (w::Net)(input::Tuple)
    x, y = input
    dat = (w.input(x)[1:end-1,:,:,:], y)
    w.layers(dat)
    softmax(w.output(reduce(+, w.layers))[end, 1, :, :])
end

function Base.show(io::IO, wave::Net)
    ln1 = ":: Wave.Net ::\n"
    ln2 = "Hidden layers:     " * string(length(wave.layers)) * "\n"
    ln3 = "Receptive field:   " * string(2^length(wave.layers)) * "\n"
    ln4 = "Residual channels: " * string(size(wave.layers[end].resconv.weight, 4)) * "\n"
    ln5 = "Skip channels:     " * string(size(wave.layers[end].skpconv.weight, 4)) * "\n"
    ln6 = "Conditioning vars: " * string(size(wave.layers[end].condconv.weight, 3)) * "\n"
    ln7 = "GPU enabled:       " * ifelse(typeof(wave.input.weight) <: Array, "no", "yes")
    print(io, ln1 * ln2 * ln3 * ln4 * ln5 * ln6 * ln7)
end
