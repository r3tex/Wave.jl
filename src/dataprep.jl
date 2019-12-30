using IndexedTables
using StructArrays
using ProgressMeter
using Flux

"""
`μlawcmp(x::T; clip = 10f0, mu = 128f0) where T <: AbstractArray`

* __input:__ array of Float32 with a mean of 0
* __args:__ 
  * `clip` lets you remove outliers beyond a threshold
  * `mu` lets you specify the quantization after compression e.g. -128:127
* __output:__ array of Float32
"""
function μlawcmp(x::T; clip = 1f0, mu = 128f0) where T <: AbstractArray
    μlcmp.(clamp.(x, -clip, clip)./clip, mu)
end
μlcmp(x, μ = 128f0) = sign(x) * log(1 + μ * abs(x)) / log(1 + μ)

"""
`μlawexp(x::T; clip = 10f0, mu = 128f0) where T <: AbstractArray`

* __input:__ array of Float32 with a mean of 0
* __args:__ 
  * `clip` lets you remove outliers beyond a threshold
  * `mu` lets you specify the quantization after expansion e.g. -128:127
* __output:__ array of Float32
"""
function μlawexp(x::T; clip = 1f0, mu = 128f0) where T <: AbstractArray
    μlexp.(x, mu) .* clip
end
μlexp(y, μ = 128f0) = sign(y) * (1/μ) * ((1 + μ)^abs(y) - 1)

"""
`preptbl(tbl; dt, lc, gc, samples, batchsize, stride, lim)`

* __input:__
  * `tbl` a JuliaDB table
* __args:__
  * `dt` the symbol for the column containing data
  * `lc` a tuple of symbols for columns with local conditioning data
  * `gc` a tuple of symbols of global conditioning classes
  * `samples` set the size of the window that you want the model to train with
  * `batchsize` set the batch size (this is the 4th dimension)
  * `stride` set the data augmentation level 
        (e.g. a stride:samples ratio of 64:256 will quadruple the training data)
  * `lim` limit the size of the resulting dataset (so it fits in memory)

  Example: Wave.preptbl(tbl, dt=:sin, lc=(:lc1, :lc2), gc=(:sin, :sin2, :sin4), 
            samples=1024, batchsize=8, stride=128)

* __output:__ an object ready for consumption by Flux
The format of the data created by `Wave.preptbl` below is:
[                            <== an array:    dat
    [                        <== a batch:     dat[1]
        [                    <== the data:    dat[1][1]
            x1,              <== timeseries:  dat[1][1][1] 
            x2               <== condvars:    dat[1][1][2]
        ]          
        [
            y                <== labels:      dat[1][2]
        ]                  
    ]
    ...
]
"""
function preptbl(tbl; dt, lc, gc, samples, batchsize, stride, lim = 4194304)  
    tbl_ = tbl
    out = []
    println("Data points in input: $(length(tbl_))")
    println("Augmenting data in strides of $stride")
    @showprogress for i in 0:samples÷stride-1
        tbl_ = transform(tbl_, Symbol("stride_$i") => 
               [(j+i*stride)÷(samples+1) for j in 1:length(tbl_)])
        push!(out, 
            select(
                groupby((idx = x->rand(), cols = x->identity(x)),
                    tbl_, Symbol("stride_$i"); select=(dt, lc...)),
            (:idx, :cols)))
    end
    sleep(0.1); println("Merging, sorting, and filtering...")
    tbl_ = reduce(merge, out)
    tbl_ = sort(tbl_, :idx)
    tbl_ = rows(filter(x -> length(x.cols) == samples+1, tbl_))
    out = []
    println("Data points after augmentation: $(length(tbl_) * samples+1)")
    println("Grouping into batches of $batchsize")
    gcvec = zeros(Float32, samples, 1, length(gc), batchsize)
    gcvec[:, :, findfirst(dt .== gc), :] .= 1f0
    @showprogress for i in 1:batchsize:length(tbl_)-length(tbl_)%batchsize
        vec = cat([reduce((x, y) -> cat(x, y, dims=3), fieldarrays(row.cols)) 
                  for row in tbl_[i:i+batchsize-1]]..., dims=4)
        push!(out, 
            (
                (
                μlawcmp(vec[1:end-1,:,1:1,:]), 
                cat(vec[1:end-1,:,2:end,:], gcvec, dims=3)
                ), 
            reduce((x, y) -> hcat(x, y), map(x -> Flux.onehot(x, -128:127), round.(Int8, μlawcmp(vec[end,1,1,:]) .* 127)))
            )
        )
        if sizeof(out) > lim
            println("-- Size limit reached after $i of ",
                    "$(length(tbl_)-length(tbl_)%batchsize) rows --"); break
        end
    end
    sleep(0.1)
    println("Size of dataset: $(round(Int, sizeof(out)/1024)) MB")
    println("Training samples: $(size(out, 1) * batchsize)")
    return out
end

"""
`testmodel(m, data; dt, lc, gc)`

* __input:__
  * `m` a Wave net model
  * `data` a JuliaDB table
* __args:__
  * `dt` the symbol for the column containing data
  * `lc` a tuple of symbols for columns with local conditioning data
  * `gc` a tuple of symbols of global conditioning classes
* __output:__ an object ready for consumption by Flux
"""
function testmodel(m, data; dt, lc, gc)  
    field = 2^length(m.layers)
    tbl_ = select(data, (dt, lc...))
    gcvec = zeros(Float32, field, 1, length(gc), 1)
    gcvec[:, :, findfirst(dt .== gc), :] .= 1f0
    dat = []
    lbl = Array{Float32, 1}(undef, 0)
    for i in 1:field÷2
        vec = reshape(cat(columns(tbl_[i:i+field])..., dims=3), field+1, 1, :, 1)
        push!(dat, (
            μlawcmp(vec[1:end-1,:,1:1,:]),
            cat(vec[1:end-1,:,2:end,:], gcvec, dims=3)
        ))
        push!(lbl, μlawcmp(vec[end:end,1,1,1])[1])
    end
    
    pred = Array{Float32, 1}(undef, 0)
    @showprogress for sample in dat
        s = vcat(reshape(sample[1][1:end-length(pred)], :), pred)
        push!(pred, cpu(Flux.onecold(
                        m((gpu(reshape(s, :, 1, 1, 1)), gpu(sample[2]))), 
                    -128:127))[1] / 128)
    end
    return pred, lbl
end