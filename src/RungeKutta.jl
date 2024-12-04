module RungeKutta

using LinearAlgebra
using Parameters

struct ExplicitMethod
    s
    a
    b
    c
end

function ExplicitMethod(a, b, c)
    s = length(c)

    @assert length(b) == s
    @assert size(a) == (s, s)
    @assert all([a[j, i] == 0 for j in 1:s for i in j:s])

    return ExplicitMethod(s, a, b, c)
end

function f!(xnew, method::ExplicitMethod, dyn!, x, u, h)
    # unpacked parameters of method
    @unpack s, a, b, c = method

    # promoted type
    S = promote_type(eltype(x), eltype(u), eltype(h))

    # vector for temporarly holding intermediate steps
    ξ = Vector{S}(undef, length(x))

    # vector and matrix for storing stages
    k = Matrix{S}(undef, length(x), s)

    @views begin
        # first stage
        dyn!(k[:, 1], x, u)

        # remaining stages
        for j in 2:s
            ξ .= x
            mul!(ξ, k[:, 1:j-1], a[j, 1:j-1], h, 1)
            dyn!(k[:, j], ξ, u)
        end
    end

    # new state
    xnew .= x
    mul!(xnew, k, b, h, 1)

    return nothing
end

function f(method::ExplicitMethod, dyn!, x, u, h)
    # unpacked parameters of method
    @unpack s, a, b, c = method

    # promoted type
    S = promote_type(eltype(x), eltype(u), eltype(h))

    # vector for temporarly holding intermediate steps
    ξ = Vector{S}(undef, length(x))

    # vector and matrix for storing stages
    k = Matrix{S}(undef, length(x), s)

    @views begin
        # first stage
        dyn!(k[:, 1], x, u)

        # remaining stages
        for j in 2:s
            ξ .= x
            mul!(ξ, k[:, 1:j-1], a[j, 1:j-1], h, 1)
            dyn!(k[:, j], ξ, u)
        end
    end

    # new state
    xnew = copy(x)
    mul!(xnew, k, b, h, 1)
    return xnew
end

# Methods with coefficients

function RK4()
    a = [
        0 0 0 0 0 0
        1/4 0 0 0 0 0
        3/32 9/32 0 0 0 0
        1932/2197 -7200/2197 7296/2197 0 0 0
        439/216 -8 3680/513 -845/4104 0 0
        -8/27 2 3544/2565 1859/4104 -11/40 0
    ]
    b = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]
    c = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]

    return RungeKutta.ExplicitMethod(a, b, c)
end

function Tsit5()
    a = [
        0 0 0 0 0 0 0
        0.161 0 0 0 0 0 0
        -0.008480655492356989 0.335480655492357 0 0 0 0 0
        2.8971530571054935 -6.359448489975075 4.3622954328695815 0 0 0 0
        5.325864828439257 -11.748883564062828 7.4955393428898365 -0.09249506636175525 0 0 0
        5.86145544294642 -12.92096931784711 8.159367898576159 -0.071584973281401 -0.028269050394068383 0 0
        0.09646076681806523 0.01 0.4798896504144996 1.379008574103742 -3.290069515436081 2.324710524099774 0
    ]
    b = [
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0
    ]
    c = [0, 0.161, 0.327, 0.9, 0.9800255409045097, 1, 1]

    return RungeKutta.ExplicitMethod(a, b, c)
end

end

