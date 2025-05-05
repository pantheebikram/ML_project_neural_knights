using JuMP, Ipopt, Plots

# model (https://lanl-ansi.github.io/PowerModels.jl/stable/math-model/)
function run_opf_with_bounds(up_active, lo_active, S_load, V_up, V_lo)
    model = Model(Ipopt.Optimizer)
    empty!(model)
    # set_optimizer_attribute(model, "max_iter", 1000)
    # set_optimizer_attribute(model, "max_cpu_time", 100.0)
    # set_optimizer_attribute(model, "tol", 1e-9)
    # set_optimizer_attribute(model, "acceptable_tol", 1e-9)
    # set_optimizer_attribute(model, "mu_strategy", "adaptive")
    # set_optimizer_attribute(model, "hessian_approximation", "exact")

    # -------------------------------
    # Decision Variables
    # -------------------------------
    @variable(model, V_mag[1:n_bus] >= 0)       # Voltage magnitudes
    @variable(model, V_angle[1:n_bus])          # Voltage angles (radians) 
    V__initial = (V_up + V_lo) ./ 2
    for i in 1:n_bus
        set_start_value(V_mag[i], 1)
        set_start_value(V_angle[i], 0)
    end

    @variable(model, P_gen[1:n_gen])                # Active power generation for each generator
    @variable(model, Q_gen[1:n_gen])                # Reactive power generation for each generator

    # branch flows:
    # Column 1 corresponds to flow from f_bus to t_bus, and
    # Column 2 corresponds to flow from t_bus to f_bus.
    @variable(model, P_branch[1:n_branch, 1:2])
    @variable(model, Q_branch[1:n_branch, 1:2])

    @expression(model, V_real[i=1:n_bus], V_mag[i] * cos(V_angle[i]))
    @expression(model, V_imag[i=1:n_bus], V_mag[i] * sin(V_angle[i]))
    # -------------------------------
    # 1. Slack Bus Constraint
    # -------------------------------
    slack = findfirst(bus_type .== 3)
        @constraint(model, V_angle[slack] == va[slack])


    # -------------------------------
    # 2. Bus Voltage Limits
    # -------------------------------
    @constraints(model, begin
    [i in lo_active], vmin[i] <= V_mag[i]
    [i in up_active], V_mag[i] <= vmax[i]
    end)


    # # -------------------------------
    # # 3. Generator Operating Limits
    # # -------------------------------
    @constraints(model, begin
    [i=1:n_gen], pmin[i] <= P_gen[i]
    [i=1:n_gen], P_gen[i] <= pmax[i]

    [i=1:n_gen], qmin[i] <= Q_gen[i]
    [i=1:n_gen], Q_gen[i] <= qmax[i]
    end)


    # -------------------------------
    # 4. Branch Thermal Limits
    # -------------------------------
    # for j in 1:n_branch
    #     @NLconstraint(model, P_branch[j,1]^2 + Q_branch[j,1]^2 <= u_app_limit[j]^2)
    #     @NLconstraint(model, P_branch[j,2]^2 + Q_branch[j,2]^2 <= u_app_limit[j]^2)
    # end


    # -------------------------------
    # 5. Nodal Power Balance Constraints
    # -------------------------------
    for i in 1:n_bus
        # set of generators connected to bus i.
        gen_idx = [k for k in 1:n_gen if gen_bus[k] == i]
        
        # active and reactive generation at bus i
        if !(isempty(gen_idx))
            sum_gen_real = sum( P_gen[k] for k in gen_idx )
            sum_gen_imag = sum( Q_gen[k] for k in gen_idx )
        else
            sum_gen_real = 0
            sum_gen_imag = 0
        end
        
        # load injection at bus i:
        load_real = real(S_load[i])
        load_imag = imag(S_load[i])
        
        #  Y_shunt[i] * |V_i|^2
        shunt_real = V_mag[i]^2 * real(conj(Y_shunt[i]))
        shunt_imag = V_mag[i]^2 * imag(conj(Y_shunt[i]))
        
        # Sum flows leaving bus i
        sum_from_real = 0
        sum_from_imag = 0

        br_idx_from = [k for k in 1:n_branch if f_bus[k] == i]
        if !(isempty(br_idx_from))
            sum_from_real += sum( P_branch[b, 1] for b in br_idx_from )
            sum_from_imag += sum( Q_branch[b, 1] for b in br_idx_from )
        end
        
        # Sum flows coming to bus i
        sum_to_real = 0
        sum_to_imag = 0
        br_idx_to = [k for k in 1:n_branch if t_bus[k] == i]
        if !(isempty(br_idx_to))
            sum_from_real += sum( P_branch[b, 2] for b in br_idx_to )
            sum_from_imag += sum( Q_branch[b, 2] for b in br_idx_to )
        end

        # net branch flow:
        sum_branch_real = sum_from_real + sum_to_real
        sum_branch_imag = sum_from_imag + sum_to_imag
        
        # Real power balance at bus i:
        @NLconstraint(model, sum_gen_real - load_real - shunt_real  == sum_branch_real)
        
        # Reactive power balance at bus i:
        @NLconstraint(model, sum_gen_imag - load_imag - shunt_imag  == sum_branch_imag)
    end

    S_ij_re = Vector{NonlinearExpr}(undef,n_branch)
    S_ij_im = Vector{NonlinearExpr}(undef,n_branch)
    for b in 1:n_branch
        # bus indices for branch b:
        i = f_bus[b]
        j = t_bus[b]
        
        #  Tap ratio
        T_abs_sq = real(Tij[b])^2 + imag(Tij[b])^2

        # A = Yij[b] + Ycij[b]
        A = Yij[b] + Ycij[b]
        A_conj_real = real(conj(A))
        A_conj_imag = imag(conj(A))

        # X = conj(Yij[b]) / Tij[b]
        X = conj(Yij[b]) / Tij[b]
        X_re = real(X)
        X_im = imag(X)
        
        # |V_i|^2 
        V_i_sq = V_mag[i]^2

        # V_i * conj(V_j) in terms of its parts:
        Vprod_re = V_real[i]*V_real[j] + V_imag[i]*V_imag[j]    # Vprod_rev_re[i,j] = V[i] V[j] cos(θ_i - θ_j)
        Vprod_im = V_imag[i]*V_real[j] - V_real[i]*V_imag[j]    # Vprod_rev_im[i,j] = V[i] V[j] sin(θ_i - θ_j)
        
        # S_ij for branch b:
        S_ij_re[b] = (A_conj_real * V_i_sq / T_abs_sq) - ( X_re * Vprod_re - X_im * Vprod_im )
        S_ij_im[b] = (A_conj_imag * V_i_sq / T_abs_sq) - ( X_re * Vprod_im + X_im * Vprod_re )
    end

    S_ji_re = Vector{NonlinearExpr}(undef,n_branch)
    S_ji_im = Vector{NonlinearExpr}(undef,n_branch)
    for b in 1:n_branch
        # bus indices:
        i = f_bus[b]
        j = t_bus[b]
        
        # Tap ratio
        T_abs_sq = real(Tij[b])^2 + imag(Tij[b])^2

        # B = Yij[b] + Ycji[b]
        B = Yij[b] + Ycji[b]
        B_conj_real = real(conj(B))
        B_conj_imag = imag(conj(B))
        
        # second term: X_rev = conj(Yij[b]) / conj(Tij[b])
        X_rev = conj(Yij[b]) / conj(Tij[b])
        X_rev_re = real(X_rev)
        X_rev_im = imag(X_rev)
        
        # |V_j|^2:
        V_j_sq = V_mag[j]^2
        
        # Define V_j * conj(V_i):
        Vprod_rev_re = V_real[j]*V_real[i] + V_imag[j]*V_imag[i]    # Vprod_rev_re[j,i] = V[i] V[j] cos(θ_i - θ_j)
        Vprod_rev_im = V_imag[j]*V_real[i] - V_real[j]*V_imag[i]    # Vprod_rev_im[j,i] = V[i] V[j] sin(θ_j - θ_i)
        
        # S_ji for branch b:
        S_ji_re[b] = (B_conj_real * V_j_sq) - ( X_rev_re * Vprod_rev_re - X_rev_im * Vprod_rev_im )
        S_ji_im[b] = (B_conj_imag * V_j_sq) - ( X_rev_re * Vprod_rev_im + X_rev_im * Vprod_rev_re )
    end

    # branch flow constraints (to_bus -> from_bus)
    @NLconstraint(model, [b=1:n_branch], P_branch[b,1] == S_ij_re[b])
    @NLconstraint(model, [b=1:n_branch], Q_branch[b,1] == S_ij_im[b])
    @NLconstraint(model, [b=1:n_branch], P_branch[b,2] == S_ji_re[b])
    @NLconstraint(model, [b=1:n_branch], Q_branch[b,2] == S_ji_im[b])


    @objective(model, Min, sum(c2[k] * P_gen[k]^2 + c1[k] * P_gen[k] + c0[k] for k in 1:n_gen))


    runtime = @timed(begin
    optimize!(model)
    end)
    println("Solution status: ", termination_status(model))
    num_viloations = 0
    new_up_active = copy(up_active)
    new_lo_active = copy(lo_active)

    for i in 1:n_bus
        if value.(V_mag[i]) < vmin[i]-1e-4 && !(i in lo_active)
            println("minimum voltage violation at bus ", i, ": ", value.(V_mag[i]))
            push!(new_lo_active, i)
            num_viloations += 1
        end
        if value.(V_mag[i]) > vmax[i]+1e-4 && !(i in up_active)
            println("maximum voltage violation at bus ", i, ": ", value.(V_mag[i]))
            push!(new_up_active, i)
            num_viloations += 1
        end
    end

    runtime = solve_time(model)
    num_const = count_inequality_constraints(model)
    num_iter = MOI.get(model, MOI.BarrierIterations())
    return objective_value(model), termination_status(model), num_viloations, runtime, num_const, num_iter, new_up_active, new_lo_active

end

function count_inequality_constraints(model::Model)
    count = 0
    for (F, S) in list_of_constraint_types(model)  # JuMP → (function, set)-pairs :contentReference[oaicite:0]{index=0}
        if S <: MOI.LessThan || S <: MOI.GreaterThan || S <: MOI.Interval
            count += num_constraints(model, F, S)
        end
    end
    return count
end