
clear; close all; clc;
rng(1234)

nBuses     = 14;       
nTime      = 24;      
nWaves     = 4;       
nScenarios = 120;     
totalSamples = nWaves * nScenarios * nTime;  

% Load the Base Case Data 
mpc   = loadcase('case14');  
baseP = mpc.bus(:, 3);     % base active power 
baseQ = mpc.bus(:, 4);     % base reactive power

% Sine Wave 
f_peak      = [1.10, 1.30, 1.40, 1.60];  %  peak factors 
amplitudes  = f_peak - 0.8;              % f(t)=amplitude*sin(pi*warp)+0.8 
peak_times  = [0.50, 0.59, 0.43, 0.53];

t = linspace(0,1, nTime);
s_warp = @(t, p) (t <= p) .* (0.5 * t / p) + (t > p) .* (0.5 + 0.5*(t - p)/(1 - p));

% Active and reactive load initialization
P_matrix = zeros(nBuses, totalSamples);
Q_matrix = zeros(nBuses, totalSamples);

%% Generate Load Samples 
col_index = 1;
for wave = 1:nWaves
    s = s_warp(t, peak_times(wave));
    base_factor = amplitudes(wave) * sin(pi * s) + 0.8;
    base_cycle{wave} = base_factor;  % cell array
    
    for scenario = 1:nScenarios
        for time_idx = 1:nTime
            % Apply random noise per bus 
            f1 = rand(nBuses, 1); 
            noiseP = 1 + (f1- 0.5) * 0.1;
            noiseQ = noiseP + (rand(nBuses, 1)-0.5)*0.1;

            % Scaling factor
            factor = base_factor(time_idx);
            
            P_matrix(:, col_index) = factor * baseP .* noiseP;
            Q_matrix(:, col_index) = factor * baseQ .* noiseQ;
            
            col_index = col_index + 1;
        end
    end
end

%% Load plot example
selected_bus = 2;  
figure;
hold on;
colors = lines(nWaves);
for wave = 1:nWaves
    plot(1:nTime, base_cycle{wave}, '-o', 'LineWidth', 1.5, 'Color', colors(wave,:));
end
xlabel('Time (Hour)');
ylabel('Load Factor');
title(['Base Load Sine Cycles (No Noise) for Bus ', num2str(selected_bus)]);
legend('Wave 1', 'Wave 2', 'Wave 3', 'Wave 4','Location','Best');
grid on;
hold off;

% Plot sample load 
figure;
for wave = 1:nWaves
    start_col   = (wave - 1) * (nScenarios * nTime) + 1;
    scenario_cols = start_col : start_col + nTime - 1;
    subplot(2,2,wave);
    plot(1:nTime, P_matrix(selected_bus, scenario_cols), '-o', 'LineWidth', 1.5);
    xlabel('Time (Hour)');
    ylabel('Active Load P');
    title(['Wave ' num2str(wave) ' - Scenario 1']);
    grid on;
end


%% Save Results
opf_results(totalSamples) = struct('V_complex', [], 'gen_P', [], 'gen_Q', [], ...
                                     'f', [], 'success', false, 'mpc_out', []);

fprintf('Starting OPF for %d scenarios...\n', totalSamples);
for k = 1:totalSamples
    % temporary case 
    mpc_temp = mpc; 
    
    % update the loads
    mpc_temp.bus(:, 3) = P_matrix(:, k);  % update active load (Pd)
    mpc_temp.bus(:, 4) = Q_matrix(:, k);  % update reactive load (Qd)
    
    % Run OPF 
    results = runopf(mpc_temp, mpoption('verbose', 0, 'out.all', 0));
    
    % Extract outputs:
    % voltages
    V_mag = results.bus(:, 8);     
    V_angle = results.bus(:, 9);    
    V_complex = V_mag .* exp(1j * V_angle * pi/180);
    
    % generator outputs
    gen_P = results.gen(:, 2);      
    gen_Q = results.gen(:, 3);      
    
    % objective function value
    f_val = results.f;
    
    % feasibility
    feasible = results.success;
    
    % Save outputs
    opf_results(k).V_complex = V_complex;
    opf_results(k).gen_P     = gen_P;
    opf_results(k).gen_Q     = gen_Q;
    opf_results(k).f         = f_val;
    opf_results(k).success   = feasible;
    opf_results(k).mpc_out   = results;  
    
    %  print progress every 100 scenarios
    if mod(k, 100) == 0
        fprintf('Completed %d OPF runs.\n', k);
    end
end
fprintf('All OPF runs completed.\n');

%% 
save('opf_results.mat', 'opf_results');

% Save to CSV
nSamples = numel(opf_results);
scenario = (1:nSamples).';  % scenario index


objective      = zeros(nSamples, 1); 
feasible       = false(nSamples, 1);    
V_abs_str      = cell(nSamples, 1);   
V_angle_str    = cell(nSamples, 1);      
gen_P_str      = cell(nSamples, 1);      
gen_Q_str      = cell(nSamples, 1);      
mpc_out_str    = cell(nSamples, 1);      

for k = 1:nSamples
    objective(k) = opf_results(k).f;
    feasible(k)  = opf_results(k).success;
    
    % Convert arrays to string
    V_abs_str{k}     = mat2str(abs(opf_results(k).V_complex), 4);
    V_angle_str{k}   = mat2str(angle(opf_results(k).V_complex), 4);
    gen_P_str{k}     = mat2str(opf_results(k).gen_P, 4);
    gen_Q_str{k}     = mat2str(opf_results(k).gen_Q, 4);

end

full_summary = table(scenario, objective, feasible, V_abs_str, V_angle_str, gen_P_str, gen_Q_str);
csv_filename = 'opf_results_all_fields_2.csv';
writetable(full_summary, csv_filename);
fprintf('Full OPF results summary saved to %s\n', csv_filename);
%% Save 
excelFile = 'OPF_Outputs_2.xlsx';

nBuses = length(opf_results(1).V_complex);
V_abs_matrix = zeros(nBuses, nSamples);
V_ang_matrix = zeros(nBuses, nSamples);
for k = 1:nSamples
    V_abs_matrix(:, k) = abs(opf_results(k).V_complex);
    V_ang_matrix(:, k) = angle(opf_results(k).V_complex);
end

%-----------------------------
% Extract Generator Active Powers (gen_P)
%-----------------------------.
nGen = length(opf_results(1).gen_P);
gen_P_matrix = zeros(nGen, nSamples);
for k = 1:nSamples
    gen_P_matrix(:, k) = opf_results(k).gen_P;
end

%-----------------------------
% Extract Generator Reactive Powers (gen_Q)
%-----------------------------
gen_Q_matrix = zeros(nGen, nSamples);
for k = 1:nSamples
    gen_Q_matrix(:, k) = opf_results(k).gen_Q;
end

%-----------------------------
% Extract Feasibility Flag (success)
%-----------------------------
feasibility_vector = false(nSamples, 1);
for k = 1:nSamples
    feasibility_vector(k) = opf_results(k).success;
end

%-----------------------------
% Extract Objective Function Value (f)
%-----------------------------
objective_vector = zeros(nSamples, 1);
for k = 1:nSamples
    objective_vector(k) = opf_results(k).f;
end

%-----------------------------
% Write each extracted output to a separate sheet in the Excel file
%-----------------------------
writematrix(V_abs_matrix, excelFile, 'Sheet', 'V_abs');
writematrix(V_ang_matrix, excelFile, 'Sheet', 'V_angle');
writematrix(gen_P_matrix, excelFile, 'Sheet', 'gen_P');
writematrix(gen_Q_matrix, excelFile, 'Sheet', 'gen_Q');
writematrix(feasibility_vector, excelFile, 'Sheet', 'feasibility');
writematrix(objective_vector, excelFile, 'Sheet', 'objective');

fprintf('All output fields saved to %s in separate sheets.\n', excelFile);


%% Save P_load and Q_load Matrices to Separate CSV Files
% These matrices have dimensions (number of buses x number of samples).
writematrix(P_matrix, 'P_load_2.csv');
writematrix(Q_matrix, 'Q_load_2.csv');

fprintf('P_load and Q_load matrices saved to P_load.csv and Q_load.csv respectively.\n');

