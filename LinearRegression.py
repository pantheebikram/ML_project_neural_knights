
def run_linear_regression_function():

    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pandas as pd
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import re
    import matplotlib.pyplot as plt


    # Load CSV data for loads
    df = pd.read_csv('P_load_2.csv', header=None)
    P_load = df.values.T  

    df = pd.read_csv('Q_load_2.csv', header=None)
    Q_load = df.values.T  

    # Load Excel file for various outputs
    xls = pd.ExcelFile('OPF_Outputs_2.xlsx')
    V_abs = pd.read_excel(xls, 'V_abs', header=None).values.T 
    V_angle = pd.read_excel(xls, 'V_angle', header=None).values.T  
    gen_P = pd.read_excel(xls, 'gen_P', header=None).values.T  
    gen_Q = pd.read_excel(xls, 'gen_Q', header=None).values.T  
    feasibility = pd.read_excel(xls, 'feasibility', header=None).values
    objective = pd.read_excel(xls, 'objective', header=None).values

    # Print shapes of loaded data
    print(f"P_load shape: {P_load.shape}")
    print(f"Q_load shape: {Q_load.shape}")
    print(f"V_abs shape: {V_abs.shape}")
    print(f"V_angle shape: {V_angle.shape}")
    print(f"gen_P shape: {gen_P.shape}")
    print(f"gen_Q shape: {gen_Q.shape}")

   
    # 2. Extend Generator Data to 14 buses
    num_buses = 14
    gen_P_extended = np.zeros((gen_P.shape[0], num_buses))
    gen_Q_extended = np.zeros((gen_Q.shape[0], num_buses))
    bus_indices = [1, 2, 3, 6, 8]
    for i, bus in enumerate(bus_indices):
        gen_P_extended[:, bus - 1] = gen_P[:, i]
        gen_Q_extended[:, bus - 1] = gen_Q[:, i]
    gen_P = gen_P_extended
    gen_Q = gen_Q_extended

    # 3. Load Ybus Matrix
    def parse_complex(element_str: str) -> complex:
        s = element_str.strip()
        if s in ['0', '0+0i', '0-0i']:
            return 0+0j
        pattern = r'^([-]?\d+(?:\.\d+)?)([-+]\d+(?:\.\d+)?)i$'
        match = re.match(pattern, s)
        if match:
            real_part = float(match.group(1))
            imag_part = float(match.group(2))
            return complex(real_part, imag_part)
        raise ValueError(f"Could not parse '{element_str}' as a complex number.")

    df_raw = pd.read_csv("Ybus_14 1.csv", header=None, dtype=str)
    Ybus = df_raw.applymap(parse_complex).to_numpy()
    Ybus_tensor = torch.tensor(Ybus, dtype=torch.complex64)


    # 4. Normalize Data Functions
    def normalize_tensor(tensor, method='minmax'):
        if method == 'minmax':
            min_val = tensor.min(dim=0, keepdim=True)[0]
            max_val = tensor.max(dim=0, keepdim=True)[0]
            return (tensor - min_val) / (max_val - min_val + 1e-6), min_val, max_val
    
    def unnormalize_tensor(norm_tensor, min_val, max_val, method='minmax'):
        if method == 'minmax':
            result = norm_tensor * (max_val - min_val + 1e-6) + min_val
            if result.dim() > 1 and result.shape[0] == 1:
                result = result.squeeze(0)
            return result
      

    # 5. Convert to Tensors and Normalize
    P_load = torch.tensor(P_load, dtype=torch.float32)
    Q_load = torch.tensor(Q_load, dtype=torch.float32)
    V_abs = torch.tensor(V_abs, dtype=torch.float32)
    V_angle = torch.tensor(V_angle, dtype=torch.float32)
    gen_P = torch.tensor(gen_P, dtype=torch.float32)
    gen_Q = torch.tensor(gen_Q, dtype=torch.float32)
    feasibility = torch.tensor(feasibility, dtype=torch.float32)
    objective = torch.tensor(objective, dtype=torch.float32)

    P_load_norm, P_min, P_max = normalize_tensor(P_load, 'minmax')
    Q_load_norm, Q_min, Q_max = normalize_tensor(Q_load, 'minmax')
    V_abs_norm, V_min, V_max = normalize_tensor(V_abs, 'minmax')
    V_angle_norm, theta_min, theta_max = normalize_tensor(V_angle, 'minmax')
    gen_P_norm, gen_P_min, gen_P_max = normalize_tensor(gen_P, 'minmax')
    gen_Q_norm, gen_Q_min, gen_Q_max = normalize_tensor(gen_Q, 'minmax')

   
    # 6. Define the Dataset
    class PowerFlowDataset(Dataset):
        def __init__(self, P_load, Q_load, V_abs, V_angle, gen_P, gen_Q):
            assert P_load.shape[0] == Q_load.shape[0] == V_abs.shape[0] == V_angle.shape[0] \
                == gen_P.shape[0] == gen_Q.shape[0], "Mismatch in number of samples."
            assert P_load.shape[1] == Q_load.shape[1] == V_abs.shape[1] == V_angle.shape[1] == 14, "Mismatch in number of buses."
            self.P_load = P_load
            self.Q_load = Q_load
            self.V_abs = V_abs
            self.V_angle = V_angle
            self.gen_P = gen_P
            self.gen_Q = gen_Q

        def __len__(self):
            return self.P_load.shape[0]

        def __getitem__(self, idx):
            load = torch.cat((self.P_load[idx], self.Q_load[idx]), dim=0)
            sample = {
                'load': load,
                'V_opt': self.V_abs[idx],
                'theta_opt': self.V_angle[idx],
                'P_gen': self.gen_P[idx],
                'Q_gen': self.gen_Q[idx],
                'P_load': self.P_load[idx],
                'Q_load': self.Q_load[idx]
            }
            return sample

    # Create full dataset and split it
    full_dataset = PowerFlowDataset(P_load_norm, Q_load_norm, V_abs_norm, V_angle_norm, gen_P_norm, gen_Q_norm)
    print(f"Total number of samples in dataset: {len(full_dataset)}")

    train_size = min(7600, int(0.8 * 9000))
    val_size = min(1900, 9000 - train_size)
    test_start = 9500
    test_size = max(0, len(full_dataset) - test_start)

    if test_size > 0:
        train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
        val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + val_size))
        test_dataset = torch.utils.data.Subset(full_dataset, range(test_start, len(full_dataset)))
    else:
        print("Warning: No samples available beyond index 9000 for testing.")
        train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
        val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
        test_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) if test_dataset is not None else None


    # 7. Define the Linear Regression Model

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim=28, output_dim=56):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
            self.n_buses = 14

        def forward(self, x):
            out = self.linear(x)
            return out

  
    # 8. Physics-Informed Loss Function
    def physics_informed_loss(pred_bounds, optimal, load_data, Ybus, 
                            lambda_data=30, lambda_phy=3, lambda_bound=1, lambda_width=0.5):
        batch_size = pred_bounds.shape[0]
        n_buses = 14
        Vmin = pred_bounds[:, 0:n_buses]
        Vmax = pred_bounds[:, n_buses:2*n_buses]
        theta_min = pred_bounds[:, 2*n_buses:3*n_buses]
        theta_max = pred_bounds[:, 3*n_buses:4*n_buses]

        # Data loss: 
        loss_data = torch.mean(torch.relu(Vmin - optimal['V'])**2 + torch.relu(optimal['V'] - Vmax)**2)
        loss_data += torch.mean(torch.relu(theta_min - optimal['theta'])**2 + torch.relu(optimal['theta'] - theta_max)**2)

        # Bound violation loss:
        bound_violation_loss = torch.mean(torch.relu(Vmin - Vmax)**2 + torch.relu(theta_min - theta_max)**2)

        # Width penalty: 
        V_width = Vmax - Vmin
        theta_width = theta_max - theta_min
        width_loss = torch.mean(V_width**2 + theta_width**2)

        # Physics-based loss: 
        V_mid = (Vmin + Vmax) / 2.0
        theta_mid = (theta_min + theta_max) / 2.0
        G = torch.real(Ybus).clone().detach()
        B = torch.imag(Ybus).clone().detach()
        
        theta_diff = theta_mid.unsqueeze(2) - theta_mid.unsqueeze(1)
        V_prod = V_mid.unsqueeze(2) * V_mid.unsqueeze(1)
        
        P_calc = torch.sum(V_prod * (G * torch.cos(theta_diff) + B * torch.sin(theta_diff)), dim=2)
        Q_calc = torch.sum(V_prod * (G * torch.sin(theta_diff) - B * torch.cos(theta_diff)), dim=2)
        
        P_exp = optimal['P_gen'] - load_data['P_load']
        Q_exp = optimal['Q_gen'] - load_data['Q_load']
        loss_phy = torch.mean((P_calc - P_exp)**2 + (Q_calc - Q_exp)**2)

        total_loss = lambda_data * loss_data + lambda_phy * loss_phy + lambda_bound * bound_violation_loss + lambda_width * width_loss
        return total_loss

   
    # 9. Training Setup

    model = LinearRegressionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Early stopping
    class EarlyStopping:
        def __init__(self, patience=20, min_delta=1e-2):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.counter = 0

        def __call__(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
            return self.counter >= self.patience

  
    # 10. Training Loop
    num_epochs = 100
    early_stopper = EarlyStopping(patience=20)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch['load']
            optimal = {
                'V': batch['V_opt'],
                'theta': batch['theta_opt'],
                'P_gen': batch['P_gen'],
                'Q_gen': batch['Q_gen']
            }
            load_data = {
                'P_load': batch['P_load'],
                'Q_load': batch['Q_load']
            }

            pred_bounds = model(inputs)
            loss = physics_informed_loss(pred_bounds, optimal, load_data, Ybus_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['load']
                optimal = {
                    'V': batch['V_opt'],
                    'theta': batch['theta_opt'],
                    'P_gen': batch['P_gen'],
                    'Q_gen': batch['Q_gen']
                }
                load_data = {
                    'P_load': batch['P_load'],
                    'Q_load': batch['Q_load']
                }
                pred_bounds = model(inputs)
                loss = physics_informed_loss(pred_bounds, optimal, load_data, Ybus_tensor)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
      

    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training vs Validation Loss (Linear Regression)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_curve_linear_regression.png')  
    plt.show()

   
    # 11. Test Evaluation
    def evaluate_test_set(model, test_loader, Ybus):
    

        model.eval()
        test_loss = 0.0
        all_results = []

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['load']
                optimal = {
                    'V': batch['V_opt'],
                    'theta': batch['theta_opt'],
                    'P_gen': batch['P_gen'],
                    'Q_gen': batch['Q_gen']
                }
                load_data = {
                    'P_load': batch['P_load'],
                    'Q_load': batch['Q_load']
                }
                
                pred_bounds = model(inputs)
                loss = physics_informed_loss(pred_bounds, optimal, load_data, Ybus)
                test_loss += loss.item()

                # Unnormalize predictions
                num_buses = 14
                Vmin_pred = pred_bounds[:, :num_buses]
                Vmax_pred = pred_bounds[:, num_buses:2*num_buses]
                theta_min_pred = pred_bounds[:, 2*num_buses:3*num_buses]
                theta_max_pred = pred_bounds[:, 3*num_buses:]

                Vmin_pred_unnorm = unnormalize_tensor(Vmin_pred, V_min, V_max, 'minmax')
                Vmax_pred_unnorm = unnormalize_tensor(Vmax_pred, V_min, V_max, 'minmax')
                theta_min_pred_unnorm = unnormalize_tensor(theta_min_pred, theta_min, theta_max, 'minmax')
                theta_max_pred_unnorm = unnormalize_tensor(theta_max_pred, theta_min, theta_max, 'minmax')
                V_opt_unnorm = unnormalize_tensor(optimal['V'], V_min, V_max, 'minmax')
                theta_opt_unnorm = unnormalize_tensor(optimal['theta'], theta_min, theta_max, 'minmax')

                # Process each sample in the batch
                for i in range(inputs.shape[0]):
                    P_load_unnorm = unnormalize_tensor(load_data['P_load'][i], P_min, P_max, 'minmax')
                    Q_load_unnorm = unnormalize_tensor(load_data['Q_load'][i], Q_min, Q_max, 'minmax')

    
                    V_within_bounds = (Vmin_pred_unnorm[i] <= V_opt_unnorm[i]) & (V_opt_unnorm[i] <= Vmax_pred_unnorm[i])
                    theta_within_bounds = (theta_min_pred_unnorm[i] <= theta_opt_unnorm[i]) & (theta_opt_unnorm[i] <= theta_max_pred_unnorm[i])
                    
                    result = {
                        'P_load': P_load_unnorm.tolist(),
                        'Q_load': Q_load_unnorm.tolist(),
                        'Vmin_pred': Vmin_pred_unnorm[i].tolist(),
                        'V_opt': V_opt_unnorm[i].tolist(),
                        'Vmax_pred': Vmax_pred_unnorm[i].tolist(),
                        'theta_min_pred': theta_min_pred_unnorm[i].tolist(),
                        'theta_opt': theta_opt_unnorm[i].tolist(),
                        'theta_max_pred': theta_max_pred_unnorm[i].tolist(),
                        'V_within_bounds': V_within_bounds.tolist(),
                        'theta_within_bounds': theta_within_bounds.tolist()
                    }
                    # Verify that each result list has length equal to num_buses
                    lengths = {k: len(v) for k, v in result.items()}
                    all_results.append(result)

        avg_test_loss = test_loss / len(test_loader)
        print(f"Average Test Loss on samples 9000-end: {avg_test_loss:.4f}")

        # Create a DataFrame for test results
        results_df = pd.DataFrame(all_results)
        results_df['Sample'] = range(9000, 9000 + len(all_results))
        
        # Explode columns so each bus appears in its own row
        results_df = results_df.explode(['P_load', 'Q_load', 'Vmin_pred', 'V_opt', 'Vmax_pred', 
                                        'theta_min_pred', 'theta_opt', 'theta_max_pred', 
                                        'V_within_bounds', 'theta_within_bounds'])
        results_df['Bus'] = (results_df.groupby('Sample').cumcount() % num_buses) + 1
        
        # print("\nTest Set Results (first few samples):")
        # print(results_df.head(50).to_string(index=False))
        
        # Summary statistics
        v_bounds_accuracy = results_df['V_within_bounds'].mean() * 100
        theta_bounds_accuracy = results_df['theta_within_bounds'].mean() * 100
        print(f"\nPercentage of V values within predicted bounds: {v_bounds_accuracy:.2f}%")
        print(f"Percentage of theta values within predicted bounds: {theta_bounds_accuracy:.2f}%")
        
        return v_bounds_accuracy, theta_bounds_accuracy, avg_test_loss,results_df

    if test_loader is not None:
        v_bounds_accuracy, theta_bounds_accuracy, avg_test_loss, test_results = evaluate_test_set(model, test_loader, Ybus_tensor)
    else:
     print("Skipping test evaluation due to empty test set.")
     v_bounds_accuracy = None
     theta_bounds_accuracy = None
     avg_test_loss = None

    return train_losses, val_losses, v_bounds_accuracy, theta_bounds_accuracy, avg_test_loss, test_results

