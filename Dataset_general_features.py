import torch
# Process unit --> number
process_unit_library = {
    "valve": 2,
    "column": 3,
    "HX": 4,
    "condenser": 5,
    "pump": 6,
    "reboiler": 7,
    "tank": 8,
    "compressor": 9, 
    "evaporator": 10,
    "knockout_drum": 11, 
    "CSTR": 12, 
    "jacket": 13, 
    "splitter": 14,
    "decanter": 15,
    "filter": 16, 
    "dosing_unit": 17, 
    "reactor": 18, 
    "furnace": 19, 
    "bubbler": 20, 
    "spectroscopic_ellipsometer": 21, 
    "steam_drum_HX": 22, 
    "settling_tank": 23, 
    "PSV": 24, 
    "mixer": 25, 
    "cooler": 26, 
    "liq_reciever": 27, 
    "turbine": 28
    #"3-way_valve": , 
    #"adiabatic_reactor": 100, 
}

control_unit_library = {
    "SP": 29, 
    "FT": 30, 
    "FC": 31, 
    "TT": 32, 
    "TC": 33, 
    "LT": 34, 
    "LC": 35, 
    "PT": 36, 
    "PC": 37, 
    "CT": 38, 
    "CC": 39, 
    "z": 40,       # valve position 
    "w": 41,      # pump frequency 
    "PDT": 42,    # Pressure differential transmitter 
    "GRT": 43, 
    "GRC": 44, 
    "IT": 45, 
    "IC": 46, 
    "X_multiplication_block": 47,
    "summation_block": 48, 
    "addition_block": 49,
    "subtraction_block": 50,  
    "VPC": 51,  
    "SRC": 52, 
    "MIN": 53, 
    "MAX": 54, 
}

full_unit_lib = {**process_unit_library, **control_unit_library}


# EX1 ---- DISTILLATION COLUMN ----
# Process: 
process1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10  11  12  13
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0,  0], #1 Valve
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0,   0,  0,  0], #2 Column
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0,   1,  0,  0], #3 Heat exchanger
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0,   0,  0,  0], #4 Condenser  
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0,   0,  0,  1], #5 Pump
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0,  0], #6 Valve 
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0,   0,  1,  0], #7 Heat exchanger
[0, 1, 0, 0, 0, 0, 0, 0, 1, 0,   0,  0,  0], #8 Reboiler
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1,   0,  0,  0], #9 Pump
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0,  0], #10 Valve  
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0,  0], #11 Valve  
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0,  0], #12 Valve   
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0,  0], #13 Valve 
], dtype=torch.float32)

u_process1 = ["valve", "column", "HX", "condenser", "pump", "valve", "HX", "reboiler", 
            "pump", "valve", "valve", "valve", "valve"]

features_p1 = torch.tensor([[full_unit_lib[unit]] for unit in u_process1], dtype=torch.float32)

# Control: 
control1_1_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control1_1 =  ["SP", "FC", "z", "FT"]
features_c1_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_1], dtype=torch.float32)
reg_u_c1_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c1_1 = torch.cat([features_c1_1, reg_u_c1_1], dim=1)  # shape [N, 2]


control1_2_adj= torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP (L/F)_s
[0, 0, 1, 0, 0, 0],  #2 X multiplication block
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 1],  #4 z
[0, 1, 0, 0, 0, 0],  #5 FT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control1_2 = ["SP", "X_multiplication_block", "FC", "z", "FT", "FT"]
features_c1_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_2], dtype=torch.float32)
reg_u_c1_2 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c1_2 = torch.cat([features_c1_2, reg_u_c1_2], dim=1)  # shape [N, 2]

control1_3_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control1_3 =  ["SP", "LC", "z", "LT"]
features_c1_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_3], dtype=torch.float32)
reg_u_c1_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c1_3 = torch.cat([features_c1_3, reg_u_c1_3], dim=1)  # shape [N, 2]

control1_4_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control1_4 = ["SP", "PC", "z", "PT"]
features_c1_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_4], dtype=torch.float32)
reg_u_c1_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c1_4 = torch.cat([features_c1_4, reg_u_c1_4], dim=1)  # shape [N, 2]

control1_5_adj= torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP (V/F)_s
[0, 0, 1, 0, 0, 0],  #2 X multiplication block
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 1],  #4 z
[0, 1, 0, 0, 0, 0],  #5 FT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control1_5 = ["SP", "X_multiplication_block", "FC", "z", "FT", "FT"]
features_c1_5 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_5], dtype=torch.float32)
reg_u_c1_5 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c1_5 = torch.cat([features_c1_5, reg_u_c1_5], dim=1)  # shape [N, 2]

control1_6_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control1_6 = ["SP", "LC", "z", "LT"]
features_c1_6 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_6], dtype=torch.float32)
reg_u_c1_6 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c1_6 = torch.cat([features_c1_6, reg_u_c1_6], dim=1)  # shape [N, 2]


#------------------------------------------------------------------
# EX2  ----  TWO HEATED TANKS ----
# Process:
process2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 Valve  
[0, 0, 0, 0],  #2 Tank
[0, 1, 0, 0],  #3 HX
[0, 0, 1, 0],  #4 Valve
], dtype=torch.float32)
u_process2 = ["valve", "tank", "HX", "valve"]
features_p2 = torch.tensor([[full_unit_lib[unit]] for unit in u_process2], dtype=torch.float32)

#Control:
control2_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0],  #2 LC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 z (valve position) 
[0, 1, 0, 0, 0, 0],  #5 LT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control2_1 = ["SP", "LC", "FC", "z", "LT", "FT"]
features_c2_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control2_1], dtype=torch.float32)
reg_u_c2_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c2_1 = torch.cat([features_c2_1, reg_u_c2_1], dim=1)  # shape [N, 2]

control2_2_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 z (valve position) 
[0, 1, 0, 0, 0, 0],  #5 TT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control2_2 = ["SP", "TC", "FC", "z", "TT", "FT"]
features_c2_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control2_2], dtype=torch.float32)
reg_u_c2_2 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c2_2 = torch.cat([features_c2_2, reg_u_c2_2], dim=1)  # shape [N, 2]



#------------------------------------------------------------------
# EX3 ---- COMPRESSOR ----
# Process:
process3_adj = torch.tensor([
 #1  2  3  4  
[0, 1, 0, 0],  #1 Valve 
[0, 0, 1, 0],  #2 Compressor
[0, 0, 0, 1],  #3 Tank 
[0, 0, 0, 0],  #4 Valve 
], dtype=torch.float32)
u_process3 = ["valve", "compressor", "tank", "valve"]
features_p3 = torch.tensor([[full_unit_lib[unit]] for unit in u_process3], dtype=torch.float32)

# Control:
control3_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 1],  #3 z (valve position)
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control3_1 = ["SP", "PC", "z", "PT"]
features_c3_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control3_1], dtype=torch.float32)
reg_u_c3_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c3_1 = torch.cat([features_c3_1, reg_u_c3_1], dim=1)  # shape [N, 2]

control3_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control3_2 = ["SP", "PC", "z", "PT"]
features_c3_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control3_2], dtype=torch.float32)
reg_u_c3_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c3_2 = torch.cat([features_c3_2, reg_u_c3_2], dim=1)  # shape [N, 2]


#------------------------------------------------------------------
# EX4 ---- Anti-surge ----
# Process: 
process4_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 Evaporator  
[0, 0, 1, 0, 0, 0],  #2 Splitter 
[0, 0, 0, 1, 0, 0],  #3 Compressor
[0, 0, 0, 0, 1, 1],  #4 Splitter
[0, 1, 0, 0, 0, 0],  #5 Valve
[0, 0, 0, 0, 0, 0],  #6 Knockout drum 
], dtype=torch.float32)
u_process4 = ["evaporator", "splitter", "compressor", "splitter", "valve", "knockout_drum"]
features_p4 = torch.tensor([[full_unit_lib[unit]] for unit in u_process4], dtype=torch.float32)

# Control: 
control4_1_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 SP = F_min 
[0, 0, 1, 0, 0, 0, 0],  #2 FC
[0, 0, 0, 0, 0, 0, 0],  #3 z
[0, 1, 0, 0, 0, 0, 0],  #4 PDT
[0, 0, 0, 1, 0, 0, 0],  #5 PT
[0, 1, 0, 0, 0, 0, 0],  #6 FT
[0, 0, 0, 1, 0, 0, 0],  #7 PT
], dtype=torch.float32)
u_control4_1 = ["SP", "FC", "z", "PDT", "PT", "FT", "PT"]
features_c4_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control4_1], dtype=torch.float32)
reg_u_c4_1 = torch.tensor([[0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c4_1 = torch.cat([features_c4_1, reg_u_c4_1], dim=1)  # shape [N, 2]


#------------------------------------------------------------------
# EX5 ---- SHELL & TUBE HX ----
# Process: 
process5_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 HX
], dtype=torch.float32)
u_process5 = ["valve", "HX"]
features_p5 = torch.tensor([[full_unit_lib[unit]] for unit in u_process5], dtype=torch.float32)


#Control: 
control5_1_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0],  #3 PC
[0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 0],  #5 TT
[0, 0, 1, 0, 0, 0],  #6 PT
], dtype=torch.float32)
u_control5_1 = ["SP", "TC", "PC", "z", "TT", "PT"]
features_c5_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control5_1], dtype=torch.float32)
reg_u_c5_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c5_1 = torch.cat([features_c5_1, reg_u_c5_1], dim=1)  # shape [N, 2]


#------------------------------------------------------------------
# EX6 ----JACKETED CSTR ----
process6_adj = torch.tensor([
# Process: 
#1  2  3  4  
[0, 1, 0, 0],  #1 Valve 
[0, 0, 0, 0],  #2 CSTR
[0, 1, 0, 1],  #3 Jacket
[0, 0, 0, 0],  #4 Valve
], dtype=torch.float32)
u_process6 = ["valve", "CSTR", "jacket", "valve"]
features_p6 = torch.tensor([[full_unit_lib[unit]] for unit in u_process6], dtype=torch.float32)

#Control:
control6_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control6_1 = ["SP", "PC", "z", "PT"]
features_c6_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control6_1], dtype=torch.float32)
reg_u_c6_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c6_1 = torch.cat([features_c6_1, reg_u_c6_1], dim=1)  # shape [N, 2]

control6_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)

u_control6_2 = ["SP", "TC", "z", "TT"]
features_c6_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control6_2], dtype=torch.float32)
reg_u_c6_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c6_2 = torch.cat([features_c6_2, reg_u_c6_2], dim=1)  # shape [N, 2]

#------------------------------------------------------------------
# EX7 ---- JACKETED CSTR 2 ----
# Process: 
process7_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve 
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)
u_process7 = ["valve", "jacket", "CSTR"]
features_p7 = torch.tensor([[full_unit_lib[unit]] for unit in u_process7], dtype=torch.float32)

# Control: 
control7_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control7_1 = ["SP", "TC", "z", "TT"]
features_c7_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control7_1], dtype=torch.float32)
reg_u_c7_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c7_1 = torch.cat([features_c7_1, reg_u_c7_1], dim=1)  # shape [N, 2]


#------------------------------------------------------------------
# EX8 ---- DISTILLATION COLUMN 2 ----
# Process: 
process8_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12 
[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #1 Column 
[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  #2 HX
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #3 Condenser
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #4 Pump
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  #5 Splitter
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #6 Valve
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  #7 Splitter
[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #8 Reboiler
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #11 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #12 Valve
], dtype=torch.float32)

u_process8 = ["column", "HX", "condenser", "pump", "splitter", "valve", "splitter", "reboiler", "valve", "valve", "valve", "valve"]
features_p8 = torch.tensor([[full_unit_lib[unit]] for unit in u_process8], dtype=torch.float32)

# Control 1:
control8_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control8_1 = ["SP", "CC", "z", "CT"]
features_c8_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_1], dtype=torch.float32)
reg_u_c8_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c8_1 = torch.cat([features_c8_1, reg_u_c8_1], dim=1)

# Control 2:
control8_2_adj = torch.tensor([
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 0],
[0, 1, 0, 0],
], dtype=torch.float32)
u_control8_2 = ["SP", "CC", "z", "CT"]
features_c8_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_2], dtype=torch.float32)
reg_u_c8_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c8_2 = torch.cat([features_c8_2, reg_u_c8_2], dim=1)

# Control 3:
control8_3_adj = torch.tensor([
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 0],
[0, 1, 0, 0],
], dtype=torch.float32)
u_control8_3 = ["SP", "LC", "z", "LT"]
features_c8_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_3], dtype=torch.float32)
reg_u_c8_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c8_3 = torch.cat([features_c8_3, reg_u_c8_3], dim=1)

# Control 4:
control8_4_adj = torch.tensor([
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 0],
[0, 1, 0, 0],
], dtype=torch.float32)
u_control8_4 = ["SP", "LC", "z", "LT"]
features_c8_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_4], dtype=torch.float32)
reg_u_c8_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c8_4 = torch.cat([features_c8_4, reg_u_c8_4], dim=1)


#------------------------------------------------------------------
# EX9 ---- SHELL & TUBE HX 2 ----
# Process:
process9_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 HX
], dtype=torch.float32)
u_process9 = ["valve", "HX"]
features_p9 = torch.tensor([[full_unit_lib[unit]] for unit in u_process9], dtype=torch.float32)

# Control:
control9_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control9_1 = ["SP", "TC", "z", "TT"]
features_c9_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control9_1], dtype=torch.float32)
reg_u_c9_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c9_1 = torch.cat([features_c9_1, reg_u_c9_1], dim=1)


#------------------------------------------------------------------
# EX10 ---- HEAT EXCHANGER ----
# Process: 
process10_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 HX
[0, 1, 0],  #3 Valve
], dtype=torch.float32)
u_process10 = ["valve", "HX", "valve"]
features_p10 = torch.tensor([[full_unit_lib[unit]] for unit in u_process10], dtype=torch.float32)

# Control:
control10_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control10_1 = ["SP", "TC", "z", "TT"]
features_c10_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control10_1], dtype=torch.float32)
reg_u_c10_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c10_1 = torch.cat([features_c10_1, reg_u_c10_1], dim=1)


#------------------------------------------------------------------
# EX11 ---- FILTER AND DECANTATION ----
# Process: 
process11_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 Valve 
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #2 Tank
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #3 Pump
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #4 Splitter
[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  #5 Decanter
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #6 Pump
[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #7 Filter
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #8 Valve
[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  #9 Filter
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 Valve
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #11 Dosing unit
], dtype=torch.float32)
u_process11 = ["valve", "tank", "pump", "splitter", "decanter", "pump", "filter", "valve", "filter", "valve", "dosing_unit"]
features_p11 = torch.tensor([[full_unit_lib[unit]] for unit in u_process11], dtype=torch.float32)

# Control 1:
control11_1_adj = torch.tensor([
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 1],
[0, 1, 0, 0],
], dtype=torch.float32)
u_control11_1 = ["SP", "FC", "w", "FT"]
features_c11_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control11_1], dtype=torch.float32)
reg_u_c11_1 = torch.tensor([[0], [6], [0], [0]], dtype=torch.float32)
feature_tensor_c11_1 = torch.cat([features_c11_1, reg_u_c11_1], dim=1)

# Control 2:
control11_2_adj = torch.tensor([
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 1],
[0, 1, 0, 0],
], dtype=torch.float32)
u_control11_2 = ["SP", "FC", "w", "FT"]
features_c11_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control11_2], dtype=torch.float32)
reg_u_c11_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c11_2 = torch.cat([features_c11_2, reg_u_c11_2], dim=1)



#------------------------------------------------------------------
# EX12 ----TANK 1 ----
# Process: 
process12_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Tank
], dtype=torch.float32)
u_process12 = ["valve", "tank"]
features_p12 = torch.tensor([[full_unit_lib[unit]] for unit in u_process12], dtype=torch.float32)

#Control:
control12_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control12_1 = ["SP", "LC", "z", "LT"]
features_c12_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control12_1], dtype=torch.float32)
reg_u_c12_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c12_1 = torch.cat([features_c12_1, reg_u_c12_1], dim=1)

#------------------------------------------------------------------
# EX13 ---- TANK 2 ----
# Process: 
process13_adj = torch.tensor([
#1  2  
[0, 1],  #1 Tank   
[0, 0],  #2 Valve
], dtype=torch.float32)
u_process13 = ["tank", "valve"]
features_p13 = torch.tensor([[full_unit_lib[unit]] for unit in u_process13], dtype=torch.float32)

# Control:
control13_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control13_1 = ["SP", "LC", "z", "LT"]
features_c13_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control13_1], dtype=torch.float32)
reg_u_c13_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c13_1 = torch.cat([features_c13_1, reg_u_c13_1], dim=1)


#------------------------------------------------------------------
# EX14 ---- TANK 3 ----
# Process: 
process14_adj = torch.tensor([
#1  2  
[0, 1],  #1 Tank   
[0, 0],  #2 Valve
], dtype=torch.float32)
u_process14 = ["tank", "valve"]
features_p14 = torch.tensor([[full_unit_lib[unit]] for unit in u_process14], dtype=torch.float32)

# Control:
control14_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control14_1 = ["SP", "FC", "z", "FT"]
features_c14_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control14_1], dtype=torch.float32)
reg_u_c14_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c14_1 = torch.cat([features_c14_1, reg_u_c14_1], dim=1)


#------------------------------------------------------------------
# EX15 ---- TANK 4 ----
# Process: 
process15_adj = torch.tensor([
#1  2  
[0, 1],  #1 Tank   
[0, 0],  #2 Valve
], dtype=torch.float32)
u_process15 = ["tank", "valve"]
features_p15 = torch.tensor([[full_unit_lib[unit]] for unit in u_process15], dtype=torch.float32)

# Control:
control15_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 LC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 0],  #5 LT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control15_1 = ["SP", "LC", "FC", "z", "LT", "FT"]
features_c15_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control15_1], dtype=torch.float32)
reg_u_c15_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c15_1 = torch.cat([features_c15_1, reg_u_c15_1], dim=1)


#------------------------------------------------------------------
# EX16 ---- ADIABATIC REACTOR ----
# Process: 
process16_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 HX
[0, 1, 0],  #3 Adiabatic reactor
], dtype=torch.float32)
u_process16 = ["valve", "HX", "reactor"]
features_p16 = torch.tensor([[full_unit_lib[unit]] for unit in u_process16], dtype=torch.float32)

# Control:
control16_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control16_1 = ["SP", "CC", "z", "CT"]
features_c16_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control16_1], dtype=torch.float32)
reg_u_c16_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c16_1 = torch.cat([features_c16_1, reg_u_c16_1], dim=1)

#------------------------------------------------------------------
# EX17 ---- CSTR ----
# Process: 
process17_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 CSTR
], dtype=torch.float32)
u_process17 = ["valve", "CSTR"]
features_p17 = torch.tensor([[full_unit_lib[unit]] for unit in u_process17], dtype=torch.float32)

# Control:
control17_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control17_1 = ["SP", "CC", "z", "CT"]
features_c17_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control17_1], dtype=torch.float32)
reg_u_c17_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c17_1 = torch.cat([features_c17_1, reg_u_c17_1], dim=1)


#------------------------------------------------------------------
# EX18 ---- FURNACE ----
# Process: 
process18_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Furnace
], dtype=torch.float32)
u_process18 = ["valve", "furnace"]
features_p18 = torch.tensor([[full_unit_lib[unit]] for unit in u_process18], dtype=torch.float32)

# Control:
control18_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control18_1 = ["SP", "TC", "z", "TT"]
features_c18_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control18_1], dtype=torch.float32)
reg_u_c18_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c18_1 = torch.cat([features_c18_1, reg_u_c18_1], dim=1)


#------------------------------------------------------------------
# EX19 ---- CASCADE FURNACE ----
# Process:
process19_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Furnace
], dtype=torch.float32)
u_process19 = ["valve", "furnace"]
features_p19 = torch.tensor([[full_unit_lib[unit]] for unit in u_process19], dtype=torch.float32)

# Control:
control19_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 1, 0],  #4 z
[0, 1, 0, 0, 0, 0],  #5 TT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control19_1 = ["SP", "TC", "FC", "z", "TT", "FT"]
features_c19_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control19_1], dtype=torch.float32)
reg_u_c19_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c19_1 = torch.cat([features_c19_1, reg_u_c19_1], dim=1)


#------------------------------------------------------------------
# EX20 ---- CASCADE CSTR ----
# Process: 
process20_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)
u_process20 = ["valve", "jacket", "CSTR"]
features_p20 = torch.tensor([[full_unit_lib[unit]] for unit in u_process20], dtype=torch.float32)

# Control:
control20_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0],  #3 TC
[0, 0, 0, 0, 1, 1],  #4 z
[0, 1, 0, 0, 0, 0],  #5 TT
[0, 0, 1, 0, 0, 0],  #6 TT
], dtype=torch.float32)
u_control20_1 = ["SP", "TC", "TC", "z", "TT", "TT"]
features_c20_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control20_1], dtype=torch.float32)
reg_u_c20_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c20_1 = torch.cat([features_c20_1, reg_u_c20_1], dim=1)


#------------------------------------------------------------------
# EX21 ---- FURNACE 2 ----
# Process: 
process21_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Furnace
], dtype=torch.float32)
u_process21 = ["valve", "furnace"]
features_p21 = torch.tensor([[full_unit_lib[unit]] for unit in u_process21], dtype=torch.float32)

# Control:
control21_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP  
[0, 0, 1, 0, 0, 0, 0, 0],  #2 FC
[0, 0, 0, 1, 0, 0, 0, 0],  #3 Summation block
[0, 0, 0, 0, 0, 0, 1, 1],  #4 z
[0, 0, 1, 0, 0, 0, 0, 0],  #5 TC
[0, 0, 0, 0, 1, 0, 0, 0],  #6 SP
[0, 1, 0, 0, 0, 0, 0, 0],  #7 FT
[0, 0, 0, 0, 1, 0, 0, 0],  #8 TT
], dtype=torch.float32)
u_control21_1 = ["SP", "FC", "summation_block", "z", "TC", "SP", "FT", "TT"]
features_c21_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control21_1], dtype=torch.float32)
reg_u_c21_1 = torch.tensor([[0], [0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c21_1 = torch.cat([features_c21_1, reg_u_c21_1], dim=1)

#------------------------------------------------------------------
# EX22 ---- JACKETED CSTR 3 ----
# Process: 
process22_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)
u_process22 = ["valve", "jacket", "CSTR"]
features_p22 = torch.tensor([[full_unit_lib[unit]] for unit in u_process22], dtype=torch.float32)

# Control:
control22_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0, 0, 0],  #3 TC
[0, 0, 0, 0, 1, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 1, 1, 0],  #5 z
[0, 1, 0, 0, 0, 0, 0, 0],  #6 TT
[0, 0, 1, 0, 0, 0, 0, 0],  #7 TT
[0, 0, 0, 1, 0, 0, 0, 0],  #8 FT
], dtype=torch.float32)
u_control22_1 = ["SP", "TC", "TC", "FC", "z", "TT", "TT", "FT"]
features_c22_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control22_1], dtype=torch.float32)
reg_u_c22_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c22_1 = torch.cat([features_c22_1, reg_u_c22_1], dim=1)


#------------------------------------------------------------------
# EX23 ---- JACKETED CSTR 4 ----
# Process:
process23_adj = torch.tensor([
#1  2  3
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)
u_process23 = ["valve", "jacket", "CSTR"]
features_p23 = torch.tensor([[full_unit_lib[unit]] for unit in u_process23], dtype=torch.float32)

# Control:
control23_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #2 CC
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #3 TC
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #4 TC
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #5 FC
[0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  #6 z
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #7 CT
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #8 TT
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #9 TT
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #10 FT
], dtype=torch.float32)
u_control23_1 = ["SP", "CC", "TC", "TC", "FC", "z", "CT", "TT", "TT", "FT"]
features_c23_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control23_1], dtype=torch.float32)
reg_u_c23_1 = torch.tensor([[0], [0], [0], [0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c23_1 = torch.cat([features_c23_1, reg_u_c23_1], dim=1)


#------------------------------------------------------------------
# EX24 ---- GROWTH RATE ----
# Process:
process24_adj = torch.tensor([
#1  2  3  4  5  6
[0, 1, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0],  #2 Bubbler
[0, 0, 0, 1, 0, 0],  #3 Splitter
[0, 0, 0, 0, 1, 1],  #4 Reactor
[0, 0, 0, 0, 0, 0],  #5 Spectroscopic ellipsometer
[0, 0, 0, 0, 0, 0],  #6 Pump
], dtype=torch.float32)
u_process24 = ["valve", "bubbler", "splitter", "reactor", "spectroscopic_ellipsometer", "pump"]
features_p24 = torch.tensor([[full_unit_lib[unit]] for unit in u_process24], dtype=torch.float32)

# Control:
control24_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0],  #2 GRC
[0, 0, 0, 1, 0, 0, 0, 0],  #3 CC
[0, 0, 0, 0, 1, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 0, 0, 1],  #5 z
[0, 1, 0, 0, 0, 0, 0, 0],  #6 GRT
[0, 0, 1, 0, 0, 0, 0, 0],  #7 CT
[0, 0, 0, 1, 0, 0, 0, 0],  #8 FT
], dtype=torch.float32)
u_control24_1 = ["SP", "GRC", "CC", "FC", "z", "GRT", "CT", "FT"]
features_c24_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control24_1], dtype=torch.float32)
reg_u_c24_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c24_1 = torch.cat([features_c24_1, reg_u_c24_1], dim=1)

#------------------------------------------------------------------
# EX25 ---- BLEND STREAM ----
# Process:
process25_adj = torch.tensor([
#1  2
[0, 1],  #1 Valve
[0, 0],  #2 Splitter
], dtype=torch.float32)
u_process25 = ["valve", "splitter"]
features_p25 = torch.tensor([[full_unit_lib[unit]] for unit in u_process25], dtype=torch.float32)

# Control:
control25_1_adj = torch.tensor([
#1  2  3  4  5  6
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 Multiplication block
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 1],  #4 z
[0, 1, 0, 0, 0, 0],  #5 FT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control25_1 = ["SP", "X_multiplication_block", "FC", "z", "FT", "FT"]
features_c25_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control25_1], dtype=torch.float32)
reg_u_c25_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c25_1 = torch.cat([features_c25_1, reg_u_c25_1], dim=1)

#------------------------------------------------------------------
# EX26 ---- CSTR + SEPARATOR ----
# Process:
process26_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0],  #2 Splitter
[0, 0, 0, 1, 0, 0, 0],  #3 CSTR
[0, 0, 0, 0, 1, 0, 0],  #4 Valve
[0, 0, 0, 0, 0, 1, 0],  #5 Splitter
[0, 1, 0, 0, 0, 0, 0],  #6 Valve
[0, 0, 0, 0, 0, 0, 0],  #7 Valve
], dtype=torch.float32)
u_process26 = ["valve", "splitter", "CSTR", "valve", "splitter", "valve", "valve"]
features_p26 = torch.tensor([[full_unit_lib[unit]] for unit in u_process26], dtype=torch.float32)

# Control 1:
control26_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control26_1 = ["SP", "FC", "z", "FT"]
features_c26_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control26_1], dtype=torch.float32)
reg_u_c26_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c26_1 = torch.cat([features_c26_1, reg_u_c26_1], dim=1)

# Control 2:
control26_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control26_2 = ["SP", "LC", "z", "LT"]
features_c26_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control26_2], dtype=torch.float32)
reg_u_c26_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c26_2 = torch.cat([features_c26_2, reg_u_c26_2], dim=1)

# Control 3:
control26_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control26_3 = ["SP", "CC", "z", "CT"]
features_c26_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control26_3], dtype=torch.float32)
reg_u_c26_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c26_3 = torch.cat([features_c26_3, reg_u_c26_3], dim=1)

# Control 4:
control26_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control26_4 = ["SP", "FC", "z", "FT"]
features_c26_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control26_4], dtype=torch.float32)
reg_u_c26_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c26_4 = torch.cat([features_c26_4, reg_u_c26_4], dim=1)


#------------------------------------------------------------------
# EX27 ---- CSTR + SEPARATOR 2 ----
# Process:
process27_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0],  #2 Splitter
[0, 0, 0, 1, 0, 0, 0],  #3 CSTR
[0, 0, 0, 0, 1, 0, 0],  #4 Valve
[0, 0, 0, 0, 0, 1, 0],  #5 Splitter
[0, 1, 0, 0, 0, 0, 0],  #6 Valve
[0, 0, 0, 0, 0, 0, 0],  #7 Valve
], dtype=torch.float32)
u_process27 = ["valve", "splitter", "CSTR", "valve", "splitter", "valve", "valve"]
features_p27 = torch.tensor([[full_unit_lib[unit]] for unit in u_process27], dtype=torch.float32)

# Control 1:
control27_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control27_1 = ["SP", "FC", "z", "FT"]
features_c27_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control27_1], dtype=torch.float32)
reg_u_c27_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c27_1 = torch.cat([features_c27_1, reg_u_c27_1], dim=1)

# Control 2:
control27_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control27_2 = ["SP", "LC", "z", "LT"]
features_c27_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control27_2], dtype=torch.float32)
reg_u_c27_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c27_2 = torch.cat([features_c27_2, reg_u_c27_2], dim=1)

# Control 3:
control27_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control27_3 = ["SP", "CC", "z", "CT"]
features_c27_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control27_3], dtype=torch.float32)
reg_u_c27_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c27_3 = torch.cat([features_c27_3, reg_u_c27_3], dim=1)

# Control 4:
control27_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control27_4 = ["SP", "FC", "z", "FT"]
features_c27_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control27_4], dtype=torch.float32)
reg_u_c27_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c27_4 = torch.cat([features_c27_4, reg_u_c27_4], dim=1)


#------------------------------------------------------------------
# EX28 ---- CSTR + DISTILLATION ----
# Process:
process28_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 Splitter
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3 CSTR
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #4 Valve
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #5 Column
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #6 HX
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #7 Condenser
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #8 Pump
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  #9 Splitter
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  #11 Splitter
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #12 Valve
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #13 HX
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #14 Valve
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #15 Valve
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #16 Valve
], dtype=torch.float32)
u_process28 = ["valve", "splitter", "CSTR", "valve", "column", "HX", "condenser", 
               "pump", "splitter", "valve", "splitter", "valve", "HX", "valve", "valve", "valve"]
features_p28 = torch.tensor([[full_unit_lib[unit]] for unit in u_process28], dtype=torch.float32)

# Control 1:
control28_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control28_1 = ["SP", "FC", "z", "FT"]
features_c28_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control28_1], dtype=torch.float32)
reg_u_c28_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c28_1 = torch.cat([features_c28_1, reg_u_c28_1], dim=1)

# Control 2:
control28_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control28_2 = ["SP", "LC", "z", "LT"]
features_c28_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control28_2], dtype=torch.float32)
reg_u_c28_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c28_2 = torch.cat([features_c28_2, reg_u_c28_2], dim=1)

# Control 3:
control28_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control28_3 = ["SP", "CC", "z", "CT"]
features_c28_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control28_3], dtype=torch.float32)
reg_u_c28_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c28_3 = torch.cat([features_c28_3, reg_u_c28_3], dim=1)

# Control 4:
control28_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control28_4 = ["SP", "LC", "z", "LT"]
features_c28_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control28_4], dtype=torch.float32)
reg_u_c28_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c28_4 = torch.cat([features_c28_4, reg_u_c28_4], dim=1)

# Control 5:
control28_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control28_5 = ["SP", "CC", "z", "CT"]
features_c28_5 = torch.tensor([[full_unit_lib[unit]] for unit in u_control28_5], dtype=torch.float32)
reg_u_c28_5 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c28_5 = torch.cat([features_c28_5, reg_u_c28_5], dim=1)

# Control 6:
control28_6_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control28_6 = ["SP", "PC", "z", "PT"]
features_c28_6 = torch.tensor([[full_unit_lib[unit]] for unit in u_control28_6], dtype=torch.float32)
reg_u_c28_6 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c28_6 = torch.cat([features_c28_6, reg_u_c28_6], dim=1)

# Control 7:
control28_7_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control28_7 = ["SP", "LC", "z", "LT"]
features_c28_7 = torch.tensor([[full_unit_lib[unit]] for unit in u_control28_7], dtype=torch.float32)
reg_u_c28_7 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c28_7 = torch.cat([features_c28_7, reg_u_c28_7], dim=1)

#------------------------------------------------------------------
# EX29 ---- TANK 5 ----
# Process:
process29_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 Valve
[0, 0, 1, 0],  #2 Tank
[0, 0, 0, 1],  #3 Pump
[0, 0, 0, 0],  #4 Valve
], dtype=torch.float32)
u_process29 = ["valve", "tank", "pump", "valve"]
features_p29 = torch.tensor([[full_unit_lib[unit]] for unit in u_process29], dtype=torch.float32)

# Control 1:
control29_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control29_1 = ["SP", "FC", "z", "FT"]
features_c29_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control29_1], dtype=torch.float32)
reg_u_c29_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c29_1 = torch.cat([features_c29_1, reg_u_c29_1], dim=1)

# Control 2:
control29_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control29_2 = ["SP", "LC", "z", "LT"]
features_c29_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control29_2], dtype=torch.float32)
reg_u_c29_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c29_2 = torch.cat([features_c29_2, reg_u_c29_2], dim=1)

#------------------------------------------------------------------
# EX30 ---- BYPASS HX ----
# Process:
process30_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 1],  #1 Splitter
[0, 0, 1, 0],  #2 HX
[0, 0, 0, 0],  #3 Splitter
[0, 0, 1, 0],  #4 Valve
], dtype=torch.float32)
u_process30 = ["splitter", "HX", "splitter", "valve"]
features_p30 = torch.tensor([[full_unit_lib[unit]] for unit in u_process30], dtype=torch.float32)

# Control:
control30_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control30_1 = ["SP", "TC", "z", "TT"]
features_c30_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control30_1], dtype=torch.float32)
reg_u_c30_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c30_1 = torch.cat([features_c30_1, reg_u_c30_1], dim=1)

#------------------------------------------------------------------
# EX32 ---- REACTOR + HX ----
# Process:
process32_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 1],  #1 Valve 
[0, 0, 1, 0, 0],  #2 HX
[0, 0, 0, 1, 0],  #3 Splitter
[0, 1, 0, 0, 0],  #4 Reactor
[0, 0, 1, 0, 0],  #5 Valve
], dtype=torch.float32)
u_process32 = ["valve", "HX", "splitter", "reactor", "valve"]
features_p32 = torch.tensor([[full_unit_lib[unit]] for unit in u_process32], dtype=torch.float32)

# Control:
control32_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control32_1 = ["SP", "TC", "z", "TT"]
features_c32_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control32_1], dtype=torch.float32)
reg_u_c32_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c32_1 = torch.cat([features_c32_1, reg_u_c32_1], dim=1)

#------------------------------------------------------------------
# EX34 ---- TANK 6 ----
# Process:
process34_adj = torch.tensor([
#1  2  3  
[0, 0, 0],  #1 Tank
[0, 0, 0],  #2 Pump
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process34 = ["tank", "pump", "valve"]
features_p34 = torch.tensor([[full_unit_lib[unit]] for unit in u_process34], dtype=torch.float32)

# Control:
control34_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control34_1 = ["SP", "LC", "z", "LT"]
features_c34_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control34_1], dtype=torch.float32)
reg_u_c34_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c34_1 = torch.cat([features_c34_1, reg_u_c34_1], dim=1)

#------------------------------------------------------------------
# EX35 ---- CSTR 2 ----
# Process:
process35_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve
[0, 0],  #2 CSTR
], dtype=torch.float32)
u_process35 = ["valve", "CSTR"]
features_p35 = torch.tensor([[full_unit_lib[unit]] for unit in u_process35], dtype=torch.float32)

# Control:
control35_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control35_1 = ["SP", "LC", "z", "LT"]
features_c35_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control35_1], dtype=torch.float32)
reg_u_c35_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c35_1 = torch.cat([features_c35_1, reg_u_c35_1], dim=1)


#------------------------------------------------------------------
# EX36 ---- JACKETED CSTR 5 ----
# Process:
process36_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0],  #2 Splitter
[0, 0, 0, 1, 0, 0],  #3 Pump
[0, 0, 0, 0, 1, 1],  #4 Jacket
[0, 1, 0, 0, 0, 0],  #5 Splitter
[0, 0, 0, 0, 0, 0],  #6 CSTR
], dtype=torch.float32)
u_process36 = ["valve", "splitter", "pump", "jacket", "splitter", "CSTR"]
features_p36 = torch.tensor([[full_unit_lib[unit]] for unit in u_process36], dtype=torch.float32)

# Control:
control36_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0],  #3 TC
[0, 0, 0, 0, 0, 1],  #4 z
[0, 1, 0, 0, 0, 0],  #5 TT
[0, 0, 1, 0, 0, 0],  #6 TT
], dtype=torch.float32)
u_control36_1 = ["SP", "TC", "TC", "z", "TT", "TT"]
features_c36_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control36_1], dtype=torch.float32)
reg_u_c36_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c36_1 = torch.cat([features_c36_1, reg_u_c36_1], dim=1)

#------------------------------------------------------------------
# EX37 ---- STEAM DRUM HX ----
# Process:
process37_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve
[0, 0],  #2 Steam drum HX
], dtype=torch.float32)
u_process37 = ["valve", "steam_drum_HX"]
features_p37 = torch.tensor([[full_unit_lib[unit]] for unit in u_process37], dtype=torch.float32)

# Control:
control37_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0, 0, 0],  #2 LC
[0, 0, 0, 1, 0, 0, 0, 0],  #3 Addition block
[0, 0, 0, 0, 1, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 1, 0, 0, 0, 0, 0, 0],  #6 LT
[0, 0, 1, 0, 0, 0, 0, 0],  #7 FT
[0, 0, 0, 1, 0, 0, 0, 0],  #8 FT
], dtype=torch.float32)
u_control37_1 = ["SP", "LC", "addition_block", "FC", "z", "LT", "FT", "FT"]
features_c37_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control37_1], dtype=torch.float32)
reg_u_c37_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c37_1 = torch.cat([features_c37_1, reg_u_c37_1], dim=1)


#------------------------------------------------------------------
# EX38 ---- TANK 7 ----
# Process:
process38_adj = torch.tensor([
#1  2  3  
[0, 0, 0],  #1 Tank
[0, 0, 0],  #2 Pump
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process38 = ["tank", "pump", "valve"]
features_p38 = torch.tensor([[full_unit_lib[unit]] for unit in u_process38], dtype=torch.float32)

# Control:
control38_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control38_1 = ["SP", "FC", "z", "FT"]
features_c38_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control38_1], dtype=torch.float32)
reg_u_c38_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c38_1 = torch.cat([features_c38_1, reg_u_c38_1], dim=1)


#------------------------------------------------------------------
# EX39 ---- VPC REACTOR ----
# Process:
process39_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0],  #2 HX
[0, 1, 0, 1, 0],  #3 Jacket
[0, 0, 0, 0, 0],  #4 Reactor
[0, 0, 1, 0, 0],  #5 Valve
], dtype=torch.float32)
u_process39 = ["valve", "HX", "jacket", "reactor", "valve"]
features_p39 = torch.tensor([[full_unit_lib[unit]] for unit in u_process39], dtype=torch.float32)

# Control:
control39_1_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 1, 0],  #2 TC
[0, 0, 0, 1, 0, 0, 0],  #3 VPC
[0, 0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 0, 0],  #5 TT
[0, 0, 0, 0, 0, 0, 0],  #6 z
[0, 0, 1, 0, 0, 0, 0],  #7 SP
], dtype=torch.float32)
u_control39_1 = ["SP", "TC", "VPC", "z", "TT", "z", "SP"]
features_c39_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control39_1], dtype=torch.float32)
reg_u_c39_1 = torch.tensor([[0], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c39_1 = torch.cat([features_c39_1, reg_u_c39_1], dim=1)

#------------------------------------------------------------------
# EX40 ---- TOP COLUMN ----
# Process:
process40_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 0],  #1 Column
[0, 0, 1, 0, 0],  #2 HX
[0, 1, 0, 1, 0],  #3 Condenser
[1, 0, 0, 0, 0],  #4 Splitter
[0, 1, 0, 0, 0],  #5 Valve
], dtype=torch.float32)
u_process40 = ["column", "HX", "condenser", "splitter", "valve"]
features_p40 = torch.tensor([[full_unit_lib[unit]] for unit in u_process40], dtype=torch.float32)

# Control:
control40_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control40_1 = ["SP", "PC", "z", "PT"]
features_c40_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control40_1], dtype=torch.float32)
reg_u_c40_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c40_1 = torch.cat([features_c40_1, reg_u_c40_1], dim=1)


#------------------------------------------------------------------
# EX41 ---- TOP COLUMN 2 ----
# Process:
process41_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 Column
[0, 0, 1, 0, 0, 0, 0],  #2 HX
[0, 0, 0, 1, 0, 0, 0],  #3 Condenser
[0, 0, 0, 0, 1, 0, 1],  #4 Splitter
[1, 0, 0, 0, 0, 0, 0],  #5 Valve
[0, 1, 0, 0, 0, 0, 0],  #6 Valve
[0, 0, 0, 0, 0, 0, 0],  #7 Valve
], dtype=torch.float32)
u_process41 = ["column", "HX", "condenser", "splitter", "valve", "valve", "valve"]
features_p41 = torch.tensor([[full_unit_lib[unit]] for unit in u_process41], dtype=torch.float32)

# Control 1:
control41_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)
u_control41_1 = ["SP", "CC", "z", "CT"]
features_c41_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control41_1], dtype=torch.float32)
reg_u_c41_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c41_1 = torch.cat([features_c41_1, reg_u_c41_1], dim=1)

# Control 2:
control41_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control41_2 = ["SP", "PC", "z", "PT"]
features_c41_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control41_2], dtype=torch.float32)
reg_u_c41_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c41_2 = torch.cat([features_c41_2, reg_u_c41_2], dim=1)

# Control 3:
control41_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control41_3 = ["SP", "LC", "z", "LT"]
features_c41_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control41_3], dtype=torch.float32)
reg_u_c41_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c41_3 = torch.cat([features_c41_3, reg_u_c41_3], dim=1)

#------------------------------------------------------------------
# EX42 ---- SETTLING TANK ----
# Process:
process42_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 1, 1, 1, 1, 0],  #1 Settling tank
[0, 0, 1, 0, 0, 0, 0, 1],  #2 Splitter
[0, 0, 0, 0, 0, 0, 0, 0],  #3 Valve
[0, 0, 0, 0, 0, 0, 0, 0],  #4 Valve
[0, 0, 0, 0, 0, 0, 0, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 0, 0],  #6 Valve
[0, 0, 0, 0, 0, 0, 0, 0],  #7 Valve
[0, 0, 0, 0, 0, 0, 0, 0],  #8 PSV
], dtype=torch.float32)
u_process42 = ["settling_tank", "splitter", "valve", "valve", "valve", "valve", "valve", "PSV"]
features_p42 = torch.tensor([[full_unit_lib[unit]] for unit in u_process42], dtype=torch.float32)

# Control 1:
control42_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control42_1 = ["SP", "PC", "z", "PT"]
features_c42_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control42_1], dtype=torch.float32)
reg_u_c42_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c42_1 = torch.cat([features_c42_1, reg_u_c42_1], dim=1)

# Control 2:
control42_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control42_2 = ["SP", "LC", "z", "LT"]
features_c42_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control42_2], dtype=torch.float32)
reg_u_c42_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c42_2 = torch.cat([features_c42_2, reg_u_c42_2], dim=1)

# Control 3:
control42_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control42_3 = ["SP", "LC", "z", "LT"]
features_c42_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control42_3], dtype=torch.float32)
reg_u_c42_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c42_3 = torch.cat([features_c42_3, reg_u_c42_3], dim=1)

#------------------------------------------------------------------
# EX43 ---- TANK 8 ----
# Process:
process43_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 Tank
[0, 0, 1, 0, 0, 0],  #2 Pump
[1, 0, 0, 1, 0, 0],  #3 Splitter
[0, 0, 0, 0, 1, 1],  #4 Splitter
[0, 0, 0, 0, 0, 0],  #5 Valve
[1, 0, 0, 0, 0, 0],  #6 Valve
], dtype=torch.float32)
u_process43 = ["tank", "pump", "splitter", "splitter", "valve", "valve"]
features_p43 = torch.tensor([[full_unit_lib[unit]] for unit in u_process43], dtype=torch.float32)

# Control 1:
control43_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0],  #2 FC
[0, 0, 0, 1, 0, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 0, 0, 0, 0],  #4 w
[0, 0, 1, 0, 0, 0, 0, 0],  #5 LC
[0, 0, 0, 0, 1, 0, 0, 0],  #6 SP
[0, 1, 0, 0, 0, 0, 0, 0],  #7 FT
[0, 0, 0, 0, 1, 0, 0, 0],  #8 LT
], dtype=torch.float32)
u_control43_1 = ["SP", "FC", "MIN", "w", "LC", "SP", "FT", "LT"]
features_c43_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control43_1], dtype=torch.float32)
reg_u_c43_1 = torch.tensor([[0], [0], [6], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c43_1 = torch.cat([features_c43_1, reg_u_c43_1], dim=1)

# Control 2:
control43_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control43_2 = ["SP", "FC", "z", "FT"]
features_c43_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control43_2], dtype=torch.float32)
reg_u_c43_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c43_2 = torch.cat([features_c43_2, reg_u_c43_2], dim=1)

# Control 3:
control43_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control43_3 = ["SP", "FC", "z", "FT"]
features_c43_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control43_3], dtype=torch.float32)
reg_u_c43_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c43_3 = torch.cat([features_c43_3, reg_u_c43_3], dim=1)

#------------------------------------------------------------------
# EX44 ---- COMPRESSOR ----
# Process:
process44_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 0],  #1 Splitter 
[0, 0, 1, 0, 0],  #2 Compressor
[0, 0, 0, 1, 0],  #3 HX
[0, 0, 0, 0, 1],  #4 Splitter
[1, 0, 0, 0, 0],  #5 Valve
], dtype=torch.float32)
u_process44 = ["splitter", "compressor", "HX", "splitter", "valve"]
features_p44 = torch.tensor([[full_unit_lib[unit]] for unit in u_process44], dtype=torch.float32)

# Control:
control44_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  
[0, 1, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0],  #2 FC
[0, 0, 0, 1, 0, 0, 0, 0, 0],  #3 MAX
[0, 0, 0, 0, 0, 0, 0, 0, 0],  #4 z
[0, 0, 1, 0, 0, 0, 0, 0, 0],  #5 FC
[0, 0, 0, 0, 1, 0, 0, 0, 0],  #6 SP
[0, 1, 0, 0, 0, 0, 0, 0, 0],  #7 FT
[0, 0, 1, 0, 0, 0, 0, 0, 0],  #8 SP
[0, 0, 0, 0, 1, 0, 0, 0, 0],  #9 FT
], dtype=torch.float32)
u_control44_1 = ["SP", "FC", "MAX", "z", "FC", "SP", "FT", "SP", "FT"]
features_c44_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control44_1], dtype=torch.float32)
reg_u_c44_1 = torch.tensor([[0], [0], [2], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c44_1 = torch.cat([features_c44_1, reg_u_c44_1], dim=1)

#------------------------------------------------------------------
# EX45 ---- COMPRESSOR 2 ----
# Process:
process45_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 0],  #1 Splitter 
[0, 0, 1, 0, 0],  #2 Compressor 
[0, 0, 0, 1, 0],  #3 HX
[0, 0, 0, 0, 1],  #4 Splitter
[1, 0, 0, 0, 0],  #5 Valve
], dtype=torch.float32)
u_process45 = ["splitter", "compressor", "HX", "splitter", "valve"]
features_p45 = torch.tensor([[full_unit_lib[unit]] for unit in u_process45], dtype=torch.float32)

# Control:
control45_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control45_1 = ["SP", "FC", "z", "FT"]
features_c45_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control45_1], dtype=torch.float32)
reg_u_c45_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c45_1 = torch.cat([features_c45_1, reg_u_c45_1], dim=1)

#------------------------------------------------------------------
# EX46 ---- INVENTORY CONTROL ----
# Process:
process46_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Tank
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process46 = ["valve", "tank", "valve"]
features_p46 = torch.tensor([[full_unit_lib[unit]] for unit in u_process46], dtype=torch.float32)

# Control 1:
control46_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control46_1 = ["SP", "FC", "z", "FT"]
features_c46_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control46_1], dtype=torch.float32)
reg_u_c46_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c46_1 = torch.cat([features_c46_1, reg_u_c46_1], dim=1)

# Control 2:
control46_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control46_2 = ["SP", "LC", "z", "LT"]
features_c46_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control46_2], dtype=torch.float32)
reg_u_c46_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c46_2 = torch.cat([features_c46_2, reg_u_c46_2], dim=1)

#------------------------------------------------------------------
# EX47 ---- INVENTORY CONTROL 2 ----
# Process:
process47_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0],  #2 Tank
[0, 0, 0, 1, 0, 0, 0],  #3 Valve
[0, 0, 0, 0, 1, 0, 0],  #4 Tank
[0, 0, 0, 0, 0, 1, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 1],  #6 Tank
[0, 0, 0, 0, 0, 0, 0],  #7 Valve
], dtype=torch.float32)
u_process47 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p47 = torch.tensor([[full_unit_lib[unit]] for unit in u_process47], dtype=torch.float32)

# Control 1:
control47_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control47_1 = ["SP", "FC", "z", "FT"]
features_c47_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_1], dtype=torch.float32)
reg_u_c47_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c47_1 = torch.cat([features_c47_1, reg_u_c47_1], dim=1)

# Control 2, 3, 4 are identical in structure, just using IC/IT
control47_2_adj = control47_3_adj = control47_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 0],
[0, 1, 0, 0],
], dtype=torch.float32)

u_control47_X = ["SP", "IC", "z", "IT"]
features_c47_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_X], dtype=torch.float32)
features_c47_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_X], dtype=torch.float32)
features_c47_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_X], dtype=torch.float32)
reg_u_c47_X = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

feature_tensor_c47_2 = torch.cat([features_c47_2, reg_u_c47_X], dim=1)
feature_tensor_c47_3 = torch.cat([features_c47_3, reg_u_c47_X], dim=1)
feature_tensor_c47_4 = torch.cat([features_c47_4, reg_u_c47_X], dim=1)


#------------------------------------------------------------------
# EX48 ---- INVENTORY CONTROL 3 ----
# Process:
process48_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0],  #2 Tank
[0, 0, 0, 1, 0, 0, 0],  #3 Valve
[0, 0, 0, 0, 1, 0, 0],  #4 Tank
[0, 0, 0, 0, 0, 1, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 1],  #6 Tank
[0, 0, 0, 0, 0, 0, 0],  #7 Valve
], dtype=torch.float32)
u_process48 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p48 = torch.tensor([[full_unit_lib[unit]] for unit in u_process48], dtype=torch.float32)

# Control 1:
control48_1_adj = torch.tensor([
#1  2  3  4
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)
u_control48_1 = ["SP", "IC", "z", "IT"]
features_c48_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control48_1], dtype=torch.float32)
reg_u_c48_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c48_1 = torch.cat([features_c48_1, reg_u_c48_1], dim=1)

# Control 2:
control48_2_adj = control48_1_adj.clone()
u_control48_2 = u_control48_1
features_c48_2 = features_c48_1.clone()
reg_u_c48_2 = reg_u_c48_1.clone()
feature_tensor_c48_2 = torch.cat([features_c48_2, reg_u_c48_2], dim=1)

# Control 3:
control48_3_adj = control48_1_adj.clone()
u_control48_3 = u_control48_1
features_c48_3 = features_c48_1.clone()
reg_u_c48_3 = reg_u_c48_1.clone()
feature_tensor_c48_3 = torch.cat([features_c48_3, reg_u_c48_3], dim=1)

#------------------------------------------------------------------
# EX49 ---- INVENTORY CONTROL 4 ----
# Process:
process49_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0],  #2 Tank
[0, 0, 0, 1, 0, 0, 0],  #3 Valve
[0, 0, 0, 0, 1, 0, 0],  #4 Tank
[0, 0, 0, 0, 0, 1, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 1],  #6 Tank
[0, 0, 0, 0, 0, 0, 0],  #7 Valve
], dtype=torch.float32)
u_process49 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p49 = torch.tensor([[full_unit_lib[unit]] for unit in u_process49], dtype=torch.float32)

# Controls 1, 2, 4 — identical IC/IT pairs
control49_1_adj = control49_2_adj = control49_4_adj = torch.tensor([
#1  2  3  4
[0, 1, 0, 0],  # SP
[0, 0, 1, 0],  # IC
[0, 0, 0, 1],  # z
[0, 1, 0, 0],  # IT
], dtype=torch.float32)
u_control49_IC = ["SP", "IC", "z", "IT"]
features_c49_IC = torch.tensor([[full_unit_lib[unit]] for unit in u_control49_IC], dtype=torch.float32)
reg_u_c49_IC = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c49_1 = torch.cat([features_c49_IC, reg_u_c49_IC], dim=1)
feature_tensor_c49_2 = torch.cat([features_c49_IC, reg_u_c49_IC], dim=1)
feature_tensor_c49_4 = torch.cat([features_c49_IC, reg_u_c49_IC], dim=1)

# Control 3 — FC/FT
control49_3_adj = torch.tensor([
#1  2  3  4
[0, 1, 0, 0],  # SP
[0, 0, 1, 0],  # FC
[0, 0, 0, 0],  # z
[0, 1, 0, 0],  # FT
], dtype=torch.float32)
u_control49_3 = ["SP", "FC", "z", "FT"]
features_c49_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control49_3], dtype=torch.float32)
reg_u_c49_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c49_3 = torch.cat([features_c49_3, reg_u_c49_3], dim=1)

#------------------------------------------------------------------
# EX50 ---- INVENTORY CONTROL 5 ----
# Process:
process50_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  # Valve
[0, 0, 1, 0, 0, 0, 0],  # Tank
[0, 0, 0, 1, 0, 0, 0],  # Valve
[0, 0, 0, 0, 1, 0, 0],  # Tank
[0, 0, 0, 0, 0, 1, 0],  # Valve
[0, 0, 0, 0, 0, 0, 1],  # Tank
[0, 0, 0, 0, 0, 0, 0],  # Valve
], dtype=torch.float32)
u_process50 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p50 = torch.tensor([[full_unit_lib[unit]] for unit in u_process50], dtype=torch.float32)

control50_1_adj = torch.tensor([
#1  2  3  4
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, 1],
[0, 1, 0, 0],
], dtype=torch.float32)
u_control50_1 = ["SP", "IC", "z", "IT"]
features_c50_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control50_1], dtype=torch.float32)
reg_u_c50_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c50_1 = torch.cat([features_c50_1, reg_u_c50_1], dim=1)

control50_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)
u_control50_2 = ["SP", "IC", "z", "IT"]
features_c50_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control50_2], dtype=torch.float32)
reg_u_c50_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c50_2 = torch.cat([features_c50_2, reg_u_c50_2], dim=1)

control50_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)
u_control50_3 = ["SP", "IC", "z", "IT"]
features_c50_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control50_3], dtype=torch.float32)
reg_u_c50_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c50_3 = torch.cat([features_c50_3, reg_u_c50_3], dim=1)





#------------------------------------------------------------------
# EX51 ---- AIR + FUEL MIX ----
# Process:
process51_adj = torch.tensor([
#1  2  3  
[0, 0, 0],  #1 Valve
[0, 0, 0],  #2 Splitter
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process51 = ["valve", "splitter", "valve"]
features_p51 = torch.tensor([[full_unit_lib[unit]] for unit in u_process51], dtype=torch.float32)

# Control:
control51_1_adj = torch.tensor([
#1   2  3  4  5  6  7  8  9  10 11 12 13  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 Multiplication block
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #6 FT
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #7 MAX
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #8 FC
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 z
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #10 PC 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #11 SP
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #12 FT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #13 SP
], dtype=torch.float32)
u_control51_1 = ["SP", "X_multiplication_block", "MIN", "FC", "z", "FT", "MAX", "FC", "z", "PC", "SP", "FT", "SP"]
features_c51_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control51_1], dtype=torch.float32)
reg_u_c51_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c51_1 = torch.cat([features_c51_1, reg_u_c51_1], dim=1)

#------------------------------------------------------------------
# EX52 ---- JACKETED CSTR 6 ----
# Process:
process52_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 Jacket
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)
u_process52 = ["jacket", "valve", "CSTR"]
features_p52 = torch.tensor([[full_unit_lib[unit]] for unit in u_process52], dtype=torch.float32)

# Control:
control52_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #6 CC
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #7 SP
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #8 TT
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #9 FT
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #10 CT
], dtype=torch.float32)
u_control52_1 = ["SP", "TC", "MIN", "FC", "z", "CC", "SP", "TT", "FT", "CT"]
features_c52_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control52_1], dtype=torch.float32)
reg_u_c52_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c52_1 = torch.cat([features_c52_1, reg_u_c52_1], dim=1)


#------------------------------------------------------------------
# EX53 ---- MIXER ----
# Process:
process53_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 Mixer
[0, 1, 0],  #3 Compressor
], dtype=torch.float32)
u_process53 = ["valve", "mixer", "compressor"]
features_p53 = torch.tensor([[full_unit_lib[unit]] for unit in u_process53], dtype=torch.float32)

# Control 1:
control53_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 CC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 0],  #5 CT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control53_1 = ["SP", "CC", "FC", "z", "CT", "FT"]
features_c53_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control53_1], dtype=torch.float32)
reg_u_c53_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c53_1 = torch.cat([features_c53_1, reg_u_c53_1], dim=1)

# Control 2:
control53_2_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 FC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 w
[0, 1, 0, 0, 0, 0],  #5 FT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)
u_control53_2 = ["SP", "FC", "FC", "w", "FT", "FT"]
features_c53_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control53_2], dtype=torch.float32)
reg_u_c53_2 = torch.tensor([[0], [0], [6], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c53_2 = torch.cat([features_c53_2, reg_u_c53_2], dim=1)



#------------------------------------------------------------------
# EX54 ---- MIXER 2 ----
# Process:
process54_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 Mixer
[0, 1, 0],  #3 Compressor
], dtype=torch.float32)
u_process54 = ["valve", "mixer", "compressor"]
features_p54 = torch.tensor([[full_unit_lib[unit]] for unit in u_process54], dtype=torch.float32)

# Control: 
control54_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12 13  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 FC
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #6 SRC
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #7 SP
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #8 FC
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 w
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 FT 
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #11 FT
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #12 CT
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #13 FT
], dtype=torch.float32)
u_control54_1 = ["SP", "FC", "MIN", "FC", "z", "SRC", "SP", "FC", "w", "FT", "FT", "CT", "FT"]
features_c54_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control54_1], dtype=torch.float32)
reg_u_c54_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [6], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c54_1 = torch.cat([features_c54_1, reg_u_c54_1], dim=1)



#------------------------------------------------------------------
# EX55 ---- MIXER 3 ----
# Process:
process55_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 Mixer
[0, 1, 0],  #3 Compressor
], dtype=torch.float32)
u_process55 = ["valve", "mixer", "compressor"]
features_p55 = torch.tensor([[full_unit_lib[unit]] for unit in u_process55], dtype=torch.float32)

# Control: 
control55_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12 13 14 15  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 FC
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #6 VPC
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],  #7 FC
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #8 CC
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #9 SP
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 FT
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #11 SP
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #12 FT
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #13 FT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #14 w
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #15 CT
], dtype=torch.float32)
u_control55_1 = ["SP", "FC", "MIN", "FC", "z", "VPC", "FC", "CC", "SP", "FT", "SP", "FT", "FT", "w", "CT"]
features_c55_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control55_1], dtype=torch.float32)
reg_u_c55_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [6], [6], [0], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c55_1 = torch.cat([features_c55_1, reg_u_c55_1], dim=1)


#------------------------------------------------------------------
# EX56 ---- DISTILLATION COLUMN 3 ----
# Process:
process56_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #2 Column
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #3 HX
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  #4 Splitter
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  #6 Splitter
[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #7 HX
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #8 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #11 Valve
], dtype=torch.float32)
u_process56 = ["valve", "column", "HX", "splitter", "valve", "splitter", "HX", "valve", "valve", "valve", "valve"]
features_p56 = torch.tensor([[full_unit_lib[unit]] for unit in u_process56], dtype=torch.float32)

# Control 1:
control56_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control56_1 = ["SP", "FC", "z", "FT"]
features_c56_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control56_1], dtype=torch.float32)
reg_u_c56_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c56_1 = torch.cat([features_c56_1, reg_u_c56_1], dim=1)

# Control 2:
control56_2_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 CC
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #3 SRC
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #4 MIN
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #6 CC
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #7 MAX
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #8 SP
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 CT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 z
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #11 CT
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #12 SP
], dtype=torch.float32)
u_control56_2 = ["SP", "CC", "SRC", "MIN", "z", "CC", "MAX", "SP", "CT", "z", "CT", "SP"]
features_c56_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control56_2], dtype=torch.float32)
reg_u_c56_2 = torch.tensor([[0], [0], [2], [2], [0], [0], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c56_2 = torch.cat([features_c56_2, reg_u_c56_2], dim=1)

# Control 3:
control56_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control56_3 = ["SP", "PC", "z", "PT"]
features_c56_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control56_3], dtype=torch.float32)
reg_u_c56_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c56_3 = torch.cat([features_c56_3, reg_u_c56_3], dim=1)

# Control 4:
control56_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control56_4 = ["SP", "LC", "z", "LT"]
features_c56_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control56_4], dtype=torch.float32)
reg_u_c56_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c56_4 = torch.cat([features_c56_4, reg_u_c56_4], dim=1)

# Control 5:
control56_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control56_5 = ["SP", "LC", "z", "LT"]
features_c56_5 = torch.tensor([[full_unit_lib[unit]] for unit in u_control56_5], dtype=torch.float32)
reg_u_c56_5 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c56_5 = torch.cat([features_c56_5, reg_u_c56_5], dim=1)


#------------------------------------------------------------------
# EX57 ---- DISTILLATION COLUMN 4 ----
# Process:
process57_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #2 Column
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #3 HX
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  #4 Splitter
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  #6 Splitter
[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],  #7 HX
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #8 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #11 Valve
], dtype=torch.float32)
u_process57 = ["valve", "column", "HX", "splitter", "valve", "splitter", "HX", "valve", "valve", "valve", "valve"]
features_p57 = torch.tensor([[full_unit_lib[unit]] for unit in u_process57], dtype=torch.float32)

# Control 1:
control57_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP  
[0, 0, 1, 0, 0, 0, 0, 0],  #2 CC
[0, 0, 0, 1, 0, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 1, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 1, 0, 0, 0, 0, 0, 0],  #6 CT
[0, 0, 1, 0, 0, 0, 0, 0],  #7 SP
[0, 0, 0, 1, 0, 0, 0, 0],  #8 FT
], dtype=torch.float32)
u_control57_1 = ["SP", "CC", "MIN", "FC", "z", "CT", "SP", "FT"]
features_c57_1 = torch.tensor([[full_unit_lib[u]] for u in u_control57_1], dtype=torch.float32)
reg_u_c57_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c57_1 = torch.cat([features_c57_1, reg_u_c57_1], dim=1)

# Control 2:
control57_2_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 CC
[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],  #3 SRC
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #4 MIN
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #6 CC
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #7 MAX
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #8 SP
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 CT
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 z
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #11 CT
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #12 SP
], dtype=torch.float32)
u_control57_2 = ["SP", "CC", "SRC", "MIN", "z", "CC", "MAX", "SP", "CT", "z", "CT", "SP"]
features_c57_2 = torch.tensor([[full_unit_lib[u]] for u in u_control57_2], dtype=torch.float32)
reg_u_c57_2 = torch.tensor([[0], [0], [2], [2], [0], [0], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c57_2 = torch.cat([features_c57_2, reg_u_c57_2], dim=1)

# Control 3:
control57_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control57_3 = ["SP", "PC", "z", "PT"]
features_c57_3 = torch.tensor([[full_unit_lib[u]] for u in u_control57_3], dtype=torch.float32)
reg_u_c57_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c57_3 = torch.cat([features_c57_3, reg_u_c57_3], dim=1)

# Control 4:
control57_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control57_4 = ["SP", "LC", "z", "LT"]
features_c57_4 = torch.tensor([[full_unit_lib[u]] for u in u_control57_4], dtype=torch.float32)
reg_u_c57_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c57_4 = torch.cat([features_c57_4, reg_u_c57_4], dim=1)

# Control 5:
control57_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)
u_control57_5 = ["SP", "LC", "z", "LT"]
features_c57_5 = torch.tensor([[full_unit_lib[u]] for u in u_control57_5], dtype=torch.float32)
reg_u_c57_5 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c57_5 = torch.cat([features_c57_5, reg_u_c57_5], dim=1)


#------------------------------------------------------------------
# EX58 ---- CO2 REFRIGERATION ----
# Process:
process58_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12 13 14
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 Evaporator 
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 Splitter
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3 Compressor 
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #4 Splitter
[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  #5 Splitter
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  #6 HX
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #7 3-way valve
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #8 Gas cooler
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #9 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  #10 Liquid receiver
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  #11 Splitter
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #12 Valve
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #13 Valve
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #14 Compressor
], dtype=torch.float32)
u_process58 = [
    "evaporator", "splitter", "compressor", "splitter", "splitter", "HX",
    "valve", "cooler", "valve", "liq_reciever", "splitter", "valve", "valve", "compressor"
]
features_p58 = torch.tensor([[full_unit_lib[unit]] for unit in u_process58], dtype=torch.float32)

# Control 1:
control58_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z 
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control58_1 = ["SP", "TC", "z", "TT"]
features_c58_1 = torch.tensor([[full_unit_lib[u]] for u in u_control58_1], dtype=torch.float32)
reg_u_c58_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c58_1 = torch.cat([features_c58_1, reg_u_c58_1], dim=1)

# Control 2:
control58_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 w 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control58_2 = ["SP", "PC", "w", "PT"]
features_c58_2 = torch.tensor([[full_unit_lib[u]] for u in u_control58_2], dtype=torch.float32)
reg_u_c58_2 = torch.tensor([[0], [6], [0], [0]], dtype=torch.float32)
feature_tensor_c58_2 = torch.cat([features_c58_2, reg_u_c58_2], dim=1)

# Control 3:
control58_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z 
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control58_3 = ["SP", "TC", "z", "TT"]
features_c58_3 = torch.tensor([[full_unit_lib[u]] for u in u_control58_3], dtype=torch.float32)
reg_u_c58_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c58_3 = torch.cat([features_c58_3, reg_u_c58_3], dim=1)

# Control 4:
control58_4_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0],  #2 TC
[0, 0, 0, 0, 0, 0, 0, 0],  #3 w
[0, 1, 0, 0, 1, 0, 0, 0],  #4 TT
[0, 0, 0, 0, 0, 1, 0, 0],  #5 TC
[0, 0, 0, 0, 0, 0, 1, 0],  #6 PC
[0, 0, 0, 0, 0, 0, 0, 0],  #7 z
[0, 0, 0, 0, 0, 1, 0, 0],  #8 PT
], dtype=torch.float32)
u_control58_4 = ["SP", "TC", "w", "TT", "TC", "PC", "z", "PT"]
features_c58_4 = torch.tensor([[full_unit_lib[u]] for u in u_control58_4], dtype=torch.float32)
reg_u_c58_4 = torch.tensor([[0], [6], [0], [0], [0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c58_4 = torch.cat([features_c58_4, reg_u_c58_4], dim=1)

# Control 5:
control58_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 w 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)
u_control58_5 = ["SP", "PC", "w", "PT"]
features_c58_5 = torch.tensor([[full_unit_lib[u]] for u in u_control58_5], dtype=torch.float32)
reg_u_c58_5 = torch.tensor([[0], [6], [0], [0]], dtype=torch.float32)
feature_tensor_c58_5 = torch.cat([features_c58_5, reg_u_c58_5], dim=1)


#------------------------------------------------------------------
# EX59 ---- HEAT EXCHANGER 2 ----
# Process:
process59_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process59 = ["HX", "valve", "valve"]
features_p59 = torch.tensor([[full_unit_lib[unit]] for unit in u_process59], dtype=torch.float32)

# Control 1:
control59_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)
u_control59_1 = ["SP", "FC", "z", "FT"]
features_c59_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control59_1], dtype=torch.float32)
reg_u_c59_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c59_1 = torch.cat([features_c59_1, reg_u_c59_1], dim=1)

# Control 2:
control59_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)
u_control59_2 = ["SP", "TC", "z", "TT"]
features_c59_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control59_2], dtype=torch.float32)
reg_u_c59_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c59_2 = torch.cat([features_c59_2, reg_u_c59_2], dim=1)


#------------------------------------------------------------------
# EX60 ---- HEAT EXCHANGER 3 ----
# Process:
process60_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process60 = ["HX", "valve", "valve"]
features_p60 = torch.tensor([[full_unit_lib[unit]] for unit in u_process60], dtype=torch.float32)

# Control:
control60_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0, 1, 0],  #3 SRC
[0, 0, 0, 0, 1, 0, 0, 0],  #4 MIN
[0, 0, 0, 0, 0, 0, 0, 0],  #5 z
[0, 1, 0, 0, 0, 0, 0, 0],  #6 TT
[0, 0, 0, 0, 0, 1, 0, 0],  #7 z
[0, 0, 0, 1, 0, 0, 0, 0],  #8 SP
], dtype=torch.float32)
u_control60_1 = ["SP", "TC", "SRC", "MIN", "z", "TT", "z", "SP"]
features_c60_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control60_1], dtype=torch.float32)
reg_u_c60_1 = torch.tensor([[0], [0], [2], [2], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c60_1 = torch.cat([features_c60_1, reg_u_c60_1], dim=1)

#------------------------------------------------------------------
# EX61 ---- HEAT EXCHANGER 4 ----
# Process:
process61_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process61 = ["HX", "valve", "valve"]
features_p61 = torch.tensor([[full_unit_lib[unit]] for unit in u_process61], dtype=torch.float32)

# Control:
control61_1_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 1, 0],  #2 TC
[0, 0, 0, 1, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 0, 0],  #5 TT
[0, 0, 0, 0, 1, 0, 0],  #6 z
[0, 0, 1, 0, 0, 0, 0],  #7 SP
], dtype=torch.float32)
u_control61_1 = ["SP", "TC", "MIN", "z", "TT", "z", "SP"]
features_c61_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control61_1], dtype=torch.float32)
reg_u_c61_1 = torch.tensor([[0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c61_1 = torch.cat([features_c61_1, reg_u_c61_1], dim=1)

#------------------------------------------------------------------
# EX62 ---- HEAT EXCHANGER 5 ----
# Process:
process62_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)
u_process62 = ["HX", "valve", "valve"]
features_p62 = torch.tensor([[full_unit_lib[unit]] for unit in u_process62], dtype=torch.float32)

# Control:
control62_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  
[0, 1, 0, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0, 0, 0, 0],  #3 MIN
[0, 0, 0, 0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 1, 0, 0, 0],  #5 TT
[0, 0, 0, 0, 0, 0, 1, 0, 0],  #6 TC
[0, 0, 0, 0, 1, 0, 0, 0, 0],  #7 z
[0, 0, 1, 0, 0, 0, 0, 0, 0],  #8 SP
[0, 0, 0, 0, 0, 1, 0, 0, 0],  #9 SP
], dtype=torch.float32)
u_control62_1 = ["SP", "TC", "MIN", "z", "TT", "TC", "z", "SP", "SP"]
features_c62_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control62_1], dtype=torch.float32)
reg_u_c62_1 = torch.tensor([[0], [0], [2], [0], [0], [2], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c62_1 = torch.cat([features_c62_1, reg_u_c62_1], dim=1)

#------------------------------------------------------------------
# EX63 ---- TANK 9 ----
# Process:
process63_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve
[0, 0],  #2 Tank
], dtype=torch.float32)
u_process63 = ["valve", "tank"]
features_p63 = torch.tensor([[full_unit_lib[unit]] for unit in u_process63], dtype=torch.float32)

# Control:
control63_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  
[0, 1, 0, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0, 0],  #2 LC
[0, 0, 0, 1, 0, 0, 0, 0],  #3 Subtraction block
[0, 0, 0, 0, 1, 0, 0, 0],  #4 FC
[0, 0, 0, 0, 0, 1, 0, 0],  #5 z
[0, 1, 0, 0, 0, 0, 0, 0],  #6 LT
[0, 0, 1, 0, 0, 0, 0, 0],  #7 FT
[0, 0, 0, 1, 0, 0, 0, 0],  #8 FT
], dtype=torch.float32)
u_control63_1 = ["SP", "LC", "subtraction_block", "FC", "z", "LT", "FT", "FT"]
features_c63_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control63_1], dtype=torch.float32)
reg_u_c63_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)
feature_tensor_c63_1 = torch.cat([features_c63_1, reg_u_c63_1], dim=1)

#------------------------------------------------------------------
# EX64 ---- TURBINE ----
# Process:
process64_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 1, 0],  #1 Splitter 
[0, 0, 1, 0, 0, 0],  #2 Turbine
[0, 0, 0, 1, 0, 0],  #3 Valve
[0, 0, 0, 0, 0, 0],  #4 Splitter
[0, 0, 0, 0, 0, 1],  #5 Splitter
[0, 0, 0, 0, 0, 0],  #6 Valve
], dtype=torch.float32)
u_process64 = ["splitter", "turbine", "valve", "splitter", "splitter", "valve"]
features_p64 = torch.tensor([[full_unit_lib[unit]] for unit in u_process64], dtype=torch.float32)

# Control:
control64_1_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0, 0],  #2 PC
[0, 0, 0, 1, 0, 0, 0],  #3 z
[0, 1, 0, 0, 1, 0, 0],  #4 PT
[0, 0, 0, 0, 0, 1, 0],  #5 PC
[0, 0, 0, 1, 0, 0, 0],  #6 z
[0, 0, 0, 0, 1, 0, 0],  #7 SP
], dtype=torch.float32)
u_control64_1 = ["SP", "PC", "z", "PT", "PC", "z", "SP"]
features_c64_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control64_1], dtype=torch.float32)
reg_u_c64_1 = torch.tensor([[0], [2], [0], [0], [2], [0], [0]], dtype=torch.float32)
feature_tensor_c64_1 = torch.cat([features_c64_1, reg_u_c64_1], dim=1)






# Auto-expose all tensors named feature_tensor_*
import inspect
for name, val in list(inspect.currentframe().f_locals.items()):
    if name.startswith("feature_tensor_"):
        globals()[name] = val
