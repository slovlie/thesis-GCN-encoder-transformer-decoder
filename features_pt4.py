import torch
import Dataset_general
# define dataset-specific parameters 
# Define all possible process units  
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


def pad_to_size(tensor, target_shape):
    padded = torch.zeros(target_shape, dtype=tensor.dtype)
    rows, cols = tensor.shape
    padded[:rows, :cols] = tensor
    return padded

# EX1 ---- DISTILLATION COLUMN ----
u_process1 = ["valve", "column", "HX", "condenser", "pump", "valve", "HX", "reboiler", 
            "pump", "valve", "valve", "valve", "valve"]

features_p1 = torch.tensor([[full_unit_lib[unit]] for unit in u_process1], dtype=torch.float32)
process1_adj =Dataset_general.process1_adj
# max size for control is 6 
# expand to 6 

# Stack them
control_adj_1_stack = torch.vstack([
    pad_to_size(Dataset_general.control1_1_adj, (6, 6)),
    pad_to_size(Dataset_general.control1_2_adj, (6, 6)),
    pad_to_size(Dataset_general.control1_3_adj, (6, 6)),
    pad_to_size(Dataset_general.control1_4_adj, (6, 6)),
    pad_to_size(Dataset_general.control1_5_adj, (6, 6)),
    pad_to_size(Dataset_general.control1_6_adj, (6, 6))
 
])


u_control1_1 =  ["SP", "FC", "z", "FT"]
u_control1_2 = ["SP", "X_multiplication_block", "FC", "z", "FT", "FT"]
u_control1_3 =  ["SP", "LC", "z", "LT"]
u_control1_4 = ["SP", "PC", "z", "PT"]
u_control1_5 = ["SP", "X_multiplication_block", "FC", "z", "FT", "FT"]
u_control1_6 = ["SP", "LC", "z", "LT"]
features_c1_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_1], dtype=torch.float32)
features_c1_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_2], dtype=torch.float32)
features_c1_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_3], dtype=torch.float32)
features_c1_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_4], dtype=torch.float32)
features_c1_5 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_5], dtype=torch.float32)
features_c1_6 = torch.tensor([[full_unit_lib[unit]] for unit in u_control1_6], dtype=torch.float32)
reg_u_c1_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c1_2 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
reg_u_c1_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c1_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c1_5 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
reg_u_c1_6 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)


control1_ufeat_stack = torch.vstack([
    pad_to_size(features_c1_1, (6, 1)), 
    pad_to_size(features_c1_2, (6, 1)), 
    pad_to_size(features_c1_3, (6, 1)), 
    pad_to_size(features_c1_4, (6, 1)), 
    pad_to_size(features_c1_5, (6, 1)), 
    pad_to_size(features_c1_6, (6, 1)), 
])


control1_reg_stack = torch.vstack([
    pad_to_size(reg_u_c1_1, (6, 1)),
    pad_to_size(reg_u_c1_2, (6, 1)),
    pad_to_size(reg_u_c1_3, (6, 1)),
    pad_to_size(reg_u_c1_4, (6, 1)),
    pad_to_size(reg_u_c1_5, (6, 1)),
    pad_to_size(reg_u_c1_6, (6, 1)),
])

control1_fullfeat_cat = torch.cat((control1_ufeat_stack, control1_reg_stack), dim=1)
print("Unit features: ", control1_ufeat_stack.shape)
print("regulation: ", control1_reg_stack.shape)
# print(control_adj_1_stack)
print("full stack: ", control1_fullfeat_cat.shape)



# EX2 ---- TWO HEATED TANKS ----
u_process2 = ["valve", "tank", "HX", "valve"]
features_p2 = torch.tensor([[full_unit_lib[unit]] for unit in u_process2], dtype=torch.float32)

control2_1_adj = Dataset_general.control2_1_adj
control2_2_adj = Dataset_general.control2_2_adj

control_adj_2_stack = torch.vstack([
    pad_to_size(control2_1_adj, (6, 6)),
    pad_to_size(control2_2_adj, (6, 6))
])

u_control2_1 = ["SP", "LC", "FC", "z", "LT", "FT"]
u_control2_2 = ["SP", "TC", "FC", "z", "TT", "FT"]

features_c2_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control2_1], dtype=torch.float32)
features_c2_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control2_2], dtype=torch.float32)

reg_u_c2_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)
reg_u_c2_2 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control2_ufeat_stack = torch.vstack([
    pad_to_size(features_c2_1, (6, 1)),
    pad_to_size(features_c2_2, (6, 1)),
])

control2_reg_stack = torch.vstack([
    pad_to_size(reg_u_c2_1, (6, 1)),
    pad_to_size(reg_u_c2_2, (6, 1)),
])

control2_fullfeat_cat = torch.cat((control2_ufeat_stack, control2_reg_stack), dim=1)


# EX3 ---- COMPRESSOR ----
u_process3 = ["valve", "compressor", "tank", "valve"]
features_p3 = torch.tensor([[full_unit_lib[unit]] for unit in u_process3], dtype=torch.float32)

control3_1_adj = Dataset_general.control3_1_adj
control3_2_adj = Dataset_general.control3_2_adj

control_adj_3_stack = torch.vstack([
    pad_to_size(control3_1_adj, (4, 4)),
    pad_to_size(control3_2_adj, (4, 4))
])

# Control unit types and regulation modes
u_control3_1 = ["SP", "PC", "z", "PT"]
u_control3_2 = ["SP", "PC", "z", "PT"]

# Control unit features
features_c3_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control3_1], dtype=torch.float32)
features_c3_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control3_2], dtype=torch.float32)

# Regulation types
reg_u_c3_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c3_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

# Stack and pad control unit features and regulation
control3_ufeat_stack = torch.vstack([
    pad_to_size(features_c3_1, (4, 1)),
    pad_to_size(features_c3_2, (4, 1))
])

control3_reg_stack = torch.vstack([
    pad_to_size(reg_u_c3_1, (4, 1)),
    pad_to_size(reg_u_c3_2, (4, 1))
])

# Final combined control feature matrix [8, 2]
control3_fullfeat_cat = torch.cat((control3_ufeat_stack, control3_reg_stack), dim=1)


# EX4 ---- Anti-surge ----
u_process4 = ["evaporator", "splitter", "compressor", "splitter", "valve", "knockout_drum"]
features_p4 = torch.tensor([[full_unit_lib[unit]] for unit in u_process4], dtype=torch.float32)

control4_1_adj = Dataset_general.control4_1_adj

control_adj_4_stack = torch.vstack([
    pad_to_size(control4_1_adj, (7, 7))
])

u_control4_1 = ["SP", "FC", "z", "PDT", "PT", "FT", "PT"]

features_c4_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control4_1], dtype=torch.float32)
reg_u_c4_1 = torch.tensor([[0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)

control4_ufeat_stack = torch.vstack([
    pad_to_size(features_c4_1, (7, 1))
])

control4_reg_stack = torch.vstack([
    pad_to_size(reg_u_c4_1, (7, 1))
])

control4_fullfeat_cat = torch.cat((control4_ufeat_stack, control4_reg_stack), dim=1)


# EX5 ---- SHELL & TUBE HX ----
u_process5 = ["valve", "HX"]
features_p5 = torch.tensor([[full_unit_lib[unit]] for unit in u_process5], dtype=torch.float32)

control5_1_adj = Dataset_general.control5_1_adj

control_adj_5_stack = torch.vstack([
    pad_to_size(control5_1_adj, (6, 6))
    ])

u_control5_1 = ["SP", "TC", "PC", "z", "TT", "PT"]

features_c5_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control5_1], dtype=torch.float32)
reg_u_c5_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control5_ufeat_stack = torch.vstack([
    pad_to_size(features_c5_1, (6, 1))
])

control5_reg_stack = torch.vstack([
    pad_to_size(reg_u_c5_1, (6, 1))
])

control5_fullfeat_cat = torch.cat((control5_ufeat_stack, control5_reg_stack), dim=1)


# EX6 ---- JACKETED CSTR ----
u_process6 = ["valve", "CSTR", "jacket", "valve"]

features_p6 = torch.tensor([[full_unit_lib[unit]] for unit in u_process6], dtype=torch.float32)

control6_1_adj = Dataset_general.control6_1_adj
control6_2_adj = Dataset_general.control6_2_adj

control_adj_6_stack = torch.vstack([
    pad_to_size(control6_1_adj, (4, 4)),
    pad_to_size(control6_2_adj, (4, 4))
])

# Adjacency matrices for the control graph (only the first NxN part from each expanded block)
control6_1_adj = Dataset_general.control6_1_adj
control6_2_adj = Dataset_general.control6_2_adj

control_adj_6_stack = torch.vstack([
    pad_to_size(control6_1_adj, (6, 6)),
    pad_to_size(control6_2_adj, (6, 6))
])

# Features and regulation
u_control6_1 = ["SP", "PC", "z", "PT"]
u_control6_2 = ["SP", "TC", "z", "TT"]

features_c6_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control6_1], dtype=torch.float32)
features_c6_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control6_2], dtype=torch.float32)

reg_u_c6_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c6_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control6_ufeat_stack = torch.vstack([
    pad_to_size(features_c6_1, (6, 1)),
    pad_to_size(features_c6_2, (6, 1)),
])

control6_reg_stack = torch.vstack([
    pad_to_size(reg_u_c6_1, (6, 1)),
    pad_to_size(reg_u_c6_2, (6, 1)),
])

control6_fullfeat_cat = torch.cat((control6_ufeat_stack, control6_reg_stack), dim=1)



# EX7 ---- JACKETED CSTR 2 ----
u_process7 = ["valve", "jacket", "CSTR"]
u_control7_1 = ["SP", "TC", "z", "TT"]

features_p7 = torch.tensor([[full_unit_lib[unit]] for unit in u_process7], dtype=torch.float32)

control7_1_adj = Dataset_general.control7_1_adj

control_adj_7_stack = torch.vstack([
    pad_to_size(control7_1_adj, (6, 6))
])

features_c7_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control7_1], dtype=torch.float32)
reg_u_c7_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control7_ufeat_stack = torch.vstack([
    pad_to_size(features_c7_1, (6, 1))
])

control7_reg_stack = torch.vstack([
    pad_to_size(reg_u_c7_1, (6, 1))
])

control7_fullfeat_cat = torch.cat((control7_ufeat_stack, control7_reg_stack), dim=1)

# EX8 ----DISTILLATION COLUMN 2 ----

u_process8 = ["column", "HX", "condenser", "pump", "splitter", "valve", "splitter", "reboiler", "valve", "valve", "valve", "valve"]
features_p8 = torch.tensor([[full_unit_lib[unit]] for unit in u_process8], dtype=torch.float32)

control8_1_adj = Dataset_general.control8_1_adj
control8_2_adj = Dataset_general.control8_2_adj
control8_3_adj = Dataset_general.control8_3_adj
control8_4_adj = Dataset_general.control8_4_adj

control_adj_8_stack = torch.vstack([
    pad_to_size(control8_1_adj, (6, 6)),
    pad_to_size(control8_2_adj, (6, 6)),
    pad_to_size(control8_3_adj, (6, 6)),
    pad_to_size(control8_4_adj, (6, 6))
])

u_control8_1 = ["SP", "CC", "z", "CT"]
u_control8_2 = ["SP", "CC", "z", "CT"]
u_control8_3 = ["SP", "LC", "z", "LT"]
u_control8_4 = ["SP", "LC", "z", "LT"]

features_c8_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_1], dtype=torch.float32)
features_c8_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_2], dtype=torch.float32)
features_c8_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_3], dtype=torch.float32)
features_c8_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control8_4], dtype=torch.float32)

reg_u_c8_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c8_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c8_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c8_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control8_ufeat_stack = torch.vstack([
    pad_to_size(features_c8_1, (6, 1)),
    pad_to_size(features_c8_2, (6, 1)),
    pad_to_size(features_c8_3, (6, 1)),
    pad_to_size(features_c8_4, (6, 1)),
])

control8_reg_stack = torch.vstack([
    pad_to_size(reg_u_c8_1, (6, 1)),
    pad_to_size(reg_u_c8_2, (6, 1)),
    pad_to_size(reg_u_c8_3, (6, 1)),
    pad_to_size(reg_u_c8_4, (6, 1)),
])

control8_fullfeat_cat = torch.cat((control8_ufeat_stack, control8_reg_stack), dim=1)



# EX9 ----SHELL & TUBE HX 2 ----
u_process9 = ["valve", "HX"]
features_p9 = torch.tensor([[full_unit_lib[unit]] for unit in u_process9], dtype=torch.float32)

control9_1_adj = Dataset_general.control9_1_adj

control_adj_9_stack = torch.vstack([
    pad_to_size(control9_1_adj, (6, 6))
])

u_control9_1 = ["SP", "TC", "z", "TT"]

features_c9_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control9_1], dtype=torch.float32)
reg_u_c9_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control9_ufeat_stack = torch.vstack([
    pad_to_size(features_c9_1, (6, 1))
])

control9_reg_stack = torch.vstack([
    pad_to_size(reg_u_c9_1, (6, 1))
])

control9_fullfeat_cat = torch.cat((control9_ufeat_stack, control9_reg_stack), dim=1)



# EX10 ----HEAT EXCHANGER ----

u_process10 = ["valve", "HX", "valve"]
features_p10 = torch.tensor([[full_unit_lib[unit]] for unit in u_process10], dtype=torch.float32)

control10_1_adj = Dataset_general.control10_1_adj

control_adj_10_stack = torch.vstack([
    pad_to_size(control10_1_adj, (6, 6))
])

u_control10_1 = ["SP", "TC", "z", "TT"]

features_c10_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control10_1], dtype=torch.float32)
reg_u_c10_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control10_ufeat_stack = torch.vstack([
    pad_to_size(features_c10_1, (6, 1))
])

control10_reg_stack = torch.vstack([
    pad_to_size(reg_u_c10_1, (6, 1))
])

control10_fullfeat_cat = torch.cat((control10_ufeat_stack, control10_reg_stack), dim=1)



# EX11 ----FILTER AND DECANTATION ----
u_process11 = ["valve", "tank", "pump", "splitter", "decanter", "pump", "filter", "valve", "filter", "valve", "dosing_unit"]
features_p11 = torch.tensor([[full_unit_lib[unit]] for unit in u_process11], dtype=torch.float32)

control11_1_adj = Dataset_general.control11_1_adj
control11_2_adj = Dataset_general.control11_2_adj

control_adj_11_stack = torch.vstack([
    pad_to_size(control11_1_adj, (6, 6)),
    pad_to_size(control11_2_adj, (6, 6))
])

u_control11_1 = ["SP", "FC", "w", "FT"]
u_control11_2 = ["SP", "FC", "w", "FT"]

features_c11_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control11_1], dtype=torch.float32)
features_c11_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control11_2], dtype=torch.float32)

reg_u_c11_1 = torch.tensor([[0], [6], [0], [0]], dtype=torch.float32)
reg_u_c11_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control11_ufeat_stack = torch.vstack([
    pad_to_size(features_c11_1, (6, 1)),
    pad_to_size(features_c11_2, (6, 1)),
])

control11_reg_stack = torch.vstack([
    pad_to_size(reg_u_c11_1, (6, 1)),
    pad_to_size(reg_u_c11_2, (6, 1)),
])

control11_fullfeat_cat = torch.cat((control11_ufeat_stack, control11_reg_stack), dim=1)


# EX12 ----TANK 1 ---- 
u_process12 = ["valve", "tank"]
features_p12 = torch.tensor([[full_unit_lib[unit]] for unit in u_process12], dtype=torch.float32)

control12_1_adj = Dataset_general.control12_1_adj
control_adj_12_stack = torch.vstack([
    pad_to_size(control12_1_adj, (6, 6))
])

u_control12_1 = ["SP", "LC", "z", "LT"]
features_c12_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control12_1], dtype=torch.float32)
reg_u_c12_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control12_ufeat_stack = torch.vstack([
    pad_to_size(features_c12_1, (6, 1))
])
control12_reg_stack = torch.vstack([
    pad_to_size(reg_u_c12_1, (6, 1))
])
control12_fullfeat_cat = torch.cat((control12_ufeat_stack, control12_reg_stack), dim=1)


# EX13 ----TANK 2 ----
u_process13 = ["tank", "valve"]
features_p13 = torch.tensor([[full_unit_lib[unit]] for unit in u_process13], dtype=torch.float32)

control13_1_adj = Dataset_general.control13_1_adj
control_adj_13_stack = torch.vstack([
    pad_to_size(control13_1_adj, (6, 6))
])

u_control13_1 = ["SP", "LC", "z", "LT"]
features_c13_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control13_1], dtype=torch.float32)
reg_u_c13_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control13_ufeat_stack = torch.vstack([
    pad_to_size(features_c13_1, (6, 1))
])
control13_reg_stack = torch.vstack([
    pad_to_size(reg_u_c13_1, (6, 1))
])
control13_fullfeat_cat = torch.cat((control13_ufeat_stack, control13_reg_stack), dim=1)



# EX14 ----TANK 3 ----
u_process14 = ["tank", "valve"]
features_p14 = torch.tensor([[full_unit_lib[unit]] for unit in u_process14], dtype=torch.float32)

control14_1_adj = Dataset_general.control14_1_adj
control_adj_14_stack = torch.vstack([
    pad_to_size(control14_1_adj, (6, 6))
])

u_control14_1 = ["SP", "FC", "z", "FT"]
features_c14_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control14_1], dtype=torch.float32)
reg_u_c14_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control14_ufeat_stack = torch.vstack([
    pad_to_size(features_c14_1, (6, 1))
])
control14_reg_stack = torch.vstack([
    pad_to_size(reg_u_c14_1, (6, 1))
])
control14_fullfeat_cat = torch.cat((control14_ufeat_stack, control14_reg_stack), dim=1)



# EX15 ----TANK 4 ----
u_process15 = ["tank", "valve"]
features_p15 = torch.tensor([[full_unit_lib[unit]] for unit in u_process15], dtype=torch.float32)

control15_1_adj = Dataset_general.control15_1_adj
control_adj_15_stack = torch.vstack([
    pad_to_size(control15_1_adj, (6, 6))
])

u_control15_1 = ["SP", "LC", "FC", "z", "LT", "FT"]
features_c15_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control15_1], dtype=torch.float32)
reg_u_c15_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control15_ufeat_stack = torch.vstack([
    pad_to_size(features_c15_1, (6, 1))
])
control15_reg_stack = torch.vstack([
    pad_to_size(reg_u_c15_1, (6, 1))
])
control15_fullfeat_cat = torch.cat((control15_ufeat_stack, control15_reg_stack), dim=1)



# EX16 ----ADIABATIC REACTOR ----
u_process16 = ["valve", "HX", "reactor"]
features_p16 = torch.tensor([[full_unit_lib[unit]] for unit in u_process16], dtype=torch.float32)

control16_1_adj = Dataset_general.control16_1_adj
control_adj_16_stack = torch.vstack([
    pad_to_size(control16_1_adj, (6, 6))
])

u_control16_1 = ["SP", "CC", "z", "CT"]
features_c16_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control16_1], dtype=torch.float32)
reg_u_c16_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control16_ufeat_stack = torch.vstack([
    pad_to_size(features_c16_1, (6, 1))
])
control16_reg_stack = torch.vstack([
    pad_to_size(reg_u_c16_1, (6, 1))
])
control16_fullfeat_cat = torch.cat((control16_ufeat_stack, control16_reg_stack), dim=1)


# EX17 ---- CSTR ----
u_process17 = ["valve", "CSTR"]
features_p17 = torch.tensor([[full_unit_lib[unit]] for unit in u_process17], dtype=torch.float32)

control17_1_adj = Dataset_general.control17_1_adj
control_adj_17_stack = torch.vstack([
    pad_to_size(control17_1_adj, (6, 6))
])

u_control17_1 = ["SP", "CC", "z", "CT"]
features_c17_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control17_1], dtype=torch.float32)
reg_u_c17_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control17_ufeat_stack = torch.vstack([
    pad_to_size(features_c17_1, (6, 1))
])
control17_reg_stack = torch.vstack([
    pad_to_size(reg_u_c17_1, (6, 1))
])
control17_fullfeat_cat = torch.cat((control17_ufeat_stack, control17_reg_stack), dim=1)



# EX18 ---- FURNACE ----
u_process18 = ["valve", "furnace"]
features_p18 = torch.tensor([[full_unit_lib[unit]] for unit in u_process18], dtype=torch.float32)

control18_1_adj = Dataset_general.control18_1_adj
control_adj_18_stack = torch.vstack([
    pad_to_size(control18_1_adj, (6, 6))
])

u_control18_1 = ["SP", "TC", "z", "TT"]
features_c18_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control18_1], dtype=torch.float32)
reg_u_c18_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control18_ufeat_stack = torch.vstack([
    pad_to_size(features_c18_1, (6, 1))
])
control18_reg_stack = torch.vstack([
    pad_to_size(reg_u_c18_1, (6, 1))
])
control18_fullfeat_cat = torch.cat((control18_ufeat_stack, control18_reg_stack), dim=1)


# EX19 ---- CASCADE FURNACE ----
u_process19 = ["valve", "furnace"]
features_p19 = torch.tensor([[full_unit_lib[unit]] for unit in u_process19], dtype=torch.float32)

control19_1_adj = Dataset_general.control19_1_adj
control_adj_19_stack = torch.vstack([
    pad_to_size(control19_1_adj, (6, 6))
])

u_control19_1 = ["SP", "TC", "FC", "z", "TT", "FT"]
features_c19_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control19_1], dtype=torch.float32)
reg_u_c19_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control19_ufeat_stack = torch.vstack([
    pad_to_size(features_c19_1, (6, 1))
])
control19_reg_stack = torch.vstack([
    pad_to_size(reg_u_c19_1, (6, 1))
])
control19_fullfeat_cat = torch.cat((control19_ufeat_stack, control19_reg_stack), dim=1)

# EX20 ---- CASCADE CSTR ----
u_process20 = ["valve", "jacket", "CSTR"]
features_p20 = torch.tensor([[full_unit_lib[unit]] for unit in u_process20], dtype=torch.float32)

control20_1_adj = Dataset_general.control20_1_adj
control_adj_20_stack = torch.vstack([
    pad_to_size(control20_1_adj, (6, 6))
])

u_control20_1 = ["SP", "TC", "TC", "z", "TT", "TT"]
features_c20_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control20_1], dtype=torch.float32)
reg_u_c20_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control20_ufeat_stack = torch.vstack([
    pad_to_size(features_c20_1, (6, 1))
])
control20_reg_stack = torch.vstack([
    pad_to_size(reg_u_c20_1, (6, 1))
])
control20_fullfeat_cat = torch.cat((control20_ufeat_stack, control20_reg_stack), dim=1)

# EX21 ---- FURNACE 2 ----
u_process21 = ["valve", "furnace"]
features_p21 = torch.tensor([[full_unit_lib[unit]] for unit in u_process21], dtype=torch.float32)

control21_1_adj = Dataset_general.control21_1_adj
control_adj_21_stack = torch.vstack([
    pad_to_size(control21_1_adj, (8, 9))
])

u_control21_1 = ["SP", "FC", "summation_block", "z", "TC", "SP", "FT", "TT"]
features_c21_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control21_1], dtype=torch.float32)
reg_u_c21_1 = torch.tensor([[0], [0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)

control21_ufeat_stack = torch.vstack([
    pad_to_size(features_c21_1, (8, 1))
])
control21_reg_stack = torch.vstack([
    pad_to_size(reg_u_c21_1, (8, 1))
])
control21_fullfeat_cat = torch.cat((control21_ufeat_stack, control21_reg_stack), dim=1)


# EX22 ---- JACKETED CSTR 3 ----
u_process22 = ["valve", "jacket", "CSTR"]
features_p22 = torch.tensor([[full_unit_lib[unit]] for unit in u_process22], dtype=torch.float32)

control22_1_adj = Dataset_general.control22_1_adj
control_adj_22_stack = torch.vstack([
    pad_to_size(control22_1_adj, (8, 8))
])

u_control22_1 = ["SP", "TC", "TC", "FC", "z", "TT", "TT", "FT"]
features_c22_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control22_1], dtype=torch.float32)
reg_u_c22_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)

control22_ufeat_stack = torch.vstack([
    pad_to_size(features_c22_1, (8, 1))
])
control22_reg_stack = torch.vstack([
    pad_to_size(reg_u_c22_1, (8, 1))
])
control22_fullfeat_cat = torch.cat((control22_ufeat_stack, control22_reg_stack), dim=1)








# EX23 ---- JACKETED CSTR 4 ----
u_process23 = ["valve", "jacket", "CSTR"]
features_p23 = torch.tensor([[full_unit_lib[unit]] for unit in u_process23], dtype=torch.float32)

control23_1_adj = Dataset_general.control23_1_adj
control_adj_23_stack = torch.vstack([
    pad_to_size(control23_1_adj, (10, 10))
])

u_control23_1 = ["SP", "CC", "TC", "TC", "FC", "z", "CT", "TT", "TT", "FT"]
features_c23_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control23_1], dtype=torch.float32)
reg_u_c23_1 = torch.tensor([[0], [0], [0], [0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)

control23_ufeat_stack = torch.vstack([
    pad_to_size(features_c23_1, (10, 1))
])
control23_reg_stack = torch.vstack([
    pad_to_size(reg_u_c23_1, (10, 1))
])
control23_fullfeat_cat = torch.cat((control23_ufeat_stack, control23_reg_stack), dim=1)


# EX24 ---- GROWTH RATE ----
u_process24 = ["valve", "bubbler", "splitter", "reactor", "spectroscopic_ellipsometer", "pump"]
features_p24 = torch.tensor([[full_unit_lib[unit]] for unit in u_process24], dtype=torch.float32)

control24_1_adj = Dataset_general.control24_1_adj
control_adj_24_stack = torch.vstack([
    pad_to_size(control24_1_adj, (8, 8))
])

u_control24_1 = ["SP", "GRC", "CC", "FC", "z", "GRT", "CT", "FT"]
features_c24_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control24_1], dtype=torch.float32)
reg_u_c24_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)

control24_ufeat_stack = torch.vstack([
    pad_to_size(features_c24_1, (8, 1))
])
control24_reg_stack = torch.vstack([
    pad_to_size(reg_u_c24_1, (8, 1))
])
control24_fullfeat_cat = torch.cat((control24_ufeat_stack, control24_reg_stack), dim=1)


# EX25 ---- BLEND STREAM ----
u_process25 = ["valve", "splitter"]
features_p25 = torch.tensor([[full_unit_lib[unit]] for unit in u_process25], dtype=torch.float32)

control25_1_adj = Dataset_general.control25_1_adj
control_adj_25_stack = torch.vstack([
    pad_to_size(control25_1_adj, (6, 6))
])

u_control25_1 = ["SP", "X_multiplication_block", "FC", "z", "FT", "FT"]
features_c25_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control25_1], dtype=torch.float32)
reg_u_c25_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control25_ufeat_stack = torch.vstack([
    pad_to_size(features_c25_1, (6, 1))
])
control25_reg_stack = torch.vstack([
    pad_to_size(reg_u_c25_1, (6, 1))
])
control25_fullfeat_cat = torch.cat((control25_ufeat_stack, control25_reg_stack), dim=1)


# EX26 ---- CSTR + SEPARATOR ----
u_process26 = ["valve", "splitter", "CSTR", "valve", "splitter", "valve", "valve"]
features_p26 = torch.tensor([[full_unit_lib[unit]] for unit in u_process26], dtype=torch.float32)

control26_1_adj = Dataset_general.control26_1_adj
control26_2_adj = Dataset_general.control26_2_adj
control26_3_adj = Dataset_general.control26_3_adj
control26_4_adj = Dataset_general.control26_4_adj

control_adj_26_stack = torch.vstack([
    pad_to_size(control26_1_adj, (4, 4)),
    pad_to_size(control26_2_adj, (4, 4)),
    pad_to_size(control26_3_adj, (4, 4)),
    pad_to_size(control26_4_adj, (4, 4))
])

u_control26_1 = ["SP", "FC", "z", "FT"]
u_control26_2 = ["SP", "LC", "z", "LT"]
u_control26_3 = ["SP", "CC", "z", "CT"]
u_control26_4 = ["SP", "FC", "z", "FT"]

features_c26 = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control26_1]), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control26_2]), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control26_3]), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control26_4]), (4, 1)),
])

reg_c26 = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
])

control26_fullfeat_cat = torch.cat((features_c26, reg_c26), dim=1)


# EX27 ---- CSTR + SEPARATOR 2 ----
u_process27 = ["valve", "splitter", "CSTR", "valve", "splitter", "valve", "valve"]
features_p27 = torch.tensor([[full_unit_lib[unit]] for unit in u_process27], dtype=torch.float32)

control27_1_adj = Dataset_general.control27_1_adj
control27_2_adj = Dataset_general.control27_2_adj
control27_3_adj = Dataset_general.control27_3_adj
control27_4_adj = Dataset_general.control27_4_adj

control_adj_27_stack = torch.vstack([
    pad_to_size(control27_1_adj, (4, 4)),
    pad_to_size(control27_2_adj, (4, 4)),
    pad_to_size(control27_3_adj, (4, 4)),
    pad_to_size(control27_4_adj, (4, 4))
])

u_control27_1 = ["SP", "FC", "z", "FT"]
u_control27_2 = ["SP", "LC", "z", "LT"]
u_control27_3 = ["SP", "CC", "z", "CT"]
u_control27_4 = ["SP", "FC", "z", "FT"]

features_c27 = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control27_1]), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control27_2]), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control27_3]), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control27_4]), (4, 1)),
])

reg_c27 = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
])

control27_fullfeat_cat = torch.cat((features_c27, reg_c27), dim=1)


#------------------------------------------------------------------
# EX28 ---- CSTR + DISTILLATION----
u_process28 = ["valve", "splitter", "CSTR", "valve", "column", "HX", "condenser", 
               "pump", "splitter", "valve", "splitter", "valve", "HX", "valve", "valve", "valve"]
features_p28 = torch.tensor([[full_unit_lib[unit]] for unit in u_process28], dtype=torch.float32)

control28_1_adj = Dataset_general.control28_1_adj
control28_2_adj = Dataset_general.control28_2_adj
control28_3_adj = Dataset_general.control28_3_adj
control28_4_adj = Dataset_general.control28_4_adj
control28_5_adj = Dataset_general.control28_5_adj
control28_6_adj = Dataset_general.control28_6_adj
control28_7_adj = Dataset_general.control28_7_adj

control_adj_28_stack = torch.vstack([
    pad_to_size(control28_1_adj, (4, 4)),
    pad_to_size(control28_2_adj, (4, 4)),
    pad_to_size(control28_3_adj, (4, 4)),
    pad_to_size(control28_4_adj, (4, 4)),
    pad_to_size(control28_5_adj, (4, 4)),
    pad_to_size(control28_6_adj, (4, 4)),
    pad_to_size(control28_7_adj, (4, 4))
])

u_control28_1 = ["SP", "FC", "z", "FT"]
u_control28_2 = ["SP", "LC", "z", "LT"]
u_control28_3 = ["SP", "CC", "z", "CT"]
u_control28_4 = ["SP", "LC", "z", "LT"]
u_control28_5 = ["SP", "CC", "z", "CT"]
u_control28_6 = ["SP", "PC", "z", "PT"]
u_control28_7 = ["SP", "LC", "z", "LT"]

features_c28 = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control28_1], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control28_2], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control28_3], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control28_4], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control28_5], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control28_6], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control28_7], dtype=torch.float32), (4, 1)),
])

reg_c28 = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
])

control28_fullfeat_cat = torch.cat((features_c28, reg_c28), dim=1)


#------------------------------------------------------------------
# EX29 ----TANK 5 ----
u_process29 = ["valve", "tank", "pump", "valve"]
features_p29 = torch.tensor([[full_unit_lib[unit]] for unit in u_process29], dtype=torch.float32)

control29_1_adj = Dataset_general.control29_1_adj
control29_2_adj = Dataset_general.control29_2_adj
control_adj_29_stack = torch.vstack([
    pad_to_size(control29_1_adj, (4, 4)),
    pad_to_size(control29_2_adj, (4, 4))
])

u_control29_1 = ["SP", "FC", "z", "FT"]
u_control29_2 = ["SP", "LC", "z", "LT"]

features_c29 = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control29_1], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control29_2], dtype=torch.float32), (4, 1)),
])
reg_c29 = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1)),
])
control29_fullfeat_cat = torch.cat((features_c29, reg_c29), dim=1)

#------------------------------------------------------------------
# EX30 ----BYPASS HX ----
u_process30 = ["splitter", "HX", "splitter", "valve"]
features_p30 = torch.tensor([[full_unit_lib[unit]] for unit in u_process30], dtype=torch.float32)

control30_1_adj = Dataset_general.control30_1_adj
control_adj_30_stack = torch.vstack([
    pad_to_size(control30_1_adj, (4, 4))
])

u_control30_1 = ["SP", "TC", "z", "TT"]

features_c30 = pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control30_1], dtype=torch.float32), (4, 1))
reg_c30 = pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1))
control30_fullfeat_cat = torch.cat((features_c30, reg_c30), dim=1)

#------------------------------------------------------------------
# EX32 ----REACTOR + HX----
u_process32 = ["valve", "HX", "splitter", "reactor", "valve"]
features_p32 = torch.tensor([[full_unit_lib[unit]] for unit in u_process32], dtype=torch.float32)

control32_1_adj = Dataset_general.control32_1_adj
control_adj_32_stack = torch.vstack([
    pad_to_size(control32_1_adj, (4, 4))
])

u_control32_1 = ["SP", "TC", "z", "TT"]
features_c32_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control32_1], dtype=torch.float32)
reg_u_c32_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control32_ufeat_stack = torch.vstack([
    pad_to_size(features_c32_1, (4, 1))
])
control32_reg_stack = torch.vstack([
    pad_to_size(reg_u_c32_1, (4, 1))
])
control32_fullfeat_cat = torch.cat((control32_ufeat_stack, control32_reg_stack), dim=1)

#------------------------------------------------------------------
# EX33 ---- CSTR + DISTILLATION 2----
u_process33 = ["valve", "splitter", "splitter", "CSTR", "valve", "column", "HX", 
               "valve", "condenser", "pump", "splitter", "valve", "valve", "jacket", 
               "splitter", "HX", "valve", "valve", "valve", "valve"]
features_p33 = torch.tensor([[full_unit_lib[unit]] for unit in u_process33], dtype=torch.float32)

control33_1_adj = Dataset_general.control33_1_adj
control33_2_adj = Dataset_general.control33_2_adj
control33_3_adj = Dataset_general.control33_3_adj
control33_4_adj = Dataset_general.control33_4_adj
control33_5_adj = Dataset_general.control33_5_adj
control33_6_adj = Dataset_general.control33_6_adj
control33_7_adj = Dataset_general.control33_7_adj
control33_8_adj = Dataset_general.control33_8_adj
control33_9_adj = Dataset_general.control33_9_adj

control_adj_33_stack = torch.vstack([
    pad_to_size(control33_1_adj, (4, 4)),
    pad_to_size(control33_2_adj, (4, 4)),
    pad_to_size(control33_3_adj, (4, 4)),
    pad_to_size(control33_4_adj, (4, 4)),
    pad_to_size(control33_5_adj, (4, 4)),
    pad_to_size(control33_6_adj, (4, 4)),
    pad_to_size(control33_7_adj, (4, 4)),
    pad_to_size(control33_8_adj, (4, 4)),
    pad_to_size(control33_9_adj, (4, 4)),
])

u_controls_33 = [
    ["SP", "LC", "z", "LT"],
    ["SP", "LC", "z", "LT"],
    ["SP", "PC", "z", "PT"],
    ["SP", "FC", "z", "FT"],
    ["SP", "TC", "z", "TT"],
    ["SP", "FC", "z", "FT"],
    ["SP", "FC", "z", "FT"],
    ["SP", "LC", "z", "LT"],
    ["SP", "TC", "z", "TT"]
]

features_c33 = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in uc], dtype=torch.float32), (4, 1))
    for uc in u_controls_33
])
reg_c33 = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]], dtype=torch.float32), (4, 1))
    for _ in u_controls_33
])
control33_fullfeat_cat = torch.cat((features_c33, reg_c33), dim=1)


# EX34 ---- TANK 6 ----
u_process34 = ["tank", "pump", "valve"]
features_p34 = torch.tensor([[full_unit_lib[unit]] for unit in u_process34], dtype=torch.float32)

control34_1_adj = Dataset_general.control34_1_adj
control_adj_34_stack = torch.vstack([
    pad_to_size(control34_1_adj, (6, 6))
])

u_control34_1 = ["SP", "LC", "z", "LT"]
features_c34_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control34_1], dtype=torch.float32)
reg_u_c34_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control34_ufeat_stack = torch.vstack([
    pad_to_size(features_c34_1, (6, 1))
])
control34_reg_stack = torch.vstack([
    pad_to_size(reg_u_c34_1, (6, 1))
])
control34_fullfeat_cat = torch.cat((control34_ufeat_stack, control34_reg_stack), dim=1)

# EX35 ---- CSTR 2 ----
u_process35 = ["valve", "CSTR"]
features_p35 = torch.tensor([[full_unit_lib[unit]] for unit in u_process35], dtype=torch.float32)

control35_1_adj = Dataset_general.control35_1_adj
control_adj_35_stack = torch.vstack([
    pad_to_size(control35_1_adj, (6, 6))
])

u_control35_1 = ["SP", "LC", "z", "LT"]
features_c35_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control35_1], dtype=torch.float32)
reg_u_c35_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control35_ufeat_stack = torch.vstack([
    pad_to_size(features_c35_1, (6, 1))
])
control35_reg_stack = torch.vstack([
    pad_to_size(reg_u_c35_1, (6, 1))
])
control35_fullfeat_cat = torch.cat((control35_ufeat_stack, control35_reg_stack), dim=1)

# EX36 ---- JACKETED CSTR 5 ----
u_process36 = ["valve", "splitter", "pump", "jacket", "splitter", "CSTR"]
features_p36 = torch.tensor([[full_unit_lib[unit]] for unit in u_process36], dtype=torch.float32)

control36_1_adj = Dataset_general.control36_1_adj
control_adj_36_stack = torch.vstack([
    pad_to_size(control36_1_adj, (6, 6))
])

u_control36_1 = ["SP", "TC", "TC", "z", "TT", "TT"]
features_c36_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control36_1], dtype=torch.float32)
reg_u_c36_1 = torch.tensor([[0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control36_ufeat_stack = torch.vstack([
    pad_to_size(features_c36_1, (6, 1))
])
control36_reg_stack = torch.vstack([
    pad_to_size(reg_u_c36_1, (6, 1))
])
control36_fullfeat_cat = torch.cat((control36_ufeat_stack, control36_reg_stack), dim=1)

# EX37 ---- STEAM DRUM HX ----
u_process37 = ["valve", "steam_drum_HX"]
features_p37 = torch.tensor([[full_unit_lib[unit]] for unit in u_process37], dtype=torch.float32)

control37_1_adj = Dataset_general.control37_1_adj
control_adj_37_stack = torch.vstack([
    pad_to_size(control37_1_adj, (8, 8))
])

u_control37_1 = ["SP", "LC", "addition_block", "FC", "z", "LT", "FT", "FT"]
features_c37_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control37_1], dtype=torch.float32)
reg_u_c37_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)

control37_ufeat_stack = torch.vstack([
    pad_to_size(features_c37_1, (8, 1))
])
control37_reg_stack = torch.vstack([
    pad_to_size(reg_u_c37_1, (8, 1))
])
control37_fullfeat_cat = torch.cat((control37_ufeat_stack, control37_reg_stack), dim=1)

# EX38 ---- TANK 7 ----
u_process38 = ["tank", "pump", "valve"]
features_p38 = torch.tensor([[full_unit_lib[unit]] for unit in u_process38], dtype=torch.float32)

control38_1_adj = Dataset_general.control38_1_adj
control_adj_38_stack = torch.vstack([
    pad_to_size(control38_1_adj, (6, 6))
])

u_control38_1 = ["SP", "FC", "z", "FT"]
features_c38_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control38_1], dtype=torch.float32)
reg_u_c38_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control38_ufeat_stack = torch.vstack([
    pad_to_size(features_c38_1, (6, 1))
])
control38_reg_stack = torch.vstack([
    pad_to_size(reg_u_c38_1, (6, 1))
])
control38_fullfeat_cat = torch.cat((control38_ufeat_stack, control38_reg_stack), dim=1)


# EX39 ---- VPC REACTOR ----
u_process39 = ["valve", "HX", "jacket", "reactor", "valve"]
features_p39 = torch.tensor([[full_unit_lib[unit]] for unit in u_process39], dtype=torch.float32)

control39_1_adj = Dataset_general.control39_1_adj
control_adj_39_stack = torch.vstack([
    pad_to_size(control39_1_adj, (7, 7))
])

u_control39_1 = ["SP", "TC", "VPC", "z", "TT", "z", "SP"]
features_c39_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control39_1], dtype=torch.float32)
reg_u_c39_1 = torch.tensor([[0], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)

control39_ufeat_stack = torch.vstack([
    pad_to_size(features_c39_1, (7, 1))
])
control39_reg_stack = torch.vstack([
    pad_to_size(reg_u_c39_1, (7, 1))
])
control39_fullfeat_cat = torch.cat((control39_ufeat_stack, control39_reg_stack), dim=1)

# EX40 ---- TOP COLUMN ----
u_process40 = ["column", "HX", "condenser", "splitter", "valve"]
features_p40 = torch.tensor([[full_unit_lib[unit]] for unit in u_process40], dtype=torch.float32)

control40_1_adj = Dataset_general.control40_1_adj
control_adj_40_stack = torch.vstack([
    pad_to_size(control40_1_adj, (6, 6))
])

u_control40_1 = ["SP", "PC", "z", "PT"]
features_c40_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control40_1], dtype=torch.float32)
reg_u_c40_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control40_ufeat_stack = torch.vstack([
    pad_to_size(features_c40_1, (6, 1))
])
control40_reg_stack = torch.vstack([
    pad_to_size(reg_u_c40_1, (6, 1))
])
control40_fullfeat_cat = torch.cat((control40_ufeat_stack, control40_reg_stack), dim=1)

# EX41 ---- TOP COLUMN 2 ----
u_process41 = ["column", "HX", "condenser", "splitter", "valve", "valve", "valve"]
features_p41 = torch.tensor([[full_unit_lib[unit]] for unit in u_process41], dtype=torch.float32)

control41_1_adj = Dataset_general.control41_1_adj
control41_2_adj = Dataset_general.control41_2_adj
control41_3_adj = Dataset_general.control41_3_adj

control_adj_41_stack = torch.vstack([
    pad_to_size(control41_1_adj, (6, 6)),
    pad_to_size(control41_2_adj, (6, 6)),
    pad_to_size(control41_3_adj, (6, 6))
])

u_control41_1 = ["SP", "CC", "z", "CT"]
u_control41_2 = ["SP", "PC", "z", "PT"]
u_control41_3 = ["SP", "LC", "z", "LT"]

features_c41_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control41_1], dtype=torch.float32)
features_c41_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control41_2], dtype=torch.float32)
features_c41_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control41_3], dtype=torch.float32)

reg_u_c41_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c41_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c41_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control41_ufeat_stack = torch.vstack([
    pad_to_size(features_c41_1, (6, 1)),
    pad_to_size(features_c41_2, (6, 1)),
    pad_to_size(features_c41_3, (6, 1)),
])
control41_reg_stack = torch.vstack([
    pad_to_size(reg_u_c41_1, (6, 1)),
    pad_to_size(reg_u_c41_2, (6, 1)),
    pad_to_size(reg_u_c41_3, (6, 1)),
])
control41_fullfeat_cat = torch.cat((control41_ufeat_stack, control41_reg_stack), dim=1)

# EX42 ---- SETTLING TANK ----
u_process42 = ["settling_tank", "splitter", "valve", "valve", "valve", "valve", "valve", "PSV"]
features_p42 = torch.tensor([[full_unit_lib[unit]] for unit in u_process42], dtype=torch.float32)

control42_1_adj = Dataset_general.control42_1_adj
control42_2_adj = Dataset_general.control42_2_adj
control42_3_adj = Dataset_general.control42_3_adj

control_adj_42_stack = torch.vstack([
    pad_to_size(control42_1_adj, (6, 6)),
    pad_to_size(control42_2_adj, (6, 6)),
    pad_to_size(control42_3_adj, (6, 6))
])

u_control42_1 = ["SP", "PC", "z", "PT"]
u_control42_2 = ["SP", "LC", "z", "LT"]
u_control42_3 = ["SP", "LC", "z", "LT"]

features_c42_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control42_1], dtype=torch.float32)
features_c42_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control42_2], dtype=torch.float32)
features_c42_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control42_3], dtype=torch.float32)

reg_u_c42_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c42_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c42_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control42_ufeat_stack = torch.vstack([
    pad_to_size(features_c42_1, (6, 1)),
    pad_to_size(features_c42_2, (6, 1)),
    pad_to_size(features_c42_3, (6, 1)),
])
control42_reg_stack = torch.vstack([
    pad_to_size(reg_u_c42_1, (6, 1)),
    pad_to_size(reg_u_c42_2, (6, 1)),
    pad_to_size(reg_u_c42_3, (6, 1)),
])
control42_fullfeat_cat = torch.cat((control42_ufeat_stack, control42_reg_stack), dim=1)


# EX43 ---- TANK 8 ----
u_process43 = ["tank", "pump", "splitter", "splitter", "valve", "valve"]
features_p43 = torch.tensor([[full_unit_lib[unit]] for unit in u_process43], dtype=torch.float32)

control43_1_adj = Dataset_general.control43_1_adj
control43_2_adj = Dataset_general.control43_2_adj
control43_3_adj = Dataset_general.control43_3_adj

control_adj_43_stack = torch.vstack([
    pad_to_size(control43_1_adj, (8, 8)),
    pad_to_size(control43_2_adj, (8, 8)),
    pad_to_size(control43_3_adj, (8, 8)),
])

u_control43_1 = ["SP", "FC", "MIN", "w", "LC", "SP", "FT", "LT"]
u_control43_2 = ["SP", "FC", "z", "FT"]
u_control43_3 = ["SP", "FC", "z", "FT"]

features_c43_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control43_1], dtype=torch.float32)
features_c43_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control43_2], dtype=torch.float32)
features_c43_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control43_3], dtype=torch.float32)

reg_u_c43_1 = torch.tensor([[0], [0], [6], [0], [0], [0], [0], [0]], dtype=torch.float32)
reg_u_c43_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c43_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control43_ufeat_stack = torch.vstack([
    pad_to_size(features_c43_1, (8, 1)),
    pad_to_size(features_c43_2, (8, 1)),
    pad_to_size(features_c43_3, (8, 1)),
])
control43_reg_stack = torch.vstack([
    pad_to_size(reg_u_c43_1, (8, 1)),
    pad_to_size(reg_u_c43_2, (8, 1)),
    pad_to_size(reg_u_c43_3, (8, 1)),
])
control43_fullfeat_cat = torch.cat((control43_ufeat_stack, control43_reg_stack), dim=1)

# EX44 ---- COMPRESSOR ----
u_process44 = ["splitter", "compressor", "HX", "splitter", "valve"]
features_p44 = torch.tensor([[full_unit_lib[unit]] for unit in u_process44], dtype=torch.float32)

control44_1_adj = Dataset_general.control44_1_adj
control_adj_44_stack = torch.vstack([
    pad_to_size(control44_1_adj, (9, 9))
])

u_control44_1 = ["SP", "FC", "MAX", "z", "FC", "SP", "FT", "SP", "FT"]
features_c44_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control44_1], dtype=torch.float32)
reg_u_c44_1 = torch.tensor([[0], [0], [2], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)

control44_ufeat_stack = pad_to_size(features_c44_1, (9, 1))
control44_reg_stack = pad_to_size(reg_u_c44_1, (9, 1))
control44_fullfeat_cat = torch.cat((control44_ufeat_stack, control44_reg_stack), dim=1)

# EX45 ---- COMPRESSOR 2 ----
u_process45 = ["splitter", "compressor", "HX", "splitter", "valve"]
features_p45 = torch.tensor([[full_unit_lib[unit]] for unit in u_process45], dtype=torch.float32)

control45_1_adj = Dataset_general.control45_1_adj
control_adj_45_stack = torch.vstack([
    pad_to_size(control45_1_adj, (6, 6))
])

u_control45_1 = ["SP", "FC", "z", "FT"]
features_c45_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control45_1], dtype=torch.float32)
reg_u_c45_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control45_ufeat_stack = pad_to_size(features_c45_1, (6, 1))
control45_reg_stack = pad_to_size(reg_u_c45_1, (6, 1))
control45_fullfeat_cat = torch.cat((control45_ufeat_stack, control45_reg_stack), dim=1)

# EX46 ---- INVENTORY CONTROL ----
u_process46 = ["valve", "tank", "valve"]
features_p46 = torch.tensor([[full_unit_lib[unit]] for unit in u_process46], dtype=torch.float32)

control46_1_adj = Dataset_general.control46_1_adj
control46_2_adj = Dataset_general.control46_2_adj
control_adj_46_stack = torch.vstack([
    pad_to_size(control46_1_adj, (6, 6)),
    pad_to_size(control46_2_adj, (6, 6)),
])

u_control46_1 = ["SP", "FC", "z", "FT"]
u_control46_2 = ["SP", "LC", "z", "LT"]

features_c46_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control46_1], dtype=torch.float32)
features_c46_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control46_2], dtype=torch.float32)
reg_u_c46_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c46_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control46_ufeat_stack = torch.vstack([
    pad_to_size(features_c46_1, (6, 1)),
    pad_to_size(features_c46_2, (6, 1)),
])
control46_reg_stack = torch.vstack([
    pad_to_size(reg_u_c46_1, (6, 1)),
    pad_to_size(reg_u_c46_2, (6, 1)),
])
control46_fullfeat_cat = torch.cat((control46_ufeat_stack, control46_reg_stack), dim=1)

# EX47 ---- INVENTORY CONTROL 2 ----
u_process47 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p47 = torch.tensor([[full_unit_lib[unit]] for unit in u_process47], dtype=torch.float32)

control47_1_adj = Dataset_general.control47_1_adj
control47_2_adj = Dataset_general.control47_2_adj
control47_3_adj = Dataset_general.control47_3_adj
control47_4_adj = Dataset_general.control47_4_adj
control_adj_47_stack = torch.vstack([
    pad_to_size(control47_1_adj, (6, 6)),
    pad_to_size(control47_2_adj, (6, 6)),
    pad_to_size(control47_3_adj, (6, 6)),
    pad_to_size(control47_4_adj, (6, 6))
])

u_control47_1 = ["SP", "FC", "z", "FT"]
u_control47_2 = ["SP", "IC", "z", "IT"]
u_control47_3 = ["SP", "IC", "z", "IT"]
u_control47_4 = ["SP", "IC", "z", "IT"]

features_c47_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_1], dtype=torch.float32)
features_c47_2 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_2], dtype=torch.float32)
features_c47_3 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_3], dtype=torch.float32)
features_c47_4 = torch.tensor([[full_unit_lib[unit]] for unit in u_control47_4], dtype=torch.float32)

reg_u_c47_1 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c47_2 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c47_3 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)
reg_u_c47_4 = torch.tensor([[0], [2], [0], [0]], dtype=torch.float32)

control47_ufeat_stack = torch.vstack([
    pad_to_size(features_c47_1, (6, 1)),
    pad_to_size(features_c47_2, (6, 1)),
    pad_to_size(features_c47_3, (6, 1)),
    pad_to_size(features_c47_4, (6, 1)),
])
control47_reg_stack = torch.vstack([
    pad_to_size(reg_u_c47_1, (6, 1)),
    pad_to_size(reg_u_c47_2, (6, 1)),
    pad_to_size(reg_u_c47_3, (6, 1)),
    pad_to_size(reg_u_c47_4, (6, 1)),
])
control47_fullfeat_cat = torch.cat((control47_ufeat_stack, control47_reg_stack), dim=1)

# EX48 ---- INVENTORY CONTROL 3 ----
u_process48 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p48 = torch.tensor([[full_unit_lib[unit]] for unit in u_process48], dtype=torch.float32)

control48_1_adj = Dataset_general.control48_1_adj
control48_2_adj = Dataset_general.control48_2_adj
control48_3_adj = Dataset_general.control48_3_adj

control_adj_48_stack = torch.vstack([
    pad_to_size(control48_1_adj, (6, 6)),
    pad_to_size(control48_2_adj, (6, 6)),
    pad_to_size(control48_3_adj, (6, 6)),
])

u_control48_1 = ["SP", "IC", "z", "IT"]
u_control48_2 = ["SP", "IC", "z", "IT"]
u_control48_3 = ["SP", "IC", "z", "IT"]

features_c48_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control48_1]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control48_2]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control48_3]), (6, 1)),
])
reg_c48_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
])
control48_fullfeat_cat = torch.cat((features_c48_ufeat_stack, reg_c48_stack), dim=1)

# EX49 ---- INVENTORY CONTROL 4 ----
u_process49 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p49 = torch.tensor([[full_unit_lib[unit]] for unit in u_process49], dtype=torch.float32)

control49_1_adj = Dataset_general.control49_1_adj
control49_2_adj = Dataset_general.control49_2_adj
control49_3_adj = Dataset_general.control49_3_adj
control49_4_adj = Dataset_general.control49_4_adj

control_adj_49_stack = torch.vstack([
    pad_to_size(control49_1_adj, (6, 6)),
    pad_to_size(control49_2_adj, (6, 6)),
    pad_to_size(control49_3_adj, (6, 6)),
    pad_to_size(control49_4_adj, (6, 6)),
])

u_control49_1 = ["SP", "IC", "z", "IT"]
u_control49_2 = ["SP", "IC", "z", "IT"]
u_control49_3 = ["SP", "FC", "z", "FT"]
u_control49_4 = ["SP", "IC", "z", "IT"]

features_c49_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control49_1]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control49_2]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control49_3]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control49_4]), (6, 1)),
])
reg_c49_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
])
control49_fullfeat_cat = torch.cat((features_c49_ufeat_stack, reg_c49_stack), dim=1)

# EX50 ---- INVENTORY CONTROL 5 ----
u_process50 = ["valve", "tank", "valve", "tank", "valve", "tank", "valve"]
features_p50 = torch.tensor([[full_unit_lib[unit]] for unit in u_process50], dtype=torch.float32)

control50_1_adj = Dataset_general.control50_1_adj
control50_2_adj = Dataset_general.control50_2_adj
control50_3_adj = Dataset_general.control50_3_adj

control_adj_50_stack = torch.vstack([
    pad_to_size(control50_1_adj, (6, 6)),
    pad_to_size(control50_2_adj, (6, 6)),
    pad_to_size(control50_3_adj, (6, 6)),
])

u_control50_1 = ["SP", "IC", "z", "IT"]
u_control50_2 = ["SP", "IC", "z", "IT"]
u_control50_3 = ["SP", "IC", "z", "IT"]

features_c50_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control50_1]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control50_2]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control50_3]), (6, 1)),
])
reg_c50_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (6, 1)),
])
control50_fullfeat_cat = torch.cat((features_c50_ufeat_stack, reg_c50_stack), dim=1)

# EX51 ---- AIR + FUEL MIX ----
u_process51 = ["valve", "splitter", "valve"]
features_p51 = torch.tensor([[full_unit_lib[unit]] for unit in u_process51], dtype=torch.float32)

control51_1_adj = Dataset_general.control51_1_adj
control_adj_51_stack = torch.vstack([
    pad_to_size(control51_1_adj, (13, 13)),
])

u_control51_1 = ["SP", "X_multiplication_block", "MIN", "FC", "z", "FT", "MAX", "FC", "z", "PC", "SP", "FT", "SP"]
features_c51_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control51_1], dtype=torch.float32)
reg_u_c51_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [2], [0], [0], [0], [0], [0]], dtype=torch.float32)

control51_ufeat_stack = pad_to_size(features_c51_1, (13, 1))
control51_reg_stack = pad_to_size(reg_u_c51_1, (13, 1))
control51_fullfeat_cat = torch.cat((control51_ufeat_stack, control51_reg_stack), dim=1)

# EX52 ---- JACKETED CSTR 6 ----
u_process52 = ["jacket", "valve", "CSTR"]
features_p52 = torch.tensor([[full_unit_lib[unit]] for unit in u_process52], dtype=torch.float32)

control52_1_adj = Dataset_general.control52_1_adj
control_adj_52_stack = torch.vstack([
    pad_to_size(control52_1_adj, (10, 10)),
])

u_control52_1 = ["SP", "TC", "MIN", "FC", "z", "CC", "SP", "TT", "FT", "CT"]
features_c52_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control52_1], dtype=torch.float32)
reg_u_c52_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)

control52_ufeat_stack = pad_to_size(features_c52_1, (10, 1))
control52_reg_stack = pad_to_size(reg_u_c52_1, (10, 1))
control52_fullfeat_cat = torch.cat((control52_ufeat_stack, control52_reg_stack), dim=1)

# EX53 ---- MIXER ----
u_process53 = ["valve", "mixer", "compressor"]
features_p53 = torch.tensor([[full_unit_lib[unit]] for unit in u_process53], dtype=torch.float32)

control53_1_adj = Dataset_general.control53_1_adj
control53_2_adj = Dataset_general.control53_2_adj

control_adj_53_stack = torch.vstack([
    pad_to_size(control53_1_adj, (6, 6)),
    pad_to_size(control53_2_adj, (6, 6)),
])

u_control53_1 = ["SP", "CC", "FC", "z", "CT", "FT"]
u_control53_2 = ["SP", "FC", "FC", "w", "FT", "FT"]

features_c53_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control53_1]), (6, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control53_2]), (6, 1)),
])
reg_c53_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [0], [2], [0], [0], [0]]), (6, 1)),
    pad_to_size(torch.tensor([[0], [0], [6], [0], [0], [0]]), (6, 1)),
])
control53_fullfeat_cat = torch.cat((features_c53_ufeat_stack, reg_c53_stack), dim=1)

# EX54 ---- MIXER 2 ----
u_process54 = ["valve", "mixer", "compressor"]
features_p54 = torch.tensor([[full_unit_lib[unit]] for unit in u_process54], dtype=torch.float32)

control54_1_adj = Dataset_general.control54_1_adj
control_adj_54_stack = torch.vstack([
    pad_to_size(control54_1_adj, (13, 13))
])

u_control54_1 = ["SP", "FC", "MIN", "FC", "z", "SRC", "SP", "FC", "w", "FT", "FT", "CT", "FT"]
features_c54_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control54_1], dtype=torch.float32)
reg_u_c54_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [6], [0], [0], [0], [0], [0]], dtype=torch.float32)

control54_ufeat_stack = pad_to_size(features_c54_1, (13, 1))
control54_reg_stack = pad_to_size(reg_u_c54_1, (13, 1))
control54_fullfeat_cat = torch.cat((control54_ufeat_stack, control54_reg_stack), dim=1)

# EX55 ---- MIXER 3 ----
u_process55 = ["valve", "mixer", "compressor"]
features_p55 = torch.tensor([[full_unit_lib[unit]] for unit in u_process55], dtype=torch.float32)

control55_1_adj = Dataset_general.control55_1_adj
control_adj_55_stack = torch.vstack([
    pad_to_size(control55_1_adj, (15, 15))
])

u_control55_1 = ["SP", "FC", "MIN", "FC", "z", "VPC", "FC", "CC", "SP", "FT", "SP", "FT", "FT", "w", "CT"]
features_c55_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control55_1], dtype=torch.float32)
reg_u_c55_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [6], [6], [0], [0], [0], [0], [0], [0], [0]], dtype=torch.float32)

control55_ufeat_stack = pad_to_size(features_c55_1, (15, 1))
control55_reg_stack = pad_to_size(reg_u_c55_1, (15, 1))
control55_fullfeat_cat = torch.cat((control55_ufeat_stack, control55_reg_stack), dim=1)

# EX56 ---- DISTILLATION COLUMN 3 ----
u_process56 = ["valve", "column", "HX", "splitter", "valve", "splitter", "HX", "valve", "valve", "valve", "valve"]
features_p56 = torch.tensor([[full_unit_lib[unit]] for unit in u_process56], dtype=torch.float32)

control56_1_adj = Dataset_general.control56_1_adj
control56_2_adj = Dataset_general.control56_2_adj
control56_3_adj = Dataset_general.control56_3_adj
control56_4_adj = Dataset_general.control56_4_adj
control56_5_adj = Dataset_general.control56_5_adj

control_adj_56_stack = torch.vstack([
    pad_to_size(control56_1_adj, (12, 12)),
    pad_to_size(control56_2_adj, (12, 12)),
    pad_to_size(control56_3_adj, (12, 12)),
    pad_to_size(control56_4_adj, (12, 12)),
    pad_to_size(control56_5_adj, (12, 12))
])

u_control56_1 = ["SP", "FC", "z", "FT"]
u_control56_2 = ["SP", "CC", "SRC", "MIN", "z", "CC", "MAX", "SP", "CT", "z", "CT", "SP"]
u_control56_3 = ["SP", "PC", "z", "PT"]
u_control56_4 = ["SP", "LC", "z", "LT"]
u_control56_5 = ["SP", "LC", "z", "LT"]

features_c56_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control56_1]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control56_2]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control56_3]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control56_4]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[unit]] for unit in u_control56_5]), (12, 1)),
])
reg_c56_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [0], [2], [2], [0], [0], [0], [0], [0], [0], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (12, 1)),
])
control56_fullfeat_cat = torch.cat((features_c56_ufeat_stack, reg_c56_stack), dim=1)

# EX57 ---- DISTILLATION COLUMN 4 ----
u_process57 = ["valve", "column", "HX", "splitter", "valve", "splitter", "HX", "valve", "valve", "valve", "valve"]
features_p57 = torch.tensor([[full_unit_lib[unit]] for unit in u_process57], dtype=torch.float32)

control57_1_adj = Dataset_general.control57_1_adj
control57_2_adj = Dataset_general.control57_2_adj
control57_3_adj = Dataset_general.control57_3_adj
control57_4_adj = Dataset_general.control57_4_adj
control57_5_adj = Dataset_general.control57_5_adj

control_adj_57_stack = torch.vstack([
    pad_to_size(control57_1_adj, (12, 12)),
    pad_to_size(control57_2_adj, (12, 12)),
    pad_to_size(control57_3_adj, (12, 12)),
    pad_to_size(control57_4_adj, (12, 12)),
    pad_to_size(control57_5_adj, (12, 12))
])

u_control57_1 = ["SP", "CC", "MIN", "FC", "z", "CT", "SP", "FT"]
u_control57_2 = ["SP", "CC", "SRC", "MIN", "z", "CC", "MAX", "SP", "CT", "z", "CT", "SP"]
u_control57_3 = ["SP", "PC", "z", "PT"]
u_control57_4 = ["SP", "LC", "z", "LT"]
u_control57_5 = ["SP", "LC", "z", "LT"]

features_c57_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control57_1]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control57_2]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control57_3]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control57_4]), (12, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control57_5]), (12, 1)),
])
reg_c57_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [0], [2], [2], [0], [0], [0], [0], [0], [0], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (12, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (12, 1)),
])
control57_fullfeat_cat = torch.cat((features_c57_ufeat_stack, reg_c57_stack), dim=1)

# EX58 ---- CO2 REFRIGERATION ----
u_process58 = ["evaporator", "splitter", "compressor", "splitter", "splitter", "HX", "valve", "cooler",
               "valve", "liq_reciever", "splitter", "valve", "valve", "compressor"]
features_p58 = torch.tensor([[full_unit_lib[unit]] for unit in u_process58], dtype=torch.float32)

control58_1_adj = Dataset_general.control58_1_adj
control58_2_adj = Dataset_general.control58_2_adj
control58_3_adj = Dataset_general.control58_3_adj
control58_4_adj = Dataset_general.control58_4_adj
control58_5_adj = Dataset_general.control58_5_adj

control_adj_58_stack = torch.vstack([
    pad_to_size(control58_1_adj, (8, 8)),
    pad_to_size(control58_2_adj, (8, 8)),
    pad_to_size(control58_3_adj, (8, 8)),
    pad_to_size(control58_4_adj, (8, 8)),
    pad_to_size(control58_5_adj, (8, 8))
])

u_control58_1 = ["SP", "TC", "z", "TT"]
u_control58_2 = ["SP", "PC", "w", "PT"]
u_control58_3 = ["SP", "TC", "z", "TT"]
u_control58_4 = ["SP", "TC", "w", "TT", "TC", "PC", "z", "PT"]
u_control58_5 = ["SP", "PC", "w", "PT"]

features_c58_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control58_1]), (8, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control58_2]), (8, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control58_3]), (8, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control58_4]), (8, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control58_5]), (8, 1)),
])
reg_c58_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (8, 1)),
    pad_to_size(torch.tensor([[0], [6], [0], [0]]), (8, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (8, 1)),
    pad_to_size(torch.tensor([[0], [6], [0], [0], [0], [2], [0], [0]]), (8, 1)),
    pad_to_size(torch.tensor([[0], [6], [0], [0]]), (8, 1)),
])
control58_fullfeat_cat = torch.cat((features_c58_ufeat_stack, reg_c58_stack), dim=1)

# EX59 ---- HEAT EXCHANGER 2 ----
u_process59 = ["HX", "valve", "valve"]
features_p59 = torch.tensor([[full_unit_lib[unit]] for unit in u_process59], dtype=torch.float32)

control59_1_adj = Dataset_general.control59_1_adj
control59_2_adj = Dataset_general.control59_2_adj
control_adj_59_stack = torch.vstack([
    pad_to_size(control59_1_adj, (4, 4)),
    pad_to_size(control59_2_adj, (4, 4))
])

u_control59_1 = ["SP", "FC", "z", "FT"]
u_control59_2 = ["SP", "TC", "z", "TT"]

features_c59_ufeat_stack = torch.vstack([
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control59_1]), (4, 1)),
    pad_to_size(torch.tensor([[full_unit_lib[u]] for u in u_control59_2]), (4, 1)),
])
reg_c59_stack = torch.vstack([
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
    pad_to_size(torch.tensor([[0], [2], [0], [0]]), (4, 1)),
])
control59_fullfeat_cat = torch.cat((features_c59_ufeat_stack, reg_c59_stack), dim=1)

# EX60 ---- HEAT EXCHANGER 3 ----
u_process60 = ["HX", "valve", "valve"]
features_p60 = torch.tensor([[full_unit_lib[unit]] for unit in u_process60], dtype=torch.float32)

control60_1_adj = Dataset_general.control60_1_adj
control_adj_60_stack = torch.vstack([
    pad_to_size(control60_1_adj, (8, 8))
])

u_control60_1 = ["SP", "TC", "SRC", "MIN", "z", "TT", "z", "SP"]
features_c60_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control60_1], dtype=torch.float32)
reg_u_c60_1 = torch.tensor([[0], [0], [2], [2], [0], [0], [0], [0]], dtype=torch.float32)

control60_ufeat_stack = pad_to_size(features_c60_1, (8, 1))
control60_reg_stack = pad_to_size(reg_u_c60_1, (8, 1))
control60_fullfeat_cat = torch.cat((control60_ufeat_stack, control60_reg_stack), dim=1)

# EX61 ---- HEAT EXCHANGER 4 ----
u_process61 = ["HX", "valve", "valve"]
features_p61 = torch.tensor([[full_unit_lib[unit]] for unit in u_process61], dtype=torch.float32)

control61_1_adj = Dataset_general.control61_1_adj
control_adj_61_stack = torch.vstack([
    pad_to_size(control61_1_adj, (7, 7))
])

u_control61_1 = ["SP", "TC", "MIN", "z", "TT", "z", "SP"]
features_c61_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control61_1], dtype=torch.float32)
reg_u_c61_1 = torch.tensor([[0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)

control61_ufeat_stack = torch.vstack([
    pad_to_size(features_c61_1, (7, 1))
])
control61_reg_stack = torch.vstack([
    pad_to_size(reg_u_c61_1, (7, 1))
])
control61_fullfeat_cat = torch.cat((control61_ufeat_stack, control61_reg_stack), dim=1)

# EX62 ---- HEAT EXCHANGER 5 ----
u_process62 = ["HX", "valve", "valve"]
features_p62 = torch.tensor([[full_unit_lib[unit]] for unit in u_process62], dtype=torch.float32)

control62_1_adj = Dataset_general.control62_1_adj
control_adj_62_stack = torch.vstack([
    pad_to_size(control62_1_adj, (9, 9))
])

u_control62_1 = ["SP", "TC", "MIN", "z", "TT", "TC", "z", "SP", "SP"]
features_c62_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control62_1], dtype=torch.float32)
reg_u_c62_1 = torch.tensor([[0], [0], [2], [0], [0], [2], [0], [0], [0]], dtype=torch.float32)

control62_ufeat_stack = torch.vstack([
    pad_to_size(features_c62_1, (9, 1))
])
control62_reg_stack = torch.vstack([
    pad_to_size(reg_u_c62_1, (9, 1))
])
control62_fullfeat_cat = torch.cat((control62_ufeat_stack, control62_reg_stack), dim=1)

# EX63 ---- TANK 9 ----
u_process63 = ["tank", "valve"]
features_p63 = torch.tensor([[full_unit_lib[unit]] for unit in u_process63], dtype=torch.float32)

control63_1_adj = Dataset_general.control63_1_adj
control_adj_63_stack = torch.vstack([
    pad_to_size(control63_1_adj, (8, 8))
])

u_control63_1 = ["SP", "LC", "subtraction_block", "FC", "z", "LT", "FT", "FT"]
features_c63_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control63_1], dtype=torch.float32)
reg_u_c63_1 = torch.tensor([[0], [0], [0], [2], [0], [0], [0], [0]], dtype=torch.float32)

control63_ufeat_stack = torch.vstack([
    pad_to_size(features_c63_1, (8, 1))
])
control63_reg_stack = torch.vstack([
    pad_to_size(reg_u_c63_1, (8, 1))
])
control63_fullfeat_cat = torch.cat((control63_ufeat_stack, control63_reg_stack), dim=1)

# EX64 ---- TURBINE ----
u_process64 = ["splitter", "turbine", "valve", "splitter", "splitter", "valve"]
features_p64 = torch.tensor([[full_unit_lib[unit]] for unit in u_process64], dtype=torch.float32)

control64_1_adj = Dataset_general.control64_1_adj
control_adj_64_stack = torch.vstack([
    pad_to_size(control64_1_adj, (7, 7))
])

u_control64_1 = ["SP", "PC", "z", "PT", "PC", "z", "SP"]
features_c64_1 = torch.tensor([[full_unit_lib[unit]] for unit in u_control64_1], dtype=torch.float32)
reg_u_c64_1 = torch.tensor([[0], [2], [0], [0], [2], [0], [0]], dtype=torch.float32)

control64_ufeat_stack = torch.vstack([
    pad_to_size(features_c64_1, (7, 1))
])
control64_reg_stack = torch.vstack([
    pad_to_size(reg_u_c64_1, (7, 1))
])
control64_fullfeat_cat = torch.cat((control64_ufeat_stack, control64_reg_stack), dim=1)




MAX_INPUT_N = max(features.shape[0] for features in [
    features_p1,
    features_p2,
    features_p3,
    features_p4,
    features_p5,
    features_p6,
    features_p7,
    features_p8,
    features_p9,
    features_p10,
    features_p11,
    features_p12,
    features_p13,
    features_p14,
    features_p15,
    features_p16,
    features_p17,
    features_p18,
    features_p19,
    features_p20,
    features_p21,
    features_p22,
    features_p23,
    features_p24,
    features_p25,
    features_p26,
    features_p27,
    features_p28,
    features_p29,
    features_p30,
    #features_p31,
    features_p32,
    features_p33,
    features_p34,
    features_p35,
    features_p36,
    features_p37,
    features_p38,
    features_p39,
    features_p40,
    features_p41,
    features_p42,
    features_p43,
    features_p44,
    features_p45,
    features_p46,
    features_p47,
    features_p48,
    features_p49,
    features_p50,
    features_p51,
    features_p52,
    features_p53,
    features_p54,
    features_p55,
    features_p56,
    features_p57,
    features_p58,
    features_p59,
    features_p60,
    features_p61,
    features_p62,
    features_p63,
    features_p64,
])

print("MAX INPUT N: ", MAX_INPUT_N)









MAX_N = max(control_adj.shape[0] for control_adj in [
    control_adj_1_stack,
    control_adj_2_stack,
    control_adj_3_stack,
    control_adj_4_stack,
    control_adj_5_stack,
    control_adj_6_stack,
    control_adj_7_stack,
    control_adj_8_stack,
    control_adj_9_stack,
    control_adj_10_stack,
    control_adj_11_stack,
    control_adj_12_stack,
    control_adj_13_stack,
    control_adj_14_stack,
    control_adj_15_stack,
    control_adj_16_stack,
    control_adj_17_stack,
    control_adj_18_stack,
    control_adj_19_stack,
    control_adj_20_stack,
    control_adj_21_stack,
    control_adj_22_stack,
    control_adj_23_stack,
    control_adj_24_stack,
    control_adj_25_stack,
    control_adj_26_stack,
    control_adj_27_stack,
    control_adj_28_stack,
    control_adj_29_stack,
    control_adj_30_stack,
    #control_adj_31_stack,
    control_adj_32_stack,
    control_adj_33_stack,
    control_adj_34_stack,
    control_adj_35_stack,
    control_adj_36_stack,
    control_adj_37_stack,
    control_adj_38_stack,
    control_adj_39_stack,
    control_adj_40_stack,
    control_adj_41_stack,
    control_adj_42_stack,
    control_adj_43_stack,
    control_adj_44_stack,
    control_adj_45_stack,
    control_adj_46_stack,
    control_adj_47_stack,
    control_adj_48_stack,
    control_adj_49_stack,
    control_adj_50_stack,
    control_adj_51_stack,
    control_adj_52_stack,
    control_adj_53_stack,
    control_adj_54_stack,
    control_adj_55_stack,
    #control_adj_56_stack,
    #control_adj_57_stack,
    #control_adj_58_stack,
    control_adj_59_stack,
    control_adj_60_stack,
    control_adj_61_stack,
    control_adj_62_stack,
    control_adj_63_stack,
    control_adj_64_stack,
])




def prepare_graph_example(x_raw, adj_raw, control_raw, control_feature_raw):
    # x_raw: [N, 1], adj_raw: [N, N], control_raw: [Nc, Nc]
    #x = pad_to_size(x_raw, (126, 1))
    pad_size_process = 20
    pad_size_control = 15
    x = pad_to_size(x_raw, (pad_size_process, 1))
    #adj = pad_to_size(adj_raw, (126, 126))
    adj = pad_to_size(adj_raw, (pad_size_process, pad_size_process))
    
    control = pad_to_size(control_raw, (pad_size_control, pad_size_control))  # make square
    
    control_feature = pad_to_size(control_feature_raw, (pad_size_control, 2))
    #control = pad_to_size(control_raw, (pad_size, pad_size))
    return (x, adj, control, control_feature)


print("MAX N: ", MAX_N)
train_data = [
    #prepare_graph_example(features_p1, Dataset_general.process1_adj, control_adj_1_stack, control1_fullfeat_cat),
    #prepare_graph_example(features_p2, Dataset_general.process2_adj, control_adj_2_stack, control2_fullfeat_cat),
    #prepare_graph_example(features_p3, Dataset_general.process3_adj, control_adj_3_stack, control3_fullfeat_cat),
    prepare_graph_example(features_p4, Dataset_general.process4_adj, control_adj_4_stack, control4_fullfeat_cat),
    prepare_graph_example(features_p5, Dataset_general.process5_adj, control_adj_5_stack, control5_fullfeat_cat),
    #prepare_graph_example(features_p6, Dataset_general.process6_adj, control_adj_6_stack, control6_fullfeat_cat),
    prepare_graph_example(features_p7, Dataset_general.process7_adj, control_adj_7_stack, control7_fullfeat_cat),
    #prepare_graph_example(features_p8, Dataset_general.process8_adj, control_adj_8_stack, control8_fullfeat_cat),
    prepare_graph_example(features_p9, Dataset_general.process9_adj, control_adj_9_stack, control9_fullfeat_cat),
    prepare_graph_example(features_p10, Dataset_general.process10_adj, control_adj_10_stack, control10_fullfeat_cat),
    #prepare_graph_example(features_p11, Dataset_general.process11_adj, control_adj_11_stack, control11_fullfeat_cat),
    prepare_graph_example(features_p12, Dataset_general.process12_adj, control_adj_12_stack, control12_fullfeat_cat),
    prepare_graph_example(features_p13, Dataset_general.process13_adj, control_adj_13_stack, control13_fullfeat_cat),
    prepare_graph_example(features_p14, Dataset_general.process14_adj, control_adj_14_stack, control14_fullfeat_cat),
    prepare_graph_example(features_p15, Dataset_general.process15_adj, control_adj_15_stack, control15_fullfeat_cat),
    prepare_graph_example(features_p16, Dataset_general.process16_adj, control_adj_16_stack, control16_fullfeat_cat),
    prepare_graph_example(features_p17, Dataset_general.process17_adj, control_adj_17_stack, control17_fullfeat_cat),
    prepare_graph_example(features_p18, Dataset_general.process18_adj, control_adj_18_stack, control18_fullfeat_cat),
    prepare_graph_example(features_p19, Dataset_general.process19_adj, control_adj_19_stack, control19_fullfeat_cat),
    prepare_graph_example(features_p20, Dataset_general.process20_adj, control_adj_20_stack, control20_fullfeat_cat),
    prepare_graph_example(features_p21, Dataset_general.process21_adj, control_adj_21_stack, control21_fullfeat_cat),
    prepare_graph_example(features_p22, Dataset_general.process22_adj, control_adj_22_stack, control22_fullfeat_cat),
    prepare_graph_example(features_p23, Dataset_general.process23_adj, control_adj_23_stack, control23_fullfeat_cat),
    prepare_graph_example(features_p24, Dataset_general.process24_adj, control_adj_24_stack, control24_fullfeat_cat),
    prepare_graph_example(features_p25, Dataset_general.process25_adj, control_adj_25_stack, control25_fullfeat_cat),
    #prepare_graph_example(features_p26, Dataset_general.process26_adj, control_adj_26_stack, control26_fullfeat_cat),
    #prepare_graph_example(features_p27, Dataset_general.process27_adj, control_adj_27_stack, control27_fullfeat_cat),
    #prepare_graph_example(features_p28, Dataset_general.process28_adj, control_adj_28_stack, control28_fullfeat_cat),
    #prepare_graph_example(features_p29, Dataset_general.process29_adj, control_adj_29_stack, control29_fullfeat_cat),
    prepare_graph_example(features_p30, Dataset_general.process30_adj, control_adj_30_stack, control30_fullfeat_cat),
    prepare_graph_example(features_p32, Dataset_general.process32_adj, control_adj_32_stack, control32_fullfeat_cat),
    #prepare_graph_example(features_p33, Dataset_general.process33_adj, control_adj_33_stack, control33_fullfeat_cat),
    prepare_graph_example(features_p34, Dataset_general.process34_adj, control_adj_34_stack, control34_fullfeat_cat),
    prepare_graph_example(features_p35, Dataset_general.process35_adj, control_adj_35_stack, control35_fullfeat_cat),
    prepare_graph_example(features_p36, Dataset_general.process36_adj, control_adj_36_stack, control36_fullfeat_cat),
    prepare_graph_example(features_p37, Dataset_general.process37_adj, control_adj_37_stack, control37_fullfeat_cat),
    prepare_graph_example(features_p38, Dataset_general.process38_adj, control_adj_38_stack, control38_fullfeat_cat),
    prepare_graph_example(features_p39, Dataset_general.process39_adj, control_adj_39_stack, control39_fullfeat_cat),
    prepare_graph_example(features_p40, Dataset_general.process40_adj, control_adj_40_stack, control40_fullfeat_cat),
    #prepare_graph_example(features_p41, Dataset_general.process41_adj, control_adj_41_stack, control41_fullfeat_cat),
    #prepare_graph_example(features_p42, Dataset_general.process42_adj, control_adj_42_stack, control42_fullfeat_cat),
    #prepare_graph_example(features_p43, Dataset_general.process43_adj, control_adj_43_stack, control43_fullfeat_cat),
    prepare_graph_example(features_p44, Dataset_general.process44_adj, control_adj_44_stack, control44_fullfeat_cat),
    prepare_graph_example(features_p45, Dataset_general.process45_adj, control_adj_45_stack, control45_fullfeat_cat),
    #prepare_graph_example(features_p46, Dataset_general.process46_adj, control_adj_46_stack, control46_fullfeat_cat),
    #prepare_graph_example(features_p47, Dataset_general.process47_adj, control_adj_47_stack, control47_fullfeat_cat),
    #prepare_graph_example(features_p48, Dataset_general.process48_adj, control_adj_48_stack, control48_fullfeat_cat),
    #prepare_graph_example(features_p49, Dataset_general.process49_adj, control_adj_49_stack, control49_fullfeat_cat),
    #prepare_graph_example(features_p59, Dataset_general.process59_adj, control_adj_59_stack, control59_fullfeat_cat),
    #prepare_graph_example(features_p50, Dataset_general.process50_adj, control_adj_50_stack, control50_fullfeat_cat),
    ]


for i, (_, _, control_adj, _) in enumerate(train_data):
    num_elements = control_adj.numel()  # Total number of elements in the matrix
    shape = control_adj.shape           # Shape of the matrix
    print(f"Sample {i}: control_adj shape = {shape}, total elements = {num_elements}")


for i, (_, _, control_adj, _) in enumerate(train_data):
    col_zero = torch.all(control_adj[:, 0] == 0).item()  # Check if column 0 is all zeros
    status = "✅ column[0] is all zero" if col_zero else "❌ column[0] has non-zero entries"
    print(f"Sample {i}: {status}")

"""

train_data = [
    prepare_graph_example(features_p1, Dataset_general.process1_adj, control_adj_1_stack, control1_fullfeat_cat),
    prepare_graph_example(features_p2, Dataset_general.process2_adj, control_adj_2_stack, control2_fullfeat_cat),
    prepare_graph_example(features_p3, Dataset_general.process3_adj, control_adj_3_stack, control3_fullfeat_cat),
    prepare_graph_example(features_p4, Dataset_general.process4_adj, control_adj_4_stack, control4_fullfeat_cat),
    prepare_graph_example(features_p5, Dataset_general.process5_adj, control_adj_5_stack, control5_fullfeat_cat),
    prepare_graph_example(features_p6, Dataset_general.process6_adj, control_adj_6_stack, control6_fullfeat_cat),
    prepare_graph_example(features_p7, Dataset_general.process7_adj, control_adj_7_stack, control7_fullfeat_cat),
    prepare_graph_example(features_p8, Dataset_general.process8_adj, control_adj_8_stack, control8_fullfeat_cat),
    prepare_graph_example(features_p9, Dataset_general.process9_adj, control_adj_9_stack, control9_fullfeat_cat),
    prepare_graph_example(features_p10, Dataset_general.process10_adj, control_adj_10_stack, control10_fullfeat_cat),
    prepare_graph_example(features_p11, Dataset_general.process11_adj, control_adj_11_stack, control11_fullfeat_cat),
    prepare_graph_example(features_p12, Dataset_general.process12_adj, control_adj_12_stack, control12_fullfeat_cat),
    prepare_graph_example(features_p13, Dataset_general.process13_adj, control_adj_13_stack, control13_fullfeat_cat),
    prepare_graph_example(features_p14, Dataset_general.process14_adj, control_adj_14_stack, control14_fullfeat_cat),
    prepare_graph_example(features_p15, Dataset_general.process15_adj, control_adj_15_stack, control15_fullfeat_cat),
    prepare_graph_example(features_p16, Dataset_general.process16_adj, control_adj_16_stack, control16_fullfeat_cat),
    prepare_graph_example(features_p17, Dataset_general.process17_adj, control_adj_17_stack, control17_fullfeat_cat),
    prepare_graph_example(features_p18, Dataset_general.process18_adj, control_adj_18_stack, control18_fullfeat_cat),
    prepare_graph_example(features_p19, Dataset_general.process19_adj, control_adj_19_stack, control19_fullfeat_cat),
    prepare_graph_example(features_p20, Dataset_general.process20_adj, control_adj_20_stack, control20_fullfeat_cat),
    prepare_graph_example(features_p21, Dataset_general.process21_adj, control_adj_21_stack, control21_fullfeat_cat),
    prepare_graph_example(features_p22, Dataset_general.process22_adj, control_adj_22_stack, control22_fullfeat_cat),
    prepare_graph_example(features_p23, Dataset_general.process23_adj, control_adj_23_stack, control23_fullfeat_cat),
    prepare_graph_example(features_p24, Dataset_general.process24_adj, control_adj_24_stack, control24_fullfeat_cat),
    prepare_graph_example(features_p25, Dataset_general.process25_adj, control_adj_25_stack, control25_fullfeat_cat),
    prepare_graph_example(features_p26, Dataset_general.process26_adj, control_adj_26_stack, control26_fullfeat_cat),
    prepare_graph_example(features_p27, Dataset_general.process27_adj, control_adj_27_stack, control27_fullfeat_cat),
    prepare_graph_example(features_p28, Dataset_general.process28_adj, control_adj_28_stack, control28_fullfeat_cat),
    prepare_graph_example(features_p29, Dataset_general.process29_adj, control_adj_29_stack, control29_fullfeat_cat),
    prepare_graph_example(features_p30, Dataset_general.process30_adj, control_adj_30_stack, control30_fullfeat_cat),
    prepare_graph_example(features_p32, Dataset_general.process32_adj, control_adj_32_stack, control32_fullfeat_cat),
    prepare_graph_example(features_p33, Dataset_general.process33_adj, control_adj_33_stack, control33_fullfeat_cat),
    prepare_graph_example(features_p34, Dataset_general.process34_adj, control_adj_34_stack, control34_fullfeat_cat),
    prepare_graph_example(features_p35, Dataset_general.process35_adj, control_adj_35_stack, control35_fullfeat_cat),
    prepare_graph_example(features_p36, Dataset_general.process36_adj, control_adj_36_stack, control36_fullfeat_cat),
    prepare_graph_example(features_p37, Dataset_general.process37_adj, control_adj_37_stack, control37_fullfeat_cat),
    prepare_graph_example(features_p38, Dataset_general.process38_adj, control_adj_38_stack, control38_fullfeat_cat),
    prepare_graph_example(features_p39, Dataset_general.process39_adj, control_adj_39_stack, control39_fullfeat_cat),
    prepare_graph_example(features_p40, Dataset_general.process40_adj, control_adj_40_stack, control40_fullfeat_cat),
    prepare_graph_example(features_p41, Dataset_general.process41_adj, control_adj_41_stack, control41_fullfeat_cat),
    prepare_graph_example(features_p42, Dataset_general.process42_adj, control_adj_42_stack, control42_fullfeat_cat),
    prepare_graph_example(features_p43, Dataset_general.process43_adj, control_adj_43_stack, control43_fullfeat_cat),
    prepare_graph_example(features_p44, Dataset_general.process44_adj, control_adj_44_stack, control44_fullfeat_cat),
    prepare_graph_example(features_p45, Dataset_general.process45_adj, control_adj_45_stack, control45_fullfeat_cat),
    prepare_graph_example(features_p46, Dataset_general.process46_adj, control_adj_46_stack, control46_fullfeat_cat),
    prepare_graph_example(features_p47, Dataset_general.process47_adj, control_adj_47_stack, control47_fullfeat_cat),
    prepare_graph_example(features_p48, Dataset_general.process48_adj, control_adj_48_stack, control48_fullfeat_cat),
    prepare_graph_example(features_p49, Dataset_general.process49_adj, control_adj_49_stack, control49_fullfeat_cat),
    prepare_graph_example(features_p50, Dataset_general.process50_adj, control_adj_50_stack, control50_fullfeat_cat),]

"""









test_data = [
    prepare_graph_example(features_p51, Dataset_general.process51_adj, control_adj_51_stack, control51_fullfeat_cat),
    prepare_graph_example(features_p52, Dataset_general.process52_adj, control_adj_52_stack, control52_fullfeat_cat),
    prepare_graph_example(features_p53, Dataset_general.process53_adj, control_adj_53_stack, control53_fullfeat_cat),
    prepare_graph_example(features_p54, Dataset_general.process54_adj, control_adj_54_stack, control54_fullfeat_cat),
    prepare_graph_example(features_p55, Dataset_general.process55_adj, control_adj_55_stack, control55_fullfeat_cat),
    #prepare_graph_example(features_p56, Dataset_general.process56_adj, control_adj_56_stack, control56_fullfeat_cat),
    #prepare_graph_example(features_p57, Dataset_general.process57_adj, control_adj_57_stack, control57_fullfeat_cat),
    #prepare_graph_example(features_p58, Dataset_general.process58_adj, control_adj_58_stack, control58_fullfeat_cat),
    prepare_graph_example(features_p59, Dataset_general.process59_adj, control_adj_59_stack, control59_fullfeat_cat),
    prepare_graph_example(features_p60, Dataset_general.process60_adj, control_adj_60_stack, control60_fullfeat_cat),
    prepare_graph_example(features_p61, Dataset_general.process61_adj, control_adj_61_stack, control61_fullfeat_cat),
    prepare_graph_example(features_p62, Dataset_general.process62_adj, control_adj_62_stack, control62_fullfeat_cat),
    prepare_graph_example(features_p63, Dataset_general.process63_adj, control_adj_63_stack, control63_fullfeat_cat),
    prepare_graph_example(features_p64, Dataset_general.process64_adj, control_adj_64_stack, control64_fullfeat_cat),
]


















