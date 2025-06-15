import torch

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

# Control: 
control1_1_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control1_2_adj= torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP (L/F)_s
[0, 0, 1, 0, 0, 0],  #2 X multiplication block
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 1],  #4 z
[0, 1, 0, 0, 0, 0],  #5 FT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)

control1_3_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control1_4_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control1_5_adj= torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP (V/F)_s
[0, 0, 1, 0, 0, 0],  #2 X multiplication block
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 1],  #4 z
[0, 1, 0, 0, 0, 0],  #5 FT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)

control1_6_adj= torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


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

control2_2_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP 
[0, 0, 1, 0, 0, 0],  #2 TC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 z (valve position) 
[0, 1, 0, 0, 0, 0],  #5 TT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)



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

# Control:
control3_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 1],  #3 z (valve position)
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control3_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z (valve position)
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)


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


#------------------------------------------------------------------
# EX5 ---- SHELL & TUBE HX ----
# Process: 
process5_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 HX
], dtype=torch.float32)


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

#Control:
control6_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control6_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX7 ----JACKETED CSTR 2 ----
# Process: 
process7_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve 
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)

# Control: 
control7_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)



#------------------------------------------------------------------
# EX8 ----DISTILLATION COLUMN 2 ----
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


#Control:
control8_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)

control8_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)

control8_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control8_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX9 ----SHELL & TUBE HX 2 ----
# Process:
process9_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 HX
], dtype=torch.float32)

#Control:
control9_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX10 ----HEAT EXCHANGER ----
# Process: 
process10_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 HX
[0, 1, 0],  #3 Valve
], dtype=torch.float32)

#Control:
control10_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX11 ----FILTER AND DECANTATION ----
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


#Control:
control11_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 w             
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control11_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 w             
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX12 ----TANK 1 ----
# Process: 
process12_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Tank
], dtype=torch.float32)

#Control:
control12_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX13 ----TANK 2 ----
# Process: 
process13_adj = torch.tensor([
#1  2  
[0, 1],  #1 Tank   
[0, 0],  #2 Valve
], dtype=torch.float32)

#Control:
control13_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX14 ----TANK 3 ----
# Process: 
process14_adj = torch.tensor([
#1  2  
[0, 1],  #1 Tank   
[0, 0],  #2 Valve
], dtype=torch.float32)

#Control:
control14_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX15 ----TANK 4 ----
# Process: 
process15_adj = torch.tensor([
#1  2  
[0, 1],  #1 Tank   
[0, 0],  #2 Valve
], dtype=torch.float32)

#Control:
control15_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 LC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 0],  #5 LT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX16 ----ADIABATIC REACTOR ----
# Process: 
process16_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 HX
[0, 1, 0],  #3 Adiabatic reactor
], dtype=torch.float32)

#Control:
control16_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX17 ---- CSTR ----
# Process: 
process17_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 CSTR
], dtype=torch.float32)

#Control:
control17_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX18 ---- FURNACE ----
# Process: 
process18_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Furnace
], dtype=torch.float32)

#Control:
control18_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX19 ---- CASCADE FURNACE ----
# Process:
process19_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Furnace
], dtype=torch.float32)

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


#------------------------------------------------------------------
# EX20 ---- CASCADE CSTR ----
# Process: 
process20_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)

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


#------------------------------------------------------------------
# EX21 ---- FURNACE 2 ----
# Process: 
process21_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve   
[0, 0],  #2 Furnace
], dtype=torch.float32)

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


#------------------------------------------------------------------
# EX22 ---- JACKETED CSTR 3 ----
# Process: 
process22_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)

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


#------------------------------------------------------------------
# EX23 ---- JACKETED CSTR 4 ----
# Process: 
process23_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Jacket
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)

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


#------------------------------------------------------------------
# EX25 ---- BLEND STREAM ----
# Process: 
process25_adj = torch.tensor([
#1  2 
[0, 1], #1 Valve
[0, 0], #2 Splitter
], dtype=torch.float32)

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

# Control:
control26_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control26_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control26_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)

control26_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)



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

# Control:
control27_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control27_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control27_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)

control27_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)



#------------------------------------------------------------------
# EX28 ---- CSTR + DISTILLATION----
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

# Control:                 

control28_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control28_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control28_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)

control28_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control28_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)

control28_6_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control28_7_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX29 ----TANK 5 ----
# Process: 
process29_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 Valve
[0, 0, 1, 0],  #2 Tank
[0, 0, 0, 1],  #3 Pump
[0, 0, 0, 0],  #4 Valve
], dtype=torch.float32)


#Control:
control29_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control29_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX30 ----BYPASS HX ----
# Process: 
process30_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 1],  #1 Splitter
[0, 0, 1, 0],  #2 HX
[0, 0, 0, 0],  #3 Splitter
[0, 0, 1, 0],  #4 Valve
], dtype=torch.float32)


#Control:
control30_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
"""
# EX31 ----BYPASS HX 2---- DELETE THIS EXAMPLE 
# Process: 
process31_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 1],  #1 Splitter
[0, 0, 1, 0],  #2 HX
[0, 0, 0, 0],  #3 Splitter
[0, 0, 1, 0],  #4 Valve
], dtype=torch.float32)


#Control:
control31_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)

"""

#------------------------------------------------------------------
# EX32 ----REACTOR + HX----
# Process: 
process32_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 1],  #1 Valve 
[0, 0, 1, 0, 0],  #2 HX
[0, 0, 0, 1, 0],  #3 Splitter
[0, 1, 0, 0, 0],  #4 Reactor
[0, 0, 1, 0, 0],  #5 Valve
], dtype=torch.float32)


#Control:
control32_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX33 ---- CSTR + DISTILLATION 2----
# Process: 
process33_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20  
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #1 Valve 
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #2 Splitter
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #3 Splitter
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #4 CSTR
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  #6 Column
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #7 HX
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #8 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #9 Condenser
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #10 Pump
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  #11 Splitter
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #12 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  #13 Valve
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #14 Jacket
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],  #15 Splitter
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #16 HX
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #17 Valve
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #18 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #19 Valve
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  #20 Valve
], dtype=torch.float32)


# Control: 
control33_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control33_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control33_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control33_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control33_5_adj = torch.tensor([
#1  2  3  4                             
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z                    
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)

control33_6_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control33_7_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control33_8_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control33_9_adj = torch.tensor([
#1  2  3  4                             
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z                    
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX34 ---- TANK 6 ----
# Process: 
process34_adj = torch.tensor([
#1  2  3  
[0, 0, 0],  #1 Tank
[0, 0, 0],  #2 Pump
[0, 0, 0],  #3 Valve
], dtype=torch.float32)


# Control: 
control34_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX35 ---- CSTR 2 ----
# Process: 
process35_adj = torch.tensor([
#1  2  
[0, 0],  #1 Valve 
[0, 0],  #2 CSTR
], dtype=torch.float32)


# Control: 
control35_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX36 ---- JACKETED CSTR 5  ----
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


#------------------------------------------------------------------
# EX37 ---- STEAM DRUM HX  ----
# Process: 
process37_adj = torch.tensor([
#1  2  
[0, 1],  #1 Valve
[0, 0],  #2 Steam drum HX
], dtype=torch.float32)


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


#------------------------------------------------------------------
# EX38 ---- TANK 7  ----
# Process: 
process38_adj = torch.tensor([
#1  2  3  
[0, 0, 0],  #1 Tank
[0, 0, 0],  #2 Pump
[0, 0, 0],  #3 Valve
], dtype=torch.float32)


# Control: 
control38_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX39 ---- VPC REACTOR  ----
# Process: 
process39_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0],  #2 HX
[0, 1, 0, 1, 0],  #3 Jacket
[0, 0, 0, 0, 0],  #4 Reactor
[0, 0, 1, 0, 0],  #5 Valve
], dtype=torch.float32)

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


#------------------------------------------------------------------
# EX40 ---- TOP COLUMN  ----
# Process:
process40_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 0],  #1 Column
[0, 0, 1, 0, 0],  #2 HX
[0, 1, 0, 1, 0],  #3 Condensor
[1, 0, 0, 0, 0],  #4 Splitter
[0, 1, 0, 0, 0],  #5 Valve
], dtype=torch.float32)


# Control: 
control40_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX41 ---- TOP COLUMN 2  ----
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


# Control: 
control41_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 CC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 CT
], dtype=torch.float32)

control41_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control41_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX42 ---- SETTLING TANK  ----
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

# Control:
control42_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control42_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control42_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX43 ---- TANK 8  ----
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


# Control: 
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

control43_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control43_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX44 ---- COMPRESSOR  ----
# Process:
process44_adj = torch.tensor([
#1  2  3  4  5  
[0, 1, 0, 0, 0],  #1 Splitter 
[0, 0, 1, 0, 0],  #2 Compressor
[0, 0, 0, 1, 0],  #3 HX
[0, 0, 0, 0, 1],  #4 Splitter
[1, 0, 0, 0, 0],  #5 Valve
], dtype=torch.float32)

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

# Control: 
control45_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX46 ---- INVENTORY CONTROL ----
# Process:
process46_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 1],  #2 Tank
[0, 0, 0],  #3 Valve
], dtype=torch.float32)


# Control: 
control46_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control46_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

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

# Control: 
control47_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control47_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control47_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control47_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)


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

# Control: 
control48_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control48_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control48_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

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

# Control: 
control49_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control49_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control49_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control49_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)




#------------------------------------------------------------------
# EX50 ---- INVENTORY CONTROL 5 ----
# Process:
process50_adj = torch.tensor([
#1  2  3  4  5  6  7  
[0, 1, 0, 0, 0, 0, 0],  #1 Valve
[0, 0, 1, 0, 0, 0, 0],  #2 Tank
[0, 0, 0, 1, 0, 0, 0],  #3 Valve
[0, 0, 0, 0, 1, 0, 0],  #4 Tank
[0, 0, 0, 0, 0, 1, 0],  #5 Valve
[0, 0, 0, 0, 0, 0, 1],  #6 Tank
[0, 0, 0, 0, 0, 0, 0],  #7 Valve
], dtype=torch.float32)

# Control: 
control50_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control50_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)

control50_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 IC
[0, 0, 0, 0],  #3 z
[0, 1, 0, 0],  #4 IT
], dtype=torch.float32)



#------------------------------------------------------------------
# EX51 ---- AIR + FUEL MIX  ----
# Process:
process51_adj = torch.tensor([
#1  2  3  
[0, 0, 0],  #1 Valve
[0, 0, 0],  #2 Splitter
[0, 0, 0],  #3 Valve
], dtype=torch.float32)

# Control: 
control51_1_adj = torch.tensor([
#1  2  3  4  5  6  7  8  9  10 11 12 13  
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

 
#------------------------------------------------------------------
# EX52 ---- JACKETED CSTR 6  ----
# Process:
process52_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 Jacket
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 CSTR
], dtype=torch.float32)


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


#------------------------------------------------------------------
# EX53 ---- MIXER  ----
# Process:
process53_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 Mixer
[0, 1, 0],  #3 Compressor
], dtype=torch.float32)


# Control: 
control53_1_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 CC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 z
[0, 1, 0, 0, 0, 0],  #5 CT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)

control53_2_adj = torch.tensor([
#1  2  3  4  5  6  
[0, 1, 0, 0, 0, 0],  #1 SP
[0, 0, 1, 0, 0, 0],  #2 FC
[0, 0, 0, 1, 0, 0],  #3 FC
[0, 0, 0, 0, 0, 0],  #4 w
[0, 1, 0, 0, 0, 0],  #5 FT
[0, 0, 1, 0, 0, 0],  #6 FT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX54 ---- MIXER 2 ----
# Process:
process54_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 Mixer
[0, 1, 0],  #3 Compressor
], dtype=torch.float32)

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



#------------------------------------------------------------------
# EX55 ---- MIXER 3 ----
# Process:
process55_adj = torch.tensor([
#1  2  3  
[0, 1, 0],  #1 Valve
[0, 0, 0],  #2 Mixer
[0, 1, 0],  #3 Compressor
], dtype=torch.float32)

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

# Control: 
control56_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

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

control56_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)


control56_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control56_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)



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

# Control: 
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

control57_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control57_4_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)

control57_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 LC
[0, 0, 0, 0],  #3 z 
[0, 1, 0, 0],  #4 LT
], dtype=torch.float32)


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
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  #10 Liquid reciever
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  #11 Splitter
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #12 Valve
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #13 Valve
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #14 Compressor
], dtype=torch.float32)


# Control: 
control58_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z 
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)

control58_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 w 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

control58_3_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z 
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)

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

control58_5_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 PC
[0, 0, 0, 0],  #3 w 
[0, 1, 0, 0],  #4 PT
], dtype=torch.float32)

#------------------------------------------------------------------
# EX59 ---- HEAT EXCHANGER 2 ----
# Process:
process59_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)

#Control:
control59_1_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 FC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 FT
], dtype=torch.float32)

control59_2_adj = torch.tensor([
#1  2  3  4  
[0, 1, 0, 0],  #1 SP
[0, 0, 1, 0],  #2 TC
[0, 0, 0, 1],  #3 z
[0, 1, 0, 0],  #4 TT
], dtype=torch.float32)


#------------------------------------------------------------------
# EX60 ---- HEAT EXCHANGER 3 ----
# Process:
process60_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)

#Control:
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


#------------------------------------------------------------------
# EX61 ---- HEAT EXCHANGER 4 ----
# Process:
process61_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)


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



#------------------------------------------------------------------
# EX62 ---- HEAT EXCHANGER 5 ----
# Process:
process62_adj = torch.tensor([
#1  2  3  
[0, 1, 1],  #1 HX
[0, 0, 0],  #2 Valve
[0, 0, 0],  #3 Valve
], dtype=torch.float32)


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



#------------------------------------------------------------------
# EX63 ---- TANK 9 ----
# Process:
process63_adj = torch.tensor([
#1  2  3  
[0, 1],  #1 Valve
[0, 0],  #2 Tank
], dtype=torch.float32)


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



#------------------------------------------------------------------
# EX64 ---- TURBINE ----
process64_adj = torch.tensor([
# Process:
#1  2  3  4  5  6  
[0, 1, 0, 0, 1, 0],  #1 Splitter 
[0, 0, 1, 0, 0, 0],  #2 Turbine
[0, 0, 0, 1, 0, 0],  #3 Valve
[0, 0, 0, 0, 0, 0],  #4 Splitter
[0, 0, 0, 0, 0, 1],  #5 Splitter
[0, 0, 0, 0, 0, 0],  #6 Valve
], dtype=torch.float32)

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

