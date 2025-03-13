!
!   Arm Performance Libraries version 24.10
!   SPDX-FileCopyrightText: Copyright 2015-2024 Arm Limited and/or its affiliates
!   SPDX-FileCopyrightText: Copyright 2015-2024 NAG
!
module armpl_kinds

integer, parameter :: armpl_i4 = selected_int_kind(8)
integer, parameter :: armpl_i8 = selected_int_kind(10)
integer, parameter :: armpl_r32 = selected_real_kind(6, 37)
integer, parameter :: armpl_r64 = selected_real_kind(15, 307)

end module armpl_kinds
