################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include src/subdir.mk

OS_SUFFIX := $(subst Linux,linux,$(subst Darwin/x86_64,darwin,$(shell uname -s)/$(shell uname -m)))

-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CU_DEPS)),)
-include $(CU_DEPS)
endif
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# OpenGL+GLUT OS-specific define
ifeq ($(shell uname -s),Darwin)
GLUT_LIBS := -Xlinker -framework -Xlinker GLUT -Xlinker -framework -Xlinker OpenGL
else
GLUT_LIBS := -lGL -lGLU -lglut 
endif


# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: 06_debug_with_eclipse

# Tool invocations
06_debug_with_eclipse: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC Linker'
	/usr/local/cuda-10.0/bin/nvcc --cudart static --relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_70,code=compute_70 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -link -o  "06_debug_with_eclipse" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CU_DEPS)$(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) 06_debug_with_eclipse
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
