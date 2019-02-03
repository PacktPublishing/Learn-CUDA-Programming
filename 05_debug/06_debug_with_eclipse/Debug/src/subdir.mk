################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/simple_sgemm.cu 

CU_DEPS += \
./src/simple_sgemm.d 

OBJS += \
./src/simple_sgemm.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-10.0/bin/nvcc -G -g -O0 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-10.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_70,code=compute_70 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


