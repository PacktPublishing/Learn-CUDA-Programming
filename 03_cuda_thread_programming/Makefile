# Project folders that contain CUDA receipts
PROJECTS ?= $(shell find \
 01_performance_limiter \
 02_maximizing_cuda_occupancy \
 03_warp_and_thread_block \
 04_cuda_threads_synchronization \
 05_warp_divergence \
 06_limiter_balancing \
 07_cooperative_group \
 08_warp_synchronous_programming \
 09_loop_unrolling \
 10_atomic_operation \
 11_mixed_precision_operation \
 -name Makefile)


%.ph_build:
	+@$(MAKE) -C $(dir $*) $(MAKECMDGOALS)

%.ph_clean:
	+@$(MAKE) -C $(dir $*) clean $(USE_DEVICE)

all:  $(addsuffix .ph_build,$(PROJECTS))
	@echo "Finished building CUDA Receipts"

build: $(addsuffix .ph_build,$(PROJECTS))

tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f
	@find * | egrep "nvvp" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))

test:
	echo $(DIRECTORY)
