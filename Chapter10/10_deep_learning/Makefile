# Project folders that contain CUDA receipts
PROJECTS ?= $(shell find \
	$(shell ls -d */) \
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

.PHONY: clean
clean: tidy $(addsuffix .ph_clean,$(PROJECTS))

test:
	echo $(DIRECTORY)
